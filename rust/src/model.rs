//! DiT forward pass and rectified-flow sampler, mirroring model.py / flow.py op for op.

use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

pub struct Config {
    pub img: usize,
    pub patch: usize,
    pub in_ch: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub num_classes: usize,
}

impl Config {
    /// DiT-S/2 at 32x32, as trained.
    pub fn dit_s() -> Self {
        Self { img: 32, patch: 2, in_ch: 3, dim: 384, depth: 12, heads: 6, num_classes: 10 }
    }
}

const EPS: f64 = 1e-6;
const FREQ_DIM: usize = 256;

// ---------------------------------------------------------------- tensor helpers

/// (B, C, H, W) -> (B, N, p*p*C)
pub fn patchify(x: &Tensor, p: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let (gh, gw) = (h / p, w / p);
    let x = x.reshape((b, c, gh, p, gw, p))?.permute((0, 2, 4, 3, 5, 1))?.contiguous()?;
    Ok(x.reshape((b, gh * gw, p * p * c))?)
}

/// (B, N, p*p*C) -> (B, C, H, W)
pub fn unpatchify(x: &Tensor, p: usize, c: usize, h: usize, w: usize) -> Result<Tensor> {
    let b = x.dim(0)?;
    let (gh, gw) = (h / p, w / p);
    let x = x.reshape((b, gh, gw, p, p, c))?.permute((0, 5, 1, 3, 2, 4))?.contiguous()?;
    Ok(x.reshape((b, c, h, w))?)
}

/// LayerNorm (no affine) then x * (1 + scale) + shift. shift/scale: (B, 1, D).
fn modulated_layernorm(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let xc = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let xhat = xc.broadcast_div(&(var + EPS)?.sqrt()?)?;
    Ok(xhat.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)?)
}

/// Fixed 2D sin-cos table, (1, h*w, dim): [sin x, cos x, sin y, cos y] per position, row-major.
fn sincos_2d(h: usize, w: usize, dim: usize, dev: &Device) -> Result<Tensor> {
    let q = dim / 4;
    let omega: Vec<f32> = (0..q).map(|i| 1.0 / 10000f32.powf(i as f32 / q as f32)).collect();
    let trig: [fn(f32) -> f32; 2] = [f32::sin, f32::cos];
    let mut data = Vec::with_capacity(h * w * dim);
    for y in 0..h {
        for x in 0..w {
            for coord in [x, y] {
                for f in trig {
                    data.extend(omega.iter().map(|o| f(coord as f32 * o)));
                }
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, h * w, dim), dev)?)
}

/// t in [0, 1] -> (B, FREQ_DIM) features [cos, sin] of t*1000 at geometric frequencies.
fn timestep_embedding(t: &[f32], dev: &Device) -> Result<Tensor> {
    let half = FREQ_DIM / 2;
    let ln = 10000f32.ln();
    let mut data = Vec::with_capacity(t.len() * FREQ_DIM);
    for &ti in t {
        let args: Vec<f32> = (0..half).map(|i| ti * 1000.0 * (-ln * i as f32 / half as f32).exp()).collect();
        data.extend(args.iter().map(|a| a.cos()));
        data.extend(args.iter().map(|a| a.sin()));
    }
    Ok(Tensor::from_vec(data, (t.len(), FREQ_DIM), dev)?)
}

// ---------------------------------------------------------------- modules

struct Attention {
    qkv: Linear,
    proj: Linear,
    heads: usize,
}

impl Attention {
    fn new(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { qkv: linear(dim, 3 * dim, vb.pp("qkv"))?, proj: linear(dim, dim, vb.pp("proj"))?, heads })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        let hd = d / self.heads;
        // (B, N, 3D) -> (3, B, H, N, hd)
        let qkv = self.qkv.forward(x)?.reshape((b, n, 3, self.heads, hd))?.permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let s = (q.matmul(&k.t()?.contiguous()?)? * (hd as f64).powf(-0.5))?;
        let o = candle_nn::ops::softmax_last_dim(&s)?.matmul(&v)?;
        Ok(self.proj.forward(&o.transpose(1, 2)?.contiguous()?.reshape((b, n, d))?)?)
    }
}

struct Block {
    attn: Attention,
    fc1: Linear,
    fc2: Linear,
    ada: Linear,
}

impl Block {
    fn new(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(dim, heads, vb.pp("attn"))?,
            fc1: linear(dim, 4 * dim, vb.pp("mlp.0"))?,
            fc2: linear(4 * dim, dim, vb.pp("mlp.2"))?,
            ada: linear(dim, 6 * dim, vb.pp("ada.1"))?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // shift1, scale1, gate1, shift2, scale2, gate2, each (B, 1, D)
        let m = self.ada.forward(&c.silu()?)?.unsqueeze(1)?.chunk(6, D::Minus1)?;
        let h = self.attn.forward(&modulated_layernorm(x, &m[0], &m[1])?)?;
        let x = (x + m[2].broadcast_mul(&h)?)?;
        let h = modulated_layernorm(&x, &m[3], &m[4])?;
        let h = self.fc2.forward(&self.fc1.forward(&h)?.gelu()?)?; // candle's gelu is the tanh approximation
        Ok((&x + m[5].broadcast_mul(&h)?)?)
    }
}

pub struct DiT {
    cfg: Config,
    patch_embed: Linear,
    pos: Tensor,
    t_fc1: Linear,
    t_fc2: Linear,
    y_embed: Embedding,
    blocks: Vec<Block>,
    final_ada: Linear,
    final_linear: Linear,
}

impl DiT {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let (d, p, grid) = (cfg.dim, cfg.patch, cfg.img / cfg.patch);
        let blocks = (0..cfg.depth)
            .map(|i| Block::new(d, cfg.heads, vb.pp(format!("blocks.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed: linear(p * p * cfg.in_ch, d, vb.pp("patch_embed"))?,
            pos: sincos_2d(grid, grid, d, vb.device())?,
            t_fc1: linear(FREQ_DIM, d, vb.pp("t_embed.mlp.0"))?,
            t_fc2: linear(d, d, vb.pp("t_embed.mlp.2"))?,
            y_embed: embedding(cfg.num_classes + 1, d, vb.pp("y_embed"))?,
            blocks,
            final_ada: linear(d, 2 * d, vb.pp("final.ada.1"))?,
            final_linear: linear(d, p * p * cfg.in_ch, vb.pp("final.linear"))?,
            cfg,
        })
    }

    /// Velocity v(x_t, t, y). x: (B, C, H, W) f32, t: one value per sample, y: (B,) u32 labels.
    pub fn forward(&self, x: &Tensor, t: &[f32], y: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let mut h = self.patch_embed.forward(&patchify(x, c.patch)?)?.broadcast_add(&self.pos)?;
        let temb = timestep_embedding(t, x.device())?;
        let cond = (self.t_fc2.forward(&self.t_fc1.forward(&temb)?.silu()?)? + self.y_embed.forward(y)?)?;
        for block in &self.blocks {
            h = block.forward(&h, &cond)?;
        }
        let m = self.final_ada.forward(&cond.silu()?)?.unsqueeze(1)?.chunk(2, D::Minus1)?;
        let h = self.final_linear.forward(&modulated_layernorm(&h, &m[0], &m[1])?)?;
        unpatchify(&h, c.patch, c.in_ch, c.img, c.img)
    }

    /// Classifier-free guidance in one batched pass; label index `num_classes` is the null class.
    pub fn forward_cfg(&self, x: &Tensor, t: f32, y: &Tensor, scale: f64) -> Result<Tensor> {
        let b = x.dim(0)?;
        if scale == 1.0 {
            return self.forward(x, &vec![t; b], y);
        }
        let null = Tensor::full(self.cfg.num_classes as u32, b, x.device())?;
        let v = self.forward(&Tensor::cat(&[x, x], 0)?, &vec![t; 2 * b], &Tensor::cat(&[y, &null], 0)?)?;
        let (v_cond, v_uncond) = (v.narrow(0, 0, b)?, v.narrow(0, b, b)?);
        Ok((&v_uncond + ((v_cond - &v_uncond)? * scale)?)?)
    }
}

/// Heun integration of dx/dt = v from t=1 (noise) to t=0 (data), as flow.sample(method="heun").
pub fn sample_heun(model: &DiT, noise: &Tensor, y: &Tensor, steps: usize, cfg: f64) -> Result<Tensor> {
    let mut x = noise.clone();
    for i in 0..steps {
        let t = 1.0 - i as f32 / steps as f32;
        let t_next = 1.0 - (i + 1) as f32 / steps as f32;
        let dt = (t_next - t) as f64;
        let v = model.forward_cfg(&x, t, y, cfg)?;
        let mut x_next = (&x + (&v * dt)?)?;
        if t_next > 0.0 {
            let v_next = model.forward_cfg(&x_next, t_next, y, cfg)?;
            x_next = (&x + ((v + v_next)? * (0.5 * dt))?)?;
        }
        x = x_next;
    }
    Ok(x)
}
