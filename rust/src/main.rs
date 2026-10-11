//! dit-rs: run the trained DiT without Python.
//!
//!   dit-rs parity model.safetensors fixtures.safetensors   # compare against PyTorch outputs
//!   dit-rs sample model.safetensors out.png [cfg] [steps] [seed]   # 10x10 class grid

mod model;

use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use model::{sample_heun, Config, DiT};

// mirrored from export.py
const PARITY_STEPS: usize = 10;
const PARITY_CFG: f64 = 2.0;
const FWD_TOL: f32 = 1e-4;
const SAMPLE_TOL: f32 = 1e-3;

fn load(weights: &str, dev: &Device) -> Result<DiT> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, dev)? };
    DiT::new(Config::dit_s(), vb)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

fn parity(weights: &str, fixtures: &str, dev: &Device) -> Result<()> {
    let model = load(weights, dev)?;
    let fx = candle_core::safetensors::load(fixtures, dev)?;
    let get = |k: &str| fx.get(k).with_context(|| format!("fixture {k} missing"));

    let t: Vec<f32> = get("fwd.t")?.to_vec1()?;
    let v = model.forward(get("fwd.x")?, &t, &get("fwd.y")?.to_dtype(DType::U32)?)?;
    let d_fwd = max_abs_diff(&v, get("fwd.v")?)?;
    println!("forward       max |rust - torch| = {d_fwd:.2e}  (tol {FWD_TOL:.0e})");

    let (noise, y) = (get("smp.noise")?, get("smp.y")?.to_dtype(DType::U32)?);
    let out = sample_heun(&model, noise, &y, PARITY_STEPS, PARITY_CFG)?;
    let d_smp = max_abs_diff(&out, get("smp.out")?)?;
    println!("heun sampler  max |rust - torch| = {d_smp:.2e}  (tol {SAMPLE_TOL:.0e}, {PARITY_STEPS} steps, cfg {PARITY_CFG})");

    // timed second run: excludes one-off costs (CUDA kernel loading); to_scalar forces completion
    let start = Instant::now();
    sample_heun(&model, noise, &y, PARITY_STEPS, PARITY_CFG)?.sum_all()?.to_scalar::<f32>()?;
    println!("sampler time  {:.2}s on {:?}", start.elapsed().as_secs_f32(), dev);

    if d_fwd > FWD_TOL || d_smp > SAMPLE_TOL {
        bail!("parity FAILED");
    }
    println!("parity OK");
    Ok(())
}

/// Seeded N(0, 1) samples: splitmix64 + Box-Muller. Candle's CPU RNG can't be seeded.
fn gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (1.0 - uniform(), uniform()); // u1 in (0, 1] so ln is finite
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

/// 10 samples per class, one row per class, 2x nearest-neighbour upscale, 2px black padding.
fn sample_grid(weights: &str, out: &str, cfg: f64, steps: usize, seed: u64, dev: &Device) -> Result<()> {
    let model = load(weights, dev)?;
    let labels: Vec<u32> = (0..10).flat_map(|c| std::iter::repeat(c).take(10)).collect();
    let y = Tensor::new(labels.as_slice(), dev)?;
    let noise = Tensor::from_vec(gaussian(100 * 3 * 32 * 32, seed), (100, 3, 32, 32), dev)?;

    let start = Instant::now();
    let x = sample_heun(&model, &noise, &y, steps, cfg)?;
    // [-1, 1] -> u8, (100, 3, 32, 32) -> (100, 32, 32, 3); to_vec1 also waits for the device to finish
    let px: Vec<u8> = ((x.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .round()?
        .permute((0, 2, 3, 1))?
        .flatten_all()?
        .to_dtype(DType::U8)?
        .to_vec1()?;
    println!("sampled 100 images in {:.1}s ({steps} Heun steps, cfg {cfg}, {dev:?})", start.elapsed().as_secs_f32());

    let (s, pad, up) = (32usize, 2usize, 2usize);
    let cell = s * up + pad;
    let side = 10 * cell + pad;
    let mut img = image::RgbImage::new(side as u32, side as u32);
    for i in 0..100 {
        let (row, col) = (i / 10, i % 10);
        for yy in 0..s * up {
            for xx in 0..s * up {
                let o = ((i * s + yy / up) * s + xx / up) * 3;
                let (gx, gy) = (pad + col * cell + xx, pad + row * cell + yy);
                img.put_pixel(gx as u32, gy as u32, image::Rgb([px[o], px[o + 1], px[o + 2]]));
            }
        }
    }
    img.save(out)?;
    println!("wrote {out}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let dev = Device::Cpu;
    match args.get(1).map(String::as_str) {
        Some("parity") if args.len() == 4 => parity(&args[2], &args[3], &dev),
        Some("sample") if args.len() >= 4 => {
            let cfg = args.get(4).map_or(Ok(2.0), |s| s.parse())?;
            let steps = args.get(5).map_or(Ok(50), |s| s.parse())?;
            let seed = args.get(6).map_or(Ok(0), |s| s.parse())?;
            sample_grid(&args[2], &args[3], cfg, steps, seed, &dev)
        }
        _ => bail!(
            "usage:\n  dit-rs parity <model.safetensors> <fixtures.safetensors>\n  dit-rs sample <model.safetensors> <out.png> [cfg=2] [steps=50] [seed=0]"
        ),
    }
}
