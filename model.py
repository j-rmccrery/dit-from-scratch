"""Diffusion Transformer (DiT, Peebles & Xie 2023) written from tensor ops up."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import modulated_layernorm


# ---------------------------------------------------------------- patches

def patchify(x, p):
    """(B, C, H, W) -> (B, N, p*p*C), N = (H/p)*(W/p). Pure reshape/permute, no copy until the last reshape."""
    B, C, H, W = x.shape
    h, w = H // p, W // p
    x = x.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 3, 5, 1)  # B, h, w, p, p, C
    return x.reshape(B, h * w, p * p * C)


def unpatchify(x, p, C, H, W):
    """Exact inverse of patchify."""
    B = x.shape[0]
    h, w = H // p, W // p
    x = x.reshape(B, h, w, p, p, C)
    x = x.permute(0, 5, 1, 3, 2, 4)  # B, C, h, p, w, p
    return x.reshape(B, C, H, W)


# ---------------------------------------------------------------- embeddings

def sincos_2d(h, w, dim):
    """Fixed 2D sin-cos position table, (h*w, dim). Half the channels encode x, half encode y."""
    assert dim % 4 == 0
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4) / (dim // 4))
    x = xs.flatten()[:, None] * omega[None]
    y = ys.flatten()[:, None] * omega[None]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


def timestep_embedding(t, dim, max_period=10000):
    """t in [0, 1], shape (B,) -> (B, dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000)[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.freq_dim))


# ---------------------------------------------------------------- attention

def einsum_attention(q, k, v):
    """softmax(q k^T / sqrt(d)) v on (B, H, N, d) tensors. Softmax runs in fp32."""
    scale = q.shape[-1] ** -0.5
    s = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale
    a = s.float().softmax(dim=-1).to(q.dtype)
    return torch.einsum("bhnm,bhmd->bhnd", a, v)


class Attention(nn.Module):
    def __init__(self, dim, heads, use_sdpa=False):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.use_sdpa = use_sdpa  # False: hand-written einsum path; True: fused PyTorch kernel
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        H = self.heads
        # one matmul for q, k, v, then split heads: (B, N, 3D) -> (3, B, H, N, d)
        q, k, v = self.qkv(x).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v) if self.use_sdpa else einsum_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


# ---------------------------------------------------------------- blocks

class DiTBlock(nn.Module):
    """Transformer block conditioned by adaLN-Zero: c -> (shift, scale, gate) for attn and MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0, use_sdpa=False):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.attn = Attention(dim, heads, use_sdpa)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).unsqueeze(1).chunk(6, dim=-1)  # each (B, 1, D)
        x = x + gate1 * self.attn(modulated_layernorm(x, shift1, scale1))
        x = x + gate2 * self.mlp(modulated_layernorm(x, shift2, scale2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, patch * patch * out_ch)

    def forward(self, x, c):
        shift, scale = self.ada(c).unsqueeze(1).chunk(2, dim=-1)
        return self.linear(modulated_layernorm(x, shift, scale))


# ---------------------------------------------------------------- model

class DiT(nn.Module):
    """Predicts flow-matching velocity v(x_t, t, y). Label index `num_classes` is the null class for CFG."""

    def __init__(self, img_size=32, patch=2, in_ch=3, dim=384, depth=12, heads=6,
                 mlp_ratio=4.0, num_classes=10, use_sdpa=False):
        super().__init__()
        self.img_size, self.patch, self.in_ch = img_size, patch, in_ch
        self.num_classes = num_classes
        grid = img_size // patch

        self.patch_embed = nn.Linear(patch * patch * in_ch, dim)  # == Conv2d(k=p, stride=p) on patchified input
        self.register_buffer("pos", sincos_2d(grid, grid, dim)[None], persistent=False)
        self.t_embed = TimestepEmbedder(dim)
        self.y_embed = nn.Embedding(num_classes + 1, dim)
        self.blocks = nn.ModuleList(DiTBlock(dim, heads, mlp_ratio, use_sdpa) for _ in range(depth))
        self.final = FinalLayer(dim, patch, in_ch)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.y_embed.weight, std=0.02)
        for m in self.t_embed.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        # adaLN-Zero: every block starts as identity, the model starts by predicting 0
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight)
            nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final.ada[-1].weight)
        nn.init.zeros_(self.final.ada[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def forward(self, x, t, y):
        x = self.patch_embed(patchify(x, self.patch)) + self.pos
        c = self.t_embed(t) + self.y_embed(y)
        for block in self.blocks:
            x = block(x, c)
        x = self.final(x, c)
        return unpatchify(x, self.patch, self.in_ch, self.img_size, self.img_size)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Classifier-free guidance: one batched pass for conditional + unconditional."""
        null = torch.full_like(y, self.num_classes)
        v = self(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
        v_cond, v_uncond = v.chunk(2)
        return v_uncond + cfg_scale * (v_cond - v_uncond)


def dit_s(**kw):
    """DiT-S/2 at 32x32: ~33M params."""
    return DiT(dim=384, depth=12, heads=6, patch=2, **kw)
