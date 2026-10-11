# dit-from-scratch

A class-conditional **Diffusion Transformer** (DiT, [Peebles & Xie 2023](https://arxiv.org/abs/2212.09748)) trained with **rectified flow matching**, written from tensor ops up in PyTorch. Same model family as Stable Diffusion 3 and Flux, sized for CIFAR-10 and trained in about 3 hours on one consumer GPU.

![denoising: noise to image, one per class](assets/denoise.gif)

## Samples

DiT-S/2 (32.5M params), 100k steps, EMA weights, Heun sampler with 50 steps, classifier-free guidance 2.0. One row per class: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

![10 samples per class](assets/grid.png)

**Guidance sweep.** Same starting noise in every row; guidance scale 1.0, 1.5, 2.0, 3.0, 4.0 top to bottom. Higher guidance trades diversity for class fidelity and contrast.

![CFG sweep](assets/cfg_sweep.png)

## What's hand-written

| Piece | Where | Verified by |
|---|---|---|
| Patchify / unpatchify via `reshape` + `permute` | `model.py` | Exact round-trip; patch embedding proven equal to a strided `Conv2d` |
| Multi-head attention via `einsum` | `model.py` | Matches `F.scaled_dot_product_attention` to 1e-5, op-level and whole-model |
| adaLN-Zero conditioning (timestep + class modulate every block) | `model.py` | Model outputs exactly zero at init, every block starts as identity |
| Fused modulated LayerNorm with a **hand-derived backward** (`torch.autograd.Function`) | `ops.py` | `torch.autograd.gradcheck` in float64; matches plain PyTorch forward |
| Rectified flow: logit-normal timesteps, label dropout, Euler + Heun ODE samplers, CFG | `flow.py` | Overfit check: 16 images memorized (nearest-neighbour MSE 0.014 vs 0.235 for unrelated images) |
| GPU-resident data pipeline (no DataLoader), `foreach` EMA, bf16 autocast | `train.py` | Full run below |

The backward of `y = LN(x)·(1 + scale) + shift`, with `x̂ = (x − μ)·rstd` and `g = dy·(1 + scale)`:

```
d_shift = Σ_tokens dy
d_scale = Σ_tokens dy · x̂
dx      = rstd · (g − mean_D(g) − x̂ · mean_D(g · x̂))
```

## Training

![loss curve](assets/loss.png)

| | |
|---|---|
| Model | DiT-S/2: 12 blocks, width 384, 6 heads, 2×2 patches → 256 tokens |
| Data | CIFAR-10 train (50k), random horizontal flips |
| Optimizer | AdamW, lr 1e-4, 1k-step warmup, no weight decay, grad clip 1.0 |
| Batch / steps | 128 / 100k (≈256 epochs) |
| Precision | bf16 autocast, `torch.compile`, SDPA attention, custom norm op |
| Hardware / time | RTX 4070 Ti SUPER 16 GB, 9.4 steps/s, ≈3 h |
| Final loss | 0.136 |

## Benchmarks

RTX 4070 Ti SUPER, torch 2.6.0+cu124, Windows (`triton-windows` for `torch.compile`). Reproduce with `python bench.py`.

### Attention, forward + backward (B=8, heads=6, head_dim=64, bf16)

| tokens | einsum ms | SDPA ms | einsum MiB | SDPA MiB |
|---|---|---|---|---|
| 256 | 0.51 | 0.20 | 48 | 6 |
| 1024 | 6.74 | 1.16 | 768 | 24 |
| 4096 | 119.82 | 17.29 | 12288 | 98 |

The `einsum` path materializes the full N×N score matrix, so its memory grows quadratically: 12 GiB at 4096 tokens against 98 MiB for the fused kernel, which tiles the computation and never stores it.

### Modulated LayerNorm, forward + backward (128×256×384, bf16 inputs)

| implementation | ms | peak MiB |
|---|---|---|
| plain PyTorch (autograd) | 0.63 | 96 |
| custom autograd.Function | 3.01 | 312 |
| plain PyTorch, compiled | 0.32 | 48 |
| custom, compiled | 0.43 | 48 |

Eager, the custom op loses: it runs as ~10 separate kernels and upcasts bf16 inputs to fp32 for its statistics, where `F.layer_norm` is a single fused kernel. Compiled, Inductor fuses the hand-written forward and backward into the same footprint as the built-in.

### DiT-S/2 training step (batch 128, bf16 autocast)

| compiled | attention | norm | steps/s | peak GiB |
|---|---|---|---|---|
| no | einsum | custom | 3.87 | 9.8 |
| no | einsum | plain | 4.41 | 10.9 |
| no | SDPA | custom | 5.47 | 6.5 |
| no | SDPA | plain | 6.59 | 7.5 |
| yes | einsum | custom | 7.95 | 8.5 |
| yes | einsum | plain | 7.96 | 8.5 |
| yes | SDPA | custom | 9.37 | 6.3 |
| yes | SDPA | plain | 9.35 | 6.3 |

In the real model the residual stream is fp32 under autocast, and there the eager custom op uses **1 GiB less** than plain PyTorch: it saves only `x̂`, `rstd` and `scale` for backward, where autograd keeps every intermediate of the LayerNorm-then-modulate chain. Compiled, the two are identical in speed and memory, which is how the final model was trained: hand-written op, no speed penalty.

## Usage

```bash
pip install -r requirements.txt          # + triton-windows on Windows for --compile
python test_core.py                      # correctness checks, CPU, seconds

# sanity check: memorize 16 images (~1 min)
python train.py --overfit 16 --model xs --steps 2000 --lr 3e-4 --sample-every 500 --out runs/overfit

# full run (~3 h on a 4070 Ti SUPER)
python train.py --model s --steps 100000 --sdpa --compile --out runs/s2

python sample.py --ckpt runs/s2/ckpt.pt --out assets
python bench.py
```

## Layout

| File | Contents |
|---|---|
| `model.py` | DiT: patch embedding, 2D sin-cos positions, attention, adaLN-Zero blocks, CFG forward |
| `ops.py` | Custom autograd op + plain-PyTorch reference |
| `flow.py` | Rectified-flow loss and samplers |
| `train.py` | Training loop, overfit mode, checkpoint/resume |
| `sample.py` | Sample grid, guidance sweep, denoising GIF, loss plot |
| `bench.py` | The benchmark tables above |
| `test_core.py` | Correctness checks |

## Roadmap

- [x] Model, custom op, tests
- [x] Overfit sanity check
- [x] Full CIFAR-10 training run
- [x] Samples, denoising GIF, benchmarks
- [ ] Rust inference port (Candle) with numerical parity against PyTorch
