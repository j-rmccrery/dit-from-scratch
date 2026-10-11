# dit-from-scratch

A class-conditional Diffusion Transformer (DiT) trained with rectified flow matching, written from tensor ops up in PyTorch. Trains on CIFAR-10 (32×32) on a single consumer GPU.

> Status: milestone 1 of 5 — model, custom op, tests. No trained weights yet.

## What's hand-written

- **Patchify / unpatchify** — reshape/permute only; proven equivalent to a strided `Conv2d`.
- **Attention** — `einsum` implementation, verified against `F.scaled_dot_product_attention`.
- **adaLN-Zero conditioning** — timestep + class embeddings modulate every block; zero init makes each block start as identity.
- **Fused modulated LayerNorm** — custom `torch.autograd.Function` with a hand-derived backward, verified by `gradcheck`.
- **Rectified flow** — logit-normal timestep sampling, Euler and Heun ODE samplers, classifier-free guidance.

## Layout

| File | Contents |
|---|---|
| `model.py` | DiT: patch embedding, 2D sin-cos positions, attention, adaLN-Zero blocks |
| `ops.py` | Custom autograd op + plain-PyTorch reference |
| `flow.py` | Training loss and samplers |
| `test_core.py` | Correctness checks (`python test_core.py`) |

## Roadmap

1. Model, custom op, tests ← here
2. Overfit sanity check
3. Full CIFAR-10 training run
4. Samples, denoising GIF, benchmarks, writeup
5. Rust inference port (Candle) with numerical parity against PyTorch
