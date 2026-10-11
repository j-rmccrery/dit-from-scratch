"""Benchmarks for the README. Run on an otherwise idle GPU:  python bench.py  (prints markdown tables).

1. Attention op: hand-written einsum vs fused SDPA, forward+backward, across sequence lengths.
2. Modulated LayerNorm op: custom autograd.Function vs plain PyTorch, eager and compiled.
3. Full DiT-S training step: every combination of the above, eager and compiled.
"""
import statistics
import time

import torch
import torch.nn.functional as F

import model as M
import ops
from flow import flow_loss

DEV = "cuda"
BF16 = torch.bfloat16


def cuda_time_ms(fn, warmup=5, iters=20):
    """Median wall time of fn() in ms, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def peak_mib(fn):
    """Extra GPU memory fn() needs at its peak, above what was already allocated."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def fwd_bwd(f, *inputs):
    """Closure running f forward and backward, clearing grads each call."""
    def run():
        for t in inputs:
            t.grad = None
        out = f(*inputs)
        out.backward(torch.ones_like(out))
    return run


def measure(run):
    try:
        return f"{cuda_time_ms(run):.2f}", f"{peak_mib(run):.0f}"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM", "OOM"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


# ---------------------------------------------------------------- 1. attention

def bench_attention(B=8, H=6, d=64, seq_lens=(256, 1024, 4096)):
    rows = []
    for N in seq_lens:
        q, k, v = (torch.randn(B, H, N, d, device=DEV, dtype=BF16, requires_grad=True) for _ in range(3))
        e_ms, e_mem = measure(fwd_bwd(M.einsum_attention, q, k, v))
        s_ms, s_mem = measure(fwd_bwd(F.scaled_dot_product_attention, q, k, v))
        rows.append([N, e_ms, s_ms, e_mem, s_mem])
        del q, k, v
    print(f"### Attention, forward + backward (B={B}, heads={H}, head_dim={d}, bf16)\n")
    table(["tokens", "einsum ms", "SDPA ms", "einsum MiB", "SDPA MiB"], rows)


# ---------------------------------------------------------------- 2. modulated layernorm

def bench_norm(B=128, N=256, D=384):
    x = torch.randn(B, N, D, device=DEV, dtype=BF16, requires_grad=True)
    shift = torch.randn(B, 1, D, device=DEV, dtype=BF16, requires_grad=True)
    scale = torch.randn(B, 1, D, device=DEV, dtype=BF16, requires_grad=True)
    variants = [
        ("plain PyTorch (autograd)", ops.modulated_layernorm_ref),
        ("custom autograd.Function", ops.modulated_layernorm),
        ("plain PyTorch, compiled", torch.compile(ops.modulated_layernorm_ref)),
        ("custom, compiled", torch.compile(ops.modulated_layernorm)),
    ]
    rows = [[name, *measure(fwd_bwd(f, x, shift, scale))] for name, f in variants]
    print(f"### Modulated LayerNorm, forward + backward ({B}x{N}x{D}, bf16)\n")
    table(["implementation", "ms", "peak MiB"], rows)


# ---------------------------------------------------------------- 3. training step

def bench_train_step(batch=128, warmup=10, steps=50):
    x = torch.randn(batch, 3, 32, 32, device=DEV)
    y = torch.randint(0, 10, (batch,), device=DEV)
    rows = []
    for compiled in (False, True):
        for sdpa in (False, True):
            for ref_norm in (False, True):
                # swap the norm used inside model.py's blocks for this config
                M.modulated_layernorm = ops.modulated_layernorm_ref if ref_norm else ops.modulated_layernorm
                torch._dynamo.reset()
                torch.manual_seed(0)
                m = M.dit_s(use_sdpa=sdpa).to(DEV)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
                fm = torch.compile(m) if compiled else m

                def step():
                    with torch.autocast("cuda", dtype=BF16):
                        loss = flow_loss(fm, x, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                try:
                    for _ in range(warmup):
                        step()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    for _ in range(steps):
                        step()
                    torch.cuda.synchronize()
                    ips = f"{steps / (time.perf_counter() - t0):.2f}"
                    mem = f"{torch.cuda.max_memory_allocated() / 2**30:.1f}"
                except torch.cuda.OutOfMemoryError:
                    ips = mem = "OOM"
                rows.append(["yes" if compiled else "no", "SDPA" if sdpa else "einsum",
                             "plain" if ref_norm else "custom", ips, mem])
                del m, opt, fm
                torch.cuda.empty_cache()
    M.modulated_layernorm = ops.modulated_layernorm
    print(f"### DiT-S/2 training step (batch {batch}, bf16 autocast)\n")
    table(["compiled", "attention", "norm", "steps/s", "peak GiB"], rows)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}\n")
    bench_attention()
    bench_norm()
    bench_train_step()
