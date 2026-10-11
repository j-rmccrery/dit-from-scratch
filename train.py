"""Train DiT with rectified flow on CIFAR-10.

  Overfit sanity check:  python train.py --overfit 16 --model xs --steps 2000 --lr 3e-4 --sample-every 500 --out runs/overfit
  Full run:              python train.py --model s --out runs/s2

The whole dataset lives on the GPU as uint8 (~150 MB); batches are sampled and flipped there,
so there is no DataLoader and no worker processes (which are slow to spawn on Windows).
"""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import torch
import torchvision
from torchvision.utils import save_image

from flow import flow_loss, sample
from model import DiT

PRESETS = {
    "xs": dict(dim=256, depth=6, heads=4),   # ~10M params, for quick checks
    "s": dict(dim=384, depth=12, heads=6),   # DiT-S/2, ~33M params
}


def load_cifar(root, device):
    ds = torchvision.datasets.CIFAR10(root, train=True, download=True)
    x = torch.from_numpy(ds.data).to(device).permute(0, 3, 1, 2).contiguous()  # (50000, 3, 32, 32) uint8
    y = torch.tensor(ds.targets, device=device)
    return x, y


def to_unit(x_uint8):
    """uint8 [0, 255] -> float [-1, 1]."""
    return x_uint8.float() / 127.5 - 1.0


def get_batch(x, y, batch, gen):
    idx = torch.randint(0, x.shape[0], (batch,), device=x.device, generator=gen)
    xb = to_unit(x[idx])
    flip = torch.rand(batch, device=x.device, generator=gen) < 0.5
    xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
    return xb, y[idx]


@torch.no_grad()
def ema_update(ema, model, decay):
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1.0 - decay)


def nn_mse(a, refs):
    """Mean over `a` of the per-pixel MSE to its nearest neighbour in `refs`."""
    d = torch.cdist(a.flatten(1).float(), refs.flatten(1).float()).pow(2) / a[0].numel()
    return d.min(1).values.mean().item()


@torch.no_grad()
def make_samples(ema, labels, args, seed=0):
    g = torch.Generator(device=labels.device).manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return sample(ema, labels, (len(labels), 3, 32, 32), steps=args.sample_steps,
                      cfg_scale=args.cfg, generator=g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=PRESETS, default="s")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema", type=float, default=0.9999)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--overfit", type=int, default=0, help="train on the first N images only (sanity check)")
    p.add_argument("--cfg", type=float, default=1.0, help="guidance scale for preview samples")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--sdpa", action="store_true", help="fused attention kernel instead of the einsum path")
    p.add_argument("--compile", action="store_true", help="torch.compile (needs a working Triton)")
    p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/default")
    args = p.parse_args()

    assert torch.cuda.is_available(), "training expects a CUDA GPU"
    dev = "cuda"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    # ---- data
    x_all, y_all = load_cifar(args.data, dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    if args.overfit:
        fixed_x, fixed_y = to_unit(x_all[:args.overfit]), y_all[:args.overfit]
        unrelated = to_unit(x_all[args.overfit:2 * args.overfit])  # baseline for the memorization metric
        save_image((fixed_x + 1) / 2, out / "targets.png", nrow=8)
        preview_labels = fixed_y
    else:
        preview_labels = torch.arange(10, device=dev).repeat_interleave(8)  # one row per class

    # ---- model
    model = DiT(img_size=32, patch=2, in_ch=3, num_classes=10, use_sdpa=args.sdpa, **PRESETS[args.model]).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    start = 0
    if args.resume and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed at step {start}")
    train_model = torch.compile(model) if args.compile else model
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "args": vars(args)}, out / "ckpt.pt")

    def preview(step):
        s = make_samples(ema, preview_labels, args)
        save_image((s + 1) / 2, out / f"samples_{step:07d}.png", nrow=8)
        if args.overfit:
            print(f"  memorization  nn-mse samples->targets {nn_mse(s, fixed_x):.4f}"
                  f"  (unrelated real images: {nn_mse(unrelated, fixed_x):.4f})")

    log_f = open(out / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    if start == 0:
        log.writerow(["step", "loss", "lr", "grad_norm", "it_per_s"])

    # ---- train
    model.train()
    loss_sum = torch.zeros((), device=dev)  # accumulate on GPU; sync only when logging
    t0 = time.perf_counter()
    for step in range(start, args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        xb, yb = (fixed_x, fixed_y) if args.overfit else get_batch(x_all, y_all, args.batch, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(train_model, xb, yb, args.p_uncond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, min(args.ema, (1 + step) / (10 + step)))  # short warmup so early EMA isn't stuck at init
        loss_sum += loss.detach()

        done = step + 1
        if done % args.log_every == 0:
            dt = time.perf_counter() - t0
            avg = (loss_sum / args.log_every).item()
            ips = args.log_every / dt
            print(f"step {done:7d}  loss {avg:.4f}  lr {lr:.2e}  gnorm {gnorm.item():.3f}  {ips:.1f} it/s")
            log.writerow([done, f"{avg:.5f}", f"{lr:.3e}", f"{gnorm.item():.4f}", f"{ips:.2f}"])
            log_f.flush()
            loss_sum.zero_()
            t0 = time.perf_counter()
        if done % args.sample_every == 0 or done == args.steps:
            preview(done)
        if done % args.ckpt_every == 0 or done == args.steps:
            save_ckpt(done)

    log_f.close()


if __name__ == "__main__":
    main()
