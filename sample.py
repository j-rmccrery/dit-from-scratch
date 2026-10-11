"""Generate README assets from a trained checkpoint (EMA weights).

  python sample.py --ckpt runs/s2/ckpt.pt --out assets

Writes: grid.png (10 per class), cfg_sweep.png (guidance scales x classes, same noise),
denoise.gif (noise -> image, one per class), loss.png (training curve from log.csv).
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image

from flow import sample
from model import DiT
from train import PRESETS

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_ema(path):
    ck = torch.load(path, map_location="cuda")
    m = DiT(img_size=32, patch=2, in_ch=3, num_classes=10, use_sdpa=True, **PRESETS[ck["args"]["model"]]).cuda()
    m.load_state_dict(ck["ema"])
    return m.eval(), ck["step"]


def generate(m, labels, cfg, steps, seed, traj=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return sample(m, labels, (len(labels), 3, 32, 32), steps=steps, cfg_scale=cfg,
                      generator=g, return_traj=traj)


def to_display(x, upscale=2):
    """[-1, 1] model space -> [0, 1], nearest-neighbour upscaled so 32px images stay crisp."""
    return F.interpolate((x.clamp(-1, 1) + 1) / 2, scale_factor=upscale, mode="nearest")


def to_pil(grid):
    return Image.fromarray(grid.mul(255).round().byte().permute(1, 2, 0).cpu().numpy())


def plot_loss(log_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    loss = [float(r["loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.plot(steps, loss, lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("flow-matching loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/s2/ckpt.pt")
    p.add_argument("--out", default="assets")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m, step = load_ema(args.ckpt)
    print(f"loaded EMA weights from step {step}")
    per_class = torch.arange(10, device="cuda")

    # 10 samples per class, one row per class
    x = generate(m, per_class.repeat_interleave(10), args.cfg, args.steps, args.seed)
    save_image(to_display(x), out / "grid.png", nrow=10, padding=2)

    # guidance sweep: rows = cfg scale, columns = class, identical starting noise in every row
    scales = [1.0, 1.5, 2.0, 3.0, 4.0]
    sweep = torch.cat([generate(m, per_class, s, args.steps, args.seed + 1) for s in scales])
    save_image(to_display(sweep), out / "cfg_sweep.png", nrow=10, padding=2)

    # denoising trajectory, one image per class
    traj = generate(m, per_class, args.cfg, args.steps, args.seed + 2, traj=True)
    frames = [to_pil(make_grid(to_display(t, upscale=3), nrow=10, padding=2)) for t in traj]
    durations = [60] * (len(frames) - 1) + [2000]  # hold the final frame
    frames[0].save(out / "denoise.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)

    log = Path(args.ckpt).parent / "log.csv"
    if log.exists():
        plot_loss(log, out / "loss.png")
    print(f"wrote {', '.join(sorted(f.name for f in out.iterdir()))} to {out}/  (scales: {scales}; classes: {CLASSES})")


if __name__ == "__main__":
    main()
