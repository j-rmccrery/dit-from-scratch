"""FID of class-conditional samples vs CIFAR-10 train statistics, via clean-fid (pip install clean-fid).

  python fid.py --ckpt dit_s2_cifar10_ema.safetensors --num 50000 --cfg 1.5

Labels are drawn uniformly at random; samples are quantized to uint8 levels like real images.
"""
import argparse
import time

import numpy as np
import torch
from cleanfid import fid
from scipy import linalg

from flow import sample
from sample import load_ema


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2)). Replaces clean-fid 0.1.35's version,
    which passes `disp=` to scipy.linalg.sqrtm (removed in recent SciPy)."""
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


fid.frechet_distance = frechet_distance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/s2/ckpt.pt")
    p.add_argument("--num", type=int, default=50_000)
    p.add_argument("--cfg", type=float, nargs="+", default=[1.5])
    p.add_argument("--steps", type=int, default=50, help="Heun steps (2*steps-1 model evaluations)")
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    m, step = load_ema(args.ckpt)
    m.compile()  # in-place, so forward_with_cfg's self(...) calls hit the compiled graph
    for cfg in args.cfg:
        g = torch.Generator(device="cuda").manual_seed(args.seed)

        def generator(z):
            n = z.shape[0]
            y = torch.randint(0, 10, (n,), device="cuda", generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = sample(m, y, (n, 3, 32, 32), steps=args.steps, cfg_scale=cfg, generator=g)
            return ((x.clamp(-1, 1) + 1) * 127.5).round()

        t0 = time.perf_counter()
        score = fid.compute_fid(gen=generator, dataset_name="cifar10", dataset_res=32, dataset_split="train",
                                mode="clean", num_gen=args.num, batch_size=args.batch, z_dim=1, verbose=False)
        mins = (time.perf_counter() - t0) / 60
        print(f"FID-{args.num // 1000}k  cfg {cfg}  heun {args.steps}  weights {step}:  {score:.2f}  ({mins:.1f} min)",
              flush=True)


if __name__ == "__main__":
    main()
