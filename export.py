"""Export EMA weights and parity fixtures for the Rust port.

  python export.py --ckpt runs/s2/ckpt.pt --out rust/

Writes model.safetensors (fp32 EMA state dict) and fixtures.safetensors (inputs + PyTorch outputs, CPU fp32):
  fwd.*  one forward pass at mixed timesteps / classes
  smp.*  a 10-step Heun sample with CFG 2.0 from fixed noise
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

from flow import sample
from model import DiT
from train import PRESETS

SAMPLE_STEPS, SAMPLE_CFG = 10, 2.0  # mirrored in rust/src/main.rs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/s2/ckpt.pt")
    p.add_argument("--out", default="rust")
    args = p.parse_args()
    out = Path(args.out)

    ck = torch.load(args.ckpt, map_location="cpu")
    m = DiT(img_size=32, patch=2, in_ch=3, num_classes=10, **PRESETS[ck["args"]["model"]])
    m.load_state_dict(ck["ema"])
    m.eval()
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, out / "model.safetensors")

    with torch.no_grad():
        torch.manual_seed(0)
        fx = torch.randn(4, 3, 32, 32)
        ft = torch.tensor([0.1, 0.4, 0.7, 0.95])
        fy = torch.tensor([0, 3, 5, 9])
        fv = m(fx, ft, fy)

        sy = torch.tensor([1, 7])
        noise = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1))
        # sample() draws its starting noise from the generator, so the same seed reproduces `noise`
        sx = sample(m, sy, (2, 3, 32, 32), steps=SAMPLE_STEPS, cfg_scale=SAMPLE_CFG, method="heun",
                    generator=torch.Generator().manual_seed(1))

    save_file({"fwd.x": fx, "fwd.t": ft, "fwd.y": fy, "fwd.v": fv,
               "smp.noise": noise, "smp.y": sy, "smp.out": sx}, out / "fixtures.safetensors")
    print(f"exported step {ck['step']} EMA weights + fixtures to {out}/")


if __name__ == "__main__":
    main()
