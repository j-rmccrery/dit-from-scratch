"""Rectified-flow training objective and ODE samplers.

Convention: t=0 is data, t=1 is noise.  x_t = (1 - t) x_0 + t eps,  target velocity v = eps - x_0.
Sampling integrates dx/dt = v from t=1 down to t=0.
"""
import torch
import torch.nn.functional as F


def flow_loss(model, x, y, p_uncond=0.1):
    B = x.shape[0]
    t = torch.sigmoid(torch.randn(B, device=x.device))  # logit-normal t, as in SD3: more weight mid-trajectory
    eps = torch.randn_like(x)
    tt = t.view(B, 1, 1, 1)
    xt = (1 - tt) * x + tt * eps
    drop = torch.rand(B, device=x.device) < p_uncond  # label dropout teaches the unconditional model for CFG
    y = torch.where(drop, torch.full_like(y, model.num_classes), y)
    v = model(xt, t, y)
    return F.mse_loss(v.float(), (eps - x).float())


def _velocity(model, x, t, y, cfg_scale):
    tb = t.expand(x.shape[0])
    v = model(x, tb, y) if cfg_scale == 1.0 else model.forward_with_cfg(x, tb, y, cfg_scale)
    return v.float()


@torch.no_grad()
def sample(model, y, shape, steps=50, cfg_scale=1.0, method="heun", generator=None, return_traj=False):
    """Euler or Heun (2nd-order) integration from noise to data. Returns final x, or the full trajectory."""
    x = torch.randn(shape, device=y.device, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=y.device)
    traj = [x]
    for i in range(steps):
        t, t_next = ts[i], ts[i + 1]
        dt = t_next - t  # negative
        v = _velocity(model, x, t, y, cfg_scale)
        x_next = x + dt * v
        if method == "heun" and t_next > 0:
            v_next = _velocity(model, x_next, t_next, y, cfg_scale)
            x_next = x + dt * 0.5 * (v + v_next)
        x = x_next
        traj.append(x)
    return traj if return_traj else x
