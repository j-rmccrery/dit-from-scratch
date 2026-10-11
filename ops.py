"""Custom autograd op: fused LayerNorm + adaLN modulation with a hand-derived backward."""
import torch
import torch.nn.functional as F


class ModulatedLayerNorm(torch.autograd.Function):
    """y = LayerNorm(x) * (1 + scale) + shift   (LayerNorm has no affine params).

    x: (B, N, D); shift, scale: (B, 1, D), broadcast over the N tokens.
    Statistics are computed in fp32 (fp64 under gradcheck) regardless of input dtype.

    Backward, with xhat = (x - mu) * rstd and g = dy * (1 + scale):
        d_shift = sum_N dy
        d_scale = sum_N dy * xhat
        dx      = rstd * (g - mean_D(g) - xhat * mean_D(g * xhat))
    """

    @staticmethod
    def forward(ctx, x, shift, scale, eps):
        dt = torch.promote_types(x.dtype, torch.float32)
        xf = x.to(dt)
        xc = xf - xf.mean(-1, keepdim=True)
        rstd = torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)
        xhat = xc * rstd
        s = scale.to(dt)
        y = xhat * (1 + s) + shift.to(dt)
        ctx.save_for_backward(xhat, rstd, s)
        ctx.dtypes = (x.dtype, shift.dtype, scale.dtype)
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, dy):
        xhat, rstd, s = ctx.saved_tensors
        x_dt, shift_dt, scale_dt = ctx.dtypes
        dy = dy.to(xhat.dtype)
        d_shift = dy.sum(1, keepdim=True)
        d_scale = (dy * xhat).sum(1, keepdim=True)
        g = dy * (1 + s)
        dx = rstd * (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True))
        return dx.to(x_dt), d_shift.to(shift_dt), d_scale.to(scale_dt), None


def modulated_layernorm(x, shift, scale, eps=1e-6):
    return ModulatedLayerNorm.apply(x, shift, scale, eps)


def modulated_layernorm_ref(x, shift, scale, eps=1e-6):
    """Plain-PyTorch reference; autograd traces this one."""
    return F.layer_norm(x, x.shape[-1:], eps=eps) * (1 + scale) + shift
