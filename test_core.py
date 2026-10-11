"""Core correctness checks. Run: `python test_core.py` or `pytest test_core.py`. CPU, a few seconds."""
import torch
import torch.nn.functional as F

from flow import flow_loss, sample
from model import DiT, Attention, einsum_attention, patchify, sincos_2d, unpatchify
from ops import modulated_layernorm, modulated_layernorm_ref

torch.manual_seed(0)


def tiny(**kw):
    return DiT(img_size=8, patch=2, in_ch=3, dim=32, depth=2, heads=4, num_classes=10, **kw)


def perturb(model, std=0.05):
    """Break the zero init so outputs are non-trivial."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * std)
    return model


def test_patchify_roundtrip_and_layout():
    x = torch.randn(2, 3, 8, 8)
    p = patchify(x, 2)
    assert p.shape == (2, 16, 12)
    assert torch.equal(unpatchify(p, 2, 3, 8, 8), x)
    # token 1 is the patch at row 0, col 1, flattened as (py, px, C)
    assert torch.equal(p[:, 1], x[:, :, 0:2, 2:4].permute(0, 2, 3, 1).reshape(2, -1))


def test_patch_embed_equals_strided_conv():
    x = torch.randn(2, 3, 8, 8)
    lin = torch.nn.Linear(12, 16)
    conv = torch.nn.Conv2d(3, 16, kernel_size=2, stride=2)
    with torch.no_grad():  # linear weight (out, py*px*C) -> conv weight (out, C, py, px)
        conv.weight.copy_(lin.weight.reshape(16, 2, 2, 3).permute(0, 3, 1, 2))
        conv.bias.copy_(lin.bias)
    a = lin(patchify(x, 2))
    b = conv(x).flatten(2).transpose(1, 2)
    assert torch.allclose(a, b, atol=1e-5)


def test_sincos_unique_positions():
    pe = sincos_2d(4, 4, 32)
    assert pe.shape == (16, 32)
    assert torch.cdist(pe, pe).add(torch.eye(16)).min() > 1e-3


def test_einsum_attention_matches_sdpa():
    q, k, v = torch.randn(3, 2, 4, 16, 8).unbind(0)
    assert torch.allclose(einsum_attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_modulated_layernorm_forward_matches_reference():
    x, shift, scale = torch.randn(2, 5, 16), torch.randn(2, 1, 16), torch.randn(2, 1, 16)
    assert torch.allclose(modulated_layernorm(x, shift, scale), modulated_layernorm_ref(x, shift, scale), atol=1e-5)


def test_modulated_layernorm_gradcheck():
    args = [torch.randn(s, dtype=torch.float64, requires_grad=True) for s in [(2, 3, 8), (2, 1, 8), (2, 1, 8)]]
    assert torch.autograd.gradcheck(modulated_layernorm, args)


def test_model_shapes_and_zero_init():
    m = tiny()
    x, t, y = torch.randn(4, 3, 8, 8), torch.rand(4), torch.randint(0, 10, (4,))
    out = m(x, t, y)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0  # adaLN-Zero: model starts by predicting zero velocity


def test_einsum_and_sdpa_models_agree():
    m = perturb(tiny())
    x, t, y = torch.randn(4, 3, 8, 8), torch.rand(4), torch.randint(0, 10, (4,))
    a = m(x, t, y)
    for mod in m.modules():
        if isinstance(mod, Attention):
            mod.use_sdpa = True
    assert torch.allclose(a, m(x, t, y), atol=1e-5)


def test_cfg_scale_one_is_conditional():
    m = perturb(tiny())
    x, t, y = torch.randn(4, 3, 8, 8), torch.rand(4), torch.randint(0, 10, (4,))
    assert torch.allclose(m.forward_with_cfg(x, t, y, 1.0), m(x, t, y), atol=1e-5)


def test_loss_backward_and_sampler():
    m = perturb(tiny())
    loss = flow_loss(m, torch.randn(4, 3, 8, 8), torch.randint(0, 10, (4,)))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in m.parameters())
    traj = sample(m, torch.randint(0, 10, (2,)), (2, 3, 8, 8), steps=4, cfg_scale=2.0, return_traj=True)
    assert len(traj) == 5 and traj[-1].shape == (2, 3, 8, 8)


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(tests)} passed")
