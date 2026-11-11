# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from vllm_ascend.models.deepseek_v41 import backend


def test_hc_block_carries_the_pre_gate_and_uses_native_post(monkeypatch):
    block = backend.Block.__new__(backend.Block)
    nn.Module.__init__(block)
    block.hc_mult, block.hc_sinkhorn_iters = 4, 20
    block.norm_eps, block.hc_eps = 1e-20, 1e-6
    block.attn_norm = block.ffn_norm = nn.Identity()
    block.attn = lambda x, *_: x
    block.ffn = lambda x, *_: x

    random = torch.Generator().manual_seed(5328)
    width = 16
    for name in ("attn", "ffn"):
        setattr(block, f"hc_{name}_fn", torch.randn(24, 4 * width, generator=random) / 8)
        setattr(block, f"hc_{name}_scale", torch.tensor([0.7, -0.8, 0.5]))
        setattr(block, f"hc_{name}_base", torch.randn(24, generator=random))
    h = torch.randn(2, 3, 4, 2 * width, generator=random).bfloat16()[..., ::2]
    incoming = torch.zeros(2, 3, 4)
    incoming[..., 1] = 1
    assert not h.is_contiguous()

    def collapse(x, gate):
        return torch.einsum("...nd,...n->...d", x.float(), gate).bfloat16()

    def expand(x, residual, post, comb):
        mixed = torch.einsum("...ij,...id->...jd", comb, residual.float())
        return (mixed + post[..., None] * x.float().unsqueeze(-2)).bfloat16()

    attn_pre, attn_post, attn_comb = block.hc_mixes(h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    mid = expand(collapse(h, incoming), h, attn_post, attn_comb)
    expected_pre, ffn_post, ffn_comb = block.hc_mixes(mid, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
    expected = expand(collapse(mid, attn_pre), mid, ffn_post, ffn_comb)
    assert not torch.equal(collapse(h, incoming), collapse(h, attn_pre))

    calls = []

    def hc_pre(x, w, scale, bias, hc, iters, norm_eps, hc_eps, *, return_pre):
        assert x.is_contiguous() and (hc, iters, norm_eps, hc_eps, return_pre) == (4, 20, 1e-20, 1e-6, True)
        calls.append("pre")
        flat = x.float().flatten(-2)
        logits = torch.nn.functional.linear(flat, w) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        return backend.hc_split_sinkhorn(logits, scale, bias, hc, iters, hc_eps)

    def hc_post(x, residual, post, comb):
        assert all(t.is_contiguous() for t in (x, residual, post, comb))
        calls.append("post")
        return expand(x, residual, post, comb)

    ops = SimpleNamespace(npu_hc_pre_v2=hc_pre, npu_hc_post=hc_post)
    monkeypatch.setattr(backend, "native_hc_ops", lambda _: ops)
    actual, actual_pre = block(h, 0, incoming, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_pre, expected_pre, rtol=0, atol=0)
    assert calls == ["pre", "post", "pre", "post"]
