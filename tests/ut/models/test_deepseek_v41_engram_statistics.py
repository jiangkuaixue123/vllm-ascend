# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import torch
from torch import nn

from vllm_ascend.device import device_config
from vllm_ascend.models.deepseek_v41 import backend, native_ops


def test_cpu_engram_gate_with_mask_never_loads_accelerator(monkeypatch):
    def unexpected():
        raise AssertionError("CPU Engram tried to load the NPU kernel")

    monkeypatch.setattr(native_ops, "_engram_statistics_op", unexpected)
    module = backend.Engram.__new__(backend.Engram)
    nn.Module.__init__(module)
    module.dim, module.hc_mult, module.eps, module.clamp_value = 32, 4, 1e-6, 1e-6
    gen = torch.Generator().manual_seed(41)
    kv = torch.randn((6, 5 * 32), generator=gen).bfloat16()
    module.embed = nn.Embedding.from_pretrained(kv)
    module.wkv = nn.Identity()
    module.q_weight = nn.Parameter(torch.randn((4, 32), generator=gen))
    module.k_weight = nn.Parameter(torch.randn((4, 32), generator=gen))
    h = torch.randn((2, 3, 4, 32), generator=gen).bfloat16()
    ids = torch.tensor([[[4], [1], [5]], [[3], [0], [2]]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    actual = module(h, ids, mask)
    rows = kv[ids.squeeze(-1)].float().unflatten(-1, (5, 32))
    key, value = rows[..., :4, :], rows[..., 4:, :]
    h32 = h.float()
    norm = torch.rsqrt(h32.square().mean(-1) + module.eps) * torch.rsqrt(key.square().mean(-1) + module.eps)
    dot = (h32 * (module.q_weight * module.k_weight) * key).sum(-1) * norm * module.dim**-0.5
    magnitude = dot.abs().clamp_min(module.clamp_value).sqrt()
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    expected = h32 + gate.masked_fill(~mask.unsqueeze(-1), 0).unsqueeze(-1) * value
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)


def test_a5_without_triton_keeps_fixed_row_order(monkeypatch):
    monkeypatch.setattr(
        device_config, "get_device_config", lambda: SimpleNamespace(supports_fixed_engram_reductions=True)
    )
    monkeypatch.delitem(sys.modules, "vllm_ascend.models.deepseek_v41.engram_reduce", raising=False)
    monkeypatch.setitem(sys.modules, "triton", None)
    fallback = native_ops._engram_statistics_op.__wrapped__()
    assert fallback is native_ops._tensor_engram_statistics
    gen = torch.Generator().manual_seed(4191)
    h = torch.randn((2, 3, 4, 5120), generator=gen).bfloat16()
    kv = torch.randn((2, 3, 5, 5120), generator=gen).bfloat16()
    key = kv[..., :4, :]
    weight = torch.randn((4, 5120), generator=gen)
    paired = torch.stack(fallback(h, key, weight), -1)
    for i in range(2):
        single = torch.stack(fallback(h[i : i + 1], key[i : i + 1], weight), -1)
        assert torch.equal(single, paired[i : i + 1])
    reversed_pair = torch.stack(fallback(h.flip(0), kv.flip(0)[..., :4, :], weight), -1).flip(0)
    assert torch.equal(reversed_pair, paired)
    hd, kd = h.double(), key.double()
    weighted = hd * weight.double() * kd
    expected = torch.stack((hd.square().mean(-1), kd.square().mean(-1), weighted.sum(-1)), -1)
    magnitude = torch.stack((expected[..., 0], expected[..., 1], weighted.abs().sum(-1)), -1)
    assert torch.all((paired.double() - expected).abs() <= 32 * torch.finfo(torch.float32).eps * magnitude)
