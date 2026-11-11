# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.models.deepseek_v41.backend import Block
from vllm_ascend.models.deepseek_v41.native_ops import native_hc_ops


@pytest.mark.parametrize(("batch", "seq", "expanded"), [(1, 6, True), (2, 7, False), (1, 97, False)])
@torch.inference_mode()
def test_hc_carried_coefficients_and_full_residual_on_a5(batch, seq, expanded):
    if not get_device_config().supports_hc_pre_coefficients:
        pytest.skip("The FP32 HcPre gate output is specific to Ascend A5.")
    random = torch.Generator().manual_seed(3584 + seq)
    params = SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-20, hc_eps=1e-6)
    hidden = 5120
    source = torch.randn((batch, seq, 1 if expanded else 4, hidden), generator=random).bfloat16()
    source[:, 0].zero_()
    source[:, 1].mul_(2.0**-35)
    # Preserve tiny-row sensitivity to the checkpoint's actual norm epsilon.
    weights = torch.randn((24, 4 * hidden), generator=random).bfloat16().float() * (2.0**-7)
    scale = torch.tensor([0.75, -1.25, 0.5])
    bias = torch.linspace(-0.5, 0.6, 24)
    host = source.expand(batch, seq, 4, hidden)
    device = source.npu().expand(batch, seq, 4, hidden)
    ops = native_hc_ops(device)
    assert ops is not None
    expected = Block.hc_mixes(params, host, weights, scale, bias)
    pre, post, comb = Block.hc_mixes(params, device, weights.npu(), scale.npu(), bias.npu())
    for actual, oracle in zip((pre, post, comb), expected, strict=True):
        assert actual.dtype == torch.float32 and actual.shape == oracle.shape
        torch.testing.assert_close(actual.cpu(), oracle, atol=1e-4, rtol=1e-4)

    incoming = torch.tensor([0.0, 0.25, 0.5, 1.0]).expand(batch, seq, 4)
    collapsed = Block.hc_pre(params, device, incoming.npu())
    torch.testing.assert_close(collapsed.cpu(), Block.hc_pre(params, host, incoming), rtol=0, atol=0)
    # Supply noncontiguous sublayer output and an asymmetric comb to check both
    # the wrapper's layout contract and the input-to-output residual direction.
    sublayer = torch.randn(batch, seq, 2 * hidden, generator=random).bfloat16()
    result = Block.hc_post(params, sublayer.npu()[..., ::2], device, post, comb).cpu()
    oracle = Block.hc_post(params, sublayer[..., ::2], host, expected[1], expected[2])
    torch.testing.assert_close(result.float(), oracle.float(), rtol=0.008, atol=0.008)

    # Existing callers omit return_pre and still get a reduced BF16 stream.
    legacy, _, _ = ops.npu_hc_pre_v2(device.contiguous(), weights.npu(), scale.npu(), bias.npu(), 4, 20, 1e-20, 1e-6)
    assert legacy.dtype == torch.bfloat16 and legacy.shape == (batch, seq, hidden)
    torch.testing.assert_close(
        legacy.cpu().float(), Block.hc_pre(params, host, expected[0]).float(), rtol=0.008, atol=0.008
    )

    meta_args = tuple(t.to("meta") for t in (host, weights, scale, bias))
    meta, meta_post, meta_comb = ops.npu_hc_pre_v2(*meta_args, 4, 20, 1e-20, 1e-6, return_pre=True)
    assert [(t.shape, t.dtype) for t in (meta, meta_post, meta_comb)] == [(t.shape, t.dtype) for t in (pre, post, comb)]
