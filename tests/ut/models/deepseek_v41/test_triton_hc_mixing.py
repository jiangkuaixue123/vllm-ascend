# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import core
from vllm_ascend.ops.triton import deepseek_v41_hc as fused


@pytest.mark.parametrize("rows", [0, 1, 12, 40, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_npu_triton_hc_mixing(rows, dtype):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(742)
    hidden = torch.randn(rows, 4, 5120, dtype=dtype).npu()
    pre = torch.sigmoid(torch.randn(rows, 4)).npu()
    post = (2 * torch.sigmoid(torch.randn(rows, 4))).npu()
    comb = torch.randn(rows, 4, 4).softmax(-1).npu()
    sublayer = torch.randn(rows, 5120, dtype=dtype).npu()
    weight = torch.randn(5120, dtype=dtype).npu()
    pairs = [
        (lambda: core.hc_pre(hidden, pre), lambda: fused.hc_pre_norm(hidden, pre)),
        (lambda: core.rms_norm(core.hc_pre(hidden, pre), weight), lambda: fused.hc_pre_norm(hidden, pre, weight)),
        (lambda: core.hc_post(sublayer, hidden, post, comb), lambda: fused.hc_post(sublayer, hidden, post, comb)),
    ]
    for reference, optimized in pairs:
        expected, actual = reference(), optimized()
        # FP32 reductions may reassociate; BF16 must stay within one ULP plus
        # a small absolute floor around zero. Preserve the internal BF16 cast.
        rtol, atol = (0.008, 0.0001) if dtype == torch.bfloat16 else (3e-6, 2e-6)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        if rows == 12:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                replayed = optimized()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(replayed, expected, rtol=rtol, atol=atol)


def test_npu_triton_hc_pre_preserves_bf16_rounding_before_norm():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(743)
    hidden = torch.randn(12, 4, 5120, dtype=torch.bfloat16).npu()
    pre = torch.tensor([0.1, 0.3, 0.5, 0.7]).expand(12, -1).contiguous().npu()
    weight = torch.ones(5120, dtype=torch.bfloat16).npu()
    rounded = core.rms_norm(core.hc_pre(hidden, pre), weight)
    unrounded = core.rms_norm((hidden.float() * pre.unsqueeze(-1)).sum(-2), weight.float()).bfloat16()
    actual = fused.hc_pre_norm(hidden, pre, weight)
    assert torch.count_nonzero(rounded != unrounded) > 100
    torch.testing.assert_close(actual, rounded, rtol=0.008, atol=0.0001)
