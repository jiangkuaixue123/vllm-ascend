# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41.attention_ops import apply_rotary_emb_real, fp8_quant_dequant
from vllm_ascend.models.deepseek_v41.core import rms_norm
from vllm_ascend.ops.triton.deepseek_v41_kv import kv_norm_rope_qdq


@pytest.mark.parametrize("rows", [0, 1, 12, 48, 128, 512])
@pytest.mark.parametrize("scale", [0.001, 1.0, 100.0])
def test_npu_triton_kv_norm_rope_qdq(rows, scale):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(753)
    x = (torch.randn(rows, 512) * scale).bfloat16().npu()
    weight = torch.randn(512, dtype=torch.bfloat16).npu()
    phases = torch.randn(rows, 32).npu() * 10000
    cosine, sine = phases.cos(), phases.sin()
    expected = rms_norm(x, weight)
    apply_rotary_emb_real(expected.unsqueeze(0)[..., -64:], cosine, sine)
    expected = fp8_quant_dequant(expected) if rows else expected
    actual = kv_norm_rope_qdq(x, weight, cosine, sine)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rows == 12:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = kv_norm_rope_qdq(x, weight, cosine, sine)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


def test_npu_triton_kv_zero_and_nonfinite():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    x = torch.zeros((3, 512), dtype=torch.bfloat16, device="npu")
    x[0, 1::2] = -0.0
    x[1, 0] = torch.inf
    x[2, 0] = torch.nan
    weight = torch.ones(512, dtype=torch.bfloat16, device="npu")
    cosine = torch.ones((3, 32), device="npu")
    sine = torch.zeros_like(cosine)
    expected = rms_norm(x, weight)
    apply_rotary_emb_real(expected.unsqueeze(0)[..., -64:], cosine, sine)
    expected = fp8_quant_dequant(expected)
    actual = kv_norm_rope_qdq(x, weight, cosine, sine)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual[0]), torch.signbit(expected[0]))
