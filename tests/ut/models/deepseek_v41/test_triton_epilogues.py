# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41.core import rms_norm
from vllm_ascend.ops.triton.deepseek_v41_epilogues import attention_sink, compressor_pool_norm


@pytest.mark.parametrize("rows", [0, 1, 12, 48, 128, 256])
@pytest.mark.parametrize("score_scale", [1, 100])
def test_npu_triton_compressor_pool_norm(rows, score_scale):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(769)
    values = torch.randn(rows, 2, 512, device="npu")
    scores = torch.randn_like(values) * score_scale
    weight = torch.randn(512, dtype=torch.bfloat16, device="npu")
    expected = rms_norm((values * scores.softmax(1)).sum(1).bfloat16(), weight)
    actual = compressor_pool_norm(values, scores, weight)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-4)
    if rows == 12:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = compressor_pool_norm(values, scores, weight)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, expected, rtol=0.008, atol=1e-4)


@pytest.mark.parametrize("shape", [(1, 1, 64, 512), (12, 1, 64, 512), (1, 48, 64, 512), (40, 5, 1, 512)])
def test_npu_triton_attention_sink_mask_and_graph(shape):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(770)
    output = torch.randn(shape, dtype=torch.bfloat16, device="npu")
    maximum = torch.randn(shape[:-1], device="npu") * 10
    total = torch.rand_like(maximum) * 30
    sink = torch.randn(shape[-2], device="npu") * 10
    slots = torch.ones((*shape[:2], 1, 128), dtype=torch.int32, device="npu")
    slots.reshape(-1, 128)[::3, 0] = -1
    lse = maximum + total.log()
    expected = (output.float() * torch.sigmoid(lse - sink).unsqueeze(-1)).bfloat16()
    expected.masked_fill_((slots[..., 0] < 0).unsqueeze(-1), 0)
    actual = attention_sink(output, maximum, total, sink, slots)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = attention_sink(output, maximum, total, sink, slots)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(replayed, expected, rtol=0.008, atol=1e-5)
