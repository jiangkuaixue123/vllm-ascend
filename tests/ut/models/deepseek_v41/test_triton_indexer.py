# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.deepseek_v41_indexer import indexer_reduce


@pytest.mark.parametrize("shape", [(0, 64, 64), (1, 64, 127), (12, 64, 512), (12, 64, 8192), (48, 16, 512)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_npu_triton_indexer_reduce(shape, weight_dtype):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(761)
    per_head = torch.randn(shape, dtype=torch.bfloat16, device="npu")
    weights = torch.randn(shape[:2], dtype=weight_dtype, device="npu")
    expected = (per_head.relu() * weights.unsqueeze(-1)).sum(1)
    actual = indexer_reduce(per_head, weights)
    if weight_dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=1e-5)
    if shape == (12, 64, 512):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = indexer_reduce(per_head, weights)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, expected, rtol=3e-6, atol=1e-5)
