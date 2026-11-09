# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41.core import clipped_swiglu
from vllm_ascend.ops.triton.deepseek_v41_moe import clipped_swiglu_route


@pytest.mark.parametrize("rows", [0, 1, 12, 48, 288, 768])
@pytest.mark.parametrize("has_factor", [False, True])
@pytest.mark.parametrize("limit", [0.0, 10.0])
def test_npu_triton_clipped_swiglu_route(rows, has_factor, limit):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(749)
    gate_up = (torch.randn(rows, 4096) * 11).bfloat16().npu()
    gate, up = gate_up.chunk(2, -1)
    factor = torch.rand(rows, device="npu") if has_factor else None
    if has_factor and rows:
        factor[::3] = 0
    expected = clipped_swiglu(gate, up, limit)
    if factor is not None:
        expected = expected * factor[:, None]
    actual = clipped_swiglu_route(gate, up, factor, limit, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=4e-6, atol=2e-5)
    result = clipped_swiglu_route(gate, up, factor, limit)
    torch.testing.assert_close(result, expected.bfloat16(), rtol=0.008, atol=1e-5)
    if rows == 12:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = clipped_swiglu_route(gate, up, factor, limit)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected.bfloat16(), rtol=0.008, atol=1e-5)
