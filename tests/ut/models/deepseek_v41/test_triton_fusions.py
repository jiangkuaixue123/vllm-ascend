# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41.core import hc_split_sinkhorn
from vllm_ascend.ops.triton.deepseek_v41 import hc_gates_sinkhorn


@pytest.mark.parametrize("rows", [0, 1, 6, 12, 24, 40, 128, 512])
@pytest.mark.parametrize("magnitude", [1, 100])
def test_npu_triton_hc_gates_sinkhorn(rows, magnitude):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(260)
    mixes = (torch.randn(rows, 24) * magnitude).npu()
    scale = torch.tensor([0.7, -1.3, 0.4]).npu()
    base = torch.randn(24).npu()
    expected = hc_split_sinkhorn(mixes, scale, base, 4)
    actual = hc_gates_sinkhorn(mixes, scale, base)
    for reference, value in zip(expected, actual):
        torch.testing.assert_close(value, reference, rtol=3e-6, atol=3e-7)
    if rows == 12 and magnitude == 1:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = hc_gates_sinkhorn(mixes, scale, base)
        graph.replay()
        torch.npu.synchronize()
        for reference, value in zip(expected, replayed):
            torch.testing.assert_close(value, reference, rtol=3e-6, atol=3e-7)
