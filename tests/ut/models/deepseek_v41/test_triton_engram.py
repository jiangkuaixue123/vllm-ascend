# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import apply_engram_gate
from vllm_ascend.ops.triton.deepseek_v41_engram import engram_gate_residual


@pytest.mark.parametrize("rows", [0, 1, 12, 48, 128])
@pytest.mark.parametrize("masked", [False, True])
def test_npu_triton_engram_gate_residual(rows, masked):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(779)
    x = torch.randn(rows, 4, 5120, dtype=torch.bfloat16, device="npu")
    kv = torch.randn(rows, 5 * 5120, dtype=torch.bfloat16, device="npu")
    qw = torch.randn(4, 5120, device="npu")
    kw = torch.randn_like(qw)
    mask = torch.arange(rows, device="npu") % 3 != 0 if masked else None
    expected = apply_engram_gate(x, x[:, :1], lambda _: kv, qw, kw, eps=1e-20, token_mask=mask)
    actual = engram_gate_residual(x, kv, qw, kw, mask)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0001)
    if masked:
        assert torch.equal(actual[~mask], x[~mask])
    if rows == 12:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = engram_gate_residual(x, kv, qw, kw, mask)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, expected, rtol=0.008, atol=0.0001)
