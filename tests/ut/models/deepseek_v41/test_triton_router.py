# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ops.triton.deepseek_v41_router import draft_confidence, router_weights


@pytest.mark.parametrize("rows", [0, 1, 12, 48, 128])
@pytest.mark.parametrize("top_k", [1, 3, 6])
@pytest.mark.parametrize("renormalize", [False, True])
def test_npu_triton_router_weights(rows, top_k, renormalize):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(787)
    scores = F.softplus(torch.randn(rows, 384, device="npu")).sqrt()
    bias = torch.randn(384, device="npu")
    indices = (scores + bias).topk(top_k, -1).indices
    expected = scores.gather(-1, indices)
    if renormalize and top_k > 1:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    expected = expected * 1.5
    actual = router_weights(scores, indices, renormalize)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    if rows == 12:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = router_weights(scores, indices, renormalize)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("rows", [0, 1, 8, 40, 128])
@pytest.mark.parametrize("markov_dtype", [torch.bfloat16, torch.float32])
def test_npu_triton_draft_confidence(rows, markov_dtype):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(788)
    hidden = torch.randn(rows, 5120, dtype=torch.bfloat16, device="npu")
    markov = torch.randn(rows, 256, dtype=markov_dtype, device="npu")
    weight = torch.randn(1, 5376, device="npu") * 0.01
    expected = F.linear(torch.cat((hidden, markov), -1).float(), weight).squeeze(-1).sigmoid()
    actual = draft_confidence(hidden, markov, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    if rows == 40:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = draft_confidence(hidden, markov, weight)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, expected, rtol=2e-6, atol=2e-7)
