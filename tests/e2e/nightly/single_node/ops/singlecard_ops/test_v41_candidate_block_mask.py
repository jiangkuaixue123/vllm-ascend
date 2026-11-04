# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compact and per-token candidate masks on a real dynamically re-applied ACLGraph."""

import pytest
import torch

from vllm_ascend.models.deepseek_v41.attention import mask_candidate_columns_, select_candidate_blocks, topk_columns


@pytest.mark.parametrize(("rows", "width", "block_budget"), [(6, 16384, 2048), (12, 16384, 2048), (6, 17, 2)])
@torch.inference_mode()
def test_block_mask_graph_replay_matches_dense_causal_topk(rows, width, block_budget):
    block_size = 8
    scores = torch.zeros((rows, width), dtype=torch.bfloat16, device="npu")
    values = torch.empty_like(scores)
    lengths = torch.ones((rows, 1), dtype=torch.long, device="npu")
    columns = torch.arange(width, device="npu")

    def select(compact):
        source = scores.masked_fill(columns >= lengths, -torch.inf)
        keep = select_candidate_blocks(source, lengths, block_budget, block_size, per_token=not compact)
        follower = values.masked_fill(columns >= lengths, -torch.inf)
        if compact:
            mask_candidate_columns_(follower, keep, block_size, width)
        else:
            follower.masked_fill_(~keep, -torch.inf)
        idx = topk_columns(follower, min(512, width)).sort(-1).values
        return follower, torch.where(idx < lengths, idx, -1)

    values.zero_()
    select(True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        observed, indices = select(True)
    for step in (1, 2):
        rng = torch.Generator().manual_seed(42 + step)
        scores.copy_(torch.randint(-3, 4, scores.shape, generator=rng).bfloat16())
        scores[:, :8] = 1
        scores[::2, 8] = torch.nan
        values.copy_(torch.randint(-4, 5, values.shape, generator=rng).bfloat16())
        values[:, 1] = -0.0
        values[:, 5] = torch.nan
        lengths.copy_(torch.tensor([0, 1, 8, 9, width - 1, width] * (rows // 6)).roll(step).view(-1, 1))
        expected, expected_indices = select(False)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(observed.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
        torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
