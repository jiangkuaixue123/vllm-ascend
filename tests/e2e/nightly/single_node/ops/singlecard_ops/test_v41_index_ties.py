# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact cutoff-tie choices and repeated A3 graph compaction."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.deepseek_v41_index_ties import compact_tied_columns

K = 512


def reference(scores, selected, indices):
    columns = torch.arange(scores.shape[-1], dtype=torch.float32, device=scores.device)
    cutoff = selected[..., -1:]
    tied = torch.where(scores == cutoff, columns, scores.shape[-1])
    first = tied.topk(K, dim=-1, largest=False).values
    at_cutoff = selected == cutoff
    ordinal = at_cutoff.long().cumsum(-1) - 1
    return torch.where(at_cutoff, first.gather(-1, ordinal.clamp_min(0)).long(), indices)


def make_scores(rows, width, dtype, case):
    generator = torch.Generator().manual_seed(4107)
    scores = torch.randint(-4, 5, (rows, width), generator=generator).to(dtype)
    if case == "all_ties":
        scores.zero_()
    elif case == "empty":
        scores.fill_(-torch.inf)
    elif case == "short":
        scores[:, 37:] = -torch.inf
    elif case == "special":
        scores[:, 1:7] = torch.nan
        scores[:, 7:11] = torch.inf
        scores[:, 11] = -0.0
        scores[:, 12] = 0.0
    elif case == "nan_cutoff":
        scores[:, : K + 3] = torch.nan
    elif case == "sparse_ties":
        scores.fill_(-1)
        scores[:, 1::251] = 1
    return scores.npu()


@pytest.mark.parametrize("rows,width", [(3, 76800), (6, 153600), (12, 76800)])
@pytest.mark.parametrize("case", ["random", "all_ties", "empty", "short", "special", "nan_cutoff", "sparse_ties"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_compact_preserves_native_non_cutoff_and_exact_ties(rows, width, case, dtype):
    scores = make_scores(rows, width, dtype, case)
    top = scores.topk(K, dim=-1)
    expected = reference(scores, top.values, top.indices)
    actual = compact_tied_columns(scores, top.values, top.indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_compact_repeated_graph_replay_and_changed_cutoff():
    scores = make_scores(6, 153600, torch.bfloat16, "all_ties")

    def select():
        top = scores.topk(K, dim=-1)
        return compact_tied_columns(scores, top.values, top.indices)

    select()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        output = select()
    for case in ("short", "special", "sparse_ties", "nan_cutoff", "empty"):
        scores.copy_(make_scores(6, 153600, torch.bfloat16, case))
        top = scores.topk(K, dim=-1)
        expected = reference(scores, top.values, top.indices)
        for _ in range(64):
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_strided_scores_use_reference_fallback():
    scores = torch.zeros((6, 153600), dtype=torch.bfloat16, device="npu")[:, ::2]
    top = scores.topk(K, dim=-1)
    assert compact_tied_columns(scores, top.values, top.indices) is None
