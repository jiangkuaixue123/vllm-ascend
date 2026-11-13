# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention


def original_sorted_topk(scores, k):
    columns = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int32)
    if k == scores.shape[-1]:
        return columns.long().expand(*scores.shape[:-1], k)
    top = scores.topk(k, dim=-1)
    cutoff = top.values[..., -1:]
    tied = torch.where(scores == cutoff, columns, scores.shape[-1])
    first = tied.topk(k, dim=-1, largest=False).values
    at_cutoff = top.values == cutoff
    ordinal = at_cutoff.long().cumsum(-1) - 1
    return torch.where(at_cutoff, first.gather(-1, ordinal.clamp_min(0)).long(), top.indices).sort(-1).values


@pytest.mark.parametrize("rows,width", [(64, 8072), (129, 4036), (76, 4096), (17, 8072), (6, 8192), (1, 512)])
@pytest.mark.parametrize("case", ["ties", "nonfinite"])
def test_prefill_index_order_matches_integer_pipeline(rows, width, case):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    gen = torch.Generator().manual_seed(918)
    scores = torch.randint(-8, 9, (rows, width), generator=gen).float().npu()
    lengths = torch.arange(rows, device="npu")[:, None] * 113 % width
    scores.masked_fill_(torch.arange(width, device="npu") >= lengths, -torch.inf)
    if case == "nonfinite":
        scores[0, :513] = torch.nan
        scores[-1, :512] = torch.inf
    with torch.inference_mode():
        expected = original_sorted_topk(scores, 512)
        actual = attention.sorted_topk_columns(scores, 512)
        assert actual.dtype == torch.int64
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(
            torch.where(actual < lengths, actual, -1).cpu(),
            torch.where(expected < lengths, expected, -1).cpu(),
            rtol=0,
            atol=0,
        )


def test_prefill_index_order_changing_graph_replay():
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    gen = torch.Generator().manual_seed(771)
    source = torch.randint(-20, 21, (64, 8072), generator=gen).float().npu()
    scores = source.clone()
    with torch.inference_mode():
        attention.sorted_topk_columns(scores, 512)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = attention.sorted_topk_columns(scores, 512)
        for i in range(16):
            source.neg_()
            scores.copy_(torch.roll(source, i * 7, dims=-1))
            if i % 4 == 0:
                scores.zero_()
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), original_sorted_topk(scores, 512).cpu(), rtol=0, atol=0)


def test_narrowing_keeps_integer_exactness_and_block_topk_boundary():
    scores = SimpleNamespace(device=SimpleNamespace(type="npu"), ndim=2, shape=(64, 1 << 24))
    with torch.inference_mode():
        assert attention._use_float32_tie_columns(scores, 512)
        assert not attention._use_float32_tie_columns(scores, 2048)
        scores.shape = (64, (1 << 24) + 1)
        assert not attention._use_float32_tie_columns(scores, 512)
