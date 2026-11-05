# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact cutoff-tie TopK through FP32 column IDs and native graph replay."""

import statistics

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.attention import sorted_topk_columns

SELECTED = 512
REPLAY_REPEATS = 128
BENCHMARK_CALLS = 16
BENCHMARK_REPEATS = 7


def reference_topk_columns(scores, k):
    """Keep the FP32 top-k, preferring the lower column only at a tied cutoff."""
    columns = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.int32)
    if k == 0 or k == scores.shape[-1]:
        return columns[:k].long().expand(*scores.shape[:-1], k)
    top = scores.topk(k, dim=-1)
    cutoff = top.values[..., -1:]
    # topk may choose different equal items at different padded widths.
    # Retain all strictly higher scores; fill only the tied part by column.
    tied = torch.where(scores == cutoff, columns, scores.shape[-1])
    first = tied.topk(k, dim=-1, largest=False).values
    at_cutoff = top.values == cutoff
    ordinal = at_cutoff.long().cumsum(-1) - 1
    return torch.where(at_cutoff, first.gather(-1, ordinal.clamp_min(0)).long(), top.indices)


def reference(scores):
    return reference_topk_columns(scores, SELECTED).float().sort(-1).values.long()


def fused(scores):
    result = sorted_topk_columns(scores, SELECTED)
    assert result is not None
    return result


@pytest.mark.parametrize("rows,width", [(6, 76800), (6, 153600), (12, 76800), (12, 153600)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_exact_selected_set_and_order(rows, width, dtype):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    for mode in ("random", "few_values", "zeros", "negative_infinity", "infinities", "few_nan", "many_nan"):
        scores = torch.randn(rows, width, device="npu", dtype=dtype)
        if mode == "few_values":
            scores = scores.round()
        elif mode == "zeros":
            scores.zero_()
        elif mode == "negative_infinity":
            scores.fill_(-torch.inf)
        elif mode == "infinities":
            scores[:, ::3] = torch.inf
            scores[:, ::5] = -torch.inf
        elif mode == "few_nan":
            scores[:, :3] = torch.nan
        elif mode == "many_nan":
            scores[:, : SELECTED * 2] = torch.nan
        saved = scores.clone()
        actual, expected = fused(scores), reference(scores)
        assert torch.equal(actual, expected), (mode, rows, width, dtype)
        assert torch.equal(scores.view(torch.int16), saved.view(torch.int16))


@pytest.mark.parametrize("rows,width", [(6, 76800), (12, 153600)])
@torch.inference_mode()
def test_repeated_graph_refresh_is_deterministic(rows, width):
    torch.npu.set_device(0)
    scores = torch.randn(rows, width, device="npu", dtype=torch.bfloat16)
    fused(scores)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fused(scores)
    address = output.data_ptr()
    for step in range(REPLAY_REPEATS):
        fresh = torch.randn_like(scores).round()
        fresh[:, (step * 97) % width :] = -torch.inf
        scores.copy_(fresh)
        expected = reference(fresh)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(output, expected), step
        assert output.data_ptr() == address


def timed_graph(function, scores):
    function(scores)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(BENCHMARK_CALLS):
            function(scores)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(BENCHMARK_REPEATS):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@pytest.mark.parametrize("rows,width", [(6, 76800), (6, 153600), (12, 76800), (12, 153600)])
@torch.inference_mode()
def test_component_latency(rows, width):
    torch.npu.set_device(0)
    scores = torch.randn(rows, width, device="npu", dtype=torch.bfloat16)
    reference_ms = timed_graph(reference, scores)
    fused_ms = timed_graph(fused, scores)
    print({"rows": rows, "width": width, "calls": BENCHMARK_CALLS, "reference_ms": reference_ms, "fused_ms": fused_ms})
    assert reference_ms > 0 and fused_ms > 0
