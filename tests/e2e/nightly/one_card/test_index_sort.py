# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact FP32 index sorting: exact selection, capture and component timing."""

import statistics
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.attention import sorted_topk_columns, topk_columns

SELECTED = 512
REPLAY_REPEATS = 128
BENCHMARK_CALLS = 16
BENCHMARK_REPEATS = 7


def reference(scores):
    return topk_columns(scores, SELECTED).sort(-1).values


def fused(scores):
    result = sorted_topk_columns(scores, SELECTED)
    assert result is not None
    return result


@torch.inference_mode()
def test_exact_indices_at_fp32_boundary(monkeypatch):
    torch.npu.set_device(0)
    maximum_width = 1 << 24
    indices = (maximum_width - 1 - torch.arange(SELECTED, device="npu")).expand(6, -1).contiguous()
    monkeypatch.setitem(sorted_topk_columns.__globals__, "topk_columns", lambda *args: indices)
    scores = SimpleNamespace(device=indices.device, ndim=2, shape=(6, maximum_width))
    actual = sorted_topk_columns(scores, SELECTED)
    assert actual.dtype == torch.int64
    assert torch.equal(actual, indices.sort(-1).values)


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
