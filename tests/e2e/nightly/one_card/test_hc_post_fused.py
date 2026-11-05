# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 numerical, graph-refresh and latency regression for decode HC post."""

import itertools
import statistics

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.hc_post import hc_post_fused
from vllm_ascend.models.deepseek_v41.core import hc_post

HIDDEN = 5120
WIDTH = 4
CHAIN_LENGTH = 80
BENCHMARK_REPEATS = 7


def inputs(rows, scale=1.0):
    return (
        (torch.randn(rows, HIDDEN, device="npu") * scale).bfloat16(),
        (torch.randn(rows, WIDTH, HIDDEN, device="npu") * scale).bfloat16(),
        torch.rand(rows, WIDTH, device="npu", dtype=torch.float32),
        torch.rand(rows, WIDTH, WIDTH, device="npu", dtype=torch.float32),
    )


def exact_output(actual, expected):
    assert actual is not None and actual.dtype == expected.dtype and actual.shape == expected.shape
    actual, expected = actual.cpu(), expected.cpu()
    both_nan = actual.isnan() & expected.isnan()
    assert torch.equal(actual.view(torch.int16)[~both_nan], expected.view(torch.int16)[~both_nan])


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_reference_order_and_finite_boundaries(rows):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    for scale in (0.0, 2.0**-120, 2.0**-20, 1.0, 2.0**20, 2.0**120):
        for _ in range(3):
            args = inputs(rows, scale)
            exact_output(hc_post_fused(*args), hc_post(*args))
    for values in itertools.permutations((2.0**26, 1.0, -(2.0**26), 1.0)):
        sub, residual, post, comb = inputs(rows)
        sub.zero_()
        residual.copy_(torch.tensor(values, device="npu", dtype=torch.bfloat16).view(1, WIDTH, 1))
        comb.fill_(1.0)
        exact_output(hc_post_fused(sub, residual, post, comb), hc_post(sub, residual, post, comb))
    sub, residual, post, comb = inputs(rows)
    sub[:, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0], device="npu")
    residual[:, :, :4] = 0
    exact_output(hc_post_fused(sub, residual, post, comb), hc_post(sub, residual, post, comb))


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_graph_replay_refreshes_every_input(rows):
    torch.npu.set_device(0)
    args = inputs(rows)
    hc_post_fused(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = hc_post_fused(*args)
    address = output.data_ptr()
    for _ in range(4):
        fresh = inputs(rows)
        for destination, source in zip(args, fresh):
            destination.copy_(source)
        graph.replay()
        torch.npu.synchronize()
        exact_output(output, hc_post(*fresh))
        assert output.data_ptr() == address
        for original, saved in zip(args, fresh):
            assert torch.equal(original.cpu(), saved.cpu())


def timed_graph(function, args):
    # Match the eighty dependent residual updates in a target decode replay.
    sub, residual, post, comb = args
    comb = torch.full_like(comb, 0.25)
    function(sub, residual, post, comb)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = residual
        for _ in range(CHAIN_LENGTH):
            result = function(sub, result, post, comb)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    elapsed = []
    for _ in range(BENCHMARK_REPEATS):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return statistics.median(elapsed)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_graph_component_latency(rows):
    torch.npu.set_device(0)
    args = inputs(rows)
    reference_ms = timed_graph(hc_post, args)
    fused_ms = timed_graph(hc_post_fused, args)
    print({"rows": rows, "calls": CHAIN_LENGTH, "reference_ms": reference_ms, "fused_ms": fused_ms})
    assert fused_ms < reference_ms
