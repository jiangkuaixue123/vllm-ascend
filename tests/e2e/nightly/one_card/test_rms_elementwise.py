# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 exact arithmetic, graph refresh and latency tests for RMS elementwise fusion."""

import statistics

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.rms_elementwise import rms_elementwise_fused
from vllm_ascend.models.deepseek_v41.core import rms_norm
from vllm_ascend.ops.triton.deepseek_v41_rms_elementwise import square_fp32

HIDDEN_SIZES = (128, 512, 1280, 5120)
CHAIN_LENGTH = 80
BENCHMARK_REPEATS = 7
EPS = 1e-20


def exact_output(actual, expected):
    assert actual is not None and actual.dtype == expected.dtype and actual.shape == expected.shape
    actual, expected = actual.cpu(), expected.cpu()
    both_nan = actual.isnan() & expected.isnan()
    integer_dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(integer_dtype)[~both_nan], expected.view(integer_dtype)[~both_nan])


@torch.inference_mode()
def test_square_all_bfloat16_encodings():
    torch.npu.set_device(0)
    states = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(128, 512).npu()
    exact_output(square_fp32(states), states.float().square())


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_exact_normalization(hidden, weight_dtype):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    for rows in (1, 5, 6, 12, 48):
        weight = torch.randn(hidden, device="npu", dtype=weight_dtype)
        for scale in (0.0, 2.0**-120, 2.0**-20, 1.0, 2.0**20, 2.0**120):
            states = (torch.randn(rows, hidden, device="npu") * scale).bfloat16()
            exact_output(rms_elementwise_fused(states, weight, EPS), rms_norm(states, weight, EPS))
        states[:, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0], device="npu")
        exact_output(rms_elementwise_fused(states, weight, EPS), rms_norm(states, weight, EPS))


@pytest.mark.parametrize("rows", [5, 6, 12])
@torch.inference_mode()
def test_graph_refreshes_states_and_weights(rows):
    torch.npu.set_device(0)
    states = torch.randn(rows, HIDDEN_SIZES[-1], device="npu", dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_SIZES[-1], device="npu", dtype=torch.bfloat16)
    rms_elementwise_fused(states, weight, EPS)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = rms_elementwise_fused(states, weight, EPS)
    address = output.data_ptr()
    for _ in range(4):
        fresh_states, fresh_weight = torch.randn_like(states), torch.randn_like(weight)
        states.copy_(fresh_states)
        weight.copy_(fresh_weight)
        graph.replay()
        torch.npu.synchronize()
        exact_output(output, rms_norm(fresh_states, fresh_weight, EPS))
        assert output.data_ptr() == address
        assert torch.equal(states.cpu(), fresh_states.cpu())
        assert torch.equal(weight.cpu(), fresh_weight.cpu())


def timed_graph(function, states, weight):
    function(states, weight, EPS)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = states
        for _ in range(CHAIN_LENGTH):
            result = function(result, weight, EPS)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    elapsed = []
    for _ in range(BENCHMARK_REPEATS):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return statistics.median(elapsed)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_component_latency(rows):
    torch.npu.set_device(0)
    states = torch.randn(rows, HIDDEN_SIZES[-1], device="npu", dtype=torch.bfloat16)
    weight = torch.ones(HIDDEN_SIZES[-1], device="npu", dtype=torch.bfloat16)
    reference = timed_graph(rms_norm, states, weight)
    fused = timed_graph(rms_elementwise_fused, states, weight)
    print({"rows": rows, "calls": CHAIN_LENGTH, "reference_ms": reference, "fused_ms": fused})
    assert fused < reference
