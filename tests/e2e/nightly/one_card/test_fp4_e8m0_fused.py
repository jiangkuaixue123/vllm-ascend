# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact A3 E8M0 FP4 conversion and graph/latency coverage."""

import statistics
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention_ops
from vllm_ascend.ops.triton.deepseek_v41_fp4_qdq import fp4_e8m0_qdq

CHAIN_LENGTH = 32
REPEATS = 7


def fused(states, inplace=False):
    if tuple(states.shape) in attention_ops._FP4_INDEX_DECODE_SHAPES:
        return attention_ops.fp4_quant_dequant(states, inplace=inplace, decode_fusion=True)
    return fp4_e8m0_qdq(states, inplace)


def reference(states, inplace=False):
    with patch.object(attention_ops, "_try_fp4_e8m0_aiv", return_value=None):
        return attention_ops.fp4_quant_dequant(states, inplace=inplace)


def exact(actual, expected):
    assert actual is not None and actual.shape == expected.shape and actual.dtype == expected.dtype
    actual, expected = actual.cpu(), expected.cpu()
    both_nan = actual.isnan() & expected.isnan()
    assert torch.equal(actual.view(torch.int16)[~both_nan], expected.view(torch.int16)[~both_nan])


@torch.inference_mode()
def test_every_bfloat16_encoding_and_rounding_boundaries():
    torch.npu.set_device(0)
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(32, 16, 128).npu()
    exact(fused(values), reference(values))
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.bfloat16)
    words = thresholds.view(torch.int16)
    nearby = torch.stack((words - 1, words, words + 1), -1).flatten().view(torch.bfloat16)
    group = torch.cat((nearby, torch.tensor([6.0, -0.0], dtype=torch.bfloat16), torch.zeros(9, dtype=torch.bfloat16)))
    for exponent in (-126, -124, -20, 0, 20, 120, 125):
        for sign in (-1, 1):
            states = (group.float() * (sign * 2.0**exponent)).bfloat16().repeat(5, 4).npu()
            exact(fused(states), reference(states))


@pytest.mark.parametrize("shape", [(1, 128), (5, 128), (24, 128), (48, 128), (6, 32, 128), (12, 32, 128)])
@torch.inference_mode()
def test_random_special_values_and_input_preservation(shape):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    for scale in (0.0, 2.0**-126, 2.0**-20, 1.0, 2.0**20, 2.0**120):
        states = (torch.randn(shape, device="npu") * scale).bfloat16()
        saved = states.clone()
        exact(fused(states), reference(states))
        exact(states, saved)
    for value in (float("inf"), -float("inf"), float("nan"), -0.0):
        states.flatten()[0] = value
        exact(fused(states), reference(states))


@pytest.mark.parametrize("inplace", [False, True])
@torch.inference_mode()
def test_graph_refresh_and_aliasing(inplace):
    torch.npu.set_device(0)
    states = torch.randn(6, 32, 128, device="npu", dtype=torch.bfloat16)
    fused(states, inplace=inplace)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fused(states, inplace=inplace)
    address = output.data_ptr()
    assert (address == states.data_ptr()) is inplace
    for _ in range(4):
        fresh = torch.randn_like(states)
        states.copy_(fresh)
        expected = reference(fresh)
        graph.replay()
        torch.npu.synchronize()
        exact(output, expected)
        assert output.data_ptr() == address
        if not inplace:
            exact(states, fresh)


def timed_graph(function, states):
    function(states)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = states
        for _ in range(CHAIN_LENGTH):
            result = function(result)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    elapsed = []
    for _ in range(REPEATS):
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
    states = torch.randn(rows, 32, 128, device="npu", dtype=torch.bfloat16)
    reference_ms = timed_graph(reference, states)
    fused_ms = timed_graph(fp4_e8m0_qdq, states)
    print({"rows": rows, "calls": CHAIN_LENGTH, "reference_ms": reference_ms, "fused_ms": fused_ms})
    assert fused_ms < reference_ms
