# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 FP4 E4M3 scales, exact output, graph replay and latency."""

import statistics
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention_ops
from vllm_ascend.ops.triton.deepseek_v41_fp4_e4m3_qdq import fp4_e4m3_qdq, prepare_fp4_e4m3

CHAIN_LENGTH = 32
REPEATS = 7
HIDDEN = 512


def fused(states, inplace=False):
    if tuple(states.shape) in attention_ops._FP4_KV_DECODE_SHAPES:
        return attention_ops.fp4_quant_dequant(states, 16, scale_format="e4m3", inplace=inplace, decode_fusion=True)
    return fp4_e4m3_qdq(states, inplace)


def reference(states, inplace=False):
    with patch.object(attention_ops, "_try_fp4_e4m3_aiv", return_value=None):
        return attention_ops.fp4_quant_dequant(states, 16, scale_format="e4m3", inplace=inplace)


def exact(actual, expected):
    assert actual is not None and actual.shape == expected.shape and actual.dtype == expected.dtype
    actual, expected = actual.cpu(), expected.cpu()
    both_nan = actual.isnan() & expected.isnan()
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(integer)[~both_nan], expected.view(integer)[~both_nan])


@torch.inference_mode()
def test_all_bfloat16_group_maxima():
    torch.npu.set_device(0)
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    states = values.repeat_interleave(16).reshape(-1, HIDDEN).npu()
    fp32_states, scales = prepare_fp4_e4m3(states)
    maximum = states.float().reshape(-1, 16).abs().amax(-1)
    expected = attention_ops.round_e4m3(maximum.clamp_min(attention_ops.FP4_E4M3_AMAX_MIN) / 6.0)
    exact(fp32_states, states.float())
    exact(scales, expected)
    states = values.reshape(-1, HIDDEN).npu()
    exact(fp4_e4m3_qdq(states), reference(states))


@torch.inference_mode()
def test_each_e4m3_scale_at_e2m1_rounding_boundaries():
    torch.npu.set_device(0)
    scales = [n * 2.0**-9 for n in range(1, 8)]
    scales += [(1 + m / 8) * 2.0**e for e in range(-6, 9) for m in range(8) if (1 + m / 8) * 2.0**e <= 448]
    groups = []
    for scale in scales:
        thresholds = (torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]) * scale).bfloat16()
        words = thresholds.view(torch.int16)
        probes = torch.stack((words - 1, words, words + 1), -1).flatten().view(torch.bfloat16)
        anchor = torch.tensor([6 * scale], dtype=torch.bfloat16)
        groups.extend((torch.cat((anchor, probes[:15])), torch.cat((anchor, probes[15:], torch.zeros(9)))))
    flat = torch.cat(groups).bfloat16()
    flat = torch.cat((flat, torch.zeros((-flat.numel()) % HIDDEN, dtype=torch.bfloat16)))
    states = flat.reshape(-1, HIDDEN).npu()
    exact(fused(states), reference(states))
    exact(fused(-states), reference(-states))


@pytest.mark.parametrize("rows", [1, 5, 6, 12, 24, 48])
@torch.inference_mode()
def test_random_special_values_and_preservation(rows):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    for scale in (0.0, 2.0**-126, 2.0**-20, 1.0, 2.0**10, 2.0**120):
        states = (torch.randn(rows, HIDDEN, device="npu") * scale).bfloat16()
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
    states = torch.randn(24, HIDDEN, device="npu", dtype=torch.bfloat16)
    fused(states, inplace)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fused(states, inplace)
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


@pytest.mark.parametrize("rows", [24, 48])
@torch.inference_mode()
def test_component_latency(rows):
    torch.npu.set_device(0)
    states = torch.randn(rows, HIDDEN, device="npu", dtype=torch.bfloat16)
    reference_ms = timed_graph(reference, states)
    fused_ms = timed_graph(fp4_e4m3_qdq, states)
    print({"rows": rows, "calls": CHAIN_LENGTH, "reference_ms": reference_ms, "fused_ms": fused_ms})
    assert fused_ms < reference_ms


@pytest.mark.parametrize("rows", [127, 2048, 8192])
@torch.inference_mode()
def test_target_prefill_rows(rows):
    torch.npu.set_device(0)
    torch.manual_seed(20261113)
    states = torch.randn(rows, 512, device="npu", dtype=torch.bfloat16)
    states.flatten()[-4:] = torch.tensor([0.25, -0.0, float("inf"), float("nan")], device="npu")
    assert attention_ops._try_fp4_e4m3_aiv(states, 16, "e4m3", False, prefill=True) is not None
    actual = attention_ops.fp4_quant_dequant(states, 16, scale_format="e4m3", prefill_fusion=True)
    exact(actual, reference(states))
