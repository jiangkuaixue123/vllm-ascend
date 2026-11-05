# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native RMS numerical classes, BF16 rounding budget, and capture replay."""

import statistics

import pytest
import torch
import torch_npu

from vllm_ascend.models.common.ops.rms_native_fp32 import native_rms_norm_fp32
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41RMSNorm
from vllm_ascend.models.deepseek_v41.core import rms_norm

EPSILON = 1e-20
REPLAY_REPEATS = 64
BENCHMARK_CALLS = 80
BENCHMARK_REPEATS = 7
SHAPES = [
    (1, 128),
    (5, 5120),
    (6, 5120),
    (12, 5120),
    (20, 5120),
    (24, 128),
    (24, 512),
    (24, 1024),
    (40, 5120),
    (48, 5120),
]


def inputs(shape, weight_dtype):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    # Nonzero storage offsets exercise real contiguous views, not only allocations.
    x = torch.randn(shape[0] + 1, shape[1], device="npu", dtype=torch.bfloat16)[1:]
    weight = torch.randn(shape[1] + 1, device="npu", dtype=weight_dtype)[1:]
    return x, weight


def candidate(x, weight):
    output = native_rms_norm_fp32(x, weight, EPSILON)
    assert output is not None
    return output


def reference(x, weight):
    return rms_norm(x, weight, EPSILON)


def check_output(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a, e = actual.cpu(), expected.cpu()
    assert torch.equal(torch.isnan(a), torch.isnan(e))
    assert torch.equal(torch.isposinf(a), torch.isposinf(e))
    assert torch.equal(torch.isneginf(a), torch.isneginf(e))
    finite = torch.isfinite(a) & torch.isfinite(e)
    steps = (a.view(torch.int16).int() - e.view(torch.int16).int()).abs()
    assert bool((steps[finite] <= 1).all()), f"BF16 maximum step difference {steps[finite].max().item()}"
    zeros = (a == 0) & (e == 0)
    assert torch.equal(torch.signbit(a)[zeros], torch.signbit(e)[zeros])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_supported_shapes_and_weight_precision(shape, weight_dtype):
    x, weight = inputs(shape, weight_dtype)
    for scale in (1.0, 1e-4, 1e4):
        scaled = x * scale
        saved_x, saved_weight = scaled.clone(), weight.clone()
        check_output(candidate(scaled, weight), reference(scaled, weight))
        assert torch.equal(scaled, saved_x) and torch.equal(weight, saved_weight)


@pytest.mark.parametrize(
    "mode", ["zero", "negative_zero", "tiny", "large", "overflow", "nan", "inf", "weight_nan", "weight_inf"]
)
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_exceptional_values_and_rounding_budget(mode, weight_dtype):
    for width in (128, 512, 1024, 5120):
        x, weight = inputs((6, width), weight_dtype)
        if mode == "zero":
            x.zero_()
        elif mode == "negative_zero":
            x.fill_(-0.0)
        elif mode == "tiny":
            x.fill_(2.0**-130)
        elif mode == "large":
            x.mul_(2.0**20)
        elif mode == "overflow":
            x.fill_(2.0**70)
        elif mode == "nan":
            x[-1, 0] = torch.nan
        elif mode == "inf":
            x[-1, 0] = torch.inf
        elif mode == "weight_nan":
            weight[0] = torch.nan
        else:
            weight[0] = torch.inf
        check_output(candidate(x, weight), reference(x, weight))


@pytest.mark.parametrize("shape", [(6, 5120), (12, 5120), (24, 512), (24, 1024)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_capture_refreshes_states_and_weights(shape, weight_dtype):
    x, weight = inputs(shape, weight_dtype)
    candidate(x, weight)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = candidate(x, weight)
    address = output.data_ptr()
    for step in range(REPLAY_REPEATS):
        x.copy_(torch.randn_like(x))
        weight.copy_(torch.randn_like(weight))
        expected = reference(x, weight)
        graph.replay()
        torch.npu.synchronize()
        check_output(output, expected)
        assert output.data_ptr() == address, step


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_real_model_norm_dispatch(weight_dtype):
    x, weight = inputs((6, 5120), weight_dtype)
    module = DeepseekV41RMSNorm(5120, EPSILON).to(device=x.device, dtype=weight.dtype)
    module.weight.copy_(weight)
    check_output(module(x), reference(x, weight))


def test_cpu_training_keeps_autograd():
    module = DeepseekV41RMSNorm(128, EPSILON)
    module.weight.data.fill_(1)
    x = torch.randn(2, 128, requires_grad=True)
    output = module(x)
    torch.testing.assert_close(output, rms_norm(x, module.weight, EPSILON), rtol=0, atol=0)
    output.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())


def timed_graph(function, x, weight):
    function(x, weight)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(BENCHMARK_CALLS):
            function(x, weight)
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


@pytest.mark.parametrize("shape", [(6, 5120), (12, 5120), (24, 512), (24, 1024)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_complete_component_latency(shape, weight_dtype):
    x, weight = inputs(shape, weight_dtype)
    parent_ms, candidate_ms = timed_graph(reference, x, weight), timed_graph(candidate, x, weight)
    print(
        {
            "shape": shape,
            "weight_dtype": str(weight_dtype),
            "calls": BENCHMARK_CALLS,
            "reference_ms": parent_ms,
            "candidate_ms": candidate_ms,
        }
    )
    assert parent_ms > 0 and candidate_ms > 0


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_rounding_observation_against_explicit_parent(weight_dtype):
    x, weight = inputs((6, 5120), weight_dtype)
    expected = reference(x, weight)
    actual = candidate(x, weight)
    direct_bf16 = torch_npu.npu_rms_norm(x, weight, EPSILON)[0]
    check_output(actual, expected)
    reference_bits = expected.cpu().view(torch.int16)
    fp32_differences = torch.count_nonzero(actual.cpu().view(torch.int16) != reference_bits).item()
    bf16_differences = torch.count_nonzero(direct_bf16.cpu().view(torch.int16) != reference_bits).item()
    print(
        {
            "weight_dtype": str(weight_dtype),
            "elements": expected.numel(),
            "fp32_operand_differences": fp32_differences,
            "direct_bf16_differences": bf16_differences,
        }
    )
