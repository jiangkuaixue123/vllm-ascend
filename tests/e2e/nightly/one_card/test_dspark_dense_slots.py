# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native A3 full-block attention equivalence and complete-call latency."""

import statistics
import time

import pytest
import torch

from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention

CALLS = 24
REPEATS = 5


def inputs(batch, length, heads):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    q = torch.randn(batch, 5, heads, 512, device="npu", dtype=torch.bfloat16)
    kv = torch.randn(batch, length, 512, device="npu", dtype=torch.bfloat16)
    sink = torch.linspace(-1, 2, heads, device="npu", dtype=torch.float32)
    return q, kv, sink


def run(runner, q, kv, sink, *, cached):
    indices = None
    if not cached:
        indices = torch.arange(kv.shape[1], dtype=torch.int32, device=q.device)
        indices = indices.view(1, 1, -1).expand(q.shape[0], q.shape[1], -1)
    return runner(q, kv, sink, indices, 512**-0.5, mode="SWA draft bidirectional")


def same_words(actual, expected):
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("length", [5, 6, 127, 128, 133])
@pytest.mark.parametrize("heads", [16, 64])
@torch.inference_mode()
def test_real_native_calls_match_after_q_kv_refresh(batch, length, heads):
    q, kv, sink = inputs(batch, length, heads)
    old, new = DeepseekV41NativeAttention(), DeepseekV41NativeAttention()
    for step in range(3):
        if step:
            q.mul_(0.625)
            kv.add_(0.03125)
        expected = run(old, q, kv, sink, cached=False)
        actual = run(new, q, kv, sink, cached=True)
        same_words(actual, expected)
    assert new.full_block_indices.cache_info().currsize == 1


@torch.inference_mode()
def test_noncontiguous_inputs_and_nonfinite_values_keep_native_semantics():
    q, kv, sink = inputs(1, 133, 64)
    backing = torch.zeros(1, 133, 1024, device="npu", dtype=kv.dtype)
    backing[..., ::2].copy_(kv)
    kv = backing[..., ::2]
    q = q.transpose(1, 2).contiguous().transpose(1, 2)
    q[..., :4] = torch.tensor([0.0, -0.0, float("inf"), float("nan")], device="npu", dtype=q.dtype)
    old, new = DeepseekV41NativeAttention(), DeepseekV41NativeAttention()
    same_words(run(new, q, kv, sink, cached=True), run(old, q, kv, sink, cached=False))


def elapsed(runner, q, kv, sink, requests, cached):
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    before = time.perf_counter()
    start.record()
    for _ in range(CALLS):
        for _ in range(requests):
            run(runner, q, kv, sink, cached=cached)
    end.record()
    submitted = time.perf_counter()
    end.synchronize()
    return 1000 * (submitted - before), 1000 * (time.perf_counter() - before), start.elapsed_time(end)


@pytest.mark.parametrize("requests", [1, 4, 8])
@torch.inference_mode()
def test_complete_attention_call_latency(requests):
    q, kv, sink = inputs(1, 133, 64)
    old, new = DeepseekV41NativeAttention(), DeepseekV41NativeAttention()
    same_words(run(new, q, kv, sink, cached=True), run(old, q, kv, sink, cached=False))
    measurements = {"parent": [], "candidate": []}
    for repeat in range(REPEATS):
        order = [("parent", old, False), ("candidate", new, True)]
        if repeat % 2:
            order.reverse()
        for name, runner, cached in order:
            measurements[name].append(elapsed(runner, q, kv, sink, requests, cached))
    medians = {
        name: tuple(statistics.median(values) for values in zip(*samples)) for name, samples in measurements.items()
    }
    print(
        {"calls": CALLS, "requests": requests, "metrics": ["host_submit_ms", "wall_ready_ms", "device_ms"], **medians}
    )
    assert all(value > 0 for values in medians.values() for value in values)
