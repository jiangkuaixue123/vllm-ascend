# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact draft HC graph replay, lifecycle fallbacks, and eager latency."""

import statistics
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.common.ops.draft_hc_graph import DraftHCMixGraph, draft_hc_pre_and_mixes
from vllm_ascend.models.deepseek_v41 import dspark

PARAMETERS = (1e-20, 1e-6, 20)
ROWS = (5, 10, 20, 40)
REFRESHES = 12
CALLS = 24
REPEATS = 5


def inputs(rows, device="npu"):
    if device == "npu":
        torch.npu.set_device(0)
    torch.manual_seed(20261106)
    # Contiguous slices with nonzero offsets must also be copied correctly.
    x = torch.randn(rows + 1, 4, 5120, device=device, dtype=torch.bfloat16)[1:]
    pre = torch.randn(rows + 1, 4, device=device, dtype=torch.float32)[1:]
    fn = torch.randn(24, 20480, device=device, dtype=torch.float32) / 128
    scale = torch.tensor([0.4, 0.8, 1.2], device=device, dtype=torch.float32)
    base = torch.randn(24, device=device, dtype=torch.float32)
    return x, pre, fn, scale, base


def reference(*args):
    return draft_hc_pre_and_mixes(*args, *PARAMETERS)


def assert_bits(actual, expected):
    assert len(actual) == len(expected)
    for output, ref in zip(actual, expected):
        assert output.shape == ref.shape and output.dtype == ref.dtype
        word = torch.int16 if output.dtype == torch.bfloat16 else torch.int32
        assert torch.equal(output.cpu().view(word), ref.cpu().view(word))


@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_refreshes_inputs_pre_gates_and_weights_exactly(rows):
    args = inputs(rows)
    replay = DraftHCMixGraph(*PARAMETERS)
    first = time.perf_counter()
    outputs = replay(*args)
    torch.npu.synchronize()
    print({"rows": rows, "first_use_ms": 1000 * (time.perf_counter() - first)})
    addresses = tuple(t.data_ptr() for t in outputs)
    for step in range(REFRESHES):
        for tensor in args:
            tensor.copy_(torch.randn_like(tensor))
        if step == 0:
            args[0].zero_()
        elif step == 1:
            args[0].fill_(-0.0)
        saved = tuple(t.clone() for t in args)
        expected = reference(*args)
        actual = replay(*args)
        assert_bits(actual, expected)
        assert_bits(args, saved)
        assert tuple(t.data_ptr() for t in actual) == addresses
    assert len(replay.entries) == 1


@pytest.mark.parametrize("rows", [0, 1, 6, 15, 45])
@torch.inference_mode()
def test_other_geometries_keep_parent(rows):
    replay = DraftHCMixGraph(*PARAMETERS)
    assert replay(*inputs(rows)) is None
    assert not replay.entries


def test_cpu_and_training_fallback():
    args = inputs(5, device="cpu")
    args[0].requires_grad_(True)
    replay = DraftHCMixGraph(*PARAMETERS)
    assert replay(*args) is None
    reference(*args)[0].float().sum().backward()
    assert args[0].grad is not None
    assert not replay.entries


@torch.inference_mode()
def test_existing_outer_capture_uses_parent():
    args = inputs(5)
    replay = DraftHCMixGraph(*PARAMETERS)
    reference(*args)
    outer = torch.npu.NPUGraph()
    with torch.npu.graph(outer):
        assert replay(*args) is None
        actual = reference(*args)
    args[0].copy_(torch.randn_like(args[0]))
    outer.replay()
    assert_bits(actual, reference(*args))
    assert not replay.entries


@torch.inference_mode()
def test_stream_and_parameter_storage_changes_keep_parent():
    args = inputs(5)
    replay = DraftHCMixGraph(*PARAMETERS)
    assert replay(*args) is not None
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert replay(*args) is None
    changed = (*args[:2], args[2].clone(), *args[3:])
    assert replay(*changed) is None
    assert_bits(replay(*args), reference(*args))
    assert len(replay.entries) == 1


@torch.inference_mode()
def test_entries_and_sublayers_keep_separate_output_storage():
    attn, ffn = DraftHCMixGraph(*PARAMETERS), DraftHCMixGraph(*PARAMETERS)
    first_args, next_args = inputs(5), inputs(20)
    first = attn(*first_args)
    snapshot = tuple(t.clone() for t in first)
    attn(*next_args)
    for tensor in first_args:
        tensor.add_(0.25)
    second = ffn(*first_args)
    assert_bits(first, snapshot)
    assert_bits(second, reference(*first_args))
    assert len(attn.entries) == 2 and len(ffn.entries) == 1


@pytest.mark.parametrize("name", ["attn", "ffn"])
@pytest.mark.parametrize("dummy", [False, True])
@torch.inference_mode()
def test_real_stage_dispatch_and_dummy_fallback(monkeypatch, name, dummy):
    args = inputs(5)
    stage = dspark.DSparkStage.__new__(dspark.DSparkStage)
    torch.nn.Module.__init__(stage)
    stage.config = SimpleNamespace(rms_norm_eps=PARAMETERS[0], hc_eps=PARAMETERS[1], hc_sinkhorn_iters=PARAMETERS[2])
    for part, tensor in zip(("fn", "scale", "base"), args[2:]):
        stage.register_parameter(f"hc_{name}_{part}", torch.nn.Parameter(tensor, requires_grad=False))
    stage._draft_hc_graphs = {name: DraftHCMixGraph(*PARAMETERS)}
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: dummy)
    actual = stage.pre_and_mixes(*args[:2], name, None)
    assert_bits(actual, reference(*args))
    assert bool(stage._draft_hc_graphs[name].entries) is not dummy


def elapsed(function, args):
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin = time.perf_counter()
    start.record()
    for _ in range(CALLS):
        function(*args)
    end.record()
    submitted = time.perf_counter()
    end.synchronize()
    return (1000 * (submitted - begin), 1000 * (time.perf_counter() - begin), start.elapsed_time(end))


@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_complete_eager_component_latency(rows):
    args = inputs(rows)
    replay = DraftHCMixGraph(*PARAMETERS)
    replay(*args)
    reference(*args)
    measurements = {"parent": [], "candidate": []}
    for repeat in range(REPEATS):
        order = [("parent", reference), ("candidate", replay)]
        if repeat % 2:
            order.reverse()
        for name, function in order:
            measurements[name].append(elapsed(function, args))
    medians = {
        name: tuple(statistics.median(values) for values in zip(*samples)) for name, samples in measurements.items()
    }
    print({"rows": rows, "calls": CALLS, "metrics": ["host_submit_ms", "wall_ready_ms", "device_ms"], **medians})
    assert all(value > 0 for values in medians.values() for value in values)
