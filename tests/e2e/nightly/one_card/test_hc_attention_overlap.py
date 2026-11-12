# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native HC input bits and refreshable stream/event graph dependencies."""

import statistics
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
import vllm.forward_context as forward_context

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_hc_pre_delayed import make_inputs
from vllm_ascend.models.common.ops.hc_attention_overlap import try_start_hc_attention
from vllm_ascend.models.common.ops.hc_pre import hc_pre_delayed
from vllm_ascend.ops.triton.deepseek_v41_hc_input import hc_input4
from vllm_ascend.utils import enable_custom_op

CALLS = 20
SAMPLES = 7


def exact(actual, expected):
    word = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    left, right = actual.view(word), expected.view(word)
    assert torch.equal(left, right), f"{int((left != right).sum())} different words"


@pytest.fixture(autouse=True)
def target_context(monkeypatch):
    assert enable_custom_op()
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: SimpleNamespace(is_draft_model=False))


def inputs(rows):
    x, fn, scale, base, previous = make_inputs(rows)
    layer = SimpleNamespace(
        attn=SimpleNamespace(tp=4, _query_overlap_stream=torch.npu.Stream()),
        config=SimpleNamespace(rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20),
        hc_attn_fn=fn,
        hc_attn_scale=scale,
        hc_attn_base=base,
    )
    return layer, x, previous


def native(layer, x, previous):
    return hc_pre_delayed(x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, previous, 1e-20, 1e-6, 20)


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_input_matches_native_for_changing_gates_and_special_values(rows):
    torch.npu.set_device(0)
    layer, x, previous = inputs(rows)
    for step in range(24):
        x.normal_().mul_(2.0 ** (step - 12))
        previous.uniform_(-2, 2)
        if step in (0, 1):
            x.fill_(0.0 if step == 0 else -0.0)
            previous.fill_(1)
        elif step == 2:
            x[:, 1].copy_(-x[:, 0])
            x[:, 3].copy_(-x[:, 2])
            previous.fill_(1)
        elif step == 3:
            x[:, :, ::97] = float("inf")
            x[:, :, 1::97] = float("nan")
        expected = native(layer, x, previous)
        assert expected is not None
        exact(hc_input4(x, previous), expected[0])


@pytest.mark.parametrize("rows", [6, 12])
@torch.inference_mode()
def test_graph_replay_preserves_coefficients_and_measures_complete_chain(rows):
    torch.npu.set_device(0)
    layer, x, previous = inputs(rows)
    runtime = SimpleNamespace(decode=object())
    weight = torch.randn(1280, 5120, device="npu", dtype=torch.bfloat16) * 0.01

    def complete(overlap):
        if overlap:
            pending = try_start_hc_attention(layer, x, previous, runtime)
            assert pending is not None
            mixed, outputs, ready = pending
        else:
            outputs = native(layer, x, previous)
            mixed = outputs[0]
        # A real native BF16 consumer waits only for the delayed input. A full
        # attention consumer is required separately before service qualification.
        projection = torch.nn.functional.linear(mixed, weight)
        if overlap:
            torch.npu.current_stream().wait_event(ready)
        return projection, mixed, *outputs[1:]

    graphs, results = [], []
    for overlap in (False, True):
        complete(overlap)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = complete(overlap)
        graphs.append(graph)
        results.append(result)
    for step in range(16):
        x.normal_().mul_(2.0 ** (step - 8))
        previous.uniform_(-2, 2)
        if step == 8:
            x.zero_()
        for graph in graphs:
            graph.replay()
            torch.npu.synchronize()
        for actual, expected in zip(results[1], results[0]):
            exact(actual, expected)
    timed = []
    for overlap in (False, True):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(CALLS):
                complete(overlap)
        timed.append(graph)
    samples = [[], []]
    for repeat in range(SAMPLES):
        for index in (repeat % 2, 1 - repeat % 2):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            timed[index].replay()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end))
    print(
        {
            "rows": rows,
            "calls": CALLS,
            "parent_ms": statistics.median(samples[0]),
            "candidate_ms": statistics.median(samples[1]),
        }
    )
