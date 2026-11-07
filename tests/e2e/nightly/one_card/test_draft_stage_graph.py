# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole draft stages with native W8A8 attention and unchanged communication count."""

import json
import statistics
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.e2e.nightly.one_card.test_draft_attention_graph import HIDDEN
from tests.e2e.nightly.one_card.test_draft_output_graph import device_inputs
from vllm_ascend.models.common.ops.draft_attention_graph import DraftAttentionGraph
from vllm_ascend.models.common.ops.draft_stage_graph import DraftStageGraph
from vllm_ascend.models.deepseek_v41 import dspark
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41RMSNorm
from vllm_ascend.models.deepseek_v41.core import make_identity_pre_mix
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Block

pytest_plugins = ("tests.e2e.nightly.one_card.test_draft_output_graph",)

WIDTH = 4
MIX_WIDTH = 24
CALLS = 6
REPEATS = 5


class FeedForward(nn.Module):
    def forward(self, states):
        assert not torch.npu.is_current_stream_capturing()
        return states * 0.5


@pytest.fixture
def stage(full_model, monkeypatch):
    attention, collective = full_model
    collective.record = False
    with torch.inference_mode():
        layer = dspark.DSparkStage.__new__(dspark.DSparkStage)
        nn.Module.__init__(layer)
        layer.config = SimpleNamespace(
            hidden_size=HIDDEN, hc_mult=WIDTH, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20
        )
        layer.attn, layer.ffn = attention, FeedForward()
        layer.attn._attention_graph = DraftAttentionGraph(project_output=True)
        layer.attn_norm = DeepseekV41RMSNorm(HIDDEN, layer.config.rms_norm_eps)
        layer.ffn_norm = DeepseekV41RMSNorm(HIDDEN, layer.config.rms_norm_eps)
        for norm in (layer.attn_norm, layer.ffn_norm):
            norm.weight = nn.Parameter(torch.ones(HIDDEN, device="npu", dtype=torch.bfloat16), requires_grad=False)
        for name in ("attn", "ffn"):
            layer.register_parameter(
                f"hc_{name}_fn",
                nn.Parameter(torch.randn(MIX_WIDTH, WIDTH * HIDDEN, device="npu") * 0.001, requires_grad=False),
            )
            layer.register_parameter(
                f"hc_{name}_scale", nn.Parameter(torch.full((3,), 0.1, device="npu"), requires_grad=False)
            )
            layer.register_parameter(
                f"hc_{name}_base", nn.Parameter(torch.zeros(MIX_WIDTH, device="npu"), requires_grad=False)
            )
        layer._stage_graphs = {"attn": DraftStageGraph(), "ffn": DraftStageGraph(after_attention=True)}
        layer.eval()
    attention_call = Mock(wraps=attention.forward)
    ffn_call = Mock(wraps=layer.ffn.forward)
    monkeypatch.setattr(attention, "forward", attention_call)
    monkeypatch.setattr(layer.ffn, "forward", ffn_call)
    return layer, collective, attention_call, ffn_call


def stage_inputs(stage, batch=1, step=0, dtype=torch.int32):
    _, positions, _ = device_inputs(stage.attn, batch=batch, step=step, position_dtype=dtype)
    hidden = torch.randn(batch * 5 + 1, WIDTH, HIDDEN, device="npu", dtype=torch.bfloat16)[1:]
    pre = make_identity_pre_mix(hidden)
    return hidden, pre, positions


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    words = torch.int32 if actual.dtype == torch.float32 else torch.int16
    torch.testing.assert_close(actual.view(words), expected.view(words), rtol=0, atol=0)


def compare(stage, observed, args):
    collective, attention_call, ffn_call = observed
    before = (collective.calls, attention_call.call_count, ffn_call.call_count)
    expected = tuple(value.clone() for value in DeepseekV41Block.forward(stage, *args))
    actual = stage(*args)
    assert (collective.calls, attention_call.call_count, ffn_call.call_count) == tuple(value + 2 for value in before)
    for result, reference in zip(actual, expected):
        exact(result, reference)
    return actual


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("dummy", [False, True])
@torch.inference_mode()
def test_complete_stage_replays_changed_inputs_gates_and_native_attention(stage, monkeypatch, batch, dummy):
    layer, *observed = stage
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: dummy)
    for step in range(4):
        args = stage_inputs(layer, batch, step, torch.int32 if step < 3 else torch.int64)
        if dummy:
            layer.attn.cache.metadata = lambda: None
        if step == 1:
            args[0].zero_()
        if step == 2:
            args[1].copy_(torch.rand_like(args[1]))
            layer.hc_attn_fn.mul_(0.9)
            layer.hc_ffn_scale.add_(0.01)
            layer.attn_norm.weight.mul_(1.01)
            layer.ffn_norm.weight.mul_(0.99)
        saved = tuple(value.clone() for value in args)
        compare(layer, observed, args)
        for value, original in zip(args, saved):
            if value.dtype in (torch.float32, torch.bfloat16):
                exact(value, original)
            else:
                assert torch.equal(value, original)
        assert all(set(graph.entries) == {batch * 5} for graph in layer._stage_graphs.values())


@pytest.mark.parametrize("part,change", [("attn", "parameter"), ("ffn", "parameter"), ("ffn", "epsilon")])
@torch.inference_mode()
def test_storage_and_scalar_changes_fall_back_without_repeating_attention(stage, monkeypatch, part, change):
    layer, *observed = stage
    args = stage_inputs(layer)
    layer(*args)
    entry = layer._stage_graphs[part].entries[5]
    if change == "parameter":
        name = f"hc_{part}_fn"
        monkeypatch.setattr(layer, name, nn.Parameter(getattr(layer, name).clone(), requires_grad=False))
    else:
        monkeypatch.setattr(layer.ffn_norm, "eps", 1e-5)
    compare(layer, observed, args)
    assert layer._stage_graphs[part].entries[5] is entry


@torch.inference_mode()
def test_other_streams_and_outer_capture_keep_local_graphs_out(stage):
    layer = stage[0]
    hidden, pre, _ = stage_inputs(layer)
    graph = layer._stage_graphs["attn"]
    assert graph(layer, (hidden, pre)) is not None
    other = torch.npu.Stream()
    other.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(other):
        assert graph(layer, (hidden, pre)) is None
    outer = torch.npu.NPUGraph()
    with torch.npu.graph(outer):
        assert graph(layer, (hidden, pre)) is None
        output = hidden + 1
    outer.replay()
    exact(output, hidden + 1)


@torch.inference_mode()
def test_batch_transitions_and_empty_stage_preserve_parent_communication(stage):
    layer, *observed = stage
    for batch in (1, 4, 1, 4, 2):
        compare(layer, observed, stage_inputs(layer, batch))
    assert all(set(graph.entries) == {5, 20} for graph in layer._stage_graphs.values())
    hidden = torch.empty(0, WIDTH, HIDDEN, device="npu", dtype=torch.bfloat16)
    pre = make_identity_pre_mix(hidden)
    positions = torch.empty(0, device="npu", dtype=torch.int32)
    before = tuple(item.call_count for item in observed[1:])
    actual = layer(hidden, pre, positions)
    assert tuple(item.call_count for item in observed[1:]) == (before[0], before[1] + 1)
    assert actual[0] is hidden and actual[1] is pre


def elapsed(function, args):
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    hidden, pre, positions = args
    start.record()
    begin = time.perf_counter()
    for _ in range(CALLS):
        hidden, pre = function(hidden, pre, positions)
    submitted = time.perf_counter()
    end.record()
    end.synchronize()
    return (1000 * (submitted - begin), 1000 * (time.perf_counter() - begin), start.elapsed_time(end))


@pytest.mark.parametrize("batch", [1, 4])
@torch.inference_mode()
def test_complete_stage_with_native_attention_latency(stage, batch):
    layer = stage[0]
    args = stage_inputs(layer, batch)
    calls = {"parent": lambda *values: DeepseekV41Block.forward(layer, *values), "candidate": layer}
    for function in calls.values():
        function(*args)
    samples = {name: [] for name in calls}
    for repeat in range(REPEATS):
        for name in ("parent", "candidate") if repeat % 2 == 0 else ("candidate", "parent"):
            samples[name].append(elapsed(calls[name], args))
    print(
        "DSPARK_STAGE_LOCAL_GRAPH_TIMING "
        + json.dumps(
            {
                "batch": batch,
                "calls": CALLS,
                "metrics": ["host_ms", "ready_ms", "device_ms"],
                **{name: list(map(statistics.median, zip(*values))) for name, values in samples.items()},
            }
        ),
        flush=True,
    )
