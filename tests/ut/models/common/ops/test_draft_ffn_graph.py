# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.common.ops import draft_ffn_graph as module


def fixture_values():
    context = SimpleNamespace(
        is_draft_model=True,
        in_profile_run=False,
        moe_comm_type=module.MoECommType.FUSED_MC2,
        padded_num_tokens=20,
        mc2_mask=torch.ones(20, dtype=torch.bool),
    )
    moe = SimpleNamespace(
        communication=SimpleNamespace(
            sequence_parallel=True, replicated_shared=True, tp_group=SimpleNamespace(world_size=4), ep_size=16
        ),
        use_fused_runner=True,
        n_routed_experts=128,
        training=False,
    )
    return context, moe


@pytest.mark.parametrize("rows", [0, 1, 2, 5])
def test_capacity_key_is_shared_by_idle_and_active_replicas(monkeypatch, rows):
    context, moe = fixture_values()
    monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch, "is_inference_mode_enabled", lambda: True)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    states = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(rows, 5120))
    assert module.draft_ffn_graph_capacity(moe, states) == 20


@pytest.mark.parametrize(
    "name,value",
    [
        ("padded_num_tokens", 0),
        ("padded_num_tokens", 44),
        ("padded_num_tokens", 19),
        ("is_draft_model", False),
        ("in_profile_run", True),
        ("moe_comm_type", module.MoECommType.ALLGATHER),
    ],
)
def test_unsupported_uniform_context_falls_back(monkeypatch, name, value):
    context, moe = fixture_values()
    setattr(context, name, value)
    monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch, "is_inference_mode_enabled", lambda: True)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    assert module.draft_ffn_graph_capacity(moe, SimpleNamespace(device=SimpleNamespace(type="npu"))) is None


@pytest.mark.parametrize("failure", [False, True])
def test_capture_orders_all_widths_and_restores_original_mask(monkeypatch, failure):
    context, moe = fixture_values()
    original_mask = context.mc2_mask
    calls = []

    def execute(states, logits):
        calls.append(states.shape[0])
        assert context.mc2_mask is not original_mask
        assert logits.shape == (states.shape[0], 128)
        if failure and states.shape[0] == 2:
            raise RuntimeError("capture failed")
        return states

    moe._forward_experts = execute
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.npu, "NPUGraph", object)
    monkeypatch.setattr(torch.npu, "graph", lambda graph: nullcontext())
    graph = module.DraftFFNGraph()
    if failure:
        with pytest.raises(RuntimeError, match="capture failed"):
            graph._capture(moe, torch.zeros(2, 5120), 20, original_mask, ())
    else:
        group = graph._capture(moe, torch.zeros(2, 5120), 20, original_mask, ())
        assert list(group.entries) == list(range(6))
        assert calls == [rows for rows in range(6) for _ in range(2)]
    assert context.mc2_mask is original_mask
    assert graph.groups == {}


def test_refreshes_inputs_and_mask_without_recapture_after_storage_change(monkeypatch):
    context, _ = fixture_values()
    layer = torch.nn.Module()
    layer.experts = torch.nn.Linear(2, 2, bias=False)
    layer.shared_experts = torch.nn.Linear(2, 2, bias=False)
    layer.gate = SimpleNamespace(temperature=0.8, scaling_factor=1.5, top_k=3, renormalize=True)
    monkeypatch.setattr(module, "draft_ffn_graph_capacity", lambda *args: 20)
    monkeypatch.setattr(module, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.npu, "current_stream", lambda device: SimpleNamespace(npu_stream=7))
    graph = module.DraftFFNGraph()
    captures, replays = [], []

    def capture(moe, states, padded, mask, signature):
        captures.append(padded)
        entry = module._ExpertGraphEntry(
            torch.empty_like(states),
            torch.empty(states.shape[0], 128),
            torch.empty_like(states),
            SimpleNamespace(replay=lambda: replays.append(1)),
        )
        return module._ExpertGraphGroup(mask.clone(), {states.shape[0]: entry}, signature)

    monkeypatch.setattr(graph, "_capture", capture)
    states = torch.ones(2, 5120, dtype=torch.bfloat16)
    logits = torch.ones(2, 128)
    assert graph(layer, states, logits) is not None
    states.mul_(3)
    logits.mul_(4)
    context.mc2_mask = torch.arange(20) < 5
    assert graph(layer, states, logits) is not None
    entry = graph.groups[20].entries[2]
    torch.testing.assert_close(entry.states, states)
    torch.testing.assert_close(entry.logits, logits)
    torch.testing.assert_close(graph.groups[20].mask, context.mc2_mask)
    assert captures == [20] and len(replays) == 2
    original = layer.experts.weight
    layer.experts.weight = torch.nn.Parameter(original.clone())
    assert graph(layer, states, logits) is None
    layer.experts.weight = original
    layer.gate.temperature = 1.0
    assert graph(layer, states, logits) is None
    assert captures == [20] and len(replays) == 2
