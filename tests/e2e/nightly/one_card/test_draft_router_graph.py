# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact original routing arithmetic and bounded local graph fallbacks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.common.ops.draft_router_graph import DraftRouterGraph
from vllm_ascend.models.deepseek_v41.core import route_sqrtsoftplus
from vllm_ascend.models.deepseek_v41.moe import DeepseekV41Gate


def make_gate(prefix="test.draft"):
    config = SimpleNamespace(
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        num_experts_per_tok=3,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        gate_temp=0.8,
        n_routed_experts=128,
        hidden_size=5120,
    )
    gate = DeepseekV41Gate(config).eval().npu()
    with torch.no_grad():
        gate.weight.normal_(std=0.01)
        gate.bias.normal_(std=0.1)
    gate._draft_router_graph = DraftRouterGraph(prefix)
    return gate


def reference(gate, logits):
    return route_sqrtsoftplus(logits, gate.bias, gate.top_k, True, gate.scaling_factor, gate.temperature)


@pytest.mark.parametrize("rows", [2, 5])
@torch.inference_mode()
def test_exact_routes_refresh_logits_bias_and_dirty_outputs(rows):
    gate = make_gate()
    graph = gate._draft_router_graph
    original = None
    for step in range(5):
        logits = torch.randn(rows + 1, 128, device="npu")[1:]
        if step == 1:
            logits.zero_()
            gate.bias.zero_()  # Equal scores exercise the parent's tie order.
        if step == 2:
            logits.mul_(100)
        if step == 3:
            gate.bias.normal_()
        before = logits.clone()
        expected = reference(gate, logits)
        actual = gate.select_experts(None, logits, gate.top_k, True)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, rtol=0, atol=0)
        torch.testing.assert_close(logits, before, rtol=0, atol=0)
        if original is None:
            original = graph.entries[rows]
        assert graph.entries[rows] is original
        actual[0].fill_(-11)
        actual[1].fill_(-1)
    assert len(graph.entries) == 1


@torch.inference_mode()
def test_bias_storage_settings_and_stream_changes_use_parent():
    gate = make_gate()
    graph = gate._draft_router_graph
    logits = torch.randn(2, 128, device="npu")
    assert graph(gate, logits, 3, True) is not None
    original = gate.bias
    gate.bias = nn.Parameter(original.clone(), requires_grad=False)
    assert graph(gate, logits, 3, True) is None
    for actual, expected in zip(gate.select_experts(None, logits, 3, True), reference(gate, logits)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gate.bias = original
    gate.temperature = 1.0
    assert graph(gate, logits, 3, True) is None
    gate.temperature = 0.8
    gate.scaling_factor = 2.0
    assert graph(gate, logits, 3, True) is None
    gate.scaling_factor = 1.5
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        assert graph(gate, logits, 3, True) is None
    torch.npu.current_stream().wait_stream(stream)
    assert len(graph.entries) == 1


@pytest.mark.parametrize("rows", [0, 1, 4, 6, 10])
@torch.inference_mode()
def test_other_rows_use_original_routing(rows):
    gate = make_gate()
    logits = torch.randn(rows, 128, device="npu")
    for actual, expected in zip(gate.select_experts(None, logits, 3, True), reference(gate, logits)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not gate._draft_router_graph.entries


@torch.inference_mode()
def test_outer_capture_and_training_keep_parent():
    gate = make_gate()
    logits = torch.randn(5, 128, device="npu")
    gate.train()
    assert gate._draft_router_graph(gate, logits, 3, True) is None
    gate.eval()
    reference(gate, logits)
    outer = torch.npu.NPUGraph()
    with torch.npu.graph(outer):
        actual = gate.select_experts(None, logits, 3, True)
    assert not gate._draft_router_graph.entries
    logits.normal_()
    outer.replay()
    for value, expected in zip(actual, reference(gate, logits)):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_cpu_and_noncontiguous_logits_keep_parent():
    gate = make_gate()
    assert gate._draft_router_graph(gate, torch.randn(2, 128), 3, True) is None
    logits = torch.randn(2, 256, device="npu")[:, ::2]
    assert gate._draft_router_graph(gate, logits, 3, True) is None
    assert gate._draft_router_graph(gate, logits.contiguous(), 3, False) is None
