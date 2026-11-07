# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact draft residual fusion and complete stage integration on A3."""

import itertools
import json
import statistics
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu  # noqa: F401
from torch import nn

from tests.e2e.nightly.one_card.test_hc_post_fused import HIDDEN, WIDTH, exact_output, inputs
from vllm_ascend.models.common.ops.hc_post import hc_post_fused
from vllm_ascend.models.deepseek_v41 import dspark
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41RMSNorm
from vllm_ascend.models.deepseek_v41.core import hc_post, make_identity_pre_mix
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Block
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

MIX_WIDTH = 24
CALLS = 6
REPEATS = 5


@pytest.fixture(scope="module", autouse=True)
def device():
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(739)


def offset_inputs(rows, scale=1.0):
    return tuple(value[1:] for value in inputs(rows + 1, scale))


@pytest.mark.parametrize("rows", [5, 20])
@torch.inference_mode()
def test_draft_rows_match_parent_order_boundaries_and_offsets(rows):
    for scale in (0.0, 2.0**-120, 2.0**-20, 1.0, 2.0**20, 2.0**120):
        args = offset_inputs(rows, scale)
        saved = [value.clone() for value in args]
        exact_output(hc_post_fused(*args, draft=True), hc_post(*args))
        for original, copy in zip(args, saved):
            torch.testing.assert_close(original, copy, rtol=0, atol=0)
    for values in itertools.permutations((2.0**26, 1.0, -(2.0**26), 1.0)):
        sub, residual, post, comb = offset_inputs(rows)
        sub.zero_()
        residual.copy_(torch.tensor(values, device="npu", dtype=torch.bfloat16).view(1, WIDTH, 1))
        comb.fill_(1.0)
        exact_output(hc_post_fused(sub, residual, post, comb, draft=True), hc_post(sub, residual, post, comb))
    sub, residual, post, comb = offset_inputs(rows)
    sub[:, :4] = torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0], device="npu")
    residual[:, :, :4] = 0
    exact_output(hc_post_fused(sub, residual, post, comb, draft=True), hc_post(sub, residual, post, comb))


@pytest.mark.parametrize("rows", [5, 20])
@torch.inference_mode()
def test_replay_refreshes_draft_states_and_gates(rows):
    args = offset_inputs(rows)
    hc_post_fused(*args, draft=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = hc_post_fused(*args, draft=True)
    for _ in range(4):
        fresh = offset_inputs(rows)
        for destination, source in zip(args, fresh):
            destination.copy_(source)
        graph.replay()
        exact_output(output, hc_post(*fresh))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = SimpleNamespace(prefix="component.mtp.0.attn.cache")

    def forward(self, states, positions):
        return states * 0.75


class FeedForward(nn.Module):
    def forward(self, states):
        return states * 0.5


@pytest.fixture
def stage():
    with torch.inference_mode():
        layer = dspark.DSparkStage.__new__(dspark.DSparkStage)
        nn.Module.__init__(layer)
        layer.config = SimpleNamespace(
            hidden_size=HIDDEN, hc_mult=WIDTH, rms_norm_eps=1e-20, hc_eps=1e-6, hc_sinkhorn_iters=20
        )
        layer.attn, layer.ffn = Attention(), FeedForward()
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
        layer.eval()
        return layer


@pytest.mark.parametrize("rows", [5, 20, 10])
@torch.inference_mode()
def test_complete_stage_keeps_both_residual_updates_and_next_gate(stage, monkeypatch, rows):
    used = []

    def dispatch(*args, **kwargs):
        result = hc_post_fused(*args, **kwargs)
        used.append(result is not None)
        return result

    observed = Mock(side_effect=dispatch)
    monkeypatch.setattr(dspark, "hc_post_fused", observed)
    hidden = offset_inputs(rows)[1]
    positions = torch.arange(rows, device="npu", dtype=torch.int32)
    pre = make_identity_pre_mix(hidden)
    parent = DeepseekV41Block.post_residual.__get__(stage)
    candidate = dspark.DSparkStage.post_residual.__get__(stage)
    for _ in range(3):
        observed.reset_mock()
        used.clear()
        stage.post_residual = parent
        expected, expected_pre = stage(hidden, pre, positions)
        stage.post_residual = candidate
        actual, actual_pre = stage(hidden, pre, positions)
        assert observed.call_count == 2
        assert used == [rows in (5, 20)] * 2
        assert all(call.kwargs == {"draft": True} for call in observed.call_args_list)
        exact_output(actual, expected)
        torch.testing.assert_close(actual_pre.view(torch.int32), expected_pre.view(torch.int32), rtol=0, atol=0)
        hidden, pre = actual, actual_pre


@pytest.mark.parametrize("rows", [5, 20])
@torch.inference_mode()
def test_actual_draft_dispatch_rejects_unsupported_dtype_and_strides(stage, rows):
    args = list(offset_inputs(rows))
    runtime = torch.arange(rows, device="npu")
    args[0] = args[0].float()
    assert hc_post_fused(*args, draft=True) is None
    torch.testing.assert_close(stage.post_residual(*args, runtime), hc_post(*args), rtol=0, atol=0)
    args = list(offset_inputs(rows))
    args[2] = args[2].t().contiguous().t()
    assert not args[2].is_contiguous()
    assert hc_post_fused(*args, draft=True) is None
    exact_output(stage.post_residual(*args, runtime), hc_post(*args))


def elapsed(function, args, runtime):
    sub, residual, post, comb = args
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    begin = time.perf_counter()
    for _ in range(CALLS):
        residual = function(sub, residual, post, comb, runtime)
    submitted = time.perf_counter()
    end.record()
    end.synchronize()
    return (1000 * (submitted - begin), 1000 * (time.perf_counter() - begin), start.elapsed_time(end))


@pytest.mark.parametrize("rows", [5, 20])
@torch.inference_mode()
def test_complete_draft_residual_chain_latency(stage, rows):
    args = offset_inputs(rows)
    args[-1].fill_(0.25)
    runtime = torch.arange(rows, device="npu")
    calls = {
        "parent": DeepseekV41Block.post_residual.__get__(stage),
        "candidate": dspark.DSparkStage.post_residual.__get__(stage),
    }
    for function in calls.values():
        function(*args, runtime)
    samples = {name: [] for name in calls}
    for repeat in range(REPEATS):
        for name in ("parent", "candidate") if repeat % 2 == 0 else ("candidate", "parent"):
            samples[name].append(elapsed(calls[name], args, runtime))
    print(
        "DSPARK_HC_POST_TIMING "
        + json.dumps(
            {
                "rows": rows,
                "calls": CALLS,
                "metrics": ["host_ms", "ready_ms", "device_ms"],
                **{name: list(map(statistics.median, zip(*values))) for name, values in samples.items()},
            }
        ),
        flush=True,
    )
