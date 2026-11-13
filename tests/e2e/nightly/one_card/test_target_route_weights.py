# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import moe
from vllm_ascend.models.deepseek_v41.core import route_sqrtsoftplus
from vllm_ascend.ops.triton import deepseek_v41_route_weights as kernel


@pytest.mark.parametrize("tp_size", [4, 8])
@torch.inference_mode()
def test_model_gate_uses_fusion_only_for_tp4(tp_size):
    states = torch.ones(6, 5120, device="npu", dtype=torch.bfloat16)
    logits = torch.randn(6, 384, device="npu")
    gate = SimpleNamespace(bias=torch.randn(384, device="npu"), bias_vl=None, scaling_factor=1.5, temperature=1.0)
    expected = route_sqrtsoftplus(logits, gate.bias, 6)
    with (
        patch.object(moe, "get_tp_group", return_value=SimpleNamespace(world_size=tp_size)),
        patch.object(kernel, "selected_route_weights", wraps=kernel.selected_route_weights) as selected,
    ):
        actual = moe.DeepseekV41Gate.select_experts(gate, states, logits, 6, True)
        assert selected.call_count == (1 if tp_size == 4 else 0)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 3, 6, 12])
@torch.inference_mode()
def test_target_route_weights_replays_changed_scores_and_selection(rows):
    assert kernel.selected_route_weights is not None
    generator = torch.Generator().manual_seed(211 + rows)
    logits = torch.randn(rows, 384, generator=generator).npu()
    bias = torch.randn(384, generator=generator).npu()

    def execute(fused):
        return route_sqrtsoftplus(logits, bias, 6, fuse_weights=fused)

    execute(True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = execute(True)
    for case in ("random", "zero", "underflow", "large", "bias", "random"):
        logits.copy_(torch.randn(rows, 384, generator=generator))
        if case == "zero":
            logits.zero_()
        elif case == "underflow":
            logits.fill_(-torch.inf)
        elif case == "large":
            logits.mul_(40)
        elif case == "bias":
            bias.copy_(torch.randn(384, generator=generator) * 10)
        expected = execute(False)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        # Use the original FP32 comparison tolerance, with exact selection.
        torch.testing.assert_close(actual[0], expected[0])
        assert torch.isfinite(actual[0]).all()
    timings = {}
    for fused in (False, True):
        chain = torch.npu.NPUGraph()
        with torch.npu.graph(chain, capture_error_mode="thread_local", auto_dispatch_capture=True):
            for _ in range(40):
                execute(fused)
        chain.replay()
        torch.npu.synchronize()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        chain.replay()
        end.record()
        end.synchronize()
        timings["candidate" if fused else "parent"] = begin.elapsed_time(end)
    print("TARGET_ROUTE_CHAIN " + json.dumps({"rows": rows, "calls": 40, "ms": timings}), flush=True)


@pytest.mark.parametrize(
    "experts,top_k,rows,renormalize,scale",
    [
        (128, 3, 5, True, 1.5),
        (384, 6, 13, True, 1.5),
        (384, 6, 0, True, 1.5),
        (384, 1, 6, True, 1.5),
        (384, 6, 6, False, 1.5),
        (384, 6, 6, True, 2.0),
    ],
)
@torch.inference_mode()
def test_target_route_weights_unsupported_parameters_use_reference(experts, top_k, rows, renormalize, scale):
    logits = torch.zeros(rows, experts, device="npu")
    bias = torch.zeros(experts, device="npu")
    expected = route_sqrtsoftplus(logits, bias, top_k, renormalize, scale)
    with patch.object(kernel, "selected_route_weights", side_effect=AssertionError("unsupported fast path")):
        actual = route_sqrtsoftplus(logits, bias, top_k, renormalize, scale, fuse_weights=True)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
