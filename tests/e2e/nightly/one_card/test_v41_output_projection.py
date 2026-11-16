# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual WO_A and WO_B checkpoints keep their BF16 intermediate boundary."""

import json
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from vllm_ascend.models.common.ops.v41_output_projection import try_v41_wo_a
from vllm_ascend.models.deepseek_v41.attention_ops import fp8_quant_dequant
from vllm_ascend.utils import enable_custom_op


def weight(suffix, axis, tp):
    model = Path("/home/admin/model").resolve()
    index = json.loads(next(model.glob("*.safetensors.index.json")).read_text())["weight_map"]
    key = next(k for k in index if k.endswith("layers.3.attn." + suffix))
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
        source = f.get_slice(key)
        selection = [slice(None)] * len(source.get_shape())
        selection[axis] = slice(0, source.get_shape()[axis] // tp)
        return source[tuple(selection)].npu().contiguous()


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("rows", [12, 24, 48, 96])
@torch.inference_mode()
def test_receive_to_wo_b_layout_and_graph(tp, rows):
    assert enable_custom_op()
    torch.manual_seed(61116)
    groups = 8 // tp
    a = fp8_quant_dequant(torch.randn(rows, groups, 4096, device="npu", dtype=torch.bfloat16), 32)
    w = weight("wo_a.weight", 0, tp).view(groups, 1024, 4096)
    down = weight("wo_b.weight", 1, tp)
    saved = a.clone()

    def bounded(actual, left, right):
        # FP32 accumulation followed by one BF16 round; the oracle runs on CPU in FP64.
        left, right = left.cpu().double(), right.cpu().double()
        exact = left @ right
        abs_dot = left.abs() @ right.abs()
        k = left.shape[-1]
        gamma = k * 2**-24 / (1 - k * 2**-24)
        budget = (1 + 2**-8) * gamma * abs_dot + 2**-8 * exact.abs()
        error = (actual.cpu().double() - exact).abs()
        assert (error <= budget).all(), (error.max().item(), budget.max().item())

    def check(actual):
        bounded(actual.transpose(0, 1), a.transpose(0, 1), w.transpose(1, 2))
        partial = actual.flatten(1) @ down.T
        bounded(partial, actual.flatten(1), down.T)

    out = try_v41_wo_a(a, w)
    assert out is not None
    check(out)
    torch.testing.assert_close(a, saved, atol=0, rtol=0)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        try_v41_wo_a(a, w)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = try_v41_wo_a(a, w)
    for step in range(3):
        a.copy_(fp8_quant_dequant(torch.randn_like(a) * (step + 1), 32))
        graph.replay()
        check(captured)


@torch.inference_mode()
def test_direct_contract_rejects_bad_output_and_fallback():
    assert enable_custom_op()
    a = torch.zeros(24, 2, 4096, dtype=torch.bfloat16, device="npu")
    w = torch.zeros(2, 1024, 4096, dtype=a.dtype, device=a.device)
    assert try_v41_wo_a(a[:1], w) is None
    assert try_v41_wo_a(a.float(), w) is None
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.deepseek_v41_wo_a(a, w, torch.empty(24, 2, 512, dtype=a.dtype, device=a.device))


@pytest.mark.parametrize("rows", [12, 24, 48, 96])
@torch.inference_mode()
def test_zero_and_impulse_keep_exact_weights(rows):
    assert enable_custom_op()
    w = weight("wo_a.weight", 0, 4).view(2, 1024, 4096)
    a = torch.zeros(rows, 2, 4096, dtype=torch.bfloat16, device="npu")
    assert torch.count_nonzero(try_v41_wo_a(a, w)) == 0
    for group in range(2):
        a.zero_()
        a[:, group, 31 + group * 2017] = 1
        out = try_v41_wo_a(a, w)
        expected = torch.zeros_like(out)
        expected[:, group] = w[group, :, 31 + group * 2017]
        assert torch.equal(out, expected)


@torch.inference_mode()
def test_nonfinite_classification_matches_original():
    assert enable_custom_op()
    w = weight("wo_a.weight", 0, 4).view(2, 1024, 4096)
    a = torch.zeros(24, 2, 4096, dtype=torch.bfloat16, device="npu")
    for value in (float("inf"), -float("inf"), float("nan")):
        a.zero_()
        a[0, :, 31] = value
        new = try_v41_wo_a(a, w)
        old = torch.einsum("sgd,grd->sgr", a, w)
        assert torch.equal(torch.isnan(new), torch.isnan(old))
        assert torch.equal(torch.isposinf(new), torch.isposinf(old))
        assert torch.equal(torch.isneginf(new), torch.isneginf(old))
