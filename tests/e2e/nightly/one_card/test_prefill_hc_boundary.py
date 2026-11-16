# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve large HC residual, mixing, projection and RMS consumers."""

import json
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from vllm_ascend.models.common.ops.hc_prefill import hc_prefill_post, hc_prefill_post_prepare, hc_prefill_prepare
from vllm_ascend.models.deepseek_v41.core import hc_mixes
from vllm_ascend.ops.triton.deepseek_v41_rms_rsqrt_fused import rms_rsqrt_fused


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    word = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(actual.view(word), expected.view(word))


def checkpoint(suffix):
    model = Path("/home/admin/model").resolve()
    index = json.loads(next(model.glob("*.safetensors.index.json")).read_text())["weight_map"]
    key = next(k for k in index if k.endswith("layers.3." + suffix))
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key).npu().contiguous()


def inputs(rows):
    torch.manual_seed(251116)
    x = torch.randn(rows, 4, 5120, device="npu", dtype=torch.bfloat16)
    sub = torch.randn(rows, 5120, device="npu", dtype=torch.bfloat16)
    post = torch.rand(rows, 4, device="npu")
    comb = torch.rand(rows, 4, 4, device="npu")
    pre = torch.rand(rows, 4, device="npu")
    return x, sub, post, comb, pre


def reference(args):
    x, sub, post, comb, pre = args
    residual = hc_prefill_post(sub, x, post, comb)
    mixed, states = hc_prefill_prepare(residual, pre)
    return residual, mixed, states, mixed.float().square()


def run(args):
    x, sub, post, comb, pre = args
    return hc_prefill_post_prepare(sub, x, post, comb, pre)


@pytest.mark.parametrize("rows", [257, 513, 2008, 4016])
@torch.inference_mode()
def test_full_prefill_boundary_and_consumers(rows):
    args = inputs(rows)
    before = args[0].clone()
    old = reference(args)
    new = run(args)
    assert new is not None
    for a, b in zip(new, old):
        exact(a, b)
    fn, scale, base = (checkpoint(k) for k in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"))
    for a, b in zip(hc_mixes(new[2], fn, scale, base), hc_mixes(old[2], fn, scale, base)):
        exact(a, b)
    weight = checkpoint("ffn_norm.weight")
    exact(rms_rsqrt_fused(new[1], weight, 1e-20, new[3]), rms_rsqrt_fused(old[1], weight, 1e-20))
    exact(args[0], before)


@torch.inference_mode()
def test_cancellation_nonfinite_and_graph_refresh():
    args = inputs(257)
    run(args)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = run(args)
    for pattern in [
        (1.0, 2.0, 3.0, 4.0),
        (2**26, 1, -(2**26), 1),
        (-0.0,) * 4,
        (float("inf"), 1, -float("inf"), float("nan")),
    ]:
        args[0].copy_(torch.tensor(pattern, dtype=torch.bfloat16, device="npu")[None, :, None])
        for t in args[1:]:
            t.copy_(torch.randn_like(t))
        graph.replay()
        for a, b in zip(actual, reference(args)):
            exact(a, b)


@torch.inference_mode()
def test_unsupported_rows_fall_back():
    args = inputs(48)
    assert run(args) is None
