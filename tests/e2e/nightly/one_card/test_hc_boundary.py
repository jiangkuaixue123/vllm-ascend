# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HC post, strict projection, BF16 mixed-input squares and RMS consumers."""

import json
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from vllm_ascend.models.common.ops.hc_post import hc_post_fused
from vllm_ascend.models.common.ops.hc_pre import hc_post_pre_delayed
from vllm_ascend.models.common.ops.rms_rsqrt_fused import try_rms_rsqrt_fused
from vllm_ascend.utils import enable_custom_op


def checkpoint(suffix):
    model = Path("/home/admin/model").resolve()
    index = json.loads(next(model.glob("*.safetensors.index.json")).read_text())["weight_map"]
    key = next(k for k in index if k.endswith("layers.3." + suffix))
    with safe_open(str(model / index[key]), framework="pt", device="cpu") as f:
        return f.get_tensor(key).npu().contiguous()


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("masked", [False, True])
@torch.inference_mode()
def test_hc_boundary_and_rms_consumers(rows, masked):
    assert enable_custom_op()
    torch.manual_seed(4713)
    x = torch.randn(rows, 4, 5120, device="npu", dtype=torch.bfloat16)
    sub = torch.randn(rows, 5120, device="npu", dtype=torch.bfloat16)
    previous = torch.randn(rows, 4, device="npu", dtype=torch.float32)
    post = torch.randn(rows, 4, device="npu", dtype=torch.float32)
    comb = torch.randn(rows, 4, 4, device="npu", dtype=torch.float32)
    row_mask = torch.ones(rows, device="npu", dtype=torch.bool)
    input_mask = row_mask.clone()
    if masked:
        row_mask[1::3] = False
        input_mask[::3] = False
        sub[::3] = float("nan")
    fn, scale, base, weight = (checkpoint(k) for k in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm.weight"))

    def reference():
        residual = hc_post_fused(sub, x, post, comb, row_mask=row_mask, input_mask=input_mask)
        assert residual is not None
        parts = torch.ops._C_ascend.npu_hc_pre_delayed(residual, fn, scale, base, previous, 1e-20)
        norm = try_rms_rsqrt_fused(parts[0], weight, 1e-20)
        return (residual, *parts, parts[0].float().square(), norm)

    def candidate():
        parts = torch.ops._C_ascend.npu_hc_post_pre_delayed(
            x, fn, scale, base, previous, sub, post, comb, row_mask, input_mask, 1e-20
        )
        norm = try_rms_rsqrt_fused(parts[1], weight, 1e-20, parts[-1])
        return (*parts, norm)

    def check(actual, expected):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
        # A real following GEMM and another HC update consume the materialized outputs.
        probe = fn[:8, :5120].contiguous().bfloat16()
        torch.testing.assert_close(actual[-1] @ probe.T, expected[-1] @ probe.T, atol=0, rtol=0, equal_nan=True)
        follow_a = hc_post_fused(actual[-1], actual[0], actual[3], actual[4], row_mask=row_mask)
        follow_b = hc_post_fused(expected[-1], expected[0], expected[3], expected[4], row_mask=row_mask)
        torch.testing.assert_close(follow_a, follow_b, atol=0, rtol=0, equal_nan=True)

    expected = reference()
    check(candidate(), expected)
    wrapped = hc_post_pre_delayed(x, fn, scale, base, previous, sub, post, comb, row_mask, input_mask, 1e-20, 1e-6, 20)
    assert wrapped is not None
    for a, b in zip(wrapped, (expected[0], expected[1], expected[2], expected[3], expected[4], expected[7])):
        torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            candidate()
    torch.npu.current_stream().wait_stream(stream)
    with torch.npu.graph(graph, stream=stream):
        captured = candidate()
    for replay in range(4):
        torch.manual_seed(4720 + replay)
        x.copy_(torch.randn_like(x))
        sub.copy_(torch.randn_like(sub))
        previous.copy_(torch.randn_like(previous))
        row_mask.copy_(torch.arange(rows, device="npu") % 3 != replay % 3)
        input_mask.copy_(torch.arange(rows, device="npu") % 2 != replay % 2)
        expected = reference()
        graph.replay()
        torch.npu.synchronize()
        check(captured, expected)
