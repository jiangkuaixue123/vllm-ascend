# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE output rounding, dynamic masks and the unchanged HC consumer on A3."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.models.deepseek_v41 import moe
from vllm_ascend.ops.triton import deepseek_v41_moe_output as kernel
from vllm_ascend.ops.triton.hc_post import hc_post4
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(
    not HAS_TRITON or get_ascend_device_type() != AscendDeviceType.A3, reason="A3 Triton only"
)


def reference(routed, shared, active, dtype=torch.bfloat16):
    return torch.where(active[:, None], (routed + shared.float()).to(dtype), 0.0)


def equal_bits(actual, expected):
    assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("shared_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_moe_output_changing_graph(rows, shared_dtype):
    count = rows * 5120
    routed = torch.zeros(rows, 5120, device="npu", dtype=torch.float32)
    shared = torch.zeros_like(routed, dtype=shared_dtype)
    # TP-sharded bool masks need not have a 32B-aligned start.
    mask_store = torch.ones(rows + 3, device="npu", dtype=torch.bool)
    active = mask_store[3:]
    assert kernel.moe_output(routed, shared, active, torch.bfloat16) is not None
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        out = kernel.moe_output(routed, shared, active, torch.bfloat16)
    seq = torch.arange(count, dtype=torch.int32)
    for step in range(64):
        # Sweep every BF16 word, then vary both operands and active rows.
        left = (seq + step * count).short().view(torch.bfloat16).float().reshape_as(routed)
        right = (
            torch.zeros_like(left)
            if step < 6
            else (seq * 47 + step * 997).short().view(torch.bfloat16).float().reshape_as(routed)
        )
        live = torch.ones(rows, dtype=torch.bool) if step < 6 else (torch.arange(rows) + step) % 3 != 0
        if step % 11 == 10:
            live.fill_(False)
        routed.copy_(left)
        shared.copy_(right)
        active.copy_(live)
        expected = reference(routed, shared, active)
        graph.replay()
        torch.npu.synchronize()
        equal_bits(out, expected)
        assert torch.equal(routed.cpu().view(torch.int32), left.view(torch.int32))
        assert torch.equal(shared.cpu().view(torch.int16), right.to(shared_dtype).view(torch.int16))
        assert out.data_ptr() not in (routed.data_ptr(), shared.data_ptr())
        assert mask_store[:3].all()


@torch.inference_mode()
def test_moe_output_fp32_bf16_rounding_boundaries():
    routed = torch.empty(6, 5120, device="npu", dtype=torch.float32)
    shared = torch.zeros_like(routed, dtype=torch.bfloat16)
    active = torch.ones(6, device="npu", dtype=torch.bool)
    count = routed.numel()
    for tail in (0, 0x7FFF, 0x8000, 0x8001, 0xFFFF):
        for start in range(0, 65536, count):
            words = (((torch.arange(count, dtype=torch.int64) + start) % 65536) << 16) | tail
            routed.copy_(words.int().view(torch.float32).reshape_as(routed))
            equal_bits(kernel.moe_output(routed, shared, active, torch.bfloat16), reference(routed, shared, active))


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("shared_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_moe_output_real_decode_and_hc_consumer(rows, shared_dtype):
    torch.manual_seed(19)
    x = torch.randn(rows, 5120, device="npu", dtype=torch.bfloat16)
    routed = torch.randn_like(x, dtype=torch.float32)
    shared = torch.randn_like(x, dtype=shared_dtype)
    active = torch.ones(rows, device="npu", dtype=torch.bool)
    residual = torch.randn(rows, 4, 5120, device="npu", dtype=torch.bfloat16)
    post = torch.randn(rows, 4, device="npu", dtype=torch.float32)
    comb = torch.randn(rows, 4, 4, device="npu", dtype=torch.float32)
    calls, inputs = [], []

    def route(states, *, decode):
        assert decode
        calls.append("routed")
        inputs.append(states)
        return routed

    def share(states):
        calls.append("shared")
        assert states is inputs[-1]
        return shared

    layer = SimpleNamespace(n_routed_experts=384, hidden_size=5120, _routed_output=route, _shared_output=share)

    def run():
        out = moe.DeepseekV41MoE.forward_decode(layer, x, active)
        return out, hc_post4(out, residual, post, comb)

    with patch.object(moe, "moe_output", wraps=kernel.moe_output) as entry:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            out, mixed = run()
        assert entry.call_count == 2 and calls == ["routed", "shared"] * 2
        for step in range(64):
            active.copy_((torch.arange(rows) + step) % 3 != 0)
            if step % 7 == 0:
                active.fill_(False)
            routed.fill_(float("nan") if step % 7 == 0 else step / 32)
            shared.fill_(float("inf") if step % 7 == 0 else -step / 64)
            expected = reference(routed, shared, active)
            expected_hc = hc_post4(expected, residual, post, comb)
            graph.replay()
            torch.npu.synchronize()
            equal_bits(out, expected)
            equal_bits(mixed, expected_hc)
            equal_bits(inputs[-1], torch.where(active[:, None], x, 0.0))


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "rows",
        "width",
        "output_dtype",
        "routed_dtype",
        "shared_dtype",
        "shared_shape",
        "mask_dtype",
        "mask_shape",
        "stride",
        "alignment",
        "negative",
        "grad",
        "training",
        "batch_invariant",
        "compiling",
        "no_triton",
        "a2",
        "nz",
    ],
)
def test_moe_output_fallback(case):
    routed = torch.zeros(6, 5120, device="npu", dtype=torch.float32)
    shared = torch.zeros_like(routed, dtype=torch.bfloat16)
    active = torch.ones(6, device="npu", dtype=torch.bool)
    dtype = torch.bfloat16
    with ExitStack() as stack:
        if case != "training":
            stack.enter_context(torch.inference_mode())
        if case == "cpu":
            routed, shared, active = routed.cpu(), shared.cpu(), active.cpu()
        elif case == "rows":
            routed, shared, active = routed[:2], shared[:2], active[:2]
        elif case == "width":
            routed, shared = routed[:, :256].contiguous(), shared[:, :256].contiguous()
        elif case == "output_dtype":
            dtype = torch.float32
        elif case == "routed_dtype":
            routed = routed.bfloat16()
        elif case == "shared_dtype":
            shared = shared.half()
        elif case == "shared_shape":
            shared = shared[:3]
        elif case == "mask_dtype":
            active = active.int()
        elif case == "mask_shape":
            active = active[:, None]
        elif case == "stride":
            shared = torch.zeros(6, 10240, device="npu", dtype=torch.bfloat16)[:, ::2]
        elif case == "alignment":
            shared = torch.zeros(6 * 5120 + 1, device="npu", dtype=torch.bfloat16)[1:].view(6, 5120)
        elif case == "negative":
            shared = torch._neg_view(shared)
        elif case == "grad":
            routed.requires_grad_(True)
        elif case == "batch_invariant":
            stack.enter_context(patch.object(kernel.envs, "VLLM_BATCH_INVARIANT", True))
        elif case == "compiling":
            stack.enter_context(patch.object(torch.compiler, "is_compiling", return_value=True))
        elif case == "no_triton":
            stack.enter_context(patch.object(kernel, "HAS_TRITON", False))
        elif case == "a2":
            stack.enter_context(patch.object(kernel, "get_ascend_device_type", return_value=AscendDeviceType.A2))
        elif case == "nz":
            previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
            try:
                torch.npu.config.allow_internal_format = True
                routed = torch_npu.empty_with_format(routed.shape, dtype=routed.dtype, device="npu", acl_format=29)
            finally:
                torch.npu.config.allow_internal_format = previous is not None and previous.decode() == "enable"
            assert int(torch_npu.get_npu_format(routed)) != int(torch_npu.Format.ND)
        assert kernel.moe_output(routed, shared, active, dtype) is None


@pytest.mark.parametrize("rows", [3, 6, 12])
@torch.inference_mode()
def test_block_owned_merge_and_hc_mask(rows):
    routed = torch.randn(rows, 5120, device="npu", dtype=torch.float32)
    shared = torch.randn_like(routed, dtype=torch.bfloat16)
    active = torch.ones(rows, dtype=torch.bool, device="npu")
    residual = torch.randn(rows, 4, 5120, device="npu", dtype=torch.bfloat16)
    post = torch.randn(rows, 4, device="npu", dtype=torch.float32)
    comb = torch.randn(rows, 4, 4, device="npu", dtype=torch.float32)

    def flow():
        value = kernel.moe_output(routed, shared, None, torch.bfloat16)
        assert value is not None
        return value, hc_post4(value, residual, post, comb, active)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        flow()
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output, hc = flow()
    for step in range(4):
        routed.copy_(torch.randn_like(routed) * (step + 1))
        shared.copy_(torch.randn_like(shared) * (step + 1))
        active.copy_(torch.arange(rows, device="npu") % 3 != step % 3)
        if step == 3:
            active.zero_()
            routed.fill_(float("nan"))
        graph.replay()
        expected = (routed + shared.float()).bfloat16()
        equal_bits(output, expected)
        equal_bits(hc, hc_post4(expected, residual, post, comb, active))
