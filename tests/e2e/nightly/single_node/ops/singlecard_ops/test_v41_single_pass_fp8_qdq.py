# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the production entry and the actual A3 kernel under changing graph inputs."""

import math
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch_npu
from vllm import envs
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.models.deepseek_v41 import attention_ops as ops
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(
    not HAS_TRITON or get_ascend_device_type() != AscendDeviceType.A3, reason="A3 Triton only"
)

if HAS_TRITON:
    from vllm_ascend.ops.triton import deepseek_v41_fp8_qdq as aiv


class KernelLaunches:
    def __init__(self, kernel):
        self.kernel = kernel
        self.calls = []

    def __getitem__(self, grid):
        def call(x, out, *, NTILES, G):
            compiled = self.kernel[grid](x, out, NTILES=NTILES, G=G)
            self.calls.append((grid, NTILES, G, x.dtype, out.dtype, x.data_ptr() == out.data_ptr(), compiled))
            return compiled

        return call


@pytest.mark.parametrize(
    "shape,tiles,groups", [((24, 2, 4096), 96, 64), ((48, 2, 4096), 192, 64), ((24, 512), 48, 8), ((48, 512), 48, 16)]
)
@pytest.mark.parametrize("inplace", [False, True])
def test_actual_single_pass_replay(shape, tiles, groups, inplace):
    n = math.prod(shape)
    # Create versioned storage before entering inference mode. The aligned nonzero
    # offset and two guards exercise DMA addressing and the existing copy_ contract.
    words = torch.full((n + 64,), 12345, dtype=torch.int16, device="npu")
    x = words[32:-32].view(torch.bfloat16).reshape(shape)
    launches = KernelLaunches(aiv._kernel)
    with torch.inference_mode(), patch.object(aiv, "_kernel", launches):
        version = x._version
        with patch.object(ops, "_decode_fp8_qdq", side_effect=AssertionError("expected the single-pass path")):
            warm = ops.fp8_quant_dequant(x, inplace=inplace)
            assert (warm is x) == inplace and x._version == version + int(inplace)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                out = ops.fp8_quant_dequant(x, inplace=inplace)
        assert (out is x) == inplace and (out.data_ptr() == x.data_ptr()) == inplace
        assert [c[:-1] for c in launches.calls] == [((48,), tiles, groups, torch.bfloat16, torch.bfloat16, inplace)] * 2
        ir = launches.calls[0][-1].asm["ttir"]
        assert ir.count("tt.load ") == ir.count("tt.store ") == 1
        assert "!tt.ptr<bf16>" in ir and "arith.addf" in ir and "arith.subf" in ir and "arith.divf" not in ir
        seq = torch.arange(n, dtype=torch.int32, device="cpu")
        # For the two smaller KV inputs, shift the raw pattern through all 65536 words.
        patterns = [(seq + start).short() for start in range(0, 65536, n)]
        edge = torch.tensor(
            [448.0, 1.0625, 1.1875, 1.9375, 2**-10, -3 * 2**-10, -0.0, 0.0], dtype=torch.bfloat16, device="cpu"
        )
        patterns.append(edge.view(torch.int16).repeat(n // 8))
        special = seq.short().reshape(-1, 32).clone()
        special[::3, 0] = 0x7F81
        special[1::3, 0] = -128
        special[2::3, 0] = -32768
        patterns.extend([special.flatten(), edge.view(torch.int16).repeat(n // 8), seq.short()])
        for raw in patterns:
            raw = raw.reshape(shape)
            x.view(torch.int16).copy_(raw)
            with patch.object(ops, "_try_decode_fp8_aiv", return_value=None):
                q = ops.fp8_quant_dequant(x)
            with patch.object(envs, "VLLM_BATCH_INVARIANT", True):
                stock = ops.fp8_quant_dequant(x)
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(out.view(torch.int16), q.view(torch.int16))
            assert torch.equal(out.view(torch.int16), stock.view(torch.int16))
            if not inplace:
                assert torch.equal(x.view(torch.int16).cpu(), raw)
            assert torch.all(words[:32] == 12345) and torch.all(words[-32:] == 12345)


@pytest.mark.parametrize("case", ["unaligned", "different_hardware", "no_triton", "other_core_count", "blocked_format"])
@torch.inference_mode()
def test_unsupported_single_pass_uses_q_or_reference(case):
    backing = torch.full((24 * 512 + 2,), 12345, dtype=torch.int16, device="npu")
    begin = 1 if case == "unaligned" else 0
    x = backing[begin : begin + 24 * 512].view(torch.bfloat16).reshape(24, 512)
    x.copy_(torch.linspace(-1.25, 1.5, x.numel(), device="cpu").reshape(x.shape))
    if case == "blocked_format":
        blocked = torch_npu.empty_with_format(
            x.shape, dtype=x.dtype, device=x.device, acl_format=torch_npu.Format.FRACTAL_NZ
        )
        x = blocked.copy_(x)
        assert torch_npu.get_npu_format(x) == torch_npu.Format.FRACTAL_NZ
    with patch.object(ops, "_try_decode_fp8_aiv", return_value=None):
        expected = ops.fp8_quant_dequant(x)
    with ExitStack() as stack:
        if case == "different_hardware":
            stack.enter_context(patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A2))
        if case == "no_triton":
            stack.enter_context(patch("vllm.triton_utils.HAS_TRITON", False))
        if case == "other_core_count":
            stack.enter_context(patch.object(aiv, "get_vectorcore_num", return_value=32))
        stack.enter_context(patch.object(aiv, "_kernel", None))
        assert ops.fp8_quant_dequant(x, inplace=True) is x
    # Move to plain CPU before reinterpretation for physical blocked formats.
    assert torch.equal(x.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    if case == "unaligned":
        assert backing[0].cpu().item() == 12345 and backing[-1].cpu().item() == 12345
