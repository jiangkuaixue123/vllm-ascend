# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from unittest.mock import patch

import pytest
import torch
from vllm import envs

from vllm_ascend.models.deepseek_v41 import attention_ops as ops


@pytest.mark.parametrize("shape", [(24, 512), (48, 512), (24, 2, 4096), (48, 2, 4096)])
@pytest.mark.parametrize("inplace", [False, True])
@torch.inference_mode()
def test_decode_fp8_entry_graph_replay(shape, inplace):
    words = torch.zeros(32 + math.prod(shape), dtype=torch.int16, device="npu")
    x = words[32:].view(torch.bfloat16).reshape(shape)
    prefix = words[:32].clone()
    with patch.object(ops, "_decode_fp8_qdq", wraps=ops._decode_fp8_qdq) as selected:
        ops.fp8_quant_dequant(x, inplace=inplace)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            result = ops.fp8_quant_dequant(x, inplace=inplace)
        assert selected.call_count == 2
    assert (result is x) == inplace
    base = torch.arange(x.numel(), dtype=torch.int32, device="cpu").short().reshape(shape)
    for case in ("raw", "special", "ties", "finite", "raw"):
        src = base.clone()
        if case == "special":
            src.flatten()[::32] = 0x7F81
            src.flatten()[1::32] = -128
            src.flatten()[2::32] = -32768
        elif case == "ties":
            group = torch.tensor(
                [448.0, 1.0625, 1.1875, 1.9375, 2**-10, -3 * 2**-10, -0.0, 0.0], dtype=torch.bfloat16, device="cpu"
            )
            src = group.view(torch.int16).repeat(x.numel() // 8).reshape(shape)
        elif case == "finite":
            src = (
                torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(101), device="cpu")
                .bfloat16()
                .view(torch.int16)
            )
        x.view(torch.int16).copy_(src)
        with patch.object(envs, "VLLM_BATCH_INVARIANT", True):
            expected = ops.fp8_quant_dequant(x)
        graph.replay()
        torch.testing.assert_close(result.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0, msg=case)
        torch.testing.assert_close(words[:32], prefix, rtol=0, atol=0)
        if not inplace:
            torch.testing.assert_close(x.view(torch.int16).cpu(), src, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["strided", "invariant", "grad", "requires_grad", "fp16", "fp32", "rows", "block"])
def test_fp8_npu_fallback(case):
    dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
    backing = torch.ones((23 if case == "rows" else 24, 1024 if case == "strided" else 512), dtype=dtype, device="npu")
    x = backing[:, 1::2] if case == "strided" else backing
    if case == "requires_grad":
        x.requires_grad_(True)
    block = 16 if case == "block" else 32
    with torch.inference_mode(case != "grad"), patch.object(envs, "VLLM_BATCH_INVARIANT", case == "invariant"):
        with patch.object(ops, "_decode_fp8_qdq", side_effect=AssertionError("unexpected fast path")):
            expected = ops.fp8_quant_dequant(x, block)
            assert ops.fp8_quant_dequant(x, block, inplace=True) is x
            torch.testing.assert_close(x, expected, rtol=0, atol=0)
        if case == "strided":
            torch.testing.assert_close(backing[:, ::2], torch.ones_like(x), rtol=0, atol=0)
