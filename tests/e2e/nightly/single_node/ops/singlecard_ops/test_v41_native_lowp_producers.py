# SPDX-License-Identifier: Apache-2.0
"""Exact producer/cache contracts, including changed-input native graph replay."""

import gc
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.models.deepseek_v41 import attention_ops as ops
from vllm_ascend.ops.triton import deepseek_v41_fp4_qdq as native4
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True)
def npu():
    torch_npu.npu.set_device(0)
    assert enable_custom_op()


def assert_bits(actual, expected):
    a, b = actual.cpu().view(torch.int16), expected.cpu().view(torch.int16)
    assert torch.equal(a, b), (torch.count_nonzero(a != b).item(), a[a != b][:16], b[a != b][:16])


def reference4(x, block, fmt):
    with patch.object(native4, "fp4_qdq", return_value=None):
        return ops.fp4_quant_dequant(x, block, scale_format=fmt)


def reference8(x):
    with (
        patch.object(ops, "_try_decode_fp8_aiv", return_value=None),
        patch.object(ops, "_use_decode_fp8_qdq", return_value=False),
    ):
        return ops.fp8_quant_dequant(x)


@pytest.mark.parametrize("fmt,block", [("e8m0", 32), ("e4m3", 16)])
@pytest.mark.parametrize("anchor", [0, 2**-126, 2**-9, 0.75, 6.0, 7.0, 2784.0, float("inf"), float("nan")])
@torch.inference_mode()
def test_all_bf16_words_and_group_scales(fmt, block, anchor):
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = torch.zeros(65536, block, dtype=torch.bfloat16)
    x[:, 0], x[:, -1] = words, anchor
    x = x.reshape(-1, 128).npu()
    expected = reference4(x, block, fmt)
    assert_bits(ops.fp4_quant_dequant(x, block, scale_format=fmt), expected)


@pytest.mark.parametrize(
    "shape,fmt,block",
    [
        ((24, 512), "e4m3", 16),
        ((48, 512), "e4m3", 16),
        ((24, 128), "e8m0", 32),
        ((48, 128), "e8m0", 32),
        ((6, 32, 128), "e8m0", 32),
        ((12, 32, 128), "e8m0", 32),
        ((1020, 512), "e4m3", 16),
        ((8160, 32, 128), "e8m0", 32),
    ],
)
@torch.inference_mode()
def test_native_fp4_graph_refresh_and_inplace(shape, fmt, block):
    with torch.inference_mode(False):
        x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
    ops.fp4_quant_dequant(x, block, scale_format=fmt)
    graph = torch.npu.NPUGraph()
    with (
        patch.object(ops, "round_e2m1", side_effect=AssertionError("generic FP4 fallback")),
        torch.npu.graph(graph),
    ):
        out = ops.fp4_quant_dequant(x, block, scale_format=fmt)
    for i in range(4):
        x.copy_(torch.randn_like(x) * (i + 1))
        x.reshape(-1)[:4].copy_(torch.tensor([0.0, -0.0, float("inf"), float("nan")], dtype=x.dtype, device=x.device))
        expected = reference4(x, block, fmt)
        graph.replay()
        assert_bits(out, expected)
    expected = reference4(x, block, fmt)
    version = x._version
    assert ops.fp4_quant_dequant(x, block, scale_format=fmt, inplace=True) is x
    assert x._version > version
    assert_bits(x, expected)


@pytest.mark.parametrize("shape", [(20, 512), (20, 2, 4096), (20, 1, 4096), (8160, 512), (257, 64, 512)])
@torch.inference_mode()
def test_extended_fp8_matches_original_generic(shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device="npu")
    x.reshape(-1)[:4].copy_(torch.tensor([0.0, -0.0, float("inf"), float("nan")], dtype=x.dtype, device=x.device))
    expected = reference8(x)
    with patch.object(ops, "round_e4m3", side_effect=AssertionError("generic FP8 fallback")):
        actual = ops.fp8_quant_dequant(x)
    assert_bits(actual, expected)
    if shape[0] == 20:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out = ops.fp8_quant_dequant(x)
        for _ in range(4):
            x.copy_(torch.randn_like(x))
            expected = reference8(x)
            graph.replay()
            assert_bits(out, expected)


@pytest.mark.parametrize("anchor", [0.0, 2**-14, 448.0, 449.0, float("inf"), float("nan")])
@torch.inference_mode()
def test_extended_fp8_all_bf16_words(anchor):
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = torch.zeros(65536, 32, dtype=torch.bfloat16)
    x[:, 0], x[:, -1] = words, anchor
    x = x.reshape(-1, 512).npu()
    assert_bits(ops.fp8_quant_dequant(x), reference8(x))


@pytest.mark.parametrize("fmt,block", [("e8m0", 32), ("e4m3", 16)])
@torch.inference_mode()
def test_small_rows_and_output_guards(fmt, block):
    for rows in (1, 3, 65):
        x = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu")
        storage = torch.full((x.numel() + 64,), 71.0, dtype=x.dtype, device=x.device)
        out = storage[32:-32].view_as(x)
        torch.ops._C_ascend.deepseek_v41_fp4_qdq(x, out, fmt == "e4m3")
        assert_bits(out, reference4(x, block, fmt))
        assert (storage[:32] == 71).all() and (storage[-32:] == 71).all()


@torch.inference_mode()
def test_unsupported_fp4_stays_on_reference():
    for x in [
        torch.zeros(2, 128),
        torch.zeros(2, 128, device="npu"),
        torch.zeros(2, 256, dtype=torch.bfloat16, device="npu"),
        torch.zeros(2, 128, dtype=torch.bfloat16, device="npu")[:, ::2],
    ]:
        assert native4.fp4_qdq(x, 32, "e8m0", False) is None
    with pytest.raises(ValueError, match="scale_format"):
        ops.fp4_quant_dequant(torch.zeros(2, 128, dtype=torch.bfloat16, device="npu"), scale_format="invalid")


@torch.inference_mode()
def test_native_fp4_callback_releases_storage():
    x = torch.zeros(32, 512, dtype=torch.bfloat16, device="npu")
    out = ops.fp4_quant_dequant(x, 16, scale_format="e4m3")
    del x, out
    gc.collect()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    for _ in range(32):
        x = torch.randn(32, 512, dtype=torch.bfloat16, device="npu")
        out = ops.fp4_quant_dequant(x, 16, scale_format="e4m3")
        del x, out
    gc.collect()
    torch.npu.synchronize()
    assert torch.npu.memory_allocated() - before < 256 * 1024
