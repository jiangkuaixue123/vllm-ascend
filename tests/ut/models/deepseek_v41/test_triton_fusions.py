# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention_ops
from vllm_ascend.models.deepseek_v41.core import hc_split_sinkhorn
from vllm_ascend.ops.triton import deepseek_v41 as fused


@pytest.mark.parametrize("rows", [0, 1, 6, 12, 24, 40, 128, 512])
@pytest.mark.parametrize("magnitude", [1, 100])
def test_npu_triton_hc_gates_sinkhorn(rows, magnitude):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(260)
    mixes = (torch.randn(rows, 24) * magnitude).npu()
    scale = torch.tensor([0.7, -1.3, 0.4]).npu()
    base = torch.randn(24).npu()
    expected = hc_split_sinkhorn(mixes, scale, base, 4)
    actual = fused.hc_gates_sinkhorn(mixes, scale, base)
    for reference, value in zip(expected, actual):
        torch.testing.assert_close(value, reference, rtol=3e-6, atol=3e-7)
    if rows == 12 and magnitude == 1:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = fused.hc_gates_sinkhorn(mixes, scale, base)
        graph.replay()
        torch.npu.synchronize()
        for reference, value in zip(expected, replayed):
            torch.testing.assert_close(value, reference, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [0, 1, 12, 40, 128])
@pytest.mark.parametrize("kind", ["fp8", "fp4_e8m0", "fp4_e4m3"])
def test_npu_triton_qdq_exact(rows, dtype, kind):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(615)
    values = (torch.randn(rows, 512) * 3).to(dtype).npu()
    if kind == "fp8":
        reference = attention_ops.fp8_quant_dequant
        optimized = fused.fp8_quant_dequant
        kwargs = {}
    else:
        reference = attention_ops.fp4_quant_dequant
        optimized = fused.fp4_quant_dequant
        kwargs = {"block_size": 16 if kind == "fp4_e4m3" else 32, "scale_format": kind.split("_")[1]}
    # Empty input requires no device kernel and has no values to compare.
    expected = reference(values, **kwargs) if rows else torch.empty_like(values)
    actual = optimized(values, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if rows == 12:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = optimized(values, **kwargs)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(replayed, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("kind", ["fp8", "fp4_e8m0", "fp4_e4m3"])
def test_npu_triton_qdq_rounding_edges_and_inplace_strided(kind):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    values = torch.tensor(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
            2.5,
            -2.5,
            3.5,
            -3.5,
            5.0,
            -5.0,
            6.0,
            -6.0,
            1e-5,
            -1e-5,
            400.0,
            -400.0,
            432.0,
            -432.0,
            448.0,
            -448.0,
            1e-38,
            -1e-38,
            1e-20,
            -1e-20,
            1.0,
            -1.0,
        ]
    ).npu()
    rows = torch.stack((values, values * 1e-4, values * 100, torch.zeros_like(values)))
    # Each zero row also contains -0, which must survive conversion.
    rows[-1, 1::2] = -0.0
    storage = torch.empty((4, 64), device=rows.device)
    storage[:, ::2] = rows
    target = storage[:, ::2]
    if kind == "fp8":
        reference, optimized, kwargs = attention_ops.fp8_quant_dequant, fused.fp8_quant_dequant, {}
    else:
        reference, optimized = attention_ops.fp4_quant_dequant, fused.fp4_quant_dequant
        kwargs = {"block_size": 16 if kind == "fp4_e4m3" else 32, "scale_format": kind.split("_")[1]}
    expected = reference(target, **kwargs)
    actual = optimized(target, inplace=True, **kwargs)
    assert actual is target
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    exceptional = torch.zeros((2, 32), device=rows.device)
    exceptional[0, 0] = torch.nan
    exceptional[1, 0] = torch.inf
    torch.testing.assert_close(
        optimized(exceptional, **kwargs), reference(exceptional, **kwargs), rtol=0, atol=0, equal_nan=True
    )
