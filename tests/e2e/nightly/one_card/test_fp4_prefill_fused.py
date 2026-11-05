# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large prefill shapes for the same exact FP4 kernels used during decode."""

import statistics
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention_ops

REPEATS = 5
CASES = [
    ((512, 64, 128), 32, "e8m0"),
    ((8192, 128), 32, "e8m0"),
    ((8192, 64, 128), 32, "e8m0"),
    ((512, 512), 16, "e4m3"),
    ((8192, 512), 16, "e4m3"),
]


def reference(states, block, scale_format):
    with (
        patch.object(attention_ops, "_try_fp4_e8m0_aiv", return_value=None),
        patch.object(attention_ops, "_try_fp4_e4m3_aiv", return_value=None),
    ):
        return attention_ops.fp4_quant_dequant(states, block, scale_format=scale_format)


def elapsed(function):
    function()
    torch.npu.synchronize()
    times = []
    for _ in range(REPEATS):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del output
    return statistics.median(times)


@pytest.mark.parametrize("shape,block,scale_format", CASES)
@torch.inference_mode()
def test_prefill_exactness_and_latency(shape, block, scale_format):
    torch.npu.set_device(0)
    torch.manual_seed(20261106)
    states = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    actual = attention_ops.fp4_quant_dequant(states, block, scale_format=scale_format)
    expected = reference(states, block, scale_format)
    assert torch.equal(actual, expected)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    del actual, expected
    reference_ms = elapsed(lambda: reference(states, block, scale_format))
    fused_ms = elapsed(lambda: attention_ops.fp4_quant_dequant(states, block, scale_format=scale_format))
    print({"shape": shape, "scale_format": scale_format, "reference_ms": reference_ms, "fused_ms": fused_ms})
    assert fused_ms < reference_ms
