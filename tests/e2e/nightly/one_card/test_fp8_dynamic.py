# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic QDQ exact arithmetic, graph input refresh, reuse and latency."""

import statistics
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
from vllm import envs

from vllm_ascend.models.deepseek_v41 import attention_ops as ops
from vllm_ascend.ops.triton import deepseek_v41_fp8_qdq as aiv

ROWS = (1, 5, 20, 37, 511, 8192, 8186)
BENCHMARK_REPEATS = 7


def reference(x):
    with patch.object(envs, "VLLM_BATCH_INVARIANT", True):
        return ops.fp8_quant_dequant(x)


class Launches:
    def __init__(self, kernel):
        self.kernel = kernel
        self.compiled = {}

    def __getitem__(self, grid):
        def call(x, out, tiles, *, G):
            compiled = self.kernel[grid](x, out, tiles, G=G)
            if G in self.compiled:
                assert compiled is self.compiled[G]
            self.compiled[G] = compiled
            return compiled

        return call


@pytest.mark.parametrize("tail", [(512,), (2, 4096)])
@torch.inference_mode()
def test_changing_rows_reuses_kernel_and_matches_reference(tail):
    torch.npu.set_device(0)
    launches = Launches(aiv._dynamic_kernel)
    with patch.object(aiv, "_dynamic_kernel", launches):
        for rows in ROWS:
            x = torch.randn((rows, *tail), device="npu", dtype=torch.bfloat16)
            actual = ops.fp8_quant_dequant(x)
            assert torch.equal(actual.view(torch.int16), reference(x).view(torch.int16))
    assert len(launches.compiled) == 1


@torch.inference_mode()
def test_all_bf16_group_maxima_and_rounding_edges():
    torch.npu.set_device(0)
    words = torch.arange(65536, dtype=torch.int32).short()
    maxima = words[:, None].expand(-1, 32).contiguous().view(torch.bfloat16)
    edges = torch.tensor([448.0, 1.0625, 1.1875, 1.9375, 2**-10, -3 * 2**-10, -0.0, 0.0], dtype=torch.bfloat16)
    patterns = [
        maxima,
        words.reshape(-1, 32).view(torch.bfloat16),
        edges.repeat(65536 // edges.numel()).reshape(-1, 32),
    ]
    for grouped in patterns:
        x = grouped.reshape(-1, 512).npu()
        actual = ops.fp8_quant_dequant(x)
        assert torch.equal(actual.view(torch.int16), reference(x).view(torch.int16))


@pytest.mark.parametrize("shape", [(5, 512), (37, 2, 4096)])
@pytest.mark.parametrize("inplace", [False, True])
def test_refresh_alias_version_and_guards(shape, inplace):
    torch.npu.set_device(0)
    count = 1
    for dim in shape:
        count *= dim
    storage = torch.full((count + 64,), 12345, device="npu", dtype=torch.int16)
    x = storage[32:-32].view(torch.bfloat16).reshape(shape)
    with torch.inference_mode():
        x.normal_()
        version = x._version
        warm = ops.fp8_quant_dequant(x, inplace=inplace)
        assert (warm is x) == inplace
        assert x._version == version + int(inplace)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = ops.fp8_quant_dequant(x, inplace=inplace)
        address = output.data_ptr()
        for step in range(16):
            fresh = torch.randn_like(x) * (2.0 ** (step - 8))
            fresh.flatten()[::97] = -0.0
            x.copy_(fresh)
            expected = reference(fresh)
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
            assert output.data_ptr() == address
            assert (output is x) == inplace
            if not inplace:
                assert torch.equal(x.view(torch.int16), fresh.view(torch.int16))
            assert torch.all(storage[:32] == 12345) and torch.all(storage[-32:] == 12345)


def graph_latency(function, x, calls):
    function(x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(calls):
            function(x)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(BENCHMARK_REPEATS):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@pytest.mark.parametrize("shape", [(5, 512), (20, 512), (20, 2, 4096), (512, 2, 4096), (8192, 512), (8192, 2, 4096)])
@torch.inference_mode()
def test_component_latency(shape):
    torch.npu.set_device(0)
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    calls = 32 if shape[0] <= 20 else 1
    reference_ms = graph_latency(reference, x, calls)
    fused_ms = graph_latency(ops.fp8_quant_dequant, x, calls)
    print({"shape": shape, "calls": calls, "reference_ms": reference_ms, "fused_ms": fused_ms})
    assert fused_ms < reference_ms
