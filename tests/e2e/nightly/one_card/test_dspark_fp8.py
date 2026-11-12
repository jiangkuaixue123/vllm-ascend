# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic QDQ exact arithmetic, graph input refresh, reuse and latency."""

import statistics
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401
import vllm.distributed as distributed
import vllm.forward_context as forward_context
from torch import nn

from tests.e2e.nightly.one_card.test_draft_output_graph import device_inputs, exact
from vllm_ascend.models.common.ops import dspark_fp8
from vllm_ascend.models.common.ops.draft_attention_graph import DraftAttentionGraph
from vllm_ascend.models.deepseek_v41 import attention_ops as ops
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.triton import deepseek_v41_fp8_qdq as aiv

ROWS = (1, 5, 20, 37, 40)
pytest_plugins = ("tests.e2e.nightly.one_card.test_draft_output_graph",)
BENCHMARK_REPEATS = 7


def reference(x):
    with patch.object(dspark_fp8, "draft_fp8_fused", return_value=None):
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
@pytest.mark.parametrize("npu_format", [0, 2])
@torch.inference_mode()
def test_changing_rows_reuses_kernel_and_matches_reference(tail, npu_format):
    torch.npu.set_device(0)
    launches = Launches(aiv._draft_kernel)
    with patch.object(aiv, "_draft_kernel", launches):
        for rows in ROWS:
            x = torch.randn((rows, *tail), device="npu", dtype=torch.bfloat16)
            x = torch_npu.npu_format_cast(x, npu_format)
            assert torch_npu.get_npu_format(x) == npu_format
            actual = ops.fp8_quant_dequant(x)
            assert torch.equal(actual.view(torch.int16), reference(x).view(torch.int16))
    assert len(launches.compiled) == 1


@pytest.mark.parametrize("shape", [(5, 512), (37, 2, 4096)])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("npu_format", [0, 2])
@torch.inference_mode()
def test_dense_formats_refresh_graph_inputs(shape, inplace, npu_format):
    torch.npu.set_device(0)
    x = torch_npu.npu_format_cast(torch.randn(shape, device="npu", dtype=torch.bfloat16), npu_format)
    assert torch_npu.get_npu_format(x) == npu_format
    assert dspark_fp8.draft_fp8_fused(x, 32, inplace) is not None
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = ops.fp8_quant_dequant(x, inplace=inplace)
    for step in range(16):
        fresh = torch.randn(shape, device="npu", dtype=torch.bfloat16) * (2.0 ** (step - 8))
        if step == 8:
            fresh.zero_()
        expected = reference(fresh)
        x.copy_(fresh)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        assert (output is x) == inplace
        assert torch_npu.get_npu_format(x) == npu_format


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
        actual = aiv.draft_fp8_qdq(x, False)
        assert actual is not None
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


@pytest.mark.parametrize("shape", [(5, 512), (20, 512), (40, 512), (5, 2, 4096), (20, 2, 4096), (40, 2, 4096)])
@torch.inference_mode()
def test_component_latency(shape):
    torch.npu.set_device(0)
    x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
    calls = 32
    reference_ms = graph_latency(reference, x, calls)
    fused_ms = graph_latency(ops.fp8_quant_dequant, x, calls)
    print({"shape": shape, "calls": calls, "reference_ms": reference_ms, "fused_ms": fused_ms})
    assert fused_ms < reference_ms


@pytest.fixture(autouse=True)
def draft_context(monkeypatch):
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: SimpleNamespace(is_draft_model=True))
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: 4)


class NativeBF16Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(512, 5120, dtype=torch.bfloat16, device="npu") * 0.01, requires_grad=False
        )
        self.quant_method = AscendUnquantizedLinearMethod()

    def forward(self, states):
        return self.quant_method.apply(self, states), None


@pytest.mark.parametrize("batch", [1, 4, 8])
@torch.inference_mode()
def test_complete_draft_cache_attention_and_output(full_model, monkeypatch, batch):
    model, observed = full_model
    monkeypatch.setattr(distributed, "get_tensor_model_parallel_world_size", lambda: model.tp)
    monkeypatch.setattr(model, "wkv", NativeBF16Linear())
    storage = torch.randn(1024, 1024, dtype=torch.bfloat16, device="npu")
    monkeypatch.setattr(model.cache, "kv_cache", (storage[:, :512], model.cache.kv_cache[1]))
    parent_graph, candidate_graph = DraftAttentionGraph(project_output=True), DraftAttentionGraph(project_output=True)
    hits = []
    kernel = aiv.draft_fp8_qdq

    def record(value, inplace):
        hits.append(tuple(value.shape))
        return kernel(value, inplace)

    monkeypatch.setattr(aiv, "draft_fp8_qdq", record)
    for step, start in enumerate((8192, 127, 128, 153550)):
        hidden, positions, _ = device_inputs(model, batch, start, step)
        if step == 2:
            hidden.zero_()
        slots = (torch.arange(hidden.shape[0], device="npu") * 7 + step) % 1024
        slots[::3] = -1
        initial = storage.clone()
        observed.partials.clear()
        with patch.object(dspark_fp8, "draft_fp8_fused", return_value=None):
            model.store_context(hidden, positions, slots)
            model._attention_graph = parent_graph
            expected = model(hidden, positions).clone()
            expected_cache = storage.clone()
        storage.copy_(initial)
        model.store_context(hidden, positions, slots)
        model._attention_graph = candidate_graph
        actual = model(hidden, positions)
        exact(actual, expected)
        exact(storage, expected_cache)
        exact(observed.partials[0], observed.partials[1])
        if model.tp == 4 and batch in (1, 4):
            assert batch in parent_graph.entries and batch in candidate_graph.entries
    if model.tp == 4:
        assert (batch * 5, 512) in hits
        assert (batch * 5, 2, 4096) in hits
    else:
        assert not hits
