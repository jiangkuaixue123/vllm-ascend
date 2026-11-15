# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

import vllm_ascend  # noqa: F401


@torch.inference_mode()
def test_prefix_silu_releases_temporary_storage():
    groups = torch.zeros(24, dtype=torch.int64, device="npu")
    groups[0] = 64

    def execute():
        x = torch.empty((1024, 2304), dtype=torch.float32, device="npu").fill_(0.5)
        return torch.ops._C_ascend.prefill_prefix_silu(x, groups)

    output = execute()
    del output
    torch.npu.synchronize()
    gc.collect()
    baseline = torch.npu.memory_allocated()
    for _ in range(32):
        output = execute()
        del output
    torch.npu.synchronize()
    gc.collect()
    assert torch.npu.memory_allocated() <= baseline + 4 * 1024 * 1024


@pytest.mark.parametrize("limit", [0.0, 2.1, 10.0])
@torch.inference_mode()
def test_prefix_silu_all_bf16_patterns(limit):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = bits.repeat(2)[: 32 * 2304].reshape(32, 2304).float().npu()
    if limit:
        x = torch.where(x > limit, limit, x)
    groups = torch.zeros(24, dtype=torch.int64, device="npu")
    groups[0] = x.shape[0]
    actual = torch.ops._C_ascend.prefill_prefix_silu(x, groups)
    expected = F.silu(x)
    finite = ~torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(actual.view(torch.int32)[finite], expected.view(torch.int32)[finite])


@pytest.mark.parametrize("width", [64, 2304])
@torch.inference_mode()
def test_prefix_silu_dynamic_graph(width):
    torch.npu.set_device(0)
    x = torch.randn(128, width, device="npu", dtype=torch.float32) * 12
    # Finite extreme values, signed zero and native NaN/Inf behavior are part of the contract.
    x[0, :8] = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 88.0], device="npu")
    counts = torch.zeros(24, dtype=torch.int64, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        y = torch.ops._C_ascend.prefill_prefix_silu(x, counts)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        y = torch.ops._C_ascend.prefill_prefix_silu(x, counts)
    for live in [0, 7, 63, 128, 13, 0]:
        counts.zero_()
        counts[0] = live
        x[1:].normal_()
        graph.replay()
        expected = F.silu(x[:live])
        actual = y[:live]
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        finite = ~torch.isnan(expected)
        assert torch.equal(actual.view(torch.int32)[finite], expected.view(torch.int32)[finite])

    empty = counts[:0]
    actual = torch.ops._C_ascend.prefill_prefix_silu(x, empty)
    expected = F.silu(x)
    finite = ~torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(actual.view(torch.int32)[finite], expected.view(torch.int32)[finite])
