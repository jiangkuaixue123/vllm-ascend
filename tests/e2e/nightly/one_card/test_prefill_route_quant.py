# SPDX-License-Identifier: Apache-2.0
import gc

import pytest
import torch
import torch_npu

from vllm_ascend.models.deepseek_v41.core import clipped_swiglu
from vllm_ascend.utils import enable_custom_op


def reference(x, factor, limit):
    activation = (clipped_swiglu(*x.chunk(2, -1), limit) * factor[:, None]).bfloat16()
    return torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)


def assert_quantized(actual, expected, live):
    for found, wanted in zip(actual, expected):
        found, wanted = found[:live].cpu(), wanted[:live].cpu()
        if wanted.dtype == torch.float32:
            assert torch.equal(torch.isnan(found), torch.isnan(wanted))
            finite = ~torch.isnan(wanted)
            assert torch.equal(found.view(torch.int32)[finite], wanted.view(torch.int32)[finite])
        else:
            assert torch.equal(found, wanted)


@pytest.mark.parametrize("limit", [0.0, 2.1, 10.0])
@torch.inference_mode()
def test_native_prefill_requant_bf16_patterns(limit):
    enable_custom_op()
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gate = bits.repeat(2)[: 32 * 2304].reshape(32, 2304)
    up = bits.flip(0).repeat(2)[: 32 * 2304].reshape(32, 2304)
    x = torch.cat((gate, up), -1).npu()
    factor = torch.tensor([0.0, -0.0, 1.0, -1.0, 1 / 3, 0.73, 1.5, -0.5]).repeat(4).npu()
    groups = torch.zeros(24, dtype=torch.int64, device="npu")
    groups[0] = 32
    actual = torch.ops._C_ascend.prefill_prefix_route_quant(x, factor, groups, limit)
    assert_quantized(actual, reference(x, factor, limit), 32)


@pytest.mark.parametrize("rows", [384, 49152])
@torch.inference_mode()
def test_native_prefill_requant_counts_and_graph(rows):
    enable_custom_op()
    x = (torch.randn(rows, 4608, device="npu") * 6).bfloat16()
    factor = torch.rand(rows, device="npu")
    groups = torch.zeros(24, dtype=torch.int64, device="npu")
    torch.ops._C_ascend.prefill_prefix_route_quant(x, factor, groups, 10.0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = torch.ops._C_ascend.prefill_prefix_route_quant(x, factor, groups, 10.0)
    for i, live in enumerate([0, 7, min(1536, rows), rows, 13, 0]):
        groups.zero_()
        groups[23] = live
        x.neg_()
        factor.mul_(0.97)
        if i % 2:
            factor[::3].zero_()
        graph.replay()
        torch.npu.synchronize()
        if live:
            assert_quantized(output, reference(x[:live], factor[:live], 10.0), live)
    # A poison suffix must remain outside the numerical contract for every live prefix.
    groups[0] = 7
    x[7:].fill_(float("nan"))
    actual = torch.ops._C_ascend.prefill_prefix_route_quant(x, factor, groups, 10.0)
    assert_quantized(actual, reference(x[:7], factor[:7], 10.0), 7)


@torch.inference_mode()
def test_native_prefill_requant_releases_storage():
    enable_custom_op()
    groups = torch.zeros(24, dtype=torch.int64, device="npu")
    groups[0] = 7

    def execute():
        x = torch.randn(128, 4608, device="npu", dtype=torch.bfloat16)
        factor = torch.ones(128, device="npu")
        return torch.ops._C_ascend.prefill_prefix_route_quant(x, factor, groups, 10.0)

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
