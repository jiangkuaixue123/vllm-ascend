# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu

from vllm_ascend.models.deepseek_v41.core import clipped_swiglu
from vllm_ascend.ops.triton import deepseek_v41_prefill_experts as fused


def assert_exact(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    torch.testing.assert_close(torch.isnan(actual), torch.isnan(expected))
    finite = ~torch.isnan(expected)
    dtype = torch.int16 if expected.dtype == torch.bfloat16 else torch.int32
    torch.testing.assert_close(actual.view(dtype)[finite], expected.view(dtype)[finite], rtol=0, atol=0)


@pytest.mark.parametrize("limit", [0.0, 2.1, 10.0])
def test_prefill_activation_all_bf16_patterns(limit):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    gate = bits.repeat(2)[: 32 * 2304].view(32, 2304)
    up = bits.flip(0).repeat(2)[: 32 * 2304].view(32, 2304)
    x = torch.cat((gate, up), dim=-1).npu()
    factor = torch.tensor([0, -0.0, 1, -1, 1 / 3, 0.73, float("inf"), float("nan")]).repeat(4).npu()
    before = x.cpu().view(torch.int16).clone()
    with torch.inference_mode():
        result = fused.prefill_route_activation(x, factor, limit)
        expected = (clipped_swiglu(*x.chunk(2, -1), limit) * factor[:, None]).bfloat16()
        assert_exact(result, expected)
        torch.testing.assert_close(x.cpu().view(torch.int16), before, rtol=0, atol=0)


@pytest.mark.parametrize("rows,width", [(17, 2304), (128, 2304), (6144, 2304)])
def test_prefill_quantized_consumers_and_changing_graph_replay(rows, width):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    gen = torch.Generator().manual_seed(1029)
    x = (torch.randn(rows, 2 * width, generator=gen) * 6).bfloat16().npu()
    factor = torch.rand(rows, generator=gen).npu()

    def expected():
        act = (clipped_swiglu(*x.chunk(2, -1), 10.0) * factor[:, None]).bfloat16()
        return torch_npu.npu_dynamic_quant(act, dst_type=torch.int8)

    def execute():
        act = fused.prefill_route_activation(x, factor, 10.0)
        return torch_npu.npu_dynamic_quant(act, dst_type=torch.int8)

    with torch.inference_mode():
        for actual, reference in zip(execute(), expected()):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=0, atol=0)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = execute()
        for i in range(16):
            x.neg_()
            factor.mul_(0.97)
            if i % 4 == 0:
                factor[::3].zero_()
            graph.replay()
            torch.npu.synchronize()
            for value, reference in zip(actual, expected()):
                torch.testing.assert_close(value.cpu(), reference.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("top_k", [3, 6])
def test_prefill_combine_masks_padding_and_preserves_native_nonfinite_order(top_k):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    rows, hidden = 17, 5120
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    count = rows * top_k
    states = values.repeat((count * hidden + 65535) // 65536)[: count * hidden].reshape(count, hidden).npu()
    factor = torch.ones(count, device="npu")
    row_ids = torch.arange(count, dtype=torch.int32).flip(0).npu()
    row_ids[::3].neg_()
    owned = torch.ones((rows, top_k), device="npu")
    owned[::2, 1::2] = 0

    def expected():
        masked = torch.where(factor[:, None] != 0, states.float(), 0.0)
        return torch_npu.npu_moe_token_unpermute(masked, row_ids.abs(), probs=owned)

    with torch.inference_mode():
        assert_exact(fused.prefill_combine_output(states, factor, row_ids, owned), expected())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = fused.prefill_combine_output(states, factor, row_ids, owned)
        for i in range(16):
            factor.fill_(0.0 if i % 3 else -1.0)
            factor[i % count] = float("nan")
            row_ids.copy_(torch.roll(row_ids, 3))
            graph.replay()
            torch.npu.synchronize()
            assert_exact(output, expected())


@pytest.mark.parametrize("rows,top_k", [(1, 6), (129, 3), (264, 6), (460, 6), (2048, 6)])
def test_prefill_combine_keeps_fp32_slot_accumulation(rows, top_k):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    hidden = 5120
    gen = torch.Generator().manual_seed(217)
    values = torch.randn(rows * top_k, hidden, generator=gen)
    values *= torch.pow(2.0, torch.randint(-15, 26, values.shape, generator=gen))
    states = values.bfloat16().npu()
    factor = torch.ones(rows * top_k, device="npu")
    row_ids = torch.randperm(rows * top_k, generator=gen).int().npu()
    owned = torch.ones((rows, top_k), device="npu")
    with torch.inference_mode():
        native = torch_npu.npu_moe_token_unpermute(states.float(), row_ids, probs=owned)
        assert_exact(fused.prefill_combine_output(states, factor, row_ids, owned), native)
        states.fill_(-0.0)
        for slot in range(top_k):
            owned.fill_(1.0)
            owned[:, slot] = 0
            native = torch_npu.npu_moe_token_unpermute(states.float(), row_ids, probs=owned)
            assert_exact(fused.prefill_combine_output(states, factor, row_ids, owned), native)
        factor.zero_()
        states.fill_(float("nan"))
        assert torch.count_nonzero(fused.prefill_combine_output(states, factor, row_ids, owned)).item() == 0


def test_prefill_fusion_keeps_cpu_fallback():
    with torch.inference_mode():
        assert not fused.prefill_fusion_available(torch.empty((4096, 5120), dtype=torch.int8))


@pytest.mark.parametrize("rows", [384, 12288])
def test_live_expert_prefix_preserves_quantization_and_graph_counts(rows):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    width = 2304
    x = torch.randn(rows, 2 * width, dtype=torch.bfloat16, device="npu")
    factor = torch.rand(rows, device="npu")
    groups = torch.zeros(24, dtype=torch.int64, device="npu")
    groups[0] = 7

    def execute():
        activation = fused.prefill_route_activation(x, factor, 10.0, groups=groups)
        return torch_npu.npu_dynamic_quant(activation, dst_type=torch.int8)

    with torch.inference_mode():
        execute()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual, scale = execute()
        for count in (0, 7, min(768, rows), rows, 13, 0):
            x.copy_(torch.randn_like(x) * 6)
            x[count:].fill_(float("nan"))
            groups.zero_()
            groups[0] = count // 2
            groups[7] = count - count // 2
            factor.mul_(0.97)
            graph.replay()
            torch.npu.synchronize()
            if count:
                gate, up = x[:count].chunk(2, -1)
                expected = (clipped_swiglu(gate, up, 10.0) * factor[:count, None]).bfloat16()
                quantized, expected_scale = torch_npu.npu_dynamic_quant(expected, dst_type=torch.int8)
                torch.testing.assert_close(actual[:count].cpu(), quantized.cpu(), rtol=0, atol=0)
                assert_exact(scale[:count], expected_scale)


@pytest.mark.parametrize("rows", [17, 264])
def test_prefill_combine_writes_only_owned_output_slice(rows):
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    width, k = 5120, 6
    states = torch.randn(rows * k, width, dtype=torch.bfloat16, device="npu")
    factor = torch.ones(rows * k, device="npu")
    ids = torch.randperm(rows * k).int().npu()
    owned = torch.ones(rows, k, device="npu")
    full = torch.full((rows + 4, width), 123.0, device="npu")
    destination = full[2:-2]
    with torch.inference_mode():
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = fused.prefill_combine_output(states, factor, ids, owned, out=destination)
        assert actual is destination
        for step in range(3):
            states.neg_()
            ids.copy_(torch.roll(ids, 3))
            owned[:, step].zero_()
            graph.replay()
            native = torch_npu.npu_moe_token_unpermute(states.float(), ids, probs=owned)
            assert_exact(actual, native)
            assert torch.all(full[:2] == 123).item() and torch.all(full[-2:] == 123).item()
