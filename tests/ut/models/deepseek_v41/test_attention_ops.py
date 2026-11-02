# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical boundaries and mask behavior of the official V4.1 primitives."""

import importlib
import importlib.util
import math

import pytest
import torch

from vllm_ascend.models.deepseek_v41.attention_ops import (
    _power_of_two,
    _unbiased_exponent,
    apply_rotary_emb_real,
    fp4_quant_dequant,
    fp8_quant_dequant,
    round_e2m1,
    round_e4m3,
    sparse_attn,
)


@pytest.fixture(params=["cpu", "npu"])
def device(request):
    if request.param == "npu":
        if importlib.util.find_spec("torch_npu") is None:
            pytest.skip("torch_npu is not installed")
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("No NPU is available")
    return request.param


def test_e2m1_ties_to_even_and_saturation(device):
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7], device=device)
    expected = torch.tensor([0.0, 1, 1, 2, 2, 4, 4, 6], device=device)
    torch.testing.assert_close(round_e2m1(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(round_e2m1(-values), -expected, rtol=0, atol=0)


def test_e4m3_subnormal_and_normal_ties(device):
    values = torch.tensor([2**-10, 3 * 2**-10, 1.0625, 1.1875, 1.9375, 464], device=device)
    expected = torch.tensor([0, 2**-8, 1, 1.25, 2, 448], device=device)
    torch.testing.assert_close(round_e4m3(values), expected, rtol=0, atol=0)
    assert torch.isnan(round_e4m3(torch.tensor([465.0], device=device))).all()
    torch.testing.assert_close(
        round_e4m3(torch.tensor([500.0], device=device), saturate=True),
        torch.tensor([448.0], device=device),
        rtol=0,
        atol=0,
    )


def test_e4m3_matches_native_cpu_conversion(device):
    generator = torch.Generator().manual_seed(7)
    values = torch.rand(2048, generator=generator) * 928 - 464
    expected = values.to(torch.float8_e4m3fn).float()
    actual = round_e4m3(values.to(device)).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fp8_scale_rounds_up_at_power_of_two_boundary(device):
    x = torch.zeros(2, 32, device=device)
    x[:, 0] = 2**-9
    x[0, -1], x[1, -1] = 448, 449
    actual = fp8_quant_dequant(x)
    expected = torch.tensor([2**-9, 0], device=device)
    torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)


def test_power_of_two_preserves_every_fp32_exponent_field(device):
    exponents = torch.arange(-127, 129, dtype=torch.int32, device=device)
    expected = torch.tensor([field << 23 for field in range(256)], dtype=torch.int32, device=device)
    torch.testing.assert_close(_power_of_two(exponents).view(torch.int32), expected, atol=0, rtol=0)


def test_exponent_extraction_keeps_mantissa_bits_out_of_the_division(device):
    patterns = [field * (1 << 23) + tail for field in range(256) for tail in (0, 1, (1 << 23) - 1)]
    bits = torch.tensor(patterns, dtype=torch.int32, device=device)
    expected = torch.tensor([(value >> 23) - 127 for value in patterns], dtype=torch.int32, device=device)
    torch.testing.assert_close(_unbiased_exponent(bits), expected, atol=0, rtol=0)


@pytest.mark.parametrize("scale_format,block_size", [("e8m0", 32), ("e4m3", 16)])
def test_zero_groups_remain_zero_with_nonzero_scales(device, scale_format, block_size):
    x = torch.zeros(2, block_size * 2, dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(fp8_quant_dequant(x), x, rtol=0, atol=0)
    torch.testing.assert_close(
        fp4_quant_dequant(x, block_size, scale_format=scale_format),
        x,
        rtol=0,
        atol=0,
    )


def test_fp4_e4m3_and_e8m0_have_distinct_scale_rules(device):
    x = torch.full((1, 32), 7.0, device=device)
    e4m3 = fp4_quant_dequant(x, 16, scale_format="e4m3")
    e8m0 = fp4_quant_dequant(x, 32, scale_format="e8m0")
    torch.testing.assert_close(e4m3, torch.full_like(x, 6.75), rtol=0, atol=0)
    torch.testing.assert_close(e8m0, torch.full_like(x, 8.0), rtol=0, atol=0)


def test_quantization_copies_back_noncontiguous_rope_tail(device):
    storage = torch.arange(96, device=device, dtype=torch.float32).reshape(2, 48)
    tail = storage[:, -32:]
    prefix = storage[:, :16].clone()
    expected = fp8_quant_dequant(tail)
    actual = fp8_quant_dequant(tail, inplace=True)
    assert actual is tail
    torch.testing.assert_close(storage[:, :16], prefix, rtol=0, atol=0)
    torch.testing.assert_close(tail, expected, rtol=0, atol=0)


def test_sparse_sink_contributes_only_to_denominator(device):
    q = torch.zeros(1, 1, 2, 1, dtype=torch.bfloat16, device=device)
    kv = torch.tensor([[[2], [4]]], dtype=torch.bfloat16, device=device)
    indices = torch.tensor([[[0, 1, -1]]], dtype=torch.int32, device=device)
    sink = torch.tensor([0, math.log(2)], device=device)
    expected = torch.tensor([[[[2], [1.5]]]], dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(sparse_attn(q, kv, sink, indices, 1.0), expected, rtol=0, atol=0)


def test_duplicate_indices_are_independent_attention_slots(device):
    q = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, device=device)
    kv = torch.tensor([[[2]]], dtype=torch.bfloat16, device=device)
    indices = torch.tensor([[[0, 0, -1]]], dtype=torch.int32, device=device)
    sink = torch.zeros(1, device=device)
    expected = torch.tensor([[[[4 / 3]]]], dtype=torch.bfloat16, device=device)
    torch.testing.assert_close(sparse_attn(q, kv, sink, indices, 1.0), expected, rtol=0, atol=0)


def test_all_masked_and_empty_rows_are_zero(device):
    q = torch.ones(1, 2, 1, 2, dtype=torch.bfloat16, device=device)
    kv = torch.ones(1, 3, 2, dtype=torch.bfloat16, device=device)
    sink = torch.zeros(1, device=device)
    indices = torch.full((1, 2, 65), -1, dtype=torch.int32, device=device)
    for selected in (indices, indices[..., :0]):
        torch.testing.assert_close(
            sparse_attn(q, kv, sink, selected, 1.0),
            torch.zeros_like(q),
            rtol=0,
            atol=0,
        )


def test_sparse_value_multiply_rounds_exponentials_to_bf16(device):
    q = torch.ones(1, 1, 1, 1, dtype=torch.bfloat16, device=device)
    kv = torch.tensor([[[0], [0.5], [1]]], dtype=torch.bfloat16, device=device)
    indices = torch.tensor([[[0, 1, 2]]], dtype=torch.int32, device=device)
    sink = torch.zeros(1, device=device)
    exponentials = torch.tensor([math.exp(-1), math.exp(-0.5), 1])
    numerator = (exponentials.bfloat16().float() * torch.tensor([0, 0.5, 1])).sum()
    expected = (numerator / (exponentials.sum() + math.exp(-1))).bfloat16()
    actual = sparse_attn(q, kv, sink, indices, 1.0)
    torch.testing.assert_close(actual.cpu().flatten(), expected.view(1), rtol=0, atol=0)


def test_sparse_query_chunking_preserves_multiblock_output(device):
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(2, 5, 3, 8, generator=generator).bfloat16().to(device)
    kv = torch.randn(2, 11, 8, generator=generator).bfloat16().to(device)
    indices = torch.randint(-1, 11, (2, 5, 130), generator=generator).to(device)
    sink = torch.tensor([-1, 0, 1], device=device)
    full = sparse_attn(q, kv, sink, indices, 8**-0.5, query_chunk_size=5)
    chunked = sparse_attn(q, kv, sink, indices, 8**-0.5, query_chunk_size=1)
    torch.testing.assert_close(chunked, full, rtol=0, atol=0)


@pytest.mark.parametrize("with_heads", [False, True])
def test_real_rope_and_inverse_use_adjacent_pairs(device, with_heads):
    values = torch.tensor([[[1.0, 2, 3, 4], [5, 6, 7, 8]]], device=device)
    x = values.unsqueeze(2) if with_heads else values
    cosine = torch.tensor([[0.0, 0], [1, 1]], device=device)
    sine = torch.tensor([[1.0, 1], [0, 0]], device=device)
    expected = torch.tensor([[[-2.0, 1, -4, 3], [5, 6, 7, 8]]], device=device)
    if with_heads:
        expected = expected.unsqueeze(2)
    rotated = apply_rotary_emb_real(x, cosine, sine, inplace=False)
    torch.testing.assert_close(rotated, expected, rtol=0, atol=0)
    restored = apply_rotary_emb_real(rotated, cosine, sine, inverse=True)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


def test_invalid_group_and_attention_shapes_are_rejected():
    with pytest.raises(ValueError):
        fp8_quant_dequant(torch.zeros(17))
    with pytest.raises(ValueError):
        fp4_quant_dequant(torch.zeros(32), scale_format="unknown")
    with pytest.raises(ValueError):
        apply_rotary_emb_real(torch.zeros(1, 2, 3), torch.zeros(2, 1), torch.zeros(2, 1))
