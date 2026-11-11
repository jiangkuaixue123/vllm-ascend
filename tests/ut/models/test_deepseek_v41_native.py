# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.compact_kv import pack_window, unpack_window
from vllm_ascend.models.deepseek_v41.eager_kernels import block_scale_bytes, cache_scale_values
from vllm_ascend.models.deepseek_v41.native_ops import NativeMXLinear, quantize_activation


def test_exact_scale_table_and_neighbors_of_every_interior_bin():
    encoded = torch.arange(256, dtype=torch.int32)
    powers = (encoded * (1 << 23)).view(torch.float32)
    assert torch.equal(cache_scale_values(encoded), powers)
    boundary = powers[2:254]
    below = torch.nextafter(boundary, torch.zeros_like(boundary))
    above = torch.nextafter(boundary, torch.full_like(boundary, torch.inf))
    values = torch.stack((below, boundary, above))
    expected = torch.stack((encoded[2:254], encoded[2:254], encoded[3:255]))
    assert torch.equal(block_scale_bytes(values, 1.0), expected.to(torch.uint8))


def test_cache_and_projection_use_identical_fp8_codes_and_exponents():
    x = torch.randn((6, 512), generator=torch.Generator().manual_seed(137)).bfloat16()
    x[0].zero_()
    x[1].mul_(2.0**-20)
    codes, pairs = quantize_activation(x)
    packed = pack_window(x)
    assert torch.equal(packed[:, :512], codes.view(torch.uint8))
    assert torch.equal(packed[:, 512:], pairs.reshape(6, 16))
    assert (packed[:, 512:] != 0).all()
    reference = codes.float().reshape(6, 16, 32) * cache_scale_values(pairs).reshape(6, 16, 1)
    assert torch.equal(unpack_window(packed, 512), reference.reshape_as(x).bfloat16())


def test_quantize_activation_preserves_scale_pairs_and_bin_boundaries():
    x = torch.tensor([448.0, 449.0, -224.0, 0.0]).repeat_interleave(32).reshape(2, 64)
    encoded, scales = quantize_activation(x)
    assert scales.shape == (2, 1, 2)
    assert scales.flatten()[:3].tolist() == [127, 128, 126]
    decoded = encoded.float().reshape(2, 2, 32) * scales.view(torch.float8_e8m0fnu).float().reshape(2, 2, 1)
    torch.testing.assert_close(decoded[:, :, 0], torch.tensor([[448.0, 448.0], [-224.0, 0.0]]))


def test_projection_pairs_one_scale_row_per_output_channel():
    source = nn.Module()
    source.weight = nn.Parameter(torch.empty(64, 64), requires_grad=False)
    source.scale = nn.Parameter(torch.tensor([[121, 122], [125, 126]], dtype=torch.uint8), requires_grad=False)
    source.packed_fp4 = False
    projection = NativeMXLinear(source)
    assert projection.weight is source.weight
    assert projection.paired_scale.shape == (1, 64, 2)
    torch.testing.assert_close(projection.paired_scale[0], source.scale.repeat_interleave(32, dim=0))
