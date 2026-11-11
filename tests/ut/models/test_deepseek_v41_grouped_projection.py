# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_ascend.models.deepseek_v41.native_ops import NativeGroupedOutputProjection, grouped_output_projection


def test_grouped_projection_packs_raw_checkpoint_and_paired_scales_on_load_and_reload():
    groups, rank, width = 3, 64, 128
    layer = NativeGroupedOutputProjection(groups, rank, width)
    gen = torch.Generator().manual_seed(41)
    storage = layer.weight.data_ptr()
    for _ in range(2):
        raw = torch.randn(groups, rank, width, generator=gen).to(torch.float8_e4m3fn)
        scales = torch.randint(117, 135, layer.scale.shape, dtype=torch.uint8, generator=gen)
        layer.weight.copy_(raw.flatten(0, 1))
        layer.scale.copy_(scales)
        layer.pack_after_loading()
        assert layer.weight.data_ptr() == storage
        grouped = layer.weight.view(groups, width, rank)
        assert torch.equal(grouped.view(torch.uint8), raw.transpose(1, 2).contiguous().view(torch.uint8))
        expanded = scales.view(groups, rank // 32, width // 32).repeat_interleave(32, dim=1)
        expected = expanded.view(groups, rank, width // 64, 2).transpose(1, 2)
        assert torch.equal(layer.paired_weight_scale.view(torch.uint8), expected)


def test_dense_reference_preserves_group_boundaries_and_batch_dimensions():
    gen = torch.Generator().manual_seed(99)
    groups, rank, width = 4, 8, 64
    x = torch.randint(-4, 5, (2, 3, groups, width), generator=gen).bfloat16()
    proj = torch.nn.Linear(width, groups * rank, bias=False, dtype=torch.bfloat16)
    w = proj.weight.detach().view(groups, rank, width)
    expected = torch.stack([x[:, :, g] @ w[g].T for g in range(groups)], dim=2)
    torch.testing.assert_close(grouped_output_projection(x, proj), expected, rtol=0, atol=0)
