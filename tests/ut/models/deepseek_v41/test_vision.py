# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_ascend.models.deepseek_v41 import vision


def config():
    return SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=19,
        patch_size=2,
        downsample_ratio=3,
        rope_theta=10000,
    )


def test_rotary_grid_separates_vertical_and_horizontal_phases():
    cos, sin = vision.vision_rotary_grid(2, 3, 8, 10000, "cpu")
    expected = torch.tensor([[r, 0.01 * r, c, 0.01 * c] for r in range(2) for c in range(3)]).unsqueeze(1)
    torch.testing.assert_close(cos, expected.cos())
    torch.testing.assert_close(sin, expected.sin())
    with pytest.raises(ValueError, match="divisible"):
        vision.vision_rotary_grid(2, 3, 6, 10000, "cpu")


def test_tiled_bidirectional_attention_matches_dense_softmax(monkeypatch):
    torch.manual_seed(594)
    layer = vision.VisionAttention(config())
    x = torch.randn(6, 16)
    cos, sin = vision.vision_rotary_grid(2, 3, 8, 10000, "cpu")
    q, k, v = layer.wqkv(x).view(6, 3, 2, 8).unbind(dim=1)
    q, k = [vision.rotate_vision_heads(t, cos, sin) for t in (q, k)]
    logits = torch.einsum("nhd,mhd->hnm", q, k) / 8**0.5
    expected = layer.wo(torch.einsum("hnm,mhd->nhd", logits.softmax(-1), v).reshape(6, 16))
    monkeypatch.setattr(vision, "VISION_QUERY_TILE_SIZE", 2)
    torch.testing.assert_close(layer(x, cos, sin), expected)
    changed = x.clone()
    changed[-1].add_(5.0)
    assert not torch.allclose(layer(changed, cos, sin)[0], expected[0])


def test_aligner_channel_major_and_nondivisible_edges():
    torch.manual_seed(36)
    layer = vision.DeepseekV41Aligner(config(), 9)
    x = torch.randn(20, 16)
    plane = F.pad(x.view(4, 5, 16).permute(2, 0, 1), (0, 1, 0, 2))
    cells = F.unfold(plane[None], 3, stride=3).squeeze(0).T
    expected = layer.w2(F.gelu(layer.w1(cells)))
    torch.testing.assert_close(layer(x, 4, 5).reshape(4, 9), expected)


def test_multi_image_spans_are_independent_and_load_by_checkpoint_names():
    torch.manual_seed(37)
    model = nn.Module()
    vision.init_vision_modules(
        model, SimpleNamespace(vision_config=config(), text_config=SimpleNamespace(hidden_size=9))
    )
    model.image_start.data.fill_(1)
    model.image_newline.data.fill_(2)
    model.image_end.data.fill_(3)
    keys = set(dict(model.named_parameters()))
    assert {
        "image_start",
        "image_end",
        "image_newline",
        "vision.patch_embed.proj.weight",
        "vision.norm.weight",
        "vision.blocks.1.attn.wqkv.bias",
        "vision.blocks.0.mlp.w1.weight",
        "aligner.w2.weight",
    } <= keys
    patches = [torch.randn(20, 3, 2, 2), torch.randn(2, 3, 2, 2)]
    grids = torch.tensor([[4, 5], [1, 2]])
    images = vision.embed_images(model, pixel_values=patches, image_grid_hw=grids)
    for i, (nh, nw) in enumerate(grids.tolist()):
        cells = model.aligner(model.vision(patches[i], nh, nw), nh, nw)
        expected = [model.image_start]
        for row in cells:
            expected.extend(row.unbind(0))
            expected.append(model.image_newline)
        expected.append(model.image_end)
        torch.testing.assert_close(images[i], torch.stack(expected))
    patches[1].zero_()
    torch.testing.assert_close(vision.embed_images(model, pixel_values=patches, image_grid_hw=grids)[0], images[0])
    with pytest.raises(ValueError, match="grid"):
        vision.embed_images(model, pixel_values=patches, image_grid_hw=grids[:1])
    with pytest.raises(ValueError, match="grid"):
        model.vision(patches[0], 1, 1)
