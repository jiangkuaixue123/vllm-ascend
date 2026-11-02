# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems

from vllm_ascend.models.deepseek_v41 import processor


def config(**kwargs):
    return SimpleNamespace(
        **(
            {"patch_size": 2, "downsample_ratio": 2, "min_pixels": 0, "max_image_tokens": 12, "max_wh_ratio": None}
            | kwargs
        )
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1, 1), (1, 1, 4)),
        ((9, 5), (3, 5, 10)),
        ((1, 400), (10, 2, 12)),
        ((400, 1), (2, 18, 12)),
    ],
)
def test_image_grid_small_tall_wide_and_partial_cells(size, expected):
    assert processor.image_grid(*size, config()) == expected


def test_patchify_matches_padded_rgb_and_bf16_patches():
    pixels = np.arange(5 * 9 * 3, dtype=np.uint8).reshape(5, 9, 3)
    img = Image.fromarray(pixels)
    patches, grid = processor.patchify(img, config())
    assert grid == (3, 5)
    expected = torch.from_numpy(np.asarray(ImageOps.pad(img, (10, 6), color=(127, 127, 127)), dtype=np.float32))
    expected = (((expected / 255) - 0.5) / 0.5).bfloat16()
    for row in range(3):
        for col in range(5):
            torch.testing.assert_close(
                patches[row * 5 + col], expected[row * 2 : row * 2 + 2, col * 2 : col * 2 + 2].permute(2, 0, 1)
            )
    distorted, _ = processor.patchify(img, config(max_wh_ratio=1))
    assert not torch.equal(patches, distorted)


def test_processor_slots_and_geometry_align_for_two_different_images():
    cfg = SimpleNamespace(image_token_id=129264, vision_config=config())
    tokenizer = Mock()
    tokenizer.convert_tokens_to_ids.return_value = cfg.image_token_id
    tokenizer.encode.return_value = [907, cfg.image_token_id, 44, cfg.image_token_id]
    proc = object.__new__(processor.DeepseekV41Processor)
    proc.info = Mock()
    proc.info.get_tokenizer.return_value = tokenizer
    proc.info.get_hf_config.return_value = cfg
    images = [Image.new("RGB", (9, 5)), Image.new("RGB", (1, 1))]
    data = proc._call_hf_processor("prompt", {"images": images}, {}, {"add_special_tokens": False})
    assert data["input_ids"] == [[907, cfg.image_token_id, 44, cfg.image_token_id]]
    assert data["image_grid_hw"].tolist() == [[3, 5], [1, 1]]
    items = MultiModalDataItems({"image": ImageProcessorItems(images)})
    (update,) = proc._get_prompt_updates(items, {}, {})
    assert update.target == [cfg.image_token_id]
    assert update.replacement(0) == [cfg.image_token_id] * 10
    assert update.replacement(1) == [cfg.image_token_id] * 4
    assert proc._hf_processor_applies_updates() is False
    with pytest.raises(ValueError, match="geometry"):
        proc._call_hf_processor("prompt", {"images": images}, {"image_mask": [0]}, {})
    tokenizer.convert_tokens_to_ids.return_value = -1
    with pytest.raises(ValueError, match="placeholder"):
        proc._call_hf_processor("prompt", {}, {}, {})


@pytest.mark.parametrize("cfg", [config(), config(max_image_tokens=5, max_wh_ratio=1)])
def test_dummy_image_geometry_is_within_the_token_budget(cfg):
    info = Mock()
    info.get_hf_config.return_value = SimpleNamespace(vision_config=cfg)
    dummy = processor.DeepseekV41DummyInputs(info)
    images = dummy.get_dummy_mm_data(128, {"image": 2}, {})["image"]
    assert len(images) == 2
    assert all(processor.image_grid(*img.size, cfg)[2] <= cfg.max_image_tokens for img in images)
    assert dummy.get_dummy_text({"image": 2}) == processor.IMAGE_PLACEHOLDER * 2
