# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact patch-grid selection and vLLM placeholder expansion for V4.1 images."""

import math

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import BatchFeature
from transformers.image_transforms import to_pil_image
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"


def image_grid(width, height, config):
    """Return ViT patch height/width and the number of language-model token rows in one span."""
    patch, downsample, limit = config.patch_size, config.downsample_ratio, config.max_image_tokens
    if min(width, height, patch, downsample) < 1 or limit < 4:
        raise ValueError("Invalid V4.1 image dimensions or token limit")
    if config.max_wh_ratio is not None:
        if config.max_wh_ratio <= 0:
            raise ValueError("Invalid image aspect ratio limit")
        width = min(width, height * config.max_wh_ratio)
    if width * height < config.min_pixels:
        scale = math.sqrt(config.min_pixels / (width * height))
        width, height = int(width * scale), int(height * scale)
    nh, nw = math.ceil(height / patch), math.ceil(width / patch)

    def tokens(rows, cols):
        return math.ceil(rows / downsample) * (math.ceil(cols / downsample) + 1) + 2

    if tokens(nh, nw) > limit:
        aspect = height / width
        budget_w = math.sqrt((limit - 2) / aspect + 0.25) - 0.5
        budget_h = budget_w * aspect
        if budget_w < 1:
            nh, nw = ((limit - 2) // 2) * downsample, downsample
        elif budget_h < 1:
            nh, nw = downsample, (limit - 3) * downsample
        else:
            cell = patch * downsample
            beta = min(math.floor(budget_w) * cell / width, math.floor(budget_h) * cell / height)
            nh, nw = math.floor(height * beta / patch), math.floor(width * beta / patch)
    if min(nh, nw) < 1 or tokens(nh, nw) > limit:
        raise ValueError("Could not fit image into the configured V4.1 token budget")
    return nh, nw, tokens(nh, nw)


def patchify(image, config):
    if not isinstance(image, Image.Image):
        image = to_pil_image(image)
    image = image.convert("RGB")
    nh, nw, _ = image_grid(*image.size, config)
    p = config.patch_size
    size = (nw * p, nh * p)
    if config.max_wh_ratio is not None and image.width >= config.max_wh_ratio * image.height:
        image = image.resize(size)
    else:
        image = ImageOps.pad(image, size, color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    normalized = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patches = normalized.reshape(3, nh, p, nw, p).permute(1, 3, 0, 2, 4).reshape(nh * nw, 3, p, p)
    return patches, (nh, nw)


class DeepseekV41ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": self.get_hf_config().vision_config.max_image_tokens}


class DeepseekV41DummyInputs(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        cfg = self.info.get_hf_config().vision_config
        # Each grid row adds a newline. Choose the most square full-budget grid allowed by the config.
        available = cfg.max_image_tokens - 2
        grids = []
        for rows in range(1, available // 2 + 1):
            cols = available // rows - 1
            if cfg.max_wh_ratio is not None:
                cols = min(cols, math.floor(rows * cfg.max_wh_ratio))
            if cols > 0:
                grids.append((rows, cols))
        if not grids:
            raise ValueError("Could not fit any dummy image into the configured V4.1 token budget")
        rows, cols = max(grids, key=lambda size: (size[0] * (size[1] + 1), -abs(math.log(size[0] / size[1]))))
        cell = cfg.patch_size * cfg.downsample_ratio
        return {
            "image": self._get_dummy_images(
                width=cols * cell,
                height=rows * cell,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class DeepseekV41Processor(BaseMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if mm_kwargs:
            raise ValueError("V4.1 image processing uses the geometry from the model config")
        tokenizer = self.info.get_tokenizer()
        config = self.info.get_hf_config()
        if tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER) != config.image_token_id:
            raise ValueError("Tokenizer does not recognize the configured V4.1 image placeholder")
        result = BatchFeature({"input_ids": [tokenizer.encode(prompt, **tok_kwargs)]})
        if images := mm_data.get("images"):
            processed = [patchify(image, config.vision_config) for image in images]
            result["pixel_values"] = [patches for patches, _ in processed]
            result["image_grid_hw"] = torch.tensor([grid for _, grid in processed], dtype=torch.long)
        return result

    def _hf_processor_applies_updates(self, *args, **kwargs):
        # Preserve the raw placeholder until vLLM inserts the per-image PromptReplacement.
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "image_grid_hw": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "image_embeds": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        config = self.info.get_hf_config()
        token_id = config.image_token_id

        def replacement(index):
            items = mm_items.get_items("image", (ImageProcessorItems, ImageEmbeddingItems))
            count = (
                items.get_feature_size(index)
                if isinstance(items, ImageEmbeddingItems)
                else image_grid(*items.get_image_size(index), config.vision_config)[2]
            )
            return [token_id] * count

        return [PromptReplacement(modality="image", target=[token_id], replacement=replacement)]


register_vision_processor = MULTIMODAL_REGISTRY.register_processor(
    DeepseekV41Processor, info=DeepseekV41ProcessingInfo, dummy_inputs=DeepseekV41DummyInputs
)
