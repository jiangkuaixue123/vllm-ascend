# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated V4.1 vision tower and image-span assembly for the ModelSlim checkpoint."""

import torch
import torch.nn.functional as F
from torch import nn

from .core import rms_norm

# A query tile attends all keys in its image. Bound temporary attention storage for large patch grids.
VISION_QUERY_TILE_SIZE = 256


class VisionNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, 1e-6)


def vision_rotary_grid(height, width, head_dim, theta, device):
    if head_dim % 4:
        raise ValueError("Vision head size must be divisible by four for two-dimensional RoPE")
    axis_dim = head_dim // 2
    freqs = theta ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim)
    rows = torch.arange(height, device=device).repeat_interleave(width)
    cols = torch.arange(width, device=device).repeat(height)
    phase = (torch.stack((rows, cols), dim=1)[:, :, None] * freqs).flatten(1).unsqueeze(1)
    return phase.cos(), phase.sin()


def rotate_vision_heads(x, cos, sin):
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        q, k, v = (part.reshape(len(x), self.heads, -1) for part in self.wqkv(x).chunk(3, dim=-1))
        q = rotate_vision_heads(q, cos, sin).transpose(0, 1)
        k = rotate_vision_heads(k, cos, sin).transpose(0, 1)
        v = v.transpose(0, 1)
        output = torch.cat(
            [F.scaled_dot_product_attention(tile, k, v) for tile in q.split(VISION_QUERY_TILE_SIZE, dim=1)], dim=1
        )
        return self.wo(output.transpose(0, 1).reshape(len(x), -1))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = VisionNorm(config.hidden_size)
        self.norm2 = VisionNorm(config.hidden_size)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches):
        return self.proj(patches.flatten(1).to(self.proj.weight.dtype))


class DeepseekV41VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("Vision hidden size must be divisible by the number of attention heads")
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.theta = config.rope_theta
        self.patch_size = config.patch_size
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.num_hidden_layers))
        self.norm = VisionNorm(config.hidden_size)

    def forward(self, patches, height, width):
        if min(height, width) < 1 or patches.shape != (height * width, 3, self.patch_size, self.patch_size):
            raise ValueError("Vision patch tensor disagrees with its image grid")
        hidden_states = self.patch_embed(patches)
        cos, sin = vision_rotary_grid(height, width, self.head_dim, self.theta, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)
        return self.norm(hidden_states)


class DeepseekV41Aligner(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()
        self.ratio = config.downsample_ratio
        self.w1 = nn.Linear(config.hidden_size * self.ratio**2, output_dim)
        self.w2 = nn.Linear(output_dim, output_dim)

    def forward(self, x, height, width):
        r = self.ratio
        grid = x.reshape(height, width, -1)
        padded = F.pad(grid, (0, 0, 0, -width % r, 0, -height % r))
        rows, cols, channels = padded.shape
        # Same channel-major order inside each tile as a nonoverlapping r-by-r im2col.
        cells = padded.reshape(rows // r, r, cols // r, r, channels).permute(0, 2, 4, 1, 3)
        return self.w2(F.gelu(self.w1(cells.flatten(2))))


def init_vision_modules(model, config):
    """Install the checkpoint's root-level names for the ordinary indexed weight loader."""
    dim = config.text_config.hidden_size
    model.vision = DeepseekV41VisionEncoder(config.vision_config)
    model.aligner = DeepseekV41Aligner(config.vision_config, dim)
    model.image_start = nn.Parameter(torch.empty(dim))
    model.image_newline = nn.Parameter(torch.empty(dim))
    model.image_end = nn.Parameter(torch.empty(dim))


def embed_images(model, *, pixel_values=None, image_grid_hw=None, image_embeds=None, **_kwargs):
    """Encode each image separately. Returned rows include all learned image delimiters."""
    if image_embeds is not None:
        if pixel_values is not None or image_grid_hw is not None:
            raise ValueError("Supply image embeddings or pixels for a batch, exclusively")
        return tuple(image_embeds)
    if pixel_values is None and image_grid_hw is None:
        return ()
    if pixel_values is None or image_grid_hw is None or len(pixel_values) != len(image_grid_hw):
        raise ValueError("Each image must have both patches and its own grid")
    spans = []
    for patches, (height, width) in zip(pixel_values, image_grid_hw.tolist(), strict=True):
        cells = model.aligner(model.vision(patches, height, width), height, width)
        newline = model.image_newline.to(cells.dtype).expand(len(cells), 1, -1)
        rows = torch.cat((cells, newline), dim=1).flatten(0, 1)
        spans.append(torch.cat((model.image_start[None].to(rows.dtype), rows, model.image_end[None].to(rows.dtype))))
    return tuple(spans)
