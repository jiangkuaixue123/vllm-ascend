# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from weakref import ref

import torch
import torch_npu
from torch import nn
from vllm.logger import logger
from vllm.triton_utils import tl, triton

ACL_FORMAT_ND = 2


def kv_b_weight_views(projection, heads, nope_dim, value_dim, latent_dim):
    weight = projection.weight.data
    if torch_npu.get_npu_format(weight) != ACL_FORMAT_ND:
        weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_ND)
    return weight.reshape(heads, nope_dim + value_dim, latent_dim).split((nope_dim, value_dim), dim=1)


class MLAAbsorptionWeights(nn.Module):
    """Refresh captured weight layouts without changing attention-layer load handling."""

    def __init__(self, projection, heads, nope_dim, value_dim, latent_dim):
        super().__init__()
        self.projection = ref(projection)
        self.dims = (heads, nope_dim, value_dim, latent_dim)
        self.register_buffer("key", None, persistent=False)
        self.register_buffer("value", None, persistent=False)

    @torch.no_grad()
    def process_weights_after_loading(self):
        projection = self.projection()
        if projection.weight.device.type != "npu":
            return
        key, value = kv_b_weight_views(projection, *self.dims)
        value = value.transpose(1, 2)
        if self.key is None:
            self.key = torch.empty_like(key, memory_format=torch.contiguous_format)
            self.value = torch.empty_like(value, memory_format=torch.contiguous_format)
        self.key.copy_(key)
        self.value.copy_(value)
        logger.info_once("HYV4 Flash prepares the MLA absorption matrices after loading weights.")


@triton.jit
def _pack_rotary_query(
    query_rope,
    key_rope,
    latent,
    positions,
    cache,
    query,
    key,
    HEADS: tl.constexpr,
    ROPE: tl.constexpr,
    LATENT_DIM: tl.constexpr,
    QUERY_ROW: tl.constexpr,
    QUERY_HEAD: tl.constexpr,
    KEY_ROW: tl.constexpr,
    LATENT_ROW: tl.constexpr,
    LATENT_HEAD: tl.constexpr,
    LATENT_COL: tl.constexpr,
    POSITION_AXIS: tl.constexpr,
    POSITION_ROW: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    pair = tl.arange(0, ROPE // 2)
    column = tl.arange(0, ROPE)
    # Interleaved temporal, height and width frequencies follow the upstream M-RoPE layout.
    if POSITION_AXIS == 0:
        # DFly broadcasts the same token positions over all three M-RoPE axes.
        position = tl.load(positions + token * POSITION_ROW)
    else:
        axis = tl.where((pair % 3 == 1) & (pair <= 3 * HEIGHT), 1, 0)
        axis = tl.where((pair % 3 == 2) & (pair <= 3 * WIDTH), 2, axis)
        position = tl.load(positions + axis * POSITION_AXIS + token * POSITION_ROW)
    position = tl.where(position < 0, position + CACHE_ROWS, position)
    cosine = tl.load(cache + position * ROPE + pair)
    sine = tl.load(cache + position * ROPE + ROPE // 2 + pair)
    if head == HEADS:
        rotary = tl.load(key_rope + token * KEY_ROW + column)
    else:
        rotary = tl.load(query_rope + token * QUERY_ROW + head * QUERY_HEAD + column)
    even, odd = tl.split(tl.reshape(rotary, (ROPE // 2, 2)))
    rotated = tl.interleave(even * cosine - odd * sine, odd * cosine + even * sine)
    if head == HEADS:
        tl.store(key + token * ROPE + column, rotated)
    else:
        destination = (token * HEADS + head) * (LATENT_DIM + ROPE)
        tl.store(query + destination + LATENT_DIM + column, rotated)
        lane = tl.arange(0, LATENT_DIM)
        value = tl.load(latent + token * LATENT_ROW + head * LATENT_HEAD + lane * LATENT_COL)
        tl.store(query + destination + lane, value)


def pack_rotary_query(query_rope, key_rope, latent, positions, cache, sections):
    """Append the interleaved rotary lanes directly to a contiguous absorbed query."""
    tokens, heads, rope = query_rope.shape
    latent_dim = latent.shape[-1]
    query = latent.new_empty((tokens, heads, latent_dim + rope))
    key = key_rope.new_empty((tokens, rope))
    _pack_rotary_query[(tokens, heads + 1)](
        query_rope,
        key_rope,
        latent,
        positions,
        cache,
        query,
        key,
        heads,
        rope,
        latent_dim,
        *query_rope.stride()[:2],
        key_rope.stride(0),
        *latent.stride(),
        *positions.stride(),
        sections[1],
        sections[2],
        cache.shape[0],
        num_warps=1,
    )
    return query, key
