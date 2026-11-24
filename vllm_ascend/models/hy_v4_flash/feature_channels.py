# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _normalized_channel_products(
    projected,
    hidden,
    key_gamma,
    query_gamma,
    products,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    channel = tl.program_id(1)
    lane = tl.arange(0, TILE)
    valid = lane < WIDTH
    output_offset = (token * CHANNELS + channel) * WIDTH + lane
    key_offset = (token * (CHANNELS + 1) + channel) * WIDTH + lane
    key = tl.load(projected + key_offset, valid, 0.0).to(tl.float32)
    query = tl.load(hidden + output_offset, valid, 0.0).to(tl.float32)
    key_scale = tl.load(key_gamma + channel * WIDTH + lane, valid, 0.0).to(tl.float32)
    query_scale = tl.load(query_gamma + channel * WIDTH + lane, valid, 0.0).to(tl.float32)
    key *= tl.rsqrt(tl.sum(key * key, 0) / WIDTH + EPS)
    query *= tl.rsqrt(tl.sum(query * query, 0) / WIDTH + EPS)
    # Preserve the BF16 RMSNorm outputs and product before the native dot reduction.
    key = (key * key_scale).to(tl.bfloat16).to(tl.float32)
    query = (query * query_scale).to(tl.bfloat16).to(tl.float32)
    tl.store(products + output_offset, key * query, valid)


def normalized_channel_products(projected, hidden, key_gamma, query_gamma):
    tokens, channels, width = hidden.shape
    products = torch.empty_like(hidden)
    _normalized_channel_products[(tokens, channels)](
        projected,
        hidden,
        key_gamma,
        query_gamma,
        products,
        channels,
        width,
        triton.next_power_of_2(width),
        torch.finfo(torch.float32).eps,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return products
