# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Router post-top-k normalization and DSpark confidence projection."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["ROWS"])
def _router_weights_kernel(
    scores,
    indices,
    output,
    ROWS,
    EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
):
    columns = tl.arange(0, TILE)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        index = tl.load(indices + row * TOP_K + columns, columns < TOP_K, other=0)
        value = tl.load(scores + row * EXPERTS + index, columns < TOP_K, other=0).to(tl.float32)
        if RENORMALIZE and TOP_K > 1:
            value = tl.div_rn(value, tl.sum(value, axis=0) + 1e-20)
        tl.store(output + row * TOP_K + columns, value * SCALE, columns < TOP_K)


def router_weights(scores, indices, renormalize=True, scale=1.5):
    if scores.ndim != 2 or indices.ndim != 2 or scores.shape[0] != indices.shape[0]:
        raise ValueError("Router scores and selected indices require matching rows")
    if scores.dtype != torch.float32 or not scores.is_contiguous() or not indices.is_contiguous():
        raise ValueError("Router fusion requires contiguous FP32 scores and contiguous indices")
    rows, experts = scores.shape
    top_k = indices.shape[-1]
    if not 1 <= top_k <= experts:
        raise ValueError("Router top-k must be between one and the expert count")
    output = torch.empty(indices.shape, dtype=torch.float32, device=scores.device)
    if rows:
        _router_weights_kernel[(min(rows, 64),)](
            scores,
            indices,
            output,
            ROWS=rows,
            EXPERTS=experts,
            TOP_K=top_k,
            RENORMALIZE=renormalize,
            SCALE=scale,
            TILE=triton.next_power_of_2(top_k),
            enable_fp_fusion=False,
        )
    return output


@triton.jit(do_not_specialize=["ROWS"])
def _confidence_kernel(
    hidden,
    markov,
    weight,
    output,
    ROWS,
    HIDDEN: tl.constexpr,
    MARKOV: tl.constexpr,
    H0: tl.constexpr,
    M0: tl.constexpr,
    CONCAT_BF16: tl.constexpr,
    TILE: tl.constexpr,
):
    columns = tl.arange(0, TILE)
    w = tl.load(weight + columns, columns < HIDDEN + MARKOV, other=0).to(tl.float32)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        h = tl.load(hidden + row * H0 + columns, columns < HIDDEN, other=0).to(tl.float32)
        m = tl.load(
            markov + row * M0 + columns - HIDDEN, (columns >= HIDDEN) & (columns < HIDDEN + MARKOV), other=0
        ).to(tl.float32)
        feature = tl.where(columns < HIDDEN, h, m)
        if CONCAT_BF16:
            feature = feature.to(tl.bfloat16).to(tl.float32)
        dot = tl.sum(feature * w, axis=0)
        tl.store(output + row, dot)


def draft_confidence(hidden, markov, weight):
    if hidden.ndim != 2 or markov.ndim != 2 or hidden.shape[0] != markov.shape[0]:
        raise ValueError("Confidence inputs require matching token rows")
    if weight.shape != (1, hidden.shape[1] + markov.shape[1]) or weight.dtype != torch.float32:
        raise ValueError("Confidence fusion requires the checkpoint FP32 projection vector")
    if hidden.stride(-1) != 1 or markov.stride(-1) != 1 or not weight.is_contiguous():
        raise ValueError("Confidence input feature dimensions must be contiguous")
    if hidden.dtype not in (torch.bfloat16, torch.float32) or markov.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Confidence inputs must use BF16 or FP32")
    rows = hidden.shape[0]
    output = torch.empty((rows,), dtype=torch.float32, device=hidden.device)
    if rows:
        _confidence_kernel[(min(rows, 64),)](
            hidden,
            markov,
            weight,
            output,
            ROWS=rows,
            HIDDEN=hidden.shape[1],
            MARKOV=markov.shape[1],
            H0=hidden.stride(0),
            M0=markov.stride(0),
            CONCAT_BF16=hidden.dtype == markov.dtype == torch.bfloat16,
            TILE=triton.next_power_of_2(weight.numel()),
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return output.sigmoid()
