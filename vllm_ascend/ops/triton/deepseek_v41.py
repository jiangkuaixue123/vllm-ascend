# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit V4.1 vector fusions used by versioned flattened forward bundles."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _hc_gates_sinkhorn_kernel(
    mixes_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    lane = tl.arange(0, 4)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    comb_scale = tl.load(scale_ptr + 2)
    pre_logit = tl.load(mixes_ptr + token * 24 + lane) * pre_scale + tl.load(base_ptr + lane)
    post_logit = tl.load(mixes_ptr + token * 24 + 4 + lane) * post_scale + tl.load(base_ptr + 4 + lane)
    pre = tl.div_rn(1.0, 1.0 + tl.exp(-pre_logit)) + EPS
    post = 2.0 * tl.div_rn(1.0, 1.0 + tl.exp(-post_logit))
    matrix_offset = lane[:, None] * 4 + lane[None, :]
    matrix = tl.load(mixes_ptr + token * 24 + 8 + matrix_offset) * comb_scale
    matrix = matrix + tl.load(base_ptr + 8 + matrix_offset)
    exponential = tl.exp(matrix - tl.max(matrix, axis=1)[:, None])
    matrix = tl.div_rn(exponential, tl.sum(exponential, axis=1)[:, None]) + EPS
    matrix = tl.div_rn(matrix, tl.sum(matrix, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        matrix = tl.div_rn(matrix, tl.sum(matrix, axis=1)[:, None] + EPS)
        matrix = tl.div_rn(matrix, tl.sum(matrix, axis=0)[None, :] + EPS)
    tl.store(pre_ptr + token * 4 + lane, pre)
    tl.store(post_ptr + token * 4 + lane, post)
    tl.store(comb_ptr + token * 16 + matrix_offset, matrix)


def hc_gates_sinkhorn(mixes, scale, base, *, sinkhorn_iters=20, eps=1e-6):
    """Return FP32 next-pre, post and comb; never consume the preceding pre."""
    if mixes.ndim != 2 or mixes.shape[1] != 24 or scale.shape != (3,) or base.shape != (24,):
        raise ValueError("V4.1 HC fusion requires mixes[T,24], scale[3], base[24]")
    if sinkhorn_iters < 1 or eps < 0:
        raise ValueError("Sinkhorn iterations must be positive and eps nonnegative")
    if any(value.dtype != torch.float32 or not value.is_contiguous() for value in (mixes, scale, base)):
        raise ValueError("V4.1 HC fusion requires contiguous FP32 inputs")
    if any(value.device != mixes.device for value in (scale, base)):
        raise ValueError("HC inputs must share a device")
    rows = mixes.shape[0]
    pre = torch.empty((rows, 4), dtype=torch.float32, device=mixes.device)
    post = torch.empty_like(pre)
    comb = torch.empty((rows, 4, 4), dtype=torch.float32, device=mixes.device)
    if rows:
        _hc_gates_sinkhorn_kernel[(rows,)](
            mixes, scale, base, pre, post, comb, ITERS=sinkhorn_iters, EPS=eps, enable_fp_fusion=False
        )
    return pre, post, comb
