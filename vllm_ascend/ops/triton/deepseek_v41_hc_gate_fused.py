# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare HC gates and finalize them with the established Sinkhorn sequence."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

_HC_WIDTH = 4
_MIX_WIDTH = 24
_EPS = 1e-6
_ROW_TILE = 2

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _affine_kernel(mixes, scale, base, pre, post, comb, ROWS: tl.constexpr, ROW_TILE: tl.constexpr):
        # Two four-float gates share a 32-byte store unit. Keep them in one
        # program so separate cores cannot update adjacent half-units.
        row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
        gate_lane = tl.arange(0, 4)
        comb_lane = tl.arange(0, 16)
        valid = row[:, None] < ROWS
        pre_value = tl.load(mixes + row[:, None] * 24 + gate_lane[None, :], valid, 0)
        post_value = tl.load(mixes + row[:, None] * 24 + gate_lane[None, :] + 4, valid, 0)
        comb_value = tl.load(mixes + row[:, None] * 24 + comb_lane[None, :] + 8, valid, 0)
        # Keep stores contiguous at their actual width, without masked columns
        # crossing the next row's four-float allocation.
        pre_affine = pre_value * tl.load(scale) + tl.load(base + gate_lane)[None, :]
        post_affine = post_value * tl.load(scale + 1) + tl.load(base + gate_lane + 4)[None, :]
        comb_affine = comb_value * tl.load(scale + 2) + tl.load(base + comb_lane + 8)[None, :]
        tl.store(pre + row[:, None] * 4 + gate_lane[None, :], pre_affine, valid)
        tl.store(post + row[:, None] * 4 + gate_lane[None, :], post_affine, valid)
        tl.store(comb + row[:, None] * 16 + comb_lane[None, :], comb_affine, valid)

    @triton.jit
    def _sum4_rows(x, lanes, STRIDE: tl.constexpr):
        adjacent = x + tl.gather(x, lanes ^ STRIDE, 1)
        return adjacent + tl.gather(adjacent, lanes ^ (2 * STRIDE), 1)

    @triton.jit
    def _finalize_kernel(
        pre_activated,
        post_activated,
        logits,
        pre,
        post,
        out,
        ROWS: tl.constexpr,
        ROW_TILE: tl.constexpr,
        EPS: tl.constexpr,
    ):
        lanes = tl.broadcast_to(tl.arange(0, 16)[None, :], (ROW_TILE, 16))
        row = (tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE))[:, None]
        gate_lanes = tl.arange(0, 4)[None, :]
        pre_gate = tl.load(pre_activated + row * 4 + gate_lanes, row < ROWS, 0)
        post_gate = tl.load(post_activated + row * 4 + gate_lanes, row < ROWS, 0)
        tl.store(pre + row * 4 + gate_lanes, pre_gate + EPS, row < ROWS)
        tl.store(post + row * 4 + gate_lanes, post_gate * 2, row < ROWS)
        offsets = row * 16 + lanes
        x = tl.load(logits + offsets, row < ROWS, 0)
        adjacent_max = tl.maximum(x, tl.gather(x, lanes ^ 1, 1))
        row_max = tl.maximum(adjacent_max, tl.gather(adjacent_max, lanes ^ 2, 1))
        exp_x = tl.exp(x - row_max)
        x = exp_x / _sum4_rows(exp_x, lanes, 1) + EPS
        x = x / (_sum4_rows(x, lanes, 4) + EPS)
        for _ in range(19):
            x = x / (_sum4_rows(x, lanes, 1) + EPS)
            x = x / (_sum4_rows(x, lanes, 4) + EPS)
        tl.store(out + offsets, x, row < ROWS)

    def hc_gate_fused(mixes, scale, base):
        rows = mixes.shape[0]
        pre_affine = torch.empty((rows, _HC_WIDTH), dtype=torch.float32, device=mixes.device)
        post_affine = torch.empty_like(pre_affine)
        logits = torch.empty((rows, _HC_WIDTH, _HC_WIDTH), dtype=torch.float32, device=mixes.device)
        pre = torch.empty((rows, _HC_WIDTH), dtype=torch.float32, device=mixes.device)
        post = torch.empty_like(pre)
        comb = torch.empty_like(logits)
        _affine_kernel[(triton.cdiv(rows, _ROW_TILE),)](
            mixes, scale, base, pre_affine, post_affine, logits, ROWS=rows, ROW_TILE=_ROW_TILE, enable_fp_fusion=False
        )
        # Retain both native sigmoid calls with their original contiguous shapes.
        pre_activated = torch.sigmoid(pre_affine)
        post_activated = torch.sigmoid(post_affine)
        _finalize_kernel[(triton.cdiv(rows, _ROW_TILE),)](
            pre_activated, post_activated, logits, pre, post, comb, ROWS=rows, ROW_TILE=_ROW_TILE, EPS=_EPS
        )
        return pre, post, comb

else:
    hc_gate_fused = None
