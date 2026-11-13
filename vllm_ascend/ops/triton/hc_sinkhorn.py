# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FP32 combination-matrix normalization for the four-stream, twenty-round HC configuration."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _sum4(x, lanes, STRIDE: tl.constexpr):
        adjacent = x + tl.gather(x, lanes ^ STRIDE, 0)
        return adjacent + tl.gather(adjacent, lanes ^ (2 * STRIDE), 0)

    @triton.jit
    def _normalize4(x, lanes, EPS: tl.constexpr):
        adjacent_max = tl.maximum(x, tl.gather(x, lanes ^ 1, 0))
        row_max = tl.maximum(adjacent_max, tl.gather(adjacent_max, lanes ^ 2, 0))
        exp_x = tl.exp(x - row_max)
        x = exp_x / _sum4(exp_x, lanes, 1) + EPS
        x = x / (_sum4(x, lanes, 4) + EPS)
        for _ in range(19):
            x = x / (_sum4(x, lanes, 1) + EPS)
            x = x / (_sum4(x, lanes, 4) + EPS)
        return x

    @triton.jit
    def _sinkhorn4_kernel(logits, out, EPS: tl.constexpr):
        lanes = tl.arange(0, 16)
        offsets = tl.program_id(0) * 16 + lanes
        tl.store(out + offsets, _normalize4(tl.load(logits + offsets), lanes, EPS))

    @triton.jit(do_not_specialize=["ROWS"])
    def _sinkhorn4_batch_kernel(logits, out, ROWS, R: tl.constexpr, EPS: tl.constexpr):
        # XOR only changes the low4 bits, so independent matrices share one vector.
        lanes = tl.arange(0, R * 16)
        offsets = tl.program_id(0) * R * 16 + lanes
        x = tl.load(logits + offsets, offsets < ROWS * 16, 0)
        tl.store(out + offsets, _normalize4(x, lanes, EPS), offsets < ROWS * 16)

    def hc_sinkhorn4(logits: torch.Tensor) -> torch.Tensor:
        """Normalize freshly computed contiguous FP32 [..., 4, 4] logits."""
        rows = logits.numel() // 16
        if rows == 0:
            return logits
        out = torch.empty_like(logits)
        if rows >= 128:
            _sinkhorn4_batch_kernel[(triton.cdiv(rows, 8),)](logits, out, rows, R=8, EPS=1e-6)
        else:
            _sinkhorn4_kernel[(rows,)](logits, out, EPS=1e-6)
        return out

else:
    hc_sinkhorn4 = None
