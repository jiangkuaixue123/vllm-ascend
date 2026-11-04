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
    def _sinkhorn4_kernel(logits, out, EPS: tl.constexpr):
        # One contiguous matrix per token. In row-major order XOR 1/2 stays in a row; XOR 4/8 stays in a column.
        lanes = tl.arange(0, 16)
        offsets = tl.program_id(0) * 16 + lanes
        x = tl.load(logits + offsets)
        adjacent_max = tl.maximum(x, tl.gather(x, lanes ^ 1, 0))
        row_max = tl.maximum(adjacent_max, tl.gather(adjacent_max, lanes ^ 2, 0))
        exp_x = tl.exp(x - row_max)
        x = exp_x / _sum4(exp_x, lanes, 1) + EPS
        x = x / (_sum4(x, lanes, 4) + EPS)
        for _ in range(19):
            x = x / (_sum4(x, lanes, 1) + EPS)
            x = x / (_sum4(x, lanes, 4) + EPS)
        tl.store(out + offsets, x)

    def hc_sinkhorn4(logits: torch.Tensor) -> torch.Tensor:
        """Normalize freshly computed contiguous FP32 [..., 4, 4] logits."""
        rows = logits.numel() // 16
        if rows == 0:
            return logits
        out = torch.empty_like(logits)
        _sinkhorn4_kernel[(rows,)](logits, out, EPS=1e-6)
        return out

else:
    hc_sinkhorn4 = None
