# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse four-stream residual mixing without contracting the FP32 products."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

_HIDDEN_SIZE = 5120
_HC_WIDTH = 4
_BLOCK_SIZE = 1024

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _hc_post4_kernel(
        sub,
        residual,
        post,
        comb,
        output,
        HIDDEN: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        token = row_id // WIDTH
        stream = row_id % WIDTH
        channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = channel < HIDDEN
        offsets = row_id * HIDDEN + channel
        row = token * (WIDTH * HIDDEN) + channel
        gate = token * WIDTH * WIDTH + stream
        r0 = tl.load(residual + row, valid, 0).to(tl.float32)
        r1 = tl.load(residual + row + HIDDEN, valid, 0).to(tl.float32)
        r2 = tl.load(residual + row + 2 * HIDDEN, valid, 0).to(tl.float32)
        r3 = tl.load(residual + row + 3 * HIDDEN, valid, 0).to(tl.float32)
        p0 = tl.load(comb + gate) * r0
        p1 = tl.load(comb + gate + WIDTH) * r1
        p2 = tl.load(comb + gate + 2 * WIDTH) * r2
        p3 = tl.load(comb + gate + 3 * WIDTH) * r3
        # The qualified A3 ReduceSum evaluates the four products left to right.
        mixed = ((p0 + p1) + p2) + p3
        # ReduceSum starts from positive zero, including when all products are -0.
        mixed = tl.where(mixed == 0.0, 0.0, mixed)
        sub_value = tl.load(sub + token * HIDDEN + channel, valid, 0).to(tl.float32)
        scaled = tl.load(post + token * WIDTH + stream) * sub_value
        result = scaled + mixed
        tl.store(output + offsets, result.to(tl.bfloat16), valid)

    def hc_post4(sublayer_output, residual, post, comb):
        output = torch.empty_like(residual)
        grid = (sublayer_output.shape[0] * _HC_WIDTH, triton.cdiv(_HIDDEN_SIZE, _BLOCK_SIZE))
        _hc_post4_kernel[grid](
            sublayer_output,
            residual,
            post,
            comb,
            output,
            HIDDEN=_HIDDEN_SIZE,
            WIDTH=_HC_WIDTH,
            BLOCK=_BLOCK_SIZE,
            enable_fp_fusion=False,
        )
        return output

else:
    hc_post4 = None
