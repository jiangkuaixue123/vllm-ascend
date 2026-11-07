# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One pass for native BF16-indexed SiLU, clipped up and BF16 output."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

BLOCK_ELEMENTS = 256
BF16_SIGN_OFFSET = 1 << 15

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _shared_swiglu_kernel(
        packed,
        gate_table,
        output,
        ELEMENTS: tl.constexpr,
        WIDTH: tl.constexpr,
        LIMIT: tl.constexpr,
        SIGN_OFFSET: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < ELEMENTS
        source = (offset // WIDTH) * (2 * WIDTH) + offset % WIDTH
        gate = tl.load(packed + source, valid, 0)
        index = gate.to(tl.int16, bitcast=True).to(tl.int32) + SIGN_OFFSET
        activated = tl.load(gate_table + index, valid, 0)
        up = tl.load(packed + source + WIDTH, valid, 0).to(tl.float32)
        # Comparisons retain NaNs and signed zero as native clamp does.
        up = tl.where(up > LIMIT, LIMIT, up)
        up = tl.where(up < -LIMIT, -LIMIT, up)
        tl.store(output + offset, (activated * up).to(tl.bfloat16), valid)

    def shared_swiglu_table(packed, gate_table, limit):
        rows, packed_width = packed.shape
        width = packed_width // 2
        output = torch.empty((rows, width), device=packed.device, dtype=packed.dtype)
        _shared_swiglu_kernel[(triton.cdiv(rows * width, BLOCK_ELEMENTS),)](
            packed,
            gate_table,
            output,
            ELEMENTS=rows * width,
            WIDTH=width,
            LIMIT=limit,
            SIGN_OFFSET=BF16_SIGN_OFFSET,
            BLOCK=BLOCK_ELEMENTS,
            enable_fp_fusion=False,
        )
        return output

else:
    shared_swiglu_table = None
