# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select BF16 rows with the native A3 nonfinite and signed-zero behavior."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

_HIDDEN_SIZE = 5120
_BLOCK_SIZE = 1024
_BF16_MAGNITUDE_MASK = 0x7FFF
_BF16_INFINITY = 0x7F80
_BF16_NATIVE_NAN = 0x7FFF

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _decode_row_mask_kernel(
        values,
        active,
        output,
        HIDDEN: tl.constexpr,
        BLOCK: tl.constexpr,
        MAGNITUDE_MASK: tl.constexpr,
        INFINITY: tl.constexpr,
        NATIVE_NAN: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offset = row * HIDDEN + column
        live = tl.load(active + row) != 0
        bits = tl.load(values + offset, column < HIDDEN, 0)
        # Native SelectV2 maps every signed BF16 NaN to positive 0x7fff.
        # All finite patterns, infinities and signed zeros are copied unchanged.
        bits = tl.where((bits & MAGNITUDE_MASK) > INFINITY, NATIVE_NAN, bits)
        tl.store(output + offset, tl.where(live, bits, 0), column < HIDDEN)

    def decode_row_mask(values, active):
        output = torch.empty_like(values)
        _decode_row_mask_kernel[(values.shape[0], triton.cdiv(_HIDDEN_SIZE, _BLOCK_SIZE))](
            values.view(torch.int16),
            active,
            output.view(torch.int16),
            HIDDEN=_HIDDEN_SIZE,
            BLOCK=_BLOCK_SIZE,
            MAGNITUDE_MASK=_BF16_MAGNITUDE_MASK,
            INFINITY=_BF16_INFINITY,
            NATIVE_NAN=_BF16_NATIVE_NAN,
        )
        return output

else:
    decode_row_mask = None
