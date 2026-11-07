# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape masked cache store for captured DSpark context projection."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _store_context(
        field,
        slots,
        values,
        PHYSICAL_ROWS: tl.constexpr,
        FIELD_STRIDE: tl.constexpr,
        VALUE_STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, WIDTH)
        slot = tl.load(slots + row).to(tl.int64)
        live = (slot >= 0) & (slot < PHYSICAL_ROWS)
        destination = tl.where(live, slot, 0) * FIELD_STRIDE + columns
        bits = tl.load(values + row * VALUE_STRIDE + columns)
        tl.store(field + destination, bits, mask=live)

    def store_context_rows(field, slots, values):
        """Caller proves BF16/512-column geometry; scheduled positive slots are unique."""
        _store_context[(values.shape[0],)](
            field.view(torch.int16),
            slots,
            values.view(torch.int16),
            PHYSICAL_ROWS=field.shape[0],
            FIELD_STRIDE=field.stride(0),
            VALUE_STRIDE=values.stride(0),
            WIDTH=values.shape[1],
        )

else:
    store_context_rows = None
