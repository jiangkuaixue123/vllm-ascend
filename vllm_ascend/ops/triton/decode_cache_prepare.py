# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the original slot-zero alias rule and BF16 update selection."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

_RAW_WIDTH = 512
_BF16_MAGNITUDE_MASK = 0x7FFF
_BF16_INFINITY = 0x7F80
_BF16_NATIVE_NAN = 0x7FFF

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _prepare_decode_raw_write_kernel(
        page,
        values,
        slot_mapping,
        active,
        enabled,
        slots,
        updates,
        ROWS: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        WIDTH: tl.constexpr,
        PADDED_ROWS: tl.constexpr,
        MAGNITUDE_MASK: tl.constexpr,
        INFINITY: tl.constexpr,
        NATIVE_NAN: tl.constexpr,
    ):
        row = tl.program_id(0)
        candidates = tl.arange(0, PADDED_ROWS)
        positions = tl.load(slot_mapping + candidates, candidates < ROWS, 0)
        lives = tl.load(active + candidates, candidates < ROWS, 0) != 0
        allowed = tl.load(enabled + candidates, candidates < ROWS, 0) != 0
        valid = (candidates < ROWS) & lives & allowed & (positions >= 0) & (positions < PHYSICAL_ROWS)
        # ArgMax on the original boolean witness chooses the first live slot 0.
        zero_row = tl.min(tl.where(valid & (positions == 0), candidates, ROWS), axis=0)
        has_live_zero = zero_row < ROWS
        safe_zero_row = tl.minimum(zero_row, ROWS - 1)
        columns = tl.arange(0, WIDTH)
        live_zero = tl.load(values + safe_zero_row * WIDTH + columns, has_live_zero, 0)
        old_zero = tl.load(page + columns, ~has_live_zero, 0)
        zero_bits = tl.where(has_live_zero, live_zero, old_zero)

        position = tl.load(slot_mapping + row)
        live = (tl.load(active + row) != 0) & (tl.load(enabled + row) != 0)
        destination = tl.where(live & (position >= 0) & (position < PHYSICAL_ROWS), position, 0)
        current = tl.load(values + row * WIDTH + columns)
        selected = tl.where(destination == 0, zero_bits, current)
        # Both original SelectV2 calls normalize every BF16 NaN to +0x7fff.
        selected = tl.where((selected & MAGNITUDE_MASK) > INFINITY, NATIVE_NAN, selected)
        tl.store(slots + row, destination)
        tl.store(updates + row * WIDTH + columns, selected)

    def prepare_decode_raw_write(page, values, slot_mapping, active, enabled):
        slots = torch.empty_like(slot_mapping)
        updates = torch.empty_like(values)
        _prepare_decode_raw_write_kernel[(values.shape[0],)](
            page.view(torch.int16),
            values.view(torch.int16),
            slot_mapping,
            active,
            enabled,
            slots,
            updates.view(torch.int16),
            ROWS=values.shape[0],
            PHYSICAL_ROWS=page.shape[0],
            WIDTH=_RAW_WIDTH,
            PADDED_ROWS=triton.next_power_of_2(values.shape[0]),
            MAGNITUDE_MASK=_BF16_MAGNITUDE_MASK,
            INFINITY=_BF16_INFINITY,
            NATIVE_NAN=_BF16_NATIVE_NAN,
        )
        return slots, updates

else:
    prepare_decode_raw_write = None
