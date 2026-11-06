# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse integer ring, page and physical-slot indices for draft attention."""

import torch
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _ring_indices_kernel(
        starts,
        table,
        slots,
        indices,
        TABLE_STRIDE: tl.constexpr,
        WINDOW: tl.constexpr,
        QUERY_ROWS: tl.constexpr,
        INDEX_TILE: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, WINDOW)
        end = tl.load(starts + row).to(tl.int64) - 1
        history = end - (end - lane) % WINDOW
        page = tl.load(table + row * TABLE_STRIDE + history // WINDOW).to(tl.int64)
        tl.store(slots + row * WINDOW + lane, page * WINDOW + history % WINDOW)
        if row == 0:
            index = tl.arange(0, INDEX_TILE)
            tl.store(indices + index, index.to(tl.int32), index < WINDOW + QUERY_ROWS)

    def build_ring_indices(starts, table, window_size, block_size):
        slots = torch.empty((starts.numel(), window_size), dtype=torch.int64, device=table.device)
        indices = torch.empty(window_size + block_size, dtype=torch.int32, device=table.device)
        _ring_indices_kernel[(starts.numel(),)](
            starts,
            table,
            slots,
            indices,
            TABLE_STRIDE=table.stride(0),
            WINDOW=window_size,
            QUERY_ROWS=block_size,
            INDEX_TILE=triton.next_power_of_2(window_size + block_size),
        )
        return slots, indices.view(1, 1, -1).expand(1, block_size, -1)

else:
    build_ring_indices = None
