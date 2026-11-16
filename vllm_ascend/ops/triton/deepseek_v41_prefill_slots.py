# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile flat physical slots for the shared-KV QK and PV consumers."""

import torch
from vllm.triton_utils import tl, triton

PREFILL_RUNS_TAG = -0x563432
SLOT_LIMIT = 1 << 24


@triton.jit
def _min_position(a, b):
    return tl.minimum(a, b)


@triton.jit
def _encode_runs(
    Slots, Encoded, Table, WIDTH: tl.constexpr, TABLE_STRIDE: tl.constexpr, BLOCK: tl.constexpr, TAG: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    slot = tl.load(Slots + row * WIDTH + col, col < WIDTH, -1)
    following = tl.load(Slots + row * WIDTH + col + 1, col + 1 < WIDTH, -1)
    end = (slot < 0) | (following != slot + 1) | (col == WIDTH - 1)
    last = tl.associative_scan(tl.where(end, col, BLOCK - 1), 0, _min_position, reverse=True)
    count = tl.minimum(last - col + 1, 128)
    word = tl.where(slot >= 0, slot | ((count - 1) << 24), -1)
    tl.store(Encoded + row * WIDTH + col, word, col < WIDTH)
    tl.store(Table + row * TABLE_STRIDE, TAG)


def encode_prefill_runs(slots: torch.Tensor, table: torch.Tensor, kv_length: int) -> torch.Tensor:
    """Preserve unsupported layouts; the native tag opts into flat run addressing."""
    if (
        slots.device.type != "npu"
        or slots.dtype != torch.int32
        or slots.ndim != 3
        or slots.shape[1] != 1
        or slots.shape[0] < 128
        or slots.shape[2] not in (512, 640)
        or not slots.is_contiguous()
        or not 0 < kv_length < SLOT_LIMIT
        or table.dtype != torch.int32
        or table.device != slots.device
        or table.shape != (slots.shape[0], slots.shape[2] // 16)
        or not table.is_contiguous()
    ):
        return slots
    encoded = torch.empty_like(slots)
    _encode_runs[(slots.shape[0],)](
        slots, encoded, table, slots.shape[2], table.stride(0), triton.next_power_of_2(slots.shape[2]), PREFILL_RUNS_TAG
    )
    return encoded
