# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read only the requested slots/columns of a strided BF16/FP32 latent-cache field."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

_ROWS_PER_TILE = 8


@triton.jit
def _read_field_kernel(
    field,
    slots,
    valid,
    out,
    NUM_SLOTS: tl.constexpr,
    PHYSICAL_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COLUMN_STRIDE: tl.constexpr,
    TILE_ROWS: tl.constexpr,
):
    # field already points at the first element of the view, including its storage offset.
    columns = tl.arange(0, WIDTH)
    for tile in range(tl.program_id(0), tl.cdiv(NUM_SLOTS, TILE_ROWS), tl.num_programs(0)):
        rows = tile * TILE_ROWS + tl.arange(0, TILE_ROWS)
        in_bounds = rows < NUM_SLOTS
        slot = tl.cast(tl.load(slots + rows, mask=in_bounds, other=0), tl.int64)
        live = (tl.load(valid + rows, mask=in_bounds, other=0) != 0) & (slot >= 0) & (slot < PHYSICAL_ROWS)
        offsets = tl.where(live, slot, 0) * ROW_STRIDE
        bits = tl.load(field + offsets[:, None] + columns[None, :] * COLUMN_STRIDE, mask=live[:, None], other=0)
        tl.store(out + tl.cast(rows, tl.int64)[:, None] * WIDTH + columns[None, :], bits, mask=in_bounds[:, None])


def read_strided_cache_field(field: torch.Tensor, slots: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Gather the requested lanes while preserving the reference active/future mask semantics."""
    if field.ndim != 2 or field.shape[1] not in (128, 512) or field.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Expected a two-dimensional BF16/FP32 latent-cache field of width 128 or 512")
    if (
        slots.ndim != 2
        or valid.shape != slots.shape
        or valid.dtype != torch.bool
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Expected a per-token table of integral slots and an equally shaped validity mask")
    if field.stride(1) != 1:
        raise ValueError("Strided page reads require contiguous columns within each physical row")
    width = field.shape[1]
    bits_dtype = torch.int16 if field.dtype == torch.bfloat16 else torch.int32
    result = torch.empty((*slots.shape, width), dtype=bits_dtype, device=field.device)
    num_slots = slots.numel()
    if num_slots:
        grid = (min(triton.cdiv(num_slots, _ROWS_PER_TILE), get_vectorcore_num()),)
        _read_field_kernel[grid](
            field.view(bits_dtype),
            slots.contiguous(),
            valid.contiguous(),
            result,
            NUM_SLOTS=num_slots,
            PHYSICAL_ROWS=field.shape[0],
            WIDTH=width,
            ROW_STRIDE=field.stride(0),
            COLUMN_STRIDE=field.stride(1),
            TILE_ROWS=_ROWS_PER_TILE,
        )
    values = result.view(field.dtype)
    # The A3 BF16 SelectV2 can canonicalize a NaN payload. Preserve the actual
    # decode_read mask operation on this dtype, including otherwise-valid lanes.
    if field.dtype == torch.bfloat16 and num_slots:
        values = torch.where(valid[..., None], values, 0)
    return values
