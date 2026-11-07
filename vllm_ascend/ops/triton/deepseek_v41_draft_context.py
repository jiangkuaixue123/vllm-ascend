# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write accepted draft context rows without compacting device slot indices."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

CONTEXT_WIDTH = 512
MAX_CONTEXT_ROWS = 48
ND_FORMAT = 2


def supports_draft_context_write(field, slots, values):
    """Check geometry only; scheduled positive slots are unique and in the pool."""
    if not (
        field.device.type == "npu"
        and field.device == slots.device == values.device
        and field.ndim == values.ndim == 2
        and field.shape[1] == values.shape[1] == CONTEXT_WIDTH
        and field.shape[0] > 0
        and 0 < values.shape[0] <= MAX_CONTEXT_ROWS
        and slots.shape == (values.shape[0],)
        and field.dtype == values.dtype == torch.bfloat16
        and slots.dtype in (torch.int32, torch.int64)
        and field.stride(1) == values.stride(1) == slots.stride(0) == 1
        and field.stride(0) >= CONTEXT_WIDTH
        and values.stride(0) >= CONTEXT_WIDTH
        and all(not tensor.is_neg() and not tensor.is_conj() for tensor in (field, slots, values))
        and torch.is_inference_mode_enabled()
        and get_ascend_device_type() == AscendDeviceType.A3
    ):
        return False
    # Projections own their output. Reject overlapping views before an in-place write.
    begin = field.data_ptr()
    end = begin + ((field.shape[0] - 1) * field.stride(0) + CONTEXT_WIDTH) * field.element_size()
    source = values.data_ptr()
    source_end = source + ((values.shape[0] - 1) * values.stride(0) + CONTEXT_WIDTH) * values.element_size()
    return end <= source or source_end <= begin


if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _write_context_kernel(
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
        # Integer lanes preserve every BF16 bit, including signed zero and NaN payloads.
        bits = tl.load(values + row * VALUE_STRIDE + columns)
        tl.store(field + destination, bits, mask=live)

    def try_write_draft_context(field, slots, values):
        if not supports_draft_context_write(field, slots, values) or torch.npu.is_current_stream_capturing():
            return False
        from torch_npu import get_npu_format

        if any(get_npu_format(tensor) != ND_FORMAT for tensor in (field, slots, values)):
            return False
        _write_context_kernel[(values.shape[0],)](
            field.view(torch.int16),
            slots,
            values.view(torch.int16),
            PHYSICAL_ROWS=field.shape[0],
            FIELD_STRIDE=field.stride(0),
            VALUE_STRIDE=values.stride(0),
            WIDTH=CONTEXT_WIDTH,
        )
        return True

else:
    try_write_draft_context = None
