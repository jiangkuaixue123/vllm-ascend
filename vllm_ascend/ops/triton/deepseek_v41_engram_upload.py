# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upload bounded quantized table rows and place their exact BF16 values."""

import struct

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType


@triton.jit(do_not_specialize=["rows"], do_not_specialize_on_alignment=["positions"])
def _unpack_engram_rows(packet, positions, output, rows):
    columns = tl.arange(0, 256)
    for step in tl.static_range(4):
        row = tl.program_id(0) * 4 + step
        if row < rows:
            weight = tl.load(packet + row * 256 + columns).to(tl.int8, bitcast=True).to(tl.int32)
            scale = tl.load(packet + rows * 256 + row * 8 + columns // 32).to(tl.int32)
            magnitude = tl.abs(weight)
            shift = tl.full((256,), 0, tl.int32)
            for bit in tl.static_range(1, 8):
                shift += (magnitude >= (1 << bit)).to(tl.int32)
            exponent = scale + shift
            bits = (exponent << 7) | ((magnitude << (7 - shift)) & 127)
            bits = tl.where(exponent >= 255, 0x7F80, bits)
            bits = tl.where(exponent == 0, magnitude * 64, bits)
            bits = tl.where(magnitude == 0, 0, bits | ((weight < 0).to(tl.int32) << 15))
            # Preserve the original A3 BF16 cast's canonical NaN payload.
            bits = tl.where(scale == 255, 0x7FFF, bits)
            destination = tl.load(positions + row)
            values = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
            tl.store(output + destination * 256 + columns, values)


def try_upload_engram_rows(weights, scales, positions, output):
    """The validated embedding owner supplies unique output row positions."""
    if (
        output.device.type != "npu"
        or output.dtype != torch.bfloat16
        or output.ndim != 2
        or output.shape[1] != 256
        or not output.is_contiguous()
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return False
    rows = len(positions)
    if not rows or len(weights) != rows * 256 or len(scales) != rows * 8:
        raise ValueError("Invalid quantized Engram row packet")
    packet = bytearray(weights)
    packet.extend(scales)
    position_offset = (len(packet) + 7) // 8 * 8
    packet.extend(bytes(position_offset - len(packet)))
    packet.extend(struct.pack(f"<{rows}q", *positions))
    uploaded = torch.frombuffer(packet, dtype=torch.uint8).to(output.device)
    # The A3 pointer-cast lowering rescales byte offsets; pass a typed alias.
    destinations = uploaded[position_offset:].view(torch.int64)
    _unpack_engram_rows[(triton.cdiv(rows, 4),)](uploaded, destinations, output, rows, multibuffer=False)
    return True
