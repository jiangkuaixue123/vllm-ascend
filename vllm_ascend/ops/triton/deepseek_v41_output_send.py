# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse inverse RoPE, BF16 E4M3 QDQ, and TP4 output packing."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.deepseek_v41_fp8_qdq import _bias_qdq
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

_VECTOR_CORES = 48
_HEADS = 64
_HEAD_DIM = 512
_TP_SIZE = 4
_ROPE_DIM = 64
_GROUP_SIZE = 32


@triton.jit
def _output_send_kernel(
    x,
    cosine,
    sine,
    output,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    groups_per_head: tl.constexpr = HEAD_DIM // GROUP_SIZE
    channel = tl.arange(0, HEAD_DIM).reshape(groups_per_head, GROUP_SIZE)
    pair_columns = tl.broadcast_to(tl.arange(0, GROUP_SIZE)[None, :] ^ 1, (groups_per_head, GROUP_SIZE))
    for item in range(tl.program_id(0), ROWS * HEADS, tl.num_programs(0)):
        row = item // HEADS
        head = item % HEADS
        # Load one whole head contiguously, swap adjacent lanes in local memory,
        # then store the whole head at its destination in the exchange buffer.
        value = tl.load(x + item * HEAD_DIM + channel)
        rotary = channel >= HEAD_DIM - ROPE_DIM
        other = tl.gather(value, pair_columns, 1).to(tl.float32)
        phase_offset = row * ROPE_DIM + tl.maximum(channel - (HEAD_DIM - ROPE_DIM), 0)
        cos = tl.load(cosine + phase_offset, rotary, 1.0)
        sin = tl.load(sine + phase_offset, rotary, 0.0)
        product = other * sin
        signed_product = tl.where(channel % 2 == 0, -product, product)
        direct = value.to(tl.float32) * cos
        rotated_fp32 = direct + signed_product
        # The vector compiler may simplify additions without signed-zero
        # semantics. Native Add retains -0 when both products are negative zero.
        # Derive those signs from the operands, independently of that rewrite.
        value_word = tl.cast(value.to(tl.float32), tl.int32, bitcast=True)
        other_word = tl.cast(other, tl.int32, bitcast=True)
        direct_sign = value_word ^ tl.cast(cos, tl.int32, bitcast=True)
        paired_sign = other_word ^ tl.cast(sin, tl.int32, bitcast=True)
        paired_sign = paired_sign ^ tl.where(channel % 2 == 0, -0x8000_0000, 0)
        zero_word = direct_sign & paired_sign & -0x8000_0000
        rotated_fp32 = tl.where(
            (direct == 0.0) & (product == 0.0),
            tl.cast(zero_word, tl.float32, bitcast=True),
            rotated_fp32,
        )
        rotated = rotated_fp32.to(tl.bfloat16)
        # Retain the native RoPE BF16 rounding before the existing exact QDQ.
        quantized = _bias_qdq(tl.where(rotary, rotated, value))
        destination = head // LOCAL_HEADS
        local_head = head % LOCAL_HEADS
        output_head = (destination * ROWS + row) * LOCAL_HEADS + local_head
        tl.store(output + output_head * HEAD_DIM + channel, quantized)


def prepare_output_send(output, cosine, sine):
    init_device_properties_triton()
    cores = get_vectorcore_num()
    if cores != _VECTOR_CORES:
        return None
    rows = output.shape[0]
    send = torch.empty((_TP_SIZE * rows, _HEADS // _TP_SIZE, _HEAD_DIM), dtype=output.dtype, device=output.device)
    _output_send_kernel[(min(cores, rows * _HEADS),)](
        output,
        cosine,
        sine,
        send,
        ROWS=rows,
        HEADS=_HEADS,
        HEAD_DIM=_HEAD_DIM,
        LOCAL_HEADS=_HEADS // _TP_SIZE,
        ROPE_DIM=_ROPE_DIM,
        GROUP_SIZE=_GROUP_SIZE,
        enable_fp_fusion=False,
    )
    return send
