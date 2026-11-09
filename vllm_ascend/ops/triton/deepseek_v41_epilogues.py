# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressor pooling and native attention output fusions."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["ROWS"])
def _pool_norm_kernel(values, scores, weight, output, ROWS, EPS: tl.constexpr):
    dim = tl.arange(0, 512)
    w = tl.load(weight + dim).to(tl.float32)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        v0 = tl.load(values + row * 1024 + dim).to(tl.float32)
        v1 = tl.load(values + row * 1024 + 512 + dim).to(tl.float32)
        s0 = tl.load(scores + row * 1024 + dim).to(tl.float32)
        s1 = tl.load(scores + row * 1024 + 512 + dim).to(tl.float32)
        maximum = tl.maximum(s0, s1)
        p0, p1 = tl.exp(s0 - maximum), tl.exp(s1 - maximum)
        denominator = p0 + p1
        pooled = v0 * tl.div_rn(p0, denominator) + v1 * tl.div_rn(p1, denominator)
        pooled = pooled.to(output.dtype.element_ty).to(tl.float32)
        variance = tl.sum(pooled * pooled, axis=0) / 512.0
        result = w * (pooled * tl.rsqrt(variance + EPS))
        tl.store(output + row * 512 + dim, result)


def compressor_pool_norm(values, scores, weight, eps=1e-20, dtype=torch.bfloat16):
    if values.ndim != 3 or values.shape[1:] != (2, 512) or scores.shape != values.shape:
        raise ValueError("Compressor fusion requires two 512-dimensional tokens per group")
    if not all(x.is_contiguous() for x in (values, scores, weight)) or weight.shape != (512,):
        raise ValueError("Compressor fusion requires contiguous values, scores and norm weights")
    rows = values.shape[0]
    output = torch.empty((rows, 512), device=values.device, dtype=dtype)
    if rows:
        _pool_norm_kernel[(min(rows, 64),)](values, scores, weight, output, ROWS=rows, EPS=eps, enable_fp_fusion=False)
    return output


@triton.jit(do_not_specialize=["QUERIES"])
def _sink_kernel(
    output,
    maximum,
    total,
    sink,
    slots,
    result,
    QUERIES,
    HEADS: tl.constexpr,
    SLOT_WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    TILE: tl.constexpr,
):
    channels = tl.arange(0, TILE)
    for block in range(tl.program_id(0), tl.cdiv(QUERIES * HEADS, 4), tl.num_programs(0)):
        head_row = block * 4 + tl.arange(0, 4)
        live = head_row < QUERIES * HEADS
        query, head = head_row // HEADS, head_row % HEADS
        max_value = tl.load(maximum + head_row, live, other=0).to(tl.float32)
        sum_value = tl.load(total + head_row, live, other=1).to(tl.float32)
        lse = max_value + tl.log(sum_value)
        factor = tl.div_rn(1.0, 1.0 + tl.exp(-(lse - tl.load(sink + head))))
        value = tl.load(
            output + head_row[:, None] * DIM + channels[None, :], live[:, None] & (channels[None, :] < DIM), other=0
        ).to(tl.float32)
        valid = tl.load(slots + query * SLOT_WIDTH, live, other=-1) >= 0
        value = tl.where(valid[:, None], value * factor[:, None], 0.0)
        tl.store(result + head_row[:, None] * DIM + channels[None, :], value, live[:, None] & (channels[None, :] < DIM))


def attention_sink(output, maximum, total, sink, slots):
    batch, sequence, heads, dim = output.shape
    if maximum.numel() != batch * sequence * heads or total.numel() != maximum.numel():
        raise ValueError("SFA softmax statistics must contain one value per query and head")
    if slots.shape[:3] != (batch, sequence, 1) or sink.shape != (heads,):
        raise ValueError("SFA sink and sparse slots disagree with output geometry")
    if not all(x.is_contiguous() for x in (output, maximum, total, sink, slots)):
        raise ValueError("SFA epilogue fusion requires contiguous inputs")
    result = torch.empty_like(output)
    if batch * sequence:
        _sink_kernel[(min(triton.cdiv(batch * sequence * heads, 4), 64),)](
            output,
            maximum,
            total,
            sink,
            slots,
            result,
            QUERIES=batch * sequence,
            HEADS=heads,
            SLOT_WIDTH=slots.shape[-1],
            DIM=dim,
            TILE=triton.next_power_of_2(dim),
            enable_fp_fusion=False,
        )
    return result
