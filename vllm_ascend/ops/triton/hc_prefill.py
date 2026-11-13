# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share four residual streams across the large-batch HC producers."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.jit(do_not_specialize=["rows"])
def _prepare(x, pre, mixed, states, rows, H: tl.constexpr, BLOCK: tl.constexpr):
    for tile in range(tl.program_id(0), rows * (H // BLOCK), tl.num_programs(0)):
        row = tile // (H // BLOCK)
        col = tile % (H // BLOCK) * BLOCK + tl.arange(0, BLOCK)
        offset = row * 4 * H + col
        r0 = tl.load(x + offset).to(tl.float32)
        r1 = tl.load(x + offset + H).to(tl.float32)
        r2 = tl.load(x + offset + 2 * H).to(tl.float32)
        r3 = tl.load(x + offset + 3 * H).to(tl.float32)
        tl.store(states + offset, r0)
        tl.store(states + offset + H, r1)
        tl.store(states + offset + 2 * H, r2)
        tl.store(states + offset + 3 * H, r3)
        p0 = tl.load(pre + row * 4) * r0
        p1 = tl.load(pre + row * 4 + 1) * r1
        p2 = tl.load(pre + row * 4 + 2) * r2
        p3 = tl.load(pre + row * 4 + 3) * r3
        value = ((p0 + p1) + p2) + p3
        value = tl.where(value == 0.0, 0.0, value)
        tl.store(mixed + row * H + col, value.to(tl.bfloat16))


@triton.jit(do_not_specialize=["rows"])
def _post(sub, residual, post, comb, out, rows, H: tl.constexpr, BLOCK: tl.constexpr):
    for tile in range(tl.program_id(0), rows * (H // BLOCK), tl.num_programs(0)):
        row = tile // (H // BLOCK)
        col = tile % (H // BLOCK) * BLOCK + tl.arange(0, BLOCK)
        offset = row * 4 * H + col
        r0 = tl.load(residual + offset).to(tl.float32)
        r1 = tl.load(residual + offset + H).to(tl.float32)
        r2 = tl.load(residual + offset + 2 * H).to(tl.float32)
        r3 = tl.load(residual + offset + 3 * H).to(tl.float32)
        value = tl.load(sub + row * H + col).to(tl.float32)
        for stream in tl.static_range(4):
            p0 = tl.load(comb + row * 16 + stream) * r0
            p1 = tl.load(comb + row * 16 + 4 + stream) * r1
            p2 = tl.load(comb + row * 16 + 8 + stream) * r2
            p3 = tl.load(comb + row * 16 + 12 + stream) * r3
            mixed = ((p0 + p1) + p2) + p3
            mixed = tl.where(mixed == 0.0, 0.0, mixed)
            scaled = tl.load(post + row * 4 + stream) * value
            tl.store(out + offset + stream * H, (scaled + mixed).to(tl.bfloat16))


def prepare(x, pre):
    init_device_properties_triton()
    mixed = x.new_empty((x.shape[0], 5120))
    states = torch.empty_like(x, dtype=torch.float32)
    _prepare[(get_vectorcore_num(),)](x, pre, mixed, states, x.shape[0], H=5120, BLOCK=1024, enable_fp_fusion=False)
    return mixed, states


def post(sub, residual, gates, comb):
    init_device_properties_triton()
    out = torch.empty_like(residual)
    _post[(get_vectorcore_num(),)](
        sub, residual, gates, comb, out, sub.shape[0], H=5120, BLOCK=1024, enable_fp_fusion=False
    )
    return out
