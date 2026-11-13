# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse prefill expert memory passes while retaining native SiLU and quantization."""

import torch
import torch.nn.functional as F
from vllm import envs
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_ROWS = 8
_COLS = 256
_COMBINE_COLS = 2048


def prefill_fusion_available(states):
    return (
        HAS_TRITON
        and not envs.VLLM_BATCH_INVARIANT
        and states.device.type == "npu"
        and states.ndim == 2
        and states.shape[0] >= 128
        and states.shape[1] == 5120
        and states.dtype in (torch.int8, torch.bfloat16)
        and torch.is_inference_mode_enabled()
        and get_ascend_device_type() == AscendDeviceType.A3
    )


if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["M"])
    def _prefill_gate(x, gate, M, INTERMEDIATE: tl.constexpr, LIMIT: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
        rows = tl.program_id(0) * R + tl.arange(0, R)
        cols = tl.program_id(1) * C + tl.arange(0, C)
        valid = (rows[:, None] < M) & (cols[None, :] < INTERMEDIATE)
        value = tl.load(x + rows[:, None] * (2 * INTERMEDIATE) + cols[None, :], valid, 0).to(tl.float32)
        if LIMIT > 0:
            value = tl.where(value > LIMIT, LIMIT, value)
        tl.store(gate + rows[:, None] * INTERMEDIATE + cols[None, :], value, valid)

    @triton.jit(do_not_specialize=["M"])
    def _prefill_route(
        x,
        silu,
        factor,
        output,
        M,
        INTERMEDIATE: tl.constexpr,
        LIMIT: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
    ):
        rows = tl.program_id(0) * R + tl.arange(0, R)
        cols = tl.program_id(1) * C + tl.arange(0, C)
        valid = (rows[:, None] < M) & (cols[None, :] < INTERMEDIATE)
        offsets = rows[:, None] * INTERMEDIATE + cols[None, :]
        up = tl.load(x + rows[:, None] * (2 * INTERMEDIATE) + INTERMEDIATE + cols[None, :], valid, 0).to(tl.float32)
        if LIMIT > 0:
            up = tl.where(up < -LIMIT, -LIMIT, up)
            up = tl.where(up > LIMIT, LIMIT, up)
        activated = tl.load(silu + offsets, valid, 0)
        scale = tl.load(factor + rows, rows < M, 0)
        # Keep both FP32 products and the original BF16 boundary before quantization.
        value = (activated * up) * scale[:, None]
        tl.store(output + offsets, value.to(tl.bfloat16), valid)

    @triton.jit(do_not_specialize=["M", "EXPANDED"])
    def _prefill_combine(
        x,
        factor,
        row_ids,
        owned,
        output,
        M,
        EXPANDED,
        H: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * C + tl.arange(0, C)
        valid = (row < M) & (cols < H)
        total = tl.full((C,), 0, tl.float32)
        for slot in tl.static_range(K):
            pair = row * K + slot
            index = tl.abs(tl.load(row_ids + pair, row < M, 0))
            in_range = (row < M) & (index >= 0) & (index < EXPANDED)
            weight = tl.load(owned + pair, row < M, 0)
            active = in_range & (weight != 0)
            scale = tl.load(factor + index, active, 0)
            value = tl.load(
                x + index * H + cols,
                valid & active & (scale != 0),
                0,
            ).to(tl.float32)
            value = value * weight
            # Native unpermute skips zero weights, including NaN loads and additions.
            if slot == 0:
                total = tl.where(active, value, 0.0)
            else:
                total = tl.where(active, total + value, total)
        tl.store(output + row * H + cols, total, valid)

    def prefill_route_activation(gate_up, factor, limit):
        rows, width = gate_up.shape
        intermediate = width // 2
        gate = torch.empty((rows, intermediate), dtype=torch.float32, device=gate_up.device)
        grid = (triton.cdiv(rows, _ROWS), triton.cdiv(intermediate, _COLS))
        _prefill_gate[grid](
            gate_up, gate, M=rows, INTERMEDIATE=intermediate, LIMIT=limit, R=_ROWS, C=_COLS, enable_fp_fusion=False
        )
        silu = F.silu(gate)
        del gate
        output = torch.empty_like(silu, dtype=torch.bfloat16)
        _prefill_route[grid](
            gate_up,
            silu,
            factor,
            output,
            M=rows,
            INTERMEDIATE=intermediate,
            LIMIT=limit,
            R=_ROWS,
            C=_COLS,
            enable_fp_fusion=False,
        )
        return output

    def prefill_combine_output(states, factor, row_ids, owned):
        rows, top_k = owned.shape
        expanded, hidden = states.shape
        output = torch.empty((rows, hidden), dtype=torch.float32, device=states.device)
        if rows == 0:
            return output
        _prefill_combine[(rows, triton.cdiv(hidden, _COMBINE_COLS))](
            states,
            factor,
            row_ids,
            owned,
            output,
            M=rows,
            EXPANDED=expanded,
            H=hidden,
            K=top_k,
            C=_COMBINE_COLS,
            enable_fp_fusion=False,
        )
        return output

else:
    prefill_route_activation = None
    prefill_combine_output = None
