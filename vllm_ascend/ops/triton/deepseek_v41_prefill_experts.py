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
_COMBINE_COLS = 8192


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

    from vllm_ascend.ops.triton.triton_utils import (
        get_vectorcore_num,
        init_device_properties_triton,
    )

    @triton.jit
    def _zero_ep_padding(output, sizes, CAPACITY: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
        rank = tl.program_id(0)
        live = tl.load(sizes + rank)
        for start in range(live * H + tl.program_id(1) * B, CAPACITY * H, tl.num_programs(1) * B):
            offset = start + tl.arange(0, B)
            tl.store(output + rank * CAPACITY * H + offset, 0.0, offset < CAPACITY * H)

    def prefill_zero_ep_padding(output, sizes, capacity):
        if (
            output.dtype != torch.float32
            or not output.is_contiguous()
            or sizes.dtype != torch.int32
            or not sizes.is_contiguous()
            or sizes.device != output.device
            or sizes.shape != (16,)
            or output.shape != (16 * capacity, 5120)
            or not 0 < capacity <= 2048
        ):
            raise ValueError("Direct EP padding requires the bounded FP32 return layout")
        _zero_ep_padding[(16, 8)](output, sizes, CAPACITY=capacity, H=5120, B=8192)

    @triton.jit(do_not_specialize=["N", "EXPANDED"])
    def _prefill_route_metadata(
        row_ids,
        weights,
        ids,
        expert_map,
        factor,
        owned,
        N,
        EXPANDED,
        HAS_MAP: tl.constexpr,
        B: tl.constexpr,
    ):
        pair = tl.program_id(0) * B + tl.arange(0, B)
        valid = pair < N
        local = tl.full((B,), True, tl.int1)
        if HAS_MAP:
            expert = tl.load(ids + pair, valid, 0)
            local = tl.load(expert_map + expert, valid, -1) >= 0
        row = tl.abs(tl.load(row_ids + pair, valid, 0))
        row = tl.minimum(row, EXPANDED - 1)
        weight = tl.load(weights + pair, valid, 0)
        weight = tl.where(weight == 0, 0.0, weight)
        # The preceding zero kernel owns initialization; native local rows are unique.
        tl.store(factor + row, weight, valid & local)
        tl.store(owned + pair, local.to(tl.float32), valid)

    def prefill_dispatched_scales(row_ids, weights, ids, expert_map, num_rows):
        """Native routing gives every local pair a unique expanded row."""
        pairs = weights.numel()
        result = weights.new_zeros(num_rows)
        owned = torch.empty_like(weights, dtype=torch.float32)
        _prefill_route_metadata[(triton.cdiv(pairs, 256),)](
            row_ids,
            weights,
            ids,
            ids if expert_map is None else expert_map,
            result,
            owned,
            N=pairs,
            EXPANDED=num_rows,
            HAS_MAP=expert_map is not None,
            B=256,
            enable_fp_fusion=False,
        )
        return result, owned

    @triton.jit
    def _live_rows(groups, M, G: tl.constexpr, GB: tl.constexpr):
        if G:
            index = tl.arange(0, GB)
            count = tl.load(groups + index, index < G, 0).to(tl.int32)
            return tl.minimum(tl.maximum(tl.sum(count, 0), 0), M)
        return M

    @triton.jit(do_not_specialize=["M"])
    def _prefill_gate(
        x,
        gate,
        groups,
        M,
        G: tl.constexpr,
        GB: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        LIMIT: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
    ):
        live = _live_rows(groups, M, G, GB)
        blocks: tl.constexpr = tl.cdiv(INTERMEDIATE, C)
        for tile in range(tl.program_id(0), tl.cdiv(live, R) * blocks, tl.num_programs(0)):
            rows = tile // blocks * R + tl.arange(0, R)
            cols = tile % blocks * C + tl.arange(0, C)
            valid = (rows[:, None] < live) & (cols[None, :] < INTERMEDIATE)
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
        groups,
        M,
        G: tl.constexpr,
        GB: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        LIMIT: tl.constexpr,
        R: tl.constexpr,
        C: tl.constexpr,
    ):
        live = _live_rows(groups, M, G, GB)
        blocks: tl.constexpr = tl.cdiv(INTERMEDIATE, C)
        for tile in range(tl.program_id(0), tl.cdiv(live, R) * blocks, tl.num_programs(0)):
            rows = tile // blocks * R + tl.arange(0, R)
            cols = tile % blocks * C + tl.arange(0, C)
            valid = (rows[:, None] < live) & (cols[None, :] < INTERMEDIATE)
            offsets = rows[:, None] * INTERMEDIATE + cols[None, :]
            up = tl.load(
                x + rows[:, None] * (2 * INTERMEDIATE) + INTERMEDIATE + cols[None, :],
                valid,
                0,
            ).to(tl.float32)
            if LIMIT > 0:
                up = tl.where(up < -LIMIT, -LIMIT, up)
                up = tl.where(up > LIMIT, LIMIT, up)
            activated = tl.load(silu + offsets, valid, 0)
            scale = tl.load(factor + rows, rows < live, 0)
            value = (activated * up) * scale[:, None]
            tl.store(output + offsets, value.to(tl.bfloat16), valid)

    @triton.jit(do_not_specialize=["M", "EXPANDED"])
    def _prefill_combine(
        x,
        factor,
        row_ids,
        owned,
        output,
        output_rows,
        M,
        EXPANDED,
        H: tl.constexpr,
        K: tl.constexpr,
        C: tl.constexpr,
        MAPPED: tl.constexpr,
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
        destination = tl.load(output_rows + row, row < M, 0) if MAPPED else row
        tl.store(output + destination * H + cols, total, valid)

    def prefill_route_activation(gate_up, factor, limit, groups=None):
        rows, width = gate_up.shape
        intermediate = width // 2
        init_device_properties_triton()
        count = 0 if groups is None else groups.numel()
        source = gate_up if groups is None else groups
        gate = torch.empty((rows, intermediate), dtype=torch.float32, device=gate_up.device)
        grid = (get_vectorcore_num(),)
        _prefill_gate[grid](
            gate_up,
            gate,
            source,
            M=rows,
            G=count,
            GB=triton.next_power_of_2(max(count, 1)),
            INTERMEDIATE=intermediate,
            LIMIT=limit,
            R=_ROWS,
            C=_COLS,
            enable_fp_fusion=False,
        )
        silu = F.silu(gate) if groups is None else torch.ops._C_ascend.prefill_prefix_silu(gate, groups)
        del gate
        output = torch.empty_like(silu, dtype=torch.bfloat16)
        _prefill_route[grid](
            gate_up,
            silu,
            factor,
            output,
            source,
            M=rows,
            G=count,
            GB=triton.next_power_of_2(max(count, 1)),
            INTERMEDIATE=intermediate,
            LIMIT=limit,
            R=_ROWS,
            C=_COLS,
            enable_fp_fusion=False,
        )
        return output

    def prefill_combine_output(states, factor, row_ids, owned, out=None, output_rows=None):
        rows, top_k = owned.shape
        expanded, hidden = states.shape
        output = torch.empty((rows, hidden), dtype=torch.float32, device=states.device) if out is None else out
        if (
            (output.shape != (rows, hidden) if output_rows is None else output.ndim != 2 or output.shape[1] != hidden)
            or output.dtype != torch.float32
            or output.device != states.device
            or not output.is_contiguous()
        ):
            raise ValueError("Prefill combine output must be a contiguous FP32 token matrix")
        if output_rows is not None and (
            out is None
            or output_rows.shape != (rows,)
            or output_rows.device != states.device
            or output_rows.dtype != torch.int32
            or not output_rows.is_contiguous()
        ):
            raise ValueError("Prefill EP destinations must be a contiguous int32 row map")
        if rows == 0:
            return output
        _prefill_combine[(rows, triton.cdiv(hidden, _COMBINE_COLS))](
            states,
            factor,
            row_ids,
            owned,
            output,
            row_ids if output_rows is None else output_rows,
            M=rows,
            EXPANDED=expanded,
            H=hidden,
            K=top_k,
            C=_COMBINE_COLS,
            MAPPED=output_rows is not None,
            enable_fp_fusion=False,
        )
        return output

else:
    prefill_dispatched_scales = None
    prefill_route_activation = None
    prefill_combine_output = None
    prefill_zero_ep_padding = None
