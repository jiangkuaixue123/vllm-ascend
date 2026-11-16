# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape masked cache store for captured DSpark context projection."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _store_context(
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
        bits = tl.load(values + row * VALUE_STRIDE + columns)
        tl.store(field + destination, bits, mask=live)

    def store_context_rows(field, slots, values):
        """Caller proves BF16/512-column geometry; scheduled positive slots are unique."""
        _store_context[(values.shape[0],)](
            field.view(torch.int16),
            slots,
            values.view(torch.int16),
            PHYSICAL_ROWS=field.shape[0],
            FIELD_STRIDE=field.stride(0),
            VALUE_STRIDE=values.stride(0),
            WIDTH=values.shape[1],
        )

    @triton.jit
    def _canonical_cache_words(words, WIDTH: tl.constexpr, FP32: tl.constexpr):
        if FP32:
            value = words.to(tl.float32, bitcast=True)
            nan = tl.full((WIDTH,), 0x7FFFFFFF, tl.int32).to(tl.float32, bitcast=True)
            return tl.where(value != value, nan, value).to(tl.int32, bitcast=True)
        else:
            value = words.to(tl.bfloat16, bitcast=True).to(tl.float32)
            return tl.where(value != value, tl.full((WIDTH,), 0x7FFF, tl.int16), words)

    @triton.jit
    def _store_quantized_cache(
        field,
        slots,
        active,
        enabled,
        values,
        ROWS: tl.constexpr,
        WIDTH: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        FIELD_STRIDE: tl.constexpr,
        FP32: tl.constexpr,
        META: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, WIDTH)
        if row < ROWS:
            slot = tl.load(slots + row)
            live = tl.load(active + row) & tl.load(enabled + row)
            live = live & (slot > 0) & (slot < PHYSICAL_ROWS)
            if live:
                bits = tl.load(values + row * WIDTH + columns)
                bits = _canonical_cache_words(bits, WIDTH, FP32)
                tl.store(field + slot * FIELD_STRIDE + columns, bits)
        else:
            # Only this owner writes slot zero; the stock argmax selects its first live source.
            rows = tl.arange(0, META)
            valid_row = rows < ROWS
            mapped_slots = tl.load(slots + rows, valid_row, -1)
            meta_live = tl.load(active + rows, valid_row, False) & tl.load(enabled + rows, valid_row, False)
            meta_live = meta_live & (mapped_slots >= 0) & (mapped_slots < PHYSICAL_ROWS) & valid_row
            first = tl.min(tl.where(meta_live & (mapped_slots == 0), rows, ROWS), 0)
            masked = tl.sum(meta_live.to(tl.int32), 0) < ROWS
            if (first < ROWS) | masked:
                if first < ROWS:
                    bits = tl.load(values + first * WIDTH + columns)
                else:
                    bits = tl.load(field + columns)
                bits = _canonical_cache_words(bits, WIDTH, FP32)
                tl.store(field + columns, bits)

    def store_quantized_cache_rows(field, slots, active, enabled, values):
        """Uniform main-graph caller owns unique positive slots; preserve stock zero aliases."""
        from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
        from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

        if not (
            field.device.type == "npu"
            and field.ndim == values.ndim == 2
            and field.shape[0] > 0
            and field.shape[1] in (128, 512)
            and field.dtype in (torch.bfloat16, torch.float32)
            and field.stride(1) == 1
            and field.stride(0) in (512, 1024)
            and field.storage_offset() + (field.shape[0] - 1) * field.stride(0) + field.shape[1] < 2**31
            and values.dtype == torch.bfloat16
            and values.shape == (slots.numel(), field.shape[1])
            and slots.numel() in (6, 24, 48)
            and slots.dtype == torch.int64
            and slots.shape == active.shape == enabled.shape == (values.shape[0],)
            and active.dtype == enabled.dtype == torch.bool
            and all(t.device == field.device and t.is_contiguous() for t in (slots, active, enabled, values))
            and all(
                not t.requires_grad and not t.is_neg() and not t.is_conj()
                for t in (field, slots, active, enabled, values)
            )
            and all(
                t.untyped_storage().data_ptr() != field.untyped_storage().data_ptr()
                for t in (slots, active, enabled, values)
            )
            and torch.is_inference_mode_enabled()
            and not torch.compiler.is_compiling()
            and get_ascend_device_type() == AscendDeviceType.A3
            and enable_custom_op()
        ):
            return False
        import torch_npu

        if not all(
            int(torch_npu.get_npu_format(t)) == int(torch_npu.Format.ND)
            for t in (field, slots, active, enabled, values)
        ):
            return False
        init_device_properties_triton()
        if get_vectorcore_num() != 48:
            return False
        # Retain the stock conversion before Where/index_copy_, especially FP32 latent fields.
        updates = values.to(field.dtype)
        integer = torch.int32 if field.dtype == torch.float32 else torch.int16
        _store_quantized_cache[(values.shape[0] + 1,)](
            field.view(integer),
            slots,
            active,
            enabled,
            updates.view(integer),
            ROWS=values.shape[0],
            WIDTH=values.shape[1],
            PHYSICAL_ROWS=field.shape[0],
            FIELD_STRIDE=field.stride(0),
            FP32=field.dtype == torch.float32,
            META=triton.next_power_of_2(values.shape[0]),
            enable_fp_fusion=False,
        )
        torch.autograd.graph.increment_version(field)
        return True

else:
    store_context_rows = None
    store_quantized_cache_rows = None
