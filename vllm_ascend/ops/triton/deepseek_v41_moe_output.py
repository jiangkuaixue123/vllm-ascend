# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize the decode MoE sum, BF16 rounding and active-row selection once."""

import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

if HAS_TRITON:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _moe_output_kernel(
        routed, shared, active, output, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, HAS_MASK: tl.constexpr
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offset = row * HIDDEN + column
        left = tl.load(routed + offset, column < HIDDEN, 0)
        right = tl.load(shared + offset, column < HIDDEN, 0).to(tl.float32)
        summed = (left + right).to(tl.bfloat16)
        result = tl.where(tl.load(active + row) != 0, summed, 0.0).to(tl.bfloat16) if HAS_MASK else summed
        tl.store(output + offset, result, column < HIDDEN)


def moe_output(routed, shared, active, dtype):
    if (
        not HAS_TRITON
        or envs.VLLM_BATCH_INVARIANT
        or torch.compiler.is_compiling()
        or not torch.is_inference_mode_enabled()
        or dtype != torch.bfloat16
        or routed.device.type != "npu"
        or routed.dtype != torch.float32
        or routed.shape not in ((3, 5120), (6, 5120), (12, 5120))
        or shared.shape != routed.shape
        or shared.dtype not in (torch.bfloat16, torch.float32)
        or (active is not None and (active.shape != routed.shape[:1] or active.dtype != torch.bool))
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    import torch_npu

    for value in (routed, shared) if active is None else (routed, shared, active):
        if (
            value.device != routed.device
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.requires_grad
            or value.is_neg()
            or (value is not active and value.data_ptr() % 32)
            or int(torch_npu.get_npu_format(value)) not in (0, 2)
        ):
            return None
    output = torch.empty_like(routed, dtype=dtype)
    _moe_output_kernel[(routed.shape[0], 10)](
        routed,
        shared,
        routed if active is None else active,
        output,
        HIDDEN=5120,
        BLOCK=512,
        HAS_MASK=active is not None,
        enable_fp_fusion=False,
    )
    return output
