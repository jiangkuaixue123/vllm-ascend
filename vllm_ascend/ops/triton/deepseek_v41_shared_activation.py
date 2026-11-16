# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep native FP32 SiLU between shared gate/up producers and consumers."""

import torch
import torch.nn.functional as F
from vllm import envs
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _shared_prepare(source, gate, up, HIDDEN: tl.constexpr, BLOCK: tl.constexpr, LIMIT: tl.constexpr):
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        row = tl.program_id(0)
        g = tl.load(source + row * (2 * HIDDEN) + columns, columns < HIDDEN, 0).to(tl.float32)
        u = tl.load(source + row * (2 * HIDDEN) + HIDDEN + columns, columns < HIDDEN, 0).to(tl.float32)
        # Comparisons preserve NaNs and signed zero exactly as the original clamp.
        g = tl.where(g > LIMIT, LIMIT, g)
        u = tl.where(u > LIMIT, LIMIT, u)
        u = tl.where(u < -LIMIT, -LIMIT, u)
        tl.store(gate + row * HIDDEN + columns, g, columns < HIDDEN)
        tl.store(up + row * HIDDEN + columns, u, columns < HIDDEN)

    @triton.jit
    def _shared_product(gate, up, output, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        offsets = tl.program_id(0) * HIDDEN + columns
        g = tl.load(gate + offsets, columns < HIDDEN, 0)
        u = tl.load(up + offsets, columns < HIDDEN, 0)
        tl.store(output + offsets, (g * u).to(tl.bfloat16), columns < HIDDEN)


def shared_gate_up_activation(gate_up, limit):
    if (
        not HAS_TRITON
        or envs.VLLM_BATCH_INVARIANT
        or gate_up.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or gate_up.ndim != 2
        or not 0 < gate_up.shape[0] <= 8192
        or gate_up.shape[1] not in (1152, 2304, 4608)
        or gate_up.dtype != torch.bfloat16
        or not gate_up.is_contiguous()
        or gate_up.is_neg()
        or gate_up.is_conj()
        or limit != 10.0
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    from torch_npu import get_npu_format

    if get_npu_format(gate_up) not in (0, 2):
        return None
    rows, doubled = gate_up.shape
    hidden = doubled // 2
    gate = torch.empty((rows, hidden), dtype=torch.float32, device=gate_up.device)
    up = torch.empty_like(gate)
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=gate_up.device)
    grid = (rows, triton.cdiv(hidden, 512))
    _shared_prepare[grid](gate_up, gate, up, hidden, 512, limit, enable_fp_fusion=False)
    activated = F.silu(gate)
    _shared_product[grid](activated, up, output, hidden, 512, enable_fp_fusion=False)
    return output
