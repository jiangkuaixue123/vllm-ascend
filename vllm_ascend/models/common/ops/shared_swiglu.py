# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve native SiLU values while fusing a shared expert's activation."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_npu import get_npu_format

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.deepseek_v41_shared_swiglu import shared_swiglu_table

BF16_SIGN_OFFSET = 1 << 15
SHARED_INTERMEDIATE = 2304
ND_FORMAT = 2


class SharedClippedSwiGLU(nn.Module):
    """Use an instance-local native FP32 table for the finite BF16 domain."""

    def __init__(self):
        super().__init__()
        self.register_buffer("gate_silu", None, persistent=False)
        self._table_limit = None

    def forward(self, packed, limit):
        if (
            shared_swiglu_table is None
            or self.training
            or not torch.is_inference_mode_enabled()
            or packed.device.type != "npu"
            or packed.dtype != torch.bfloat16
            or packed.ndim != 2
            or packed.shape[0] == 0
            or packed.shape[1] != 2 * SHARED_INTERMEDIATE
            or not packed.is_contiguous()
            or packed.is_neg()
            or packed.is_conj()
            or not math.isfinite(limit)
            or limit <= 0
            or get_ascend_device_type() != AscendDeviceType.A3
            or get_npu_format(packed) != ND_FORMAT
        ):
            return None
        if (
            self.gate_silu is None
            or self.gate_silu.device != packed.device
            or self.gate_silu.dtype != torch.float32
            or self._table_limit != limit
        ):
            # Warmup initializes this once. Never add table construction to an
            # outer graph or keep using a table changed by Module.to(dtype).
            if torch.npu.is_current_stream_capturing():
                return None
            words = torch.arange(-BF16_SIGN_OFFSET, BF16_SIGN_OFFSET, device=packed.device, dtype=torch.int32)
            gate = words.to(torch.int16).view(torch.bfloat16).float()
            self.gate_silu = F.silu(gate.clamp(max=limit))
            self._table_limit = limit
        return shared_swiglu_table(packed, self.gate_silu, limit)
