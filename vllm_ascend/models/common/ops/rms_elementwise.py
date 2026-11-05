# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small V4.1 RMSNorm with native reduction and reciprocal square root."""

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_HIDDEN_SIZES = (128, 512, 1280, 5120)
_MAX_ROWS = 48
_TEXT_EPS = 1e-20
_ND_FORMAT = 2


def rms_elementwise_fused(hidden_states, weight, eps):
    """Return the fused result, or None when the reference path is required."""
    if (
        hidden_states.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or hidden_states.ndim != 2
        or not 0 < hidden_states.shape[0] <= _MAX_ROWS
        or hidden_states.shape[1] not in _HIDDEN_SIZES
        or hidden_states.dtype != torch.bfloat16
        or weight.shape != (hidden_states.shape[1],)
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or eps != _TEXT_EPS
        or any(
            tensor.device != hidden_states.device or not tensor.is_contiguous() or tensor.is_neg() or tensor.is_conj()
            for tensor in (hidden_states, weight)
        )
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    from torch_npu import get_npu_format

    if any(get_npu_format(tensor) != _ND_FORMAT for tensor in (hidden_states, weight)):
        return None
    from vllm_ascend.ops.triton.deepseek_v41_rms_elementwise import rms_elementwise

    if rms_elementwise is None:
        return None
    return rms_elementwise(hidden_states, weight, eps)
