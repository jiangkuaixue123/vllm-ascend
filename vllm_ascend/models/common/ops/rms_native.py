# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native RMSNorm for bounded V4.1 decode and draft normalization shapes."""

import math

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_MAX_ROWS = 48
_HIDDEN_SIZES = (128, 512, 1024, 5120)
_ND_FORMAT = 2
_MIN_FP32_POSITIVE = 2.0**-149
_MAX_FP32_FINITE = (2.0 - 2.0**-23) * 2.0**127


def native_rms_norm(x, weight, eps):
    """Keep parameter values and return dtype; unsupported calls use the parent.

    The native reduction need not be bitwise identical to the explicit FP32
    chain. Supported shapes require separate numerical and model evaluation.
    """
    if (
        x.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or x.ndim != 2
        or not 0 < x.shape[0] <= _MAX_ROWS
        or x.shape[-1] not in _HIDDEN_SIZES
        or x.dtype != torch.bfloat16
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or weight.device != x.device
        or weight.shape != (x.shape[-1],)
        or any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (x, weight))
        or not math.isfinite(eps)
        or not _MIN_FP32_POSITIVE <= eps <= _MAX_FP32_FINITE
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    # Discover the NPU runtime only in a qualified worker call.
    from torch_npu import get_npu_format, npu_rms_norm

    if any(get_npu_format(t) != _ND_FORMAT for t in (x, weight)):
        return None
    return npu_rms_norm(x, weight, eps)[0]
