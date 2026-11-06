# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse small HC gate preparation while retaining the native sigmoid."""

import os

import torch
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_HC_WIDTH = 4
_MIX_WIDTH = 24
_ROUNDS = 20
_EPS = 1e-6
_MAX_ROWS = 48
_ND_FORMAT = 2


def try_hc_gate_fused(mixes, scale, base, hc_mult, sinkhorn_iters, eps):
    if (
        mixes.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or mixes.ndim != 2
        or not 0 < mixes.shape[0] <= _MAX_ROWS
        or mixes.shape[1] != _MIX_WIDTH
        or scale.shape != (3,)
        or base.shape != (_MIX_WIDTH,)
        or hc_mult != _HC_WIDTH
        or sinkhorn_iters != _ROUNDS
        or eps != _EPS
        or any(
            tensor.device != mixes.device
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or tensor.is_neg()
            or tensor.is_conj()
            for tensor in (mixes, scale, base)
        )
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    from torch_npu import get_npu_format

    if any(get_npu_format(tensor) != _ND_FORMAT for tensor in (mixes, scale, base)):
        return None
    from vllm_ascend.ops.triton.deepseek_v41_hc_gate_fused import hc_gate_fused

    if hc_gate_fused is None:
        return None
    result = hc_gate_fused(mixes, scale, base)
    logger.info_once("HC_GATE_FUSED_RUNTIME pid=%s", os.getpid(), scope="process")
    return result
