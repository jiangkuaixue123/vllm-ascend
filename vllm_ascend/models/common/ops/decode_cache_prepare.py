# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare raw-cache writes while retaining the original native scatter."""

import os

import torch
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_DECODE_ROWS = (24, 48)
_RAW_WIDTH = 512
_ND_FORMAT = 2


def try_prepare_decode_raw_write(page, values, slot_mapping, active, enabled):
    """Return native-equivalent slots/updates, or None for unsupported layouts."""
    if not (
        values.device.type == "npu"
        and values.dtype == page.dtype == torch.bfloat16
        and values.ndim == 2
        and values.shape[0] in _DECODE_ROWS
        and values.shape[1] == _RAW_WIDTH
        and values.is_contiguous()
        and page.ndim == 2
        and page.shape[0] > 0
        and page.shape[1] == _RAW_WIDTH
        and page.stride(1) == 1
        and page.stride(0) >= _RAW_WIDTH
        and slot_mapping.dtype == torch.int64
        and active.dtype == enabled.dtype == torch.bool
        and all(t.shape == values.shape[:1] and t.is_contiguous() for t in (slot_mapping, active, enabled))
        and all(
            t.device == values.device and not t.requires_grad and not t.is_neg() and not t.is_conj()
            for t in (page, values, slot_mapping, active, enabled)
        )
        and torch.is_inference_mode_enabled()
        and not torch.compiler.is_compiling()
        and get_ascend_device_type() == AscendDeviceType.A3
    ):
        return None
    from torch_npu import get_npu_format

    if any(get_npu_format(t) != _ND_FORMAT for t in (page, values, slot_mapping, active, enabled)):
        return None
    from vllm_ascend.ops.triton.decode_cache_prepare import prepare_decode_raw_write

    if prepare_decode_raw_write is None:
        return None
    result = prepare_decode_raw_write(page, values, slot_mapping, active, enabled)
    logger.info_once("TARGET_CACHE_PREPARE_RUNTIME pid=%d rows=%d", os.getpid(), values.shape[0], scope="process")
    return result
