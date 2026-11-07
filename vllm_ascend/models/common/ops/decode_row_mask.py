# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small contiguous decode row masks without scalar-broadcast selection."""

import os

import torch
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_DECODE_ROWS = (6, 12)
_HIDDEN_SIZE = 5120
_ND_FORMAT = 2


def mask_decode_rows(values, active):
    """Match native BF16 selection, including NaN canonicalization and positive zero."""
    if (
        values.device.type == "npu"
        and values.dtype == torch.bfloat16
        and values.ndim == 2
        and values.shape[0] in _DECODE_ROWS
        and values.shape[1] == _HIDDEN_SIZE
        and active.shape == values.shape[:1]
        and active.dtype == torch.bool
        and active.device == values.device
        and values.is_contiguous()
        and active.is_contiguous()
        and not values.is_neg()
        and not values.is_conj()
        and torch.is_inference_mode_enabled()
        and not torch.compiler.is_compiling()
        and get_ascend_device_type() == AscendDeviceType.A3
    ):
        from torch_npu import get_npu_format

        if get_npu_format(values) == _ND_FORMAT:
            from vllm_ascend.ops.triton.decode_row_mask import decode_row_mask

            if decode_row_mask is not None:
                output = decode_row_mask(values, active)
                logger.info_once(
                    "TARGET_ROW_MASK_RUNTIME pid=%d rows=%d", os.getpid(), values.shape[0], scope="process"
                )
                return output
    return torch.where(active[:, None], values, 0.0)
