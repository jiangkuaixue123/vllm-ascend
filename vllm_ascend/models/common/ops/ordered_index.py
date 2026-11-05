# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dispatch for deterministic ascending decode index selection."""

import os

import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_ROWS = (6, 12)
_WIDTHS = (76800, 153600)
_SELECTED_COLUMNS = 512
_ND_FORMAT = 2


def ordered_index_fused(scores, k):
    if (
        scores.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or scores.ndim != 2
        or scores.shape[0] not in _ROWS
        or scores.shape[1] not in _WIDTHS
        or scores.dtype not in (torch.bfloat16, torch.float32)
        or k != _SELECTED_COLUMNS
        or not scores.is_contiguous()
        or scores.is_neg()
        or scores.is_conj()
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    from torch_npu import get_npu_format
    from vllm import envs
    from vllm.logger import logger
    from vllm.triton_utils import HAS_TRITON

    if not HAS_TRITON or envs.VLLM_BATCH_INVARIANT or get_npu_format(scores) != _ND_FORMAT:
        return None
    from vllm_ascend.ops.triton.deepseek_v41_ordered_index import ordered_index_columns

    output = ordered_index_columns(scores, k)
    logger.info_once("DEVICE_INDEX_RUNTIME pid=%s", os.getpid(), scope="process")
    return output
