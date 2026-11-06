# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare complete DSpark ring indices without eager arithmetic launches."""

import os

import torch
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_WINDOW = 128
_QUERY_ROWS = 5
_MAX_REQUESTS = 8
_ND_FORMAT = 2


def try_build_ring_indices(starts, table, *, window_size, block_size):
    """Return slots and logical indices, or None for the original tensor path."""
    batch = len(starts)
    if (
        table.device.type != "npu"
        or not torch.is_inference_mode_enabled()
        or window_size != _WINDOW
        or block_size != _QUERY_ROWS
        or not 1 < batch <= _MAX_REQUESTS
        or table.ndim != 2
        or table.shape[0] < batch
        or table.dtype not in (torch.int32, torch.int64)
        or table.stride(1) != 1
        or table.is_neg()
        or table.is_conj()
        or any(not isinstance(start, int) or not _WINDOW <= start <= table.shape[1] * _WINDOW for start in starts)
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return None
    from torch_npu import get_npu_format

    if get_npu_format(table) != _ND_FORMAT:
        return None
    from vllm.utils.torch_utils import async_tensor_h2d

    from vllm_ascend.ops.triton.deepseek_v41_dspark_ring_indices import build_ring_indices

    if build_ring_indices is None:
        return None
    device_starts = async_tensor_h2d(starts, device=table.device, dtype=torch.int64)
    result = build_ring_indices(device_starts, table, _WINDOW, _QUERY_ROWS)
    logger.info_once("DSPARK_RING_INDICES_RUNTIME pid=%s", os.getpid(), scope="process")
    return result
