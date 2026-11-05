# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extend exact BF16 E4M3 QDQ to draft and prefill row counts."""

import os

import torch

_MAX_ROWS = 8192
_GROUP_SIZE = 32
_ND_FORMAT = 2
_LAYOUTS = ((512,), (2, 4096))


def dynamic_fp8_fused(x, block_size, inplace):
    if (
        x.device.type != "npu"
        or x.dtype != torch.bfloat16
        or x.ndim not in (2, 3)
        or not 0 < x.shape[0] <= _MAX_ROWS
        or x.shape[1:] not in _LAYOUTS
        or block_size != _GROUP_SIZE
        or x.layout != torch.strided
        or not x.is_contiguous()
        or x.is_neg()
        or x.is_conj()
        or x.requires_grad
        or not torch.is_inference_mode_enabled()
        or torch.compiler.is_compiling()
        or x.storage_offset() % _GROUP_SIZE
        or x.data_ptr() % (_GROUP_SIZE * x.element_size())
    ):
        return None
    from torch_npu import get_npu_format
    from vllm import envs
    from vllm.logger import logger
    from vllm.triton_utils import HAS_TRITON

    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType

    if (
        not HAS_TRITON
        or envs.VLLM_BATCH_INVARIANT
        or get_ascend_device_type() != AscendDeviceType.A3
        or get_npu_format(x) != _ND_FORMAT
    ):
        return None
    from vllm_ascend.ops.triton.deepseek_v41_fp8_qdq import dynamic_fp8_qdq

    output = dynamic_fp8_qdq(x, inplace)
    if output is not None:
        logger.info_once("FP8_DYNAMIC_RUNTIME pid=%s", os.getpid(), scope="process")
    return output
