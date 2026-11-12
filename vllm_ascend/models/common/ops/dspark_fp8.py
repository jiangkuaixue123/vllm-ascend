# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse small DSpark E4M3 QDQ while retaining target and large-prefill paths."""

import os

import torch

_MAX_ROWS = 40
_GROUP_SIZE = 32
# Native attention can return contiguous NCHW (0); allocations use ND (2).
_DENSE_FORMATS = (0, 2)
_LAYOUTS = ((512,), (2, 4096))


def draft_fp8_fused(x, block_size, inplace):
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
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.logger import logger
    from vllm.triton_utils import HAS_TRITON

    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType

    if (
        not is_forward_context_available()
        or getattr(get_forward_context(), "is_draft_model", False) is not True
        or get_tensor_model_parallel_world_size() != 4
        or not HAS_TRITON
        or envs.VLLM_BATCH_INVARIANT
        or get_ascend_device_type() != AscendDeviceType.A3
        or get_npu_format(x) not in _DENSE_FORMATS
    ):
        return None
    from vllm_ascend.ops.triton.deepseek_v41_fp8_qdq import draft_fp8_qdq

    output = draft_fp8_qdq(x, inplace)
    if output is not None:
        logger.info_once("DSPARK_FP8_DECODE_RUNTIME pid=%s", os.getpid(), scope="process")
    return output
