# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Large A3 prefill HC paths; preserve the library FP32 gate projection."""

import torch
from vllm import envs

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType


def _supported(x, *tensors):
    if (
        x.device.type != "npu"
        or x.ndim != 3
        or x.shape[0] <= 256
        or x.shape[1:] != (4, 5120)
        or x.dtype != torch.bfloat16
        or not torch.is_inference_mode_enabled()
        or envs.VLLM_BATCH_INVARIANT
        or torch.compiler.is_compiling()
        or any(t.device != x.device or not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (x, *tensors))
        or get_ascend_device_type() != AscendDeviceType.A3
    ):
        return False
    from torch_npu import get_npu_format
    from vllm.triton_utils import HAS_TRITON

    return HAS_TRITON and all(get_npu_format(t) == 2 for t in (x, *tensors))


def hc_prefill_prepare(x, previous_pre):
    if previous_pre.shape != x.shape[:-1] or previous_pre.dtype != torch.float32 or not _supported(x, previous_pre):
        return None
    from vllm_ascend.ops.triton.hc_prefill import prepare

    return prepare(x, previous_pre)


def hc_prefill_post(sub, residual, post, comb):
    rows = residual.shape[0]
    if (
        sub.shape != (rows, 5120)
        or sub.dtype != torch.bfloat16
        or post.shape != (rows, 4)
        or comb.shape != (rows, 4, 4)
        or post.dtype != torch.float32
        or comb.dtype != torch.float32
        or not _supported(residual, sub, post, comb)
    ):
        return None
    from vllm_ascend.ops.triton.hc_prefill import post as post_kernel

    return post_kernel(sub, residual, post, comb)


def hc_prefill_post_prepare(sub, residual, post, comb, previous_pre):
    rows = residual.shape[0]
    if (
        sub.shape != (rows, 5120)
        or sub.dtype != torch.bfloat16
        or post.shape != (rows, 4)
        or comb.shape != (rows, 4, 4)
        or previous_pre.shape != (rows, 4)
        or any(t.dtype != torch.float32 for t in (post, comb, previous_pre))
        or not _supported(residual, sub, post, comb, previous_pre)
    ):
        return None
    from vllm_ascend.ops.triton.hc_prefill import post_prepare

    return post_prepare(sub, residual, post, comb, previous_pre)
