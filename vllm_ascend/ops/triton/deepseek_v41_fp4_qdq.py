# SPDX-License-Identifier: Apache-2.0
"""A3 native BF16 FP4 producer shared by index Q/K and compressed KV."""

import torch
import torch_npu
from vllm import envs

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type


def fp4_qdq(x, block_size, scale_format, inplace):
    if (
        (block_size, scale_format) not in ((32, "e8m0"), (16, "e4m3"))
        or x.device.type != "npu"
        or x.dtype != torch.bfloat16
        or x.layout != torch.strided
        or x.ndim not in (2, 3)
        or x.shape[-1] not in (128, 512)
        or not 0 < x.numel() <= 2**32 - 1
        or not x.is_contiguous()
        or x.is_neg()
        or x.requires_grad
        or not torch.is_inference_mode_enabled()
        or torch.compiler.is_compiling()
        or envs.VLLM_BATCH_INVARIANT
        or x.storage_offset() % 32
        or get_ascend_device_type() != AscendDeviceType.A3
        or int(torch_npu.get_npu_format(x)) != int(torch_npu.Format.ND)
    ):
        return None
    init_device_properties_triton()
    if get_vectorcore_num() != 48 or not enable_custom_op():
        return None
    out = x if inplace else torch.empty_like(x)
    torch.ops._C_ascend.deepseek_v41_fp4_qdq(x, out, scale_format == "e4m3")
    if inplace:
        torch.autograd.graph.increment_version(x)
    return out
