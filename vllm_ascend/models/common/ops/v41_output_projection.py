# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the TP receive and following WO_B layouts through WO_A."""

import torch
from vllm import envs

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType


def try_v41_wo_a(x, weight):
    if (
        x.device.type != "npu"
        or x.ndim != 3
        or x.shape[0] not in (12, 24, 48, 96)
        or x.shape[1] not in (1, 2)
        or x.shape[2] != 4096
        or weight.shape != (x.shape[1], 1024, 4096)
        or not torch.is_inference_mode_enabled()
        or envs.VLLM_BATCH_INVARIANT
        or torch.compiler.is_compiling()
        or any(
            t.device != x.device
            or t.dtype != torch.bfloat16
            or not t.is_contiguous()
            or t.is_neg()
            or t.is_conj()
            or t.data_ptr() % 32
            for t in (x, weight)
        )
        or get_ascend_device_type() != AscendDeviceType.A3
        or not hasattr(torch.ops._C_ascend, "deepseek_v41_wo_a")
    ):
        return None
    from torch_npu import get_npu_format

    if any(get_npu_format(t) not in (0, 2) for t in (x, weight)):
        return None
    out = x.new_empty((x.shape[0], x.shape[1], 1024))
    torch.ops._C_ascend.deepseek_v41_wo_a(x, weight, out)
    return out
