# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare the TP4 target output exchange in one vector pass."""

import os

import torch

_ROWS = (6, 12)
_HEADS = 64
_HEAD_DIM = 512
_ROPE_DIM = 64
_TP_SIZE = 4
_OUTPUT_GROUPS = 2
_FP8_GROUP = 32
_BF16_BYTES = 2
_DENSE_FORMATS = (0, 2)


def try_target_output_send(layer, output, positions):
    if (
        not layer.dsa_cp
        or layer.tp != _TP_SIZE
        or layer.sequence_group is None
        or layer.sequence_group.world_size != _TP_SIZE
        or (layer.heads, layer.local_heads, layer.dim, layer.groups, layer.rope_dim)
        != (_HEADS, _HEADS // _TP_SIZE, _HEAD_DIM, _OUTPUT_GROUPS, _ROPE_DIM)
        or output.device.type != "npu"
        or output.dtype != torch.bfloat16
        or output.ndim != 3
        or output.shape[0] not in _ROWS
        or output.shape[1:] != (_HEADS, _HEAD_DIM)
        or positions.shape != (output.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or layer.frequencies.shape != (_ROPE_DIM // 2,)
        or layer.frequencies.dtype != torch.float32
        or any(
            value.device != output.device
            or value.layout != torch.strided
            or not value.is_contiguous()
            or value.is_neg()
            or value.is_conj()
            or value.requires_grad
            for value in (output, positions, layer.frequencies)
        )
        or output.storage_offset() % _FP8_GROUP
        or output.data_ptr() % (_FP8_GROUP * _BF16_BYTES)
        or not torch.is_inference_mode_enabled()
        or torch.compiler.is_compiling()
    ):
        return None
    from torch_npu import get_npu_format
    from vllm import envs
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.logger import logger
    from vllm.triton_utils import HAS_TRITON

    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.utils import enable_custom_op

    if (
        not HAS_TRITON
        or envs.VLLM_BATCH_INVARIANT
        or not is_forward_context_available()
        or getattr(get_forward_context(), "is_draft_model", None) is not False
        or get_ascend_device_type() != AscendDeviceType.A3
        or not enable_custom_op()
        or not hasattr(torch.ops._C_ascend, "inplace_partial_rotary_mul")
        or any(get_npu_format(value) not in _DENSE_FORMATS for value in (output, positions, layer.frequencies))
    ):
        return None
    from vllm_ascend.ops.triton.deepseek_v41_output_send import prepare_output_send

    # Keep the parent's phase construction and native sine/cosine arithmetic.
    phases = positions.float().unsqueeze(-1) * layer.frequencies
    phases = torch.stack((phases, phases), dim=-1).view(output.shape[0], 1, 1, _ROPE_DIM)
    sine = phases.sin()
    send = prepare_output_send(output, phases.cos(), -sine)
    if send is not None:
        logger.info_once("TARGET_OUTPUT_ROPE_FP8_RUNTIME pid=%s", os.getpid(), scope="process")
    return send
