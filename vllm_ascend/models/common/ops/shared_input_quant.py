# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse native input quantization for two unchanged shared-expert projections."""

import os

import torch
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_ascend.ops.linear import AscendReplicatedLinear
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod

_TARGET_ROWS = (6, 12)
_HIDDEN_SIZE = 5120


def shared_input_projections(gate, up, states):
    """Return the original two projections, or None for the normal layer path."""
    if (
        states.device.type != "npu"
        or states.dtype != torch.bfloat16
        or states.ndim != 2
        or states.shape[0] not in _TARGET_ROWS
        or states.shape[1] != _HIDDEN_SIZE
        or not states.is_contiguous()
        or not torch.is_inference_mode_enabled()
    ):
        return None
    for layer in (gate, up):
        if (
            type(layer) not in (ReplicatedLinear, AscendReplicatedLinear)
            or getattr(layer, "custom_op", None) is not None
            or type(layer.quant_method) is not AscendLinearMethod
            or type(layer.quant_method.quant_method) is not AscendW8A8DynamicLinearMethod
            or layer.quant_method.quant_method.act_quant_type != torch.int8
            or layer.bias is not None
            or layer.return_bias
            or layer._forward_hooks
            or layer._forward_pre_hooks
        ):
            return None
    # Both original calls use exactly these activations and per-token scales.
    # Preserve two separate GEMMs, including their weight storage and N geometry.
    quantized, scales = torch_npu.npu_dynamic_quant(states, dst_type=torch.int8)
    outputs = tuple(
        torch_npu.npu_quant_matmul(
            quantized,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=scales,
            bias=None,
            output_dtype=states.dtype,
        )
        for layer in (gate, up)
    )
    logger.info_once("SHARED_INPUT_QUANT_REUSE_RUNTIME pid=%s", os.getpid(), scope="process")
    return outputs
