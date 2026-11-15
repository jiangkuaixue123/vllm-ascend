# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 native Vector QDQ shared by KV and final prefill TP send consumers."""

import torch
import torch_npu
from vllm import envs

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type

_QUALIFIED_VECTOR_CORES = 48
_GROUP_SIZE = 32


def decode_fp8_qdq(x: torch.Tensor, inplace: bool):
    init_device_properties_triton()
    if get_vectorcore_num() != _QUALIFIED_VECTOR_CORES or not enable_custom_op():
        return None
    out = x if inplace else torch.empty_like(x)
    torch.ops._C_ascend.deepseek_v41_fp8_qdq(x, out)
    if inplace:
        torch.autograd.graph.increment_version(x)
    return out


def supports_prefill_projection_input(x: torch.Tensor, tp: int):
    if (
        x.device.type != "npu"
        or x.dtype != torch.bfloat16
        or x.layout != torch.strided
        or x.ndim != 3
        or x.shape[0] <= 256
        or x.shape[1:] != (64, 512)
        or tp not in (4, 8)
        or not x.is_contiguous()
        or x.is_neg()
        or x.requires_grad
        or not torch.is_inference_mode_enabled()
        or torch.compiler.is_compiling()
        or envs.VLLM_BATCH_INVARIANT
        or x.storage_offset() % _GROUP_SIZE
        or get_ascend_device_type() != AscendDeviceType.A3
        or int(torch_npu.get_npu_format(x)) != int(torch_npu.Format.ND)
    ):
        return False
    init_device_properties_triton()
    cores = get_vectorcore_num()
    return cores == _QUALIFIED_VECTOR_CORES and enable_custom_op()


def pack_prefill_projection_input(x: torch.Tensor, tp: int):
    """Quantize complete per-head groups while writing the BF16 all-to-all send layout."""
    if not supports_prefill_projection_input(x, tp):
        return None
    rows, heads, width = x.shape
    out = torch.empty((tp * rows, heads // tp, width), dtype=x.dtype, device=x.device)
    torch.ops._C_ascend.deepseek_v41_fp8_pack(x, None, out, 0, tp)
    return out


def pack_rotated_prefill_output(x, positions, frequencies, out, start, tp):
    """Write one request span directly into its final prequantized TP send rows."""
    rows = x.shape[0]
    total_rows = out.shape[0] // tp
    if x.shape != (rows, 64, 512) or positions.shape != (rows,) or not 0 <= start <= total_rows - rows:
        raise ValueError("Invalid prefill projection span")
    if not enable_custom_op():
        raise RuntimeError("Native prefill packing requires the Ascend custom-op extension")
    # Keep the stock rotary phase shape and sine/cosine implementation.
    phases = positions.float().unsqueeze(-1) * frequencies
    phases = torch.stack((phases, phases), dim=-1).view(rows, 1, 1, 64)
    cosine, sine = phases.cos(), -phases.sin()
    tail = x[..., 448:].float()
    partner = torch.stack((-tail[..., 1::2], tail[..., ::2]), dim=-1).flatten(-2)
    rotated = (tail * cosine.view(rows, 1, 64) + partner * sine.view(rows, 1, 64)).bfloat16()
    torch.ops._C_ascend.deepseek_v41_fp8_pack(x.contiguous(), rotated, out, start, tp)
