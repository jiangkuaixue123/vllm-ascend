# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One A3 vector pass over complete BF16 groups for V4.1 window/output E4M3 QDQ."""

import torch
import torch_npu
from vllm import envs
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_QUALIFIED_VECTOR_CORES = 48
_GROUP_SIZE = 32


@triton.jit
def _bias_qdq(src):
    bits = tl.cast(tl.cast(src, tl.int16, bitcast=True), tl.int32)
    max_word = tl.max(bits & 0x7FFF, axis=1)
    # For positive finite BF16 words, ceil(log2(amax/448)) crosses when
    # mantissa > 96. Adding 31 encodes that comparison plus exponent carry.
    exp = tl.maximum(-22, ((max_word + 31) >> 7) - 135)
    inf_group = max_word == 0x7F80
    nan_group = max_word > 0x7F80
    # IEEE FP32 bit patterns for signed zero, infinity, exponent and sign masks.
    scale_word = tl.where(nan_group, -0x8000_0000, tl.where(inf_group, 0x7F80_0000, (127 + exp) << 23))
    reciprocal_word = tl.where(nan_group, -0x0080_0000, tl.where(inf_group, 0, (127 - exp) << 23))
    scale = tl.cast(scale_word, tl.float32, bitcast=True)
    inv = tl.cast(reciprocal_word, tl.float32, bitcast=True)
    scaled = tl.cast(src, tl.float32) * inv[:, None]
    mag = tl.abs(scaled)
    # Keep the NaN class through clamp. For finite groups scale ensures |x/scale| <= 448.
    mag = tl.where(mag > 448.0, 448.0, mag)
    field = tl.cast(mag, tl.int32, bitcast=True) & 0x7F80_0000
    # A normal exponent field of 121 is E4M3's minimum. Twenty more FP32
    # exponent steps give the addition exactly three retained fraction bits.
    bias = tl.cast(tl.maximum(field, 0x3C80_0000) + 0x0A00_0000, tl.float32, bitcast=True)
    rounded = (bias + mag) - bias
    sign = tl.cast(scaled, tl.int32, bitcast=True) & -0x8000_0000
    signed = tl.cast((tl.cast(rounded, tl.int32, bitcast=True) & 0x7FFF_FFFF) | sign, tl.float32, bitcast=True)
    return tl.cast(signed * scale[:, None], tl.bfloat16)


@triton.jit
def _kernel(x, out, NTILES: tl.constexpr, G: tl.constexpr):
    for tile in range(tl.program_id(0), NTILES, tl.num_programs(0)):
        groups = tile * G + tl.arange(0, G)
        ix = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
        src = tl.load(x + ix)
        tl.store(out + ix, _bias_qdq(src))


def decode_fp8_qdq(x: torch.Tensor, inplace: bool):
    init_device_properties_triton()
    cores = get_vectorcore_num()
    if cores != _QUALIFIED_VECTOR_CORES:
        return None
    total_groups = x.numel() // _GROUP_SIZE
    groups = min(64, max(8, triton.next_power_of_2(triton.cdiv(total_groups, cores))))
    ntiles = total_groups // groups
    out = x if inplace else torch.empty_like(x)
    _kernel[(min(cores, ntiles),)](x, out, NTILES=ntiles, G=groups)
    if inplace:
        # A raw Triton store bypasses the version change made by Tensor.copy_.
        torch.autograd.graph.increment_version(x)
    return out


@triton.jit
def _pack_prefill(x, out, rows, total_groups, ROW_GROUPS: tl.constexpr, TP: tl.constexpr, G: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(total_groups, G), tl.num_programs(0)):
        group = tile * G + tl.arange(0, G)
        rank = group // (rows * ROW_GROUPS)
        within = group % (rows * ROW_GROUPS)
        source = (within // ROW_GROUPS) * (TP * ROW_GROUPS) + rank * ROW_GROUPS + within % ROW_GROUPS
        cols = tl.arange(0, 32)[None, :]
        values = tl.load(x + source[:, None] * 32 + cols, group[:, None] < total_groups, other=0)
        tl.store(out + group[:, None] * 32 + cols, _bias_qdq(values), group[:, None] < total_groups)


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
    return cores == _QUALIFIED_VECTOR_CORES


def pack_prefill_projection_input(x: torch.Tensor, tp: int):
    """Quantize complete per-head groups while writing the BF16 all-to-all send layout."""
    if not supports_prefill_projection_input(x, tp):
        return None
    rows, heads, width = x.shape
    out = torch.empty((tp * rows, heads // tp, width), dtype=x.dtype, device=x.device)
    _pack_prefill[(_QUALIFIED_VECTOR_CORES,)](
        x, out, rows, x.numel() // _GROUP_SIZE, ROW_GROUPS=heads // tp * width // _GROUP_SIZE, TP=tp, G=64
    )
    return out


@triton.jit(do_not_specialize=["rows", "total_rows", "start"])
def _pack_rotated_prefill(
    x, cosine, sine, out, rows, total_rows, start, ROW_GROUPS: tl.constexpr, TP: tl.constexpr, G: tl.constexpr
):
    rank = tl.program_id(1)
    cols = tl.arange(0, 32)
    for tile in range(tl.program_id(0), tl.cdiv(rows * ROW_GROUPS, G), tl.num_programs(0)):
        row = tile // (ROW_GROUPS // G)
        within = tile % (ROW_GROUPS // G) * G
        source = (row * (TP * ROW_GROUPS) + rank * ROW_GROUPS + within) * 32
        # Four complete heads share one phase load, without per-group GM gathers.
        values = tl.load(x + source + tl.arange(0, G * 32)).reshape(G, 32)
        phase = row * 64 + tl.arange(0, 64)
        co = tl.load(cosine + phase).reshape(1, 1, 2, 32)
        si = tl.load(sine + phase).reshape(1, 1, 2, 32)
        co = tl.broadcast_to(co, (G // 16, 8, 2, 32)).reshape(G, 32)
        si = tl.broadcast_to(si, (G // 16, 8, 2, 32)).reshape(G, 32)
        rotary = tl.arange(0, G) % 16 >= 14
        partner = tl.gather(values.to(tl.float32), tl.broadcast_to((cols ^ 1)[None, :], (G, 32)), axis=1)
        cross = partner * si
        cross = tl.where(cols[None, :] % 2 == 0, -cross, cross)
        rotated = (values.to(tl.float32) * co + cross).to(tl.bfloat16)
        values = tl.where(rotary[:, None], rotated, values)
        # Whole tiles divide each row; expose the contiguous write to avoid a scalar scatter lock.
        destination = ((rank * total_rows + start) * ROW_GROUPS + tile * G) * 32
        tl.store(out + destination + tl.arange(0, G * 32), _bias_qdq(values).reshape(G * 32))


def pack_rotated_prefill_output(x, positions, frequencies, out, start, tp):
    """Write one request span directly into its final prequantized TP send rows."""
    rows = x.shape[0]
    total_rows = out.shape[0] // tp
    if x.shape != (rows, 64, 512) or positions.shape != (rows,) or not 0 <= start <= total_rows - rows:
        raise ValueError("Invalid prefill projection span")
    # Keep the stock rotary phase shape and sine/cosine implementation.
    phases = positions.float().unsqueeze(-1) * frequencies
    phases = torch.stack((phases, phases), dim=-1).view(rows, 1, 1, 64)
    cosine, sine = phases.cos(), -phases.sin()
    _pack_rotated_prefill[(_QUALIFIED_VECTOR_CORES // tp, tp)](
        x.contiguous(),
        cosine,
        sine,
        out,
        rows,
        total_rows,
        start,
        ROW_GROUPS=64 // tp * 16,
        TP=tp,
        G=64,
        enable_fp_fusion=False,
    )
