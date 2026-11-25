# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.models.hy_v4_flash.reference import FP8_MAX, INDEX_DIM
from vllm.triton_utils import tl, triton

SMALL_ROTARY_TOKENS = 8
INDEX_ROTARY_WIDTH = 64

_MIN_AMAX = 1e-4
# The reference clamps the FP32 maximum to 1e-4, whose exponent is -14.
_MIN_ROW_EXPONENT = -14
_MAX_ROW_EXPONENT = 127
_MANTISSA_WIDTH = 23
_EXPONENT_BIAS = 127
_FP8_MAX_EXPONENT = 8
_E4M3_MIN_NORMAL = tl.constexpr(2.0**-6)
_E4M3_FRACTION_BITS = tl.constexpr(3)
_FP32_FRACTION_BITS = tl.constexpr(23)
_FP32_EXPONENT_BIAS = tl.constexpr(127)


@triton.jit
def _index_frequencies(
    positions,
    inverse,
    phase,
    AXIS_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    MULTI_AXIS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TEMPORAL: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    PAIRS: tl.constexpr,
):
    token = tl.program_id(0)
    lane = tl.arange(0, PAIRS)
    if MULTI_AXIS:
        if INTERLEAVED:
            axis = tl.where((lane % 3 == 1) & (lane < 3 * HEIGHT), 1, 0)
            axis = tl.where((lane % 3 == 2) & (lane < 3 * WIDTH), 2, axis)
        else:
            axis = tl.where(lane < TEMPORAL, 0, tl.where(lane < TEMPORAL + HEIGHT, 1, 2))
        pos = tl.load(positions + axis * AXIS_STRIDE + token * TOKEN_STRIDE)
    else:
        pos = tl.load(positions + token * TOKEN_STRIDE)
    tl.store(phase + token * PAIRS + lane, pos.to(tl.float32) * tl.load(inverse + lane))


@triton.jit
def _round_index_values(values, magnitude, scale):
    # The FP32 exponent gives the normal E4M3 spacing. Clamping at the
    # smallest normal magnitude also gives the fixed subnormal spacing.
    magnitude_bits = tl.maximum(magnitude, _E4M3_MIN_NORMAL).to(tl.int32, bitcast=True)
    exponent = (magnitude_bits >> _FP32_FRACTION_BITS) - _FP32_EXPONENT_BIAS - _E4M3_FRACTION_BITS
    unit = ((exponent + _FP32_EXPONENT_BIAS) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    inverse_unit = ((_FP32_EXPONENT_BIAS - exponent) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
    rounded = tl.extra.cann.libdevice.nearbyint(magnitude * inverse_unit)
    sign = tl.where(values > 0, 1.0, tl.where(values < 0, -1.0, 0.0))
    return tl.where(scale == scale, rounded * unit * sign * scale, float("nan"))


def index_scale_transitions(device):
    """Find the last FP32 mantissa assigned the lower E4M3 scale by native CANN."""
    exponents = torch.arange(_MIN_ROW_EXPONENT, _MAX_ROW_EXPONENT + 1, dtype=torch.int32, device=device)
    base = (exponents + _EXPONENT_BIAS) << _MANTISSA_WIDTH
    lower_scale = torch.exp2((exponents - _FP8_MAX_EXPONENT).float())
    low = torch.zeros_like(base)
    high = torch.full_like(base, (1 << _MANTISSA_WIDTH) - 1)
    # The native scale is monotone within each FP32 exponent and takes just two power-of-two values.
    for _ in range(_MANTISSA_WIDTH):
        mid = (low + high + 1) // 2
        maximum = (base + mid).view(torch.float32)
        lower = torch.exp2(torch.ceil(torch.log2(maximum / FP8_MAX))) <= lower_scale
        low = torch.where(lower, mid, low)
        high = torch.where(lower, high, mid - 1)
    return low


@triton.jit
def _indexpool_hadamard(
    vectors,
    transformed,
    maxima,
    cosine,
    sine,
    transitions,
    FUSED_ROUND: tl.constexpr,
    HEADS: tl.constexpr,
    ROTARY: tl.constexpr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    MIN_EXP: tl.constexpr,
    MAX_EXP: tl.constexpr,
    MIN_AMAX: tl.constexpr,
    FP8_EXP: tl.constexpr,
    FP8_LIMIT: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    offset = row * WIDTH + lane
    values = tl.load(vectors + offset).to(tl.float32)
    if ROTARY:
        prefix = WIDTH - ROTARY
        pair = tl.maximum((lane - prefix) % (ROTARY // 2), 0)
        phase_offset = row // HEADS * (ROTARY // 2) + pair
        c = tl.load(cosine + phase_offset)
        s = tl.load(sine + phase_offset)
        first = tl.load(vectors + row * WIDTH + prefix + pair * 2).to(tl.float32)
        second = tl.load(vectors + row * WIDTH + prefix + pair * 2 + 1).to(tl.float32)
        rotated = tl.where(lane < prefix + ROTARY // 2, first * c - second * s, second * c + first * s)
        # The checkpoint rounds both rotary halves to BF16 before the Hadamard butterflies.
        values = tl.where(lane < prefix, values, rotated.to(tl.bfloat16).to(tl.float32))
    for level in tl.static_range(LEVELS):
        stride = 1 << level
        partner = tl.gather(values, lane ^ stride, 0)
        values = tl.where((lane & stride) == 0, values + partner, partner - values)
    values = values * NORMALIZE

    magnitude = tl.abs(values)
    nonfinite = tl.max(tl.where(magnitude < float("inf"), 0, 1), 0) != 0
    amax = tl.max(magnitude, 0)
    if FUSED_ROUND:
        bits = tl.maximum(amax, MIN_AMAX).to(tl.int32, bitcast=True)
        exponent = ((bits >> _FP32_FRACTION_BITS) & 255) - _FP32_EXPONENT_BIAS
        fraction = bits & ((1 << _FP32_FRACTION_BITS) - 1)
        index = tl.minimum(tl.maximum(exponent, MIN_EXP), MAX_EXP) - MIN_EXP
        boundary = tl.load(transitions + index)
        scale_exp = exponent - FP8_EXP + (fraction > boundary).to(tl.int32)
        scale = ((scale_exp + _FP32_EXPONENT_BIAS) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
        inverse_scale = ((_FP32_EXPONENT_BIAS - scale_exp) << _FP32_FRACTION_BITS).to(tl.float32, bitcast=True)
        normalized = tl.minimum(magnitude * inverse_scale, FP8_LIMIT)
        result = _round_index_values(values, normalized, scale)
        tl.store(transformed + offset, tl.where(nonfinite, float("nan"), result))
    else:
        tl.store(transformed + offset, values)
        tl.store(maxima + row, tl.where(nonfinite, float("nan"), amax))


@triton.jit
def _indexpool_round_e4m3(
    transformed,
    normalized,
    scales,
    WIDTH: tl.constexpr,
    QUANT_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    offset = row * WIDTH + tl.arange(0, WIDTH)
    values = tl.load(transformed + offset)
    scale = tl.load(scales + row)
    magnitude = tl.minimum(tl.abs(tl.load(normalized + offset)), QUANT_MAX)
    result = _round_index_values(values, magnitude, scale)
    tl.store(normalized + offset, result)


def encode_indexpool_fp8(
    vectors: torch.Tensor,
    *,
    cosine: torch.Tensor | None = None,
    sine: torch.Tensor | None = None,
    transitions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the FP32, E4M3-rounded IndexPool rows used by the scorer."""
    rows = vectors.numel() // INDEX_DIM
    transformed = torch.empty_like(vectors, dtype=torch.float32)
    maxima = vectors.new_empty((*vectors.shape[:-1], 1), dtype=torch.float32) if transitions is None else None
    _indexpool_hadamard[(rows,)](
        vectors,
        transformed,
        maxima,
        cosine,
        sine,
        transitions,
        transitions is not None,
        vectors.shape[1] if vectors.ndim == 3 else 1,
        INDEX_ROTARY_WIDTH if cosine is not None else 0,
        INDEX_DIM,
        INDEX_DIM.bit_length() - 1,
        INDEX_DIM**-0.5,
        _MIN_ROW_EXPONENT,
        _MAX_ROW_EXPONENT,
        _MIN_AMAX,
        _FP8_MAX_EXPONENT,
        FP8_MAX,
        num_warps=1,
        enable_fp_fusion=False,
    )
    if transitions is not None:
        return transformed
    # Keep CANN's scale and division rounding at exponent and E4M3 bin boundaries.
    scale = torch.exp2(torch.ceil(torch.log2(maxima.clamp_min(_MIN_AMAX) / FP8_MAX)))
    normalized = transformed / scale
    _indexpool_round_e4m3[(rows,)](
        transformed, normalized, scale, INDEX_DIM, FP8_MAX, num_warps=1, enable_fp_fusion=False
    )
    return normalized


def index_rotary_frequencies(positions, inverse, sections, interleaved):
    """Select each rotary lane's position axis and retain the native FP32 trigonometry."""
    pairs = INDEX_ROTARY_WIDTH // 2
    tokens = positions.shape[-1]
    phase = inverse.new_empty((tokens, pairs))
    _index_frequencies[(tokens,)](
        positions,
        inverse,
        phase,
        positions.stride(0) if positions.ndim == 2 else 0,
        positions.stride(-1),
        positions.ndim == 2,
        interleaved,
        *(sections if positions.ndim == 2 else (pairs, 0, 0)),
        pairs,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return phase.cos(), phase.sin()


def shared_query_frequencies(cache, positions, inverse, sections, interleaved, theta, stream):
    """Reuse identical runner positions on one stream during one target forward."""
    key = (
        positions.device,
        positions.data_ptr(),
        positions.shape,
        positions.stride(),
        positions.dtype,
        theta,
        tuple(sections or ()),
        interleaved,
        stream,
    )
    if key not in cache:
        # Retain the view for the forward lifetime so a freed allocation cannot reuse its address.
        values = index_rotary_frequencies(positions, inverse, sections, interleaved)
        cache[key] = (positions, values)
    return cache[key][1]
