# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch reference primitives for DeepSeek-V4.1's quantized latent attention.

Semantics follow the frozen official HF inference/kernel.py and model.py. Values
are represented with ordinary FP32/BF16 tensors, so float8 device kernels are not
required. This is a correctness path, not a packed-cache implementation.
"""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_AMAX_MIN = 1e-4
FP4_E4M3_AMAX_MIN = 6 * 2**-9
FP4_E8M0_AMAX_MIN = 6 * 2**-126
FLOAT32_MANTISSA_BITS = 23
FLOAT32_EXPONENT_BIAS = 127
FLOAT32_EXPONENT_UNIT = 1 << FLOAT32_MANTISSA_BITS
FLOAT32_EXPONENT_MASK = 0xFF * FLOAT32_EXPONENT_UNIT
FLOAT32_MANTISSA_MASK = (1 << FLOAT32_MANTISSA_BITS) - 1
FLOAT32_SIGN_MASK = -(1 << 31)
FLOAT32_MAGNITUDE_MASK = (1 << 31) - 1
ATTENTION_BLOCK_SIZE = 64
ATTENTION_QUERY_CHUNK_SIZE = 16
ATTENTION_INITIAL_MAX = -1e30


def _power_of_two(exponent: torch.Tensor) -> torch.Tensor:
    # The encoded field is in [0, 255], so this INT32 multiplication is
    # exact even at +infinity. Tensor left-shift falls back to CPU on Ascend.
    bits = (exponent.to(torch.int32) + FLOAT32_EXPONENT_BIAS) * FLOAT32_EXPONENT_UNIT
    return bits.contiguous().view(torch.float32)


def _unbiased_exponent(bits: torch.Tensor) -> torch.Tensor:
    # Ascend lacks the tensor right-shift kernel too. Mask first so the
    # dividend is exactly field * 2**23 even if integer division uses FP32.
    encoded = bits & FLOAT32_EXPONENT_MASK
    return torch.div(encoded, FLOAT32_EXPONENT_UNIT, rounding_mode="trunc") - FLOAT32_EXPONENT_BIAS


def _round_scale_up(amax: torch.Tensor, reciprocal_max: float) -> torch.Tensor:
    """Match fast_round_scale, including FP32 reciprocal multiplication."""
    scaled = (amax.float() * reciprocal_max).contiguous()
    bits = scaled.view(torch.int32)
    exponent = _unbiased_exponent(bits)
    exponent += ((bits & FLOAT32_MANTISSA_MASK) != 0).to(torch.int32)
    return _power_of_two(exponent)


def _round_binary_float(x: torch.Tensor, mantissa_bits: int, minimum_exponent: int) -> torch.Tensor:
    magnitude = x.float().abs()
    bits = magnitude.contiguous().view(torch.int32)
    exponent = _unbiased_exponent(bits)
    quantum = _power_of_two(torch.clamp_min(exponent, minimum_exponent) - mantissa_bits)
    rounded = torch.round(magnitude / quantum) * quantum
    sign = x.float().contiguous().view(torch.int32) & FLOAT32_SIGN_MASK
    bits = (rounded.contiguous().view(torch.int32) & FLOAT32_MAGNITUDE_MASK) | sign
    return bits.view(torch.float32)


def round_e4m3(x: torch.Tensor, *, saturate: bool = False) -> torch.Tensor:
    """Round to E4M3FN with ties to even, returning its FP32 numeric value.

    Values above the finite conversion range become NaN unless ``saturate`` is
    set. Activation conversion clamps first; compressed-KV scale conversion does
    not. No float8 cast or float8 arithmetic runs on the device.
    """
    source = x.float().clamp(-FP8_MAX, FP8_MAX) if saturate else x.float()
    rounded = _round_binary_float(source, mantissa_bits=3, minimum_exponent=-6)
    overflow = rounded.abs() > FP8_MAX
    return torch.where(overflow, torch.full_like(rounded, torch.nan), rounded)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Saturating E2M1FN conversion with round-to-nearest, ties-to-even."""
    source = x.float().clamp(-FP4_MAX, FP4_MAX)
    return _round_binary_float(source, mantissa_bits=1, minimum_exponent=0)


def _group_input(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.ndim == 0 or block_size <= 0 or x.shape[-1] % block_size:
        raise ValueError("Last dimension must be divisible by a positive block size")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("Quantization input must use BF16, FP16 or FP32")
    return x.float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)


def fp8_quant_dequant(x: torch.Tensor, block_size: int = 32, *, inplace: bool = False) -> torch.Tensor:
    """Official E4M3 window/output QDQ with upward-rounded E8M0 scales.

    The last dimension is grouped independently for every preceding position.
    FP32 amax is clamped to 1e-4. Return dtype matches the input, including the
    final BF16 rounding used by the official in-place activation kernel.
    """
    grouped = _group_input(x, block_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(FP8_AMAX_MIN)
    scale = _round_scale_up(amax, 1.0 / FP8_MAX)
    result = round_e4m3(grouped / scale, saturate=True) * scale
    result = result.reshape(x.shape).to(x.dtype)
    if inplace:
        x.copy_(result)
        return x
    return result


def fp4_quant_dequant(
    x: torch.Tensor,
    block_size: int = 32,
    *,
    scale_format: str = "e8m0",
    inplace: bool = False,
) -> torch.Tensor:
    """Official FP4 QDQ: index Q/K use 32/E8M0, compressed KV uses 16/E4M3.

    E8M0 scales round upward to a power of two after an amax floor of
    6*2**-126. E4M3 scales use nearest-even conversion of max(amax, 6*2**-9)/6.
    Neither rule may be replaced with BF16-only cache storage.
    """
    grouped = _group_input(x, block_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    if scale_format == "e8m0":
        scale = _round_scale_up(amax.clamp_min(FP4_E8M0_AMAX_MIN), 1.0 / FP4_MAX)
    elif scale_format == "e4m3":
        scale = round_e4m3(amax.clamp_min(FP4_E4M3_AMAX_MIN) / FP4_MAX)
    else:
        raise ValueError("scale_format must be e8m0 or e4m3")
    result = round_e2m1(grouped / scale) * scale
    result = result.reshape(x.shape).to(x.dtype)
    if inplace:
        x.copy_(result)
        return x
    return result


def apply_rotary_emb_real(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
    *,
    inplace: bool = True,
) -> torch.Tensor:
    """Adjacent-pair RoPE for [B,S,D] or [B,S,H,D], without complex tensors.

    ``cos`` and ``sin`` have shape [S,D/2]. Set ``inverse`` to remove the
    query rotation from the attention output, as in the official model.
    """
    if x.ndim not in (3, 4) or x.shape[-1] % 2:
        raise ValueError("RoPE expects rank 3 or 4 and an even last dimension")
    expected = (x.shape[1], x.shape[-1] // 2)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError("RoPE cosine and sine must have shape [S,D/2]")
    shape = (1, x.shape[1], x.shape[-1] // 2)
    if x.ndim == 4:
        shape = (1, x.shape[1], 1, x.shape[-1] // 2)
    cosine = cos.float().reshape(shape)
    sine = sin.float().reshape(shape)
    if inverse:
        sine = -sine
    even, odd = x[..., 0::2].float(), x[..., 1::2].float()
    result = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2).to(x.dtype)
    if inplace:
        x.copy_(result)
        return x
    return result


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    *,
    query_chunk_size: int = ATTENTION_QUERY_CHUNK_SIZE,
) -> torch.Tensor:
    """Sparse latent attention preserving the official 64-slot kernel semantics.

    Args:
        q: BF16 queries [B,S,H,D].
        kv: BF16 shared keys/values [B,L,D].
        attn_sink: Per-head FP32 logits [H], contributing only to the denominator.
        topk_idxs: Integer slots [B,S,K]; -1 masks a slot. Other entries must be
            valid KV indices. Repeated indices contribute repeatedly, without
            deduplication. Causality is encoded entirely by these indices.
        softmax_scale: Query-key score multiplier.
        query_chunk_size: Limits gathered temporary storage during prefill.

    Returns:
        BF16 output [B,S,H,D]. All-masked rows are zero. FP32 online sums and
        accumulation retain the kernel's BF16 rounding of exponentials before
        each value multiplication; a plain FP32 softmax/matmul is not equivalent.
    """
    if q.ndim != 4 or kv.ndim != 3 or topk_idxs.ndim != 3:
        raise ValueError("Expected q[B,S,H,D], kv[B,L,D], indices[B,S,K]")
    batch, sequence, heads, dimension = q.shape
    if (
        kv.shape[0] != batch
        or kv.shape[-1] != dimension
        or topk_idxs.shape[:2] != (batch, sequence)
        or attn_sink.shape != (heads,)
    ):
        raise ValueError("Incompatible sparse-attention tensor shapes")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("Sparse attention follows the official BF16 Q/KV contract")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise TypeError("Sparse-attention indices must be integer tensors")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    output = torch.zeros_like(q)
    if topk_idxs.shape[-1] == 0 or kv.shape[1] == 0:
        return output
    for start in range(0, sequence, query_chunk_size):
        end = min(start + query_chunk_size, sequence)
        queries = q[:, start:end].float()
        running_max = torch.full(
            (batch, end - start, heads),
            ATTENTION_INITIAL_MAX,
            dtype=torch.float32,
            device=q.device,
        )
        denominator = torch.zeros_like(running_max)
        accumulator = torch.zeros_like(queries)
        all_indices = topk_idxs[:, start:end]
        for offset in range(0, all_indices.shape[-1], ATTENTION_BLOCK_SIZE):
            indices = all_indices[..., offset : offset + ATTENTION_BLOCK_SIZE]
            valid = indices != -1
            safe_indices = indices.masked_fill(~valid, 0).to(torch.int64)
            gather_indices = safe_indices.reshape(batch, -1).unsqueeze(-1)
            selected = torch.gather(kv, 1, gather_indices.expand(-1, -1, dimension)).reshape(
                batch, end - start, indices.shape[-1], dimension
            )
            selected = selected.masked_fill(~valid.unsqueeze(-1), 0).float()
            scores = torch.matmul(queries, selected.transpose(-1, -2))
            scores = (scores * softmax_scale).masked_fill(~valid.unsqueeze(-2), -torch.inf)
            new_max = torch.maximum(running_max, scores.amax(dim=-1))
            rescale = torch.exp(running_max - new_max)
            exponentials = torch.exp(scores - new_max.unsqueeze(-1))
            denominator = denominator * rescale + exponentials.sum(dim=-1)
            accumulator = accumulator * rescale.unsqueeze(-1) + torch.matmul(
                exponentials.to(torch.bfloat16).float(), selected
            )
            running_max = new_max
        denominator = denominator + torch.exp(attn_sink.float().view(1, 1, heads) - running_max)
        result = accumulator / denominator.unsqueeze(-1)
        result = result.masked_fill(~(all_indices != -1).any(dim=-1).unsqueeze(-1).unsqueeze(-1), 0)
        output[:, start:end] = result.to(q.dtype)
    return output
