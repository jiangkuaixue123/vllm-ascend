// SPDX-License-Identifier: CANN-Open-Software-License-Agreement-Version-2.0
#pragma once
#include <cstdint>

namespace KvCompactCodec {
#ifdef KV_COMPRESS_CPU_TEST
#define KV_CODEC_INLINE inline
#else
#define KV_CODEC_INLINE __simt_callee__ __attribute__((always_inline)) inline
#endif

union FloatBits {
    uint32_t word;
    float value;
};

KV_CODEC_INLINE float AsFloat(uint32_t bits)
{
    FloatBits result;
    result.word = bits;
    return result.value;
}

// BF16 group maxima permit an exact exponent comparison without division.
// 448 = 1.75 * 2^8 and 6 = 1.5 * 2^2. The floors are the HF floors
// ceil(log2(1e-4/448)) and -126, respectively.
KV_CODEC_INLINE uint8_t PowerScale(uint16_t maxWord, bool fp8)
{
    if (maxWord >= 0x7f80U) {
        return 255;
    }
    const int32_t exponent = (maxWord >> 7) - (fp8 ? 8 : 2) + ((maxWord & 127U) > (fp8 ? 96U : 64U));
    const int32_t floor = fp8 ? 105 : 1;
    return static_cast<uint8_t>(exponent < floor ? floor : exponent);
}

KV_CODEC_INLINE uint8_t E4M3Magnitude(float x)
{
    if (!(x <= 464.0f)) {
        return 127;
    }
    if (x < 0.015625f) {
        const float sub = x * 512.0f;
        const uint32_t lo = static_cast<uint32_t>(sub);
        const float remainder = sub - static_cast<float>(lo);
        return static_cast<uint8_t>(lo + (remainder > 0.5f || (remainder == 0.5f && (lo & 1U))));
    }
    FloatBits bits;
    bits.value = x;
    // Keep the FP32 signless exponent and the top three mantissa bits,
    // rounding discarded bits to even before rebiasing the exponent.
    const uint32_t rounded = (bits.word + 0x7ffffU + ((bits.word >> 20) & 1U)) >> 20;
    return static_cast<uint8_t>(rounded - 120U * 8U);
}

KV_CODEC_INLINE float E4M3Value(uint8_t code)
{
    if (code < 8) {
        return code * (1.0f / 512.0f);
    }
    if (code == 127) {
        return AsFloat(0x7fc00000U);
    }
    return AsFloat(((static_cast<uint32_t>(code >> 3) + 120U) << 23) | ((code & 7U) << 20));
}

KV_CODEC_INLINE uint8_t PayloadFP8(uint16_t word, uint8_t scale)
{
    const uint32_t sign = (word >> 8) & 128U;
    const uint16_t magnitude = word & 0x7fffU;
    if (scale == 255) {
        return static_cast<uint8_t>(sign | (magnitude >= 0x7f80U ? 127U : 0U));
    }
    const float x = AsFloat(static_cast<uint32_t>(magnitude) << 16);
    const float normalized = x * AsFloat(static_cast<uint32_t>(254 - scale) << 23);
    return static_cast<uint8_t>(sign | E4M3Magnitude(normalized > 448.0f ? 448.0f : normalized));
}

KV_CODEC_INLINE uint8_t PayloadFP4(uint16_t word, float scale)
{
    const float x = AsFloat(static_cast<uint32_t>(word & 0x7fffU) << 16);
    const uint32_t code = (x > scale * 0.25f) + (x >= scale * 0.75f) + (x > scale * 1.25f) +
        (x >= scale * 1.75f) + (x > scale * 2.5f) + (x >= scale * 3.5f) + (x > scale * 5.0f);
    return static_cast<uint8_t>(code | ((word >> 12) & 8U));
}

KV_CODEC_INLINE uint16_t PowerScaledBoundary(uint16_t base, uint8_t scale)
{
    const int32_t fraction = base & 127U;
    const int32_t exponent = static_cast<int32_t>(base >> 7) + static_cast<int32_t>(scale) - 127;
    if (exponent <= 0) {
        return static_cast<uint16_t>((128 + fraction) >> (1 - exponent));
    }
    return exponent >= 255 ? 0x7f80U : static_cast<uint16_t>((exponent << 7) | fraction);
}

KV_CODEC_INLINE uint8_t PayloadMXFP4(uint16_t word, uint8_t scale)
{
    // Every midpoint is exactly representable as a BF16 magnitude at the
    // UE8M0 floor, including subnormals. Avoid vector-core float FTZ here.
    const uint16_t x = word & 0x7fffU;
    const uint32_t code = (x > PowerScaledBoundary(0x3e80U, scale)) +
        (x >= PowerScaledBoundary(0x3f40U, scale)) + (x > PowerScaledBoundary(0x3fa0U, scale)) +
        (x >= PowerScaledBoundary(0x3fe0U, scale)) + (x > PowerScaledBoundary(0x4020U, scale)) +
        (x >= PowerScaledBoundary(0x4060U, scale)) + (x > PowerScaledBoundary(0x40a0U, scale));
    return static_cast<uint8_t>(code | ((word >> 12) & 8U));
}

KV_CODEC_INLINE uint8_t E4Scale(uint16_t maximum)
{
    const float x = AsFloat(static_cast<uint32_t>(maximum) << 16);
    if (!(x <= 2784.0f)) {  // 6 * midpoint between 448 and the FP8 overflow code.
        return 127;
    }
    // Correct a divide/cast estimate at exact ties with 6 times the adjacent
    // E4 midpoint. This threshold is dyadic, even on vector cores using a
    // reciprocal approximation. A finite BF16 maximum is separated from a
    // neighboring non-tie boundary by much more than one FP32 division ulp.
    uint8_t code = E4M3Magnitude(x / 6.0f);
    code = code < 1 ? 1 : code > 126 ? 126 : code;
    if (code > 1) {
        const float lower = 3.0f * (E4M3Value(code - 1) + E4M3Value(code));
        if (x < lower || (x == lower && (code & 1U))) {
            return code - 1;
        }
    }
    if (code < 126) {
        const float upper = 3.0f * (E4M3Value(code) + E4M3Value(code + 1));
        if (x > upper || (x == upper && (code & 1U))) {
            return code + 1;
        }
    }
    return code;
}
#undef KV_CODEC_INLINE
}  // namespace KvCompactCodec
