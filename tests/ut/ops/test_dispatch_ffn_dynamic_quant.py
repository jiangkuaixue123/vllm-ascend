# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest


def test_route_before_down_quant_uses_vector_reciprocal_and_preserves_zero_scale(tmp_path):
    """Execute the production helper with instrumented CPU vector operations.

    This verifies the operands, synchronization, cast modes, and zero branch.
    Device numerical tests remain responsible for AscendC Div accuracy.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the quantization regression")
    root = Path(__file__).resolve().parents[3]
    header = root / "csrc/mc2/dispatch_ffn_combine/op_kernel/utils/route_before_down_quant.hpp"
    (tmp_path / "kernel_operator.h").write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>
#define __aicore__
constexpr int EVENT_ID0 = 0;
constexpr int PIPE_V = 0;
// All integer quantization results are exactly representable in FP16.
// Model its width and integral values without requiring a compiler FP16 type.
using half = int16_t;
namespace AscendC {
enum class HardEvent { V_S, S_V };
enum class RoundMode { CAST_RINT, CAST_ROUND, CAST_TRUNC };
struct BinaryRepeatParams {
    uint8_t dstBlkStride, src0BlkStride, src1BlkStride;
    uint8_t dstRepStride, src0RepStride, src1RepStride;
};
inline bool scalarReadable = false;
inline int divides = 0, multiplies = 0, scaleMultiplies = 0, casts = 0;
inline float expectedMax = 0;
template <typename T> struct LocalTensor {
    T* ptr;
    template <typename U> LocalTensor<U> ReinterpretCast() const {
        return {reinterpret_cast<U*>(ptr)};
    }
    T GetValue(int i) const {
        assert(scalarReadable);  // Every vector-to-scalar read needs V_S.
        return ptr[i];
    }
};
template <HardEvent E> void SetFlag(int) {}
template <HardEvent E> void WaitFlag(int) {
    if constexpr (E == HardEvent::V_S) scalarReadable = true;
}
template <int P> void PipeBarrier() {}
inline void SetDeqScale(half value) { assert(value == 1); }
template <typename T> void Duplicate(LocalTensor<T> dst, T value, uint32_t n) {
    std::fill(dst.ptr, dst.ptr + n, value);
    scalarReadable = false;
}
inline void Div(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    ++divides;
    assert(n == 8 && expectedMax != 0);
    for (uint32_t i = 0; i < n; ++i) {
        assert(a.ptr[i] == 127.0f && b.ptr[i] == expectedMax);
        dst.ptr[i] = a.ptr[i] / b.ptr[i];
    }
    scalarReadable = false;
}
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    ++multiplies;
    for (uint32_t i = 0; i < n; ++i) dst.ptr[i] = a.ptr[i] * b.ptr[i];
    scalarReadable = false;
}
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b,
                uint64_t mask, uint8_t repeats, const BinaryRepeatParams& p) {
    ++multiplies;
    assert(mask == 64 && repeats > 0);
    for (uint32_t r = 0; r < repeats; ++r) {
        for (uint32_t i = 0; i < mask; ++i) {
            uint32_t d = r * p.dstRepStride * 8 + (i / 8) * p.dstBlkStride * 8 + i % 8;
            uint32_t x = r * p.src0RepStride * 8 + (i / 8) * p.src0BlkStride * 8 + i % 8;
            uint32_t y = r * p.src1RepStride * 8 + (i / 8) * p.src1BlkStride * 8 + i % 8;
            assert(d < mask * repeats && x < mask * repeats && y < 8);
            dst.ptr[d] = a.ptr[x] * b.ptr[y];
        }
    }
    scalarReadable = false;
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> a, float factor, uint32_t n) {
    ++scaleMultiplies;
    assert(n == 8 && factor == 1.0f / 127.0f);
    for (uint32_t i = 0; i < n; ++i) {
        assert(a.ptr[i] == expectedMax);
        dst.ptr[i] = a.ptr[i] * factor;
    }
    scalarReadable = false;
}
template <typename D, typename S>
void Cast(LocalTensor<D> dst, LocalTensor<S> src, RoundMode mode, uint32_t n) {
    ++casts;
    std::vector<S> snapshot(src.ptr, src.ptr + n);
    if constexpr (std::is_same_v<D, int32_t>) {
        static_assert(std::is_same_v<S, float>);
        assert(mode == RoundMode::CAST_RINT);
        for (uint32_t i = 0; i < n; ++i) dst.ptr[i] = static_cast<D>(std::nearbyint(snapshot[i]));
    } else {
        if constexpr (std::is_same_v<D, half>) {
            static_assert(std::is_same_v<S, int32_t>);
            assert(mode == RoundMode::CAST_ROUND);
        } else {
            static_assert(std::is_same_v<D, int8_t> && std::is_same_v<S, half>);
            assert(mode == RoundMode::CAST_TRUNC);
        }
        for (uint32_t i = 0; i < n; ++i) dst.ptr[i] = static_cast<D>(snapshot[i]);
    }
    scalarReadable = false;
}
}
"""
    )
    program = tmp_path / "quant.cpp"
    program.write_text(
        f'#include "{header}"\n'
        + r"""
template<size_t N>
void check(const std::array<float, N>& values, float maximum,
           const std::array<int8_t, N>& expected) {
    alignas(32) std::array<float, N> input = values, scratch{};
    alignas(32) std::array<float, 8> scale{};
    alignas(32) std::array<int8_t, N> output{};
    scale[0] = maximum;
    AscendC::expectedMax = maximum;
    AscendC::divides = AscendC::multiplies = AscendC::scaleMultiplies = AscendC::casts = 0;
    AscendC::scalarReadable = false;
    float returned = VllmAscend::QuantizeRouteBeforeDown(
        {input.data()}, {scratch.data()}, {scale.data()}, {output.data()}, input.size());
    assert(output == expected);
    assert(AscendC::divides == (maximum != 0));
    assert(AscendC::multiplies == (maximum != 0));
    assert(AscendC::scaleMultiplies == 1 && AscendC::casts == 3);
    float expectedScale = maximum * (1.0f / 127.0f);
    assert(returned == expectedScale && std::isfinite(returned));
    for (float x : scale) assert(x == expectedScale);
}
template<size_t N>
void checkSize() {
    const std::array<float, 8> pattern{3.484375f, -3.484375f, 1.7421875f, -1.7421875f, 0, 0, 0, 0};
    const std::array<int8_t, 8> integers{127, -127, 63, -63, 0, 0, 0, 0};
    std::array<float, N> values{};
    std::array<int8_t, N> expected{};
    for(size_t i=0; i<N; ++i) { values[i]=pattern[i%8]; expected[i]=integers[i%8]; }
    check(values, 3.484375f, expected);
    check<N>({}, 0.0f, {});
}
int main() {
    check<8>({}, 0, {});
    check<8>({3.484375f, -3.484375f, 1.7421875f, -1.7421875f, 0, 0, 0, 0},
          3.484375f, {127, -127, 63, -63, 0, 0, 0, 0});
    check<8>({.5f, 1.5f, 2.5f, 126.5f, -.5f, -1.5f, -2.5f, -126.5f},
          127.0f, {0, 2, 2, 126, 0, -2, -2, -126});
    check<8>({1.1171875f, -1.1171875f, 0, 0, 0, 0, 0, 0},
          1.1171875f, {127, -127, 0, 0, 0, 0, 0, 0});
    volatile float maxForDivision = 1.1171875f;
    assert(maxForDivision / 127.0f != maxForDivision * (1.0f / 127.0f));
    checkSize<63>(); checkSize<64>(); checkSize<65>(); checkSize<128>();
    checkSize<2304>(); checkSize<2305>(); checkSize<16320>(); checkSize<16384>();
}
"""
    )
    binary = tmp_path / "quant"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(tmp_path), str(program), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
