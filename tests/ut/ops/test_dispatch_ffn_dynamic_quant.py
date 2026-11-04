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
void check(const std::array<float, 8>& values, float maximum,
           const std::array<int8_t, 8>& expected) {
    alignas(32) std::array<float, 8> input = values, scratch{}, scale{};
    alignas(32) std::array<int8_t, 8> output{};
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
int main() {
    check({}, 0, {});
    check({3.484375f, -3.484375f, 1.7421875f, -1.7421875f, 0, 0, 0, 0},
          3.484375f, {127, -127, 63, -63, 0, 0, 0, 0});
    check({.5f, 1.5f, 2.5f, 126.5f, -.5f, -1.5f, -2.5f, -126.5f},
          127.0f, {0, 2, 2, 126, 0, -2, -2, -126});
    check({1.1171875f, -1.1171875f, 0, 0, 0, 0, 0, 0},
          1.1171875f, {127, -127, 0, 0, 0, 0, 0, 0});
    volatile float maxForDivision = 1.1171875f;
    assert(maxForDivision / 127.0f != maxForDivision * (1.0f / 127.0f));
}
"""
    )
    binary = tmp_path / "quant"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(tmp_path), str(program), "-o", str(binary)],
        check=True,
    )
    subprocess.run([str(binary)], check=True)
