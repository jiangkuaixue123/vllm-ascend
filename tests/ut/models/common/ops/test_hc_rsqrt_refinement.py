# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU syntax/arithmetic regression for the actual delayed-only vector helper.

The vector stubs deliberately supply a low precision Rsqrt estimate. They are
not an AscendC instruction simulator; real NPU validation remains required.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize("unaligned_comparison", [False, True])
def test_actual_refinement_helper_preserves_nonfinite_and_recovers_precision(tmp_path, unaligned_comparison):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h").read_text()
    start = source.index("    __aicore__ inline void RefineDelayedRsqrt(")
    end = source.index("    // Queue ownership", start)
    helper = source[start:end]
    if unaligned_comparison:
        # Reintroduce snapshot004's actual contract violation. CANN's release
        # implementation floors count / 64; its debug contract rejects it.
        expected = "MAX_FINITE, CMPMODE::LE, compareCount);"
        assert helper.count(expected) == 1
        helper = helper.replace(expected, "MAX_FINITE, CMPMODE::LE, count);")
    unit = r"""
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#define __aicore__
constexpr int PIPE_V = 0;
template<int> void PipeBarrier() {}
int64_t CeilAlign(int64_t value, int64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
enum class CMPMODE { LE };
enum class SELMODE { VSEL_TENSOR_TENSOR_MODE };
template<class T> struct LocalTensor {
    std::shared_ptr<std::vector<float>> storage;
    LocalTensor() : storage(std::make_shared<std::vector<float>>(512)) {}
    template<class U> LocalTensor<U> ReinterpretCast() const {
        LocalTensor<U> value; value.storage = storage; return value;
    }
    T& operator[](int64_t i) const { return reinterpret_cast<T*>(storage->data())[i]; }
};
void Adds(LocalTensor<float> out, LocalTensor<float> in, float v, int64_t n) {
    for (int64_t i=0;i<n;++i) out[i]=in[i]+v;
}
void Duplicate(LocalTensor<float> out, float value, int64_t n) {
    for (int64_t i=0;i<n;++i) out[i]=value;
}
void Muls(LocalTensor<float> out, LocalTensor<float> in, float v, int64_t n) {
    for (int64_t i=0;i<n;++i) out[i]=in[i]*v;
}
void Mul(LocalTensor<float> out, LocalTensor<float> a, LocalTensor<float> b, int64_t n) {
    for (int64_t i=0;i<n;++i) out[i]=a[i]*b[i];
}
void Rsqrt(LocalTensor<float> out, LocalTensor<float> in, int64_t n) {
    for (int64_t i=0;i<n;++i) {
        const float exact=1.0f/std::sqrt(in[i]);
        out[i]=std::isfinite(in[i]) ? exact*(1.0f+(i%2 ? -1.0f : 1.0f)/256.0f) : exact;
    }
}
void CompareScalar(LocalTensor<uint8_t> mask, LocalTensor<float> in, float limit, CMPMODE, int64_t n) {
    // Actual A3 L2 contract: count*sizeof(float) must divide into 256B
    // repeats. Pack eight predicates per byte, as consumed by Select mode2.
    if (n % 64) throw std::invalid_argument("unaligned CompareScalar");
    for (int64_t i=0;i<n/8;++i) mask[i]=0;
    for (int64_t i=0;i<n;++i) if (in[i]<=limit) mask[i/8] |= uint8_t(1u << (i%8));
}
void Select(LocalTensor<float> out, LocalTensor<uint8_t> mask, LocalTensor<float> a,
            LocalTensor<float> b, SELMODE, int64_t n) {
    // Select's count overload sets mask-count mode and supports short tails.
    for (int64_t i=0;i<n;++i) out[i]=(mask[i/8] & (1u << (i%8))) ? a[i] : b[i];
}
struct Kernel {
    LocalTensor<float> rsqrtLocal, rowBrcbLocal0, hcBrcbLocal1, reduceLocal, mixes01ReduceLocal;
__HELPER__
};
int main() {
    std::vector<float> values;
    for (int e=-66;e<=127;++e) {
        values.push_back(std::ldexp(1.0f,e));
        values.push_back(std::ldexp(1.375f,e));
    }
    values.push_back(1e-20f);
    values.push_back(std::numeric_limits<float>::max());
    values.push_back(std::numeric_limits<float>::infinity());
    values.push_back(std::numeric_limits<float>::quiet_NaN());
    int negative_control=0;
    // count1 is the actual tiling. Other counts exercise partial repeats and
    // packed masks; the full exponent set preserves the earlier math check.
    for (int64_t count : std::array<int64_t, 7>{1, 6, 12, 63, 64, 65, static_cast<int64_t>(values.size())}) {
      for (size_t offset : std::array<size_t, 3>{0, values.size()-2, values.size()-1}) {
        Kernel kernel;
        LocalTensor<float> input, unrefined;
        const int64_t aligned = CeilAlign(count, 64);
        for (int64_t i=0;i<aligned;++i) kernel.rowBrcbLocal0[i]=std::numeric_limits<float>::quiet_NaN();
        for (int64_t i=0;i<count;++i) input[i]=kernel.rsqrtLocal[i]=values[(offset+i)%values.size()];
        Rsqrt(unrefined,input,count);
        try { kernel.RefineDelayedRsqrt(count); }
        catch (const std::invalid_argument&) { return 11; }
        for (int64_t i=count;i<aligned;++i) if (kernel.rowBrcbLocal0[i]!=0.0f) return 12;
        for (int64_t i=0;i<count;++i) {
            const float expected=1.0f/std::sqrt(input[i]);
            const float actual=kernel.rsqrtLocal[i];
            if (std::isnan(expected)) { if (!std::isnan(actual)) return 2; continue; }
            if (std::isinf(input[i])) { if (actual!=0.0f || std::signbit(actual)) return 3; continue; }
            const double bound=1e-7+5e-6*std::abs(static_cast<double>(expected));
            if (std::abs(static_cast<double>(unrefined[i])-expected)>bound) ++negative_control;
            if (!std::isfinite(actual) || std::abs(static_cast<double>(actual)-expected)>bound) {
                std::cerr << i << " " << input[i] << " " << expected << " " << actual; return 4;
            }
        }
      }
    }
    if (negative_control==0) return 5;
    std::cout << values.size() << " cases, unrefined failures=" << negative_control << "\n";
}
""".replace("__HELPER__", helper)
    cpp = tmp_path / "refinement.cpp"
    executable = tmp_path / "refinement"
    cpp.write_text(unit)
    built = subprocess.run(
        [compiler, "-std=c++17", "-ffp-contract=off", str(cpp), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    checked = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
    assert checked.returncode == (11 if unaligned_comparison else 0), checked.stderr
    if not unaligned_comparison:
        assert "unrefined failures=" in checked.stdout
