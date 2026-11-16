// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>

void deepseek_v41_fp8_rotary_pack_launch(void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t);
void deepseek_v41_fp8_qdq_launch(void*, void*, void*, uint32_t);
void deepseek_v41_fp8_pack_launch(void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);

namespace {
void check_matrix(const at::Tensor& x) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && !x.is_neg() && x.numel() % 32 == 0
                && x.numel() <= UINT32_MAX && reinterpret_cast<uintptr_t>(x.data_ptr()) % 64 == 0,
                "V4.1 FP8 requires aligned contiguous BF16 groups of32");
}

void qdq(const at::Tensor& x, at::Tensor& out) {
    check_matrix(x); check_matrix(out);
    TORCH_CHECK(x.sizes() == out.sizes() && x.device() == out.device(), "FP8 output must match input");
    at::assert_no_partial_overlap(x, out);
    if (!x.numel()) return;
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream();
    void* input = x.data_ptr(); void* output = out.data_ptr(); const uint32_t n = x.numel();
    at_npu::native::OpCommand cmd;
    cmd.Name("deepseek_v41_fp8_qdq");
    cmd.SetCustomHandler([stream, input, output, n]() -> int {
        deepseek_v41_fp8_qdq_launch(stream, input, output, n); return 0;
    });
    cmd.Run();
}

void pack(const at::Tensor& x, const c10::optional<at::Tensor>& tail, at::Tensor& out, int64_t start, int64_t tp) {
    check_matrix(x); check_matrix(out);
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 64 && x.size(2) == 512 && (tp == 4 || tp == 8)
                && out.dim() == 3 && out.size(0) % tp == 0 && out.size(1) == 64 / tp && out.size(2) == 512
                && x.device() == out.device() && start >= 0 && start + x.size(0) <= out.size(0) / tp,
                "Invalid V4.1 TP send span");
    at::assert_no_overlap(x, out);
    void* rotary = nullptr;
    if (tail.has_value()) {
        check_matrix(*tail);
        TORCH_CHECK(tail->sizes() == at::IntArrayRef({x.size(0), 64, 64}) && tail->device() == x.device(),
                    "Invalid V4.1 rotary tail");
        at::assert_no_overlap(*tail, out);
        rotary = tail->data_ptr();
    }
    if (!x.numel()) return;
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream();
    void* input = x.data_ptr(); void* output = out.data_ptr();
    const uint32_t n = x.numel(), rows = out.size(0) / tp, begin = start, ranks = tp;
    at_npu::native::OpCommand cmd;
    cmd.Name("deepseek_v41_fp8_pack");
    cmd.SetCustomHandler([stream, input, rotary, output, n, rows, begin, ranks]() -> int {
        deepseek_v41_fp8_pack_launch(stream, input, rotary, output, n, rows, begin, ranks); return 0;
    });
    cmd.Run();
}
void pack_rotary(const at::Tensor& x, const at::Tensor& cosine, const at::Tensor& sine, at::Tensor& out, int64_t tp) {
    check_matrix(x); check_matrix(out);
    TORCH_CHECK(x.dim() == 3 && (x.size(0) == 3 || x.size(0) == 6 || x.size(0) == 12)
                && x.size(1) == 64 && x.size(2) == 512 && (tp == 4 || tp == 8)
                && out.sizes() == at::IntArrayRef({tp * x.size(0), 64 / tp, 512})
                && out.device() == x.device(), "Invalid V4.1 rotary send shape");
    for (const auto* phase : {&cosine, &sine}) {
        TORCH_CHECK(phase->sizes() == at::IntArrayRef({x.size(0), 1, 1, 64})
                    && phase->scalar_type() == at::kFloat && phase->device() == x.device()
                    && phase->is_contiguous() && !phase->is_neg() && !phase->is_conj()
                    && reinterpret_cast<uintptr_t>(phase->data_ptr()) % 32 == 0,
                    "V4.1 rotary send requires dense FP32 paired phases");
        at::assert_no_overlap(*phase, out);
    }
    at::assert_no_overlap(x, out);
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream();
    void* input = x.data_ptr(); void* co = cosine.data_ptr(); void* si = sine.data_ptr(); void* output = out.data_ptr();
    const uint32_t n = x.numel(), rows = x.size(0), ranks = tp;
    at_npu::native::OpCommand cmd;
    cmd.Name("deepseek_v41_fp8_rotary_pack");
    cmd.SetCustomHandler([stream, input, co, si, output, n, rows, ranks]() -> int {
        deepseek_v41_fp8_rotary_pack_launch(stream, input, co, si, output, n, rows, ranks); return 0;
    });
    cmd.Run();
}

}

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("deepseek_v41_fp8_rotary_pack(Tensor x, Tensor cosine, Tensor sine, Tensor(a!) out, int tp) -> ()");
    m.def("deepseek_v41_fp8_qdq(Tensor x, Tensor(a!) out) -> ()");
    m.def("deepseek_v41_fp8_pack(Tensor x, Tensor? tail, Tensor(a!) out, int start, int tp) -> ()");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("deepseek_v41_fp8_rotary_pack", &pack_rotary);
    m.impl("deepseek_v41_fp8_qdq", &qdq);
    m.impl("deepseek_v41_fp8_pack", &pack);
}
#endif
