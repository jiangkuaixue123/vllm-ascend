// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>

void deepseek_v41_fp4_qdq_launch(void*, void*, void*, uint32_t, bool);

namespace {
void qdq(const at::Tensor& x, at::Tensor& out, bool e4) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16
                && x.is_contiguous() && out.is_contiguous() && !x.is_neg() && !out.is_neg()
                && x.sizes() == out.sizes() && x.device() == out.device()
                && x.numel() % (e4 ? 16 : 32) == 0 && x.numel() <= UINT32_MAX
                && reinterpret_cast<uintptr_t>(x.data_ptr()) % 64 == 0
                && reinterpret_cast<uintptr_t>(out.data_ptr()) % 64 == 0,
                "V4.1 FP4 requires matching aligned contiguous BF16 groups");
    at::assert_no_partial_overlap(x, out);
    if (!x.numel()) return;
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream();
    void* input = x.data_ptr(); void* output = out.data_ptr(); const uint32_t n = x.numel();
    at_npu::native::OpCommand cmd;
    cmd.Name("deepseek_v41_fp4_qdq");
    cmd.SetCustomHandler([stream, input, output, n, e4]() -> int {
        deepseek_v41_fp4_qdq_launch(stream, input, output, n, e4); return 0;
    });
    cmd.Run();
}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("deepseek_v41_fp4_qdq(Tensor x, Tensor(a!) out, bool e4) -> ()");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("deepseek_v41_fp4_qdq", &qdq);
}
#endif
