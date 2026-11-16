// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>

namespace vllm_ascend {
void deepseek_v41_wo_a_launch(void*, void*, void*, void*, uint32_t, uint32_t);
}

namespace {
void wo_a(const at::Tensor& x, const at::Tensor& weight, at::Tensor& out) {
    TORCH_CHECK(x.dim() == 3 && (x.size(0) == 12 || x.size(0) == 24 || x.size(0) == 48 || x.size(0) == 96)
                && (x.size(1) == 1 || x.size(1) == 2) && x.size(2) == 4096,
                "V4.1 WO_A requires qualified token/group dimensions");
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({x.size(1), 1024, 4096})
                && out.sizes() == at::IntArrayRef({x.size(0), x.size(1), 1024}), "Invalid WO_A weight/output");
    for (const auto* t : {&x, &weight, static_cast<const at::Tensor*>(&out)}) {
        TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() && !t->is_neg() && !t->is_conj()
                    && t->device() == x.device() && reinterpret_cast<uintptr_t>(t->data_ptr()) % 32 == 0,
                    "V4.1 WO_A requires aligned contiguous BF16 tensors on one device");
    }
    at::assert_no_overlap(x, out);
    at::assert_no_overlap(weight, out);
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream().stream();
    void* a = x.data_ptr(); void* b = weight.data_ptr(); void* c = out.data_ptr();
    const uint32_t rows = x.size(0), groups = x.size(1);
    at_npu::native::OpCommand cmd;
    cmd.Name("deepseek_v41_wo_a");
    cmd.SetCustomHandler([stream, a, b, c, rows, groups]() -> int {
        vllm_ascend::deepseek_v41_wo_a_launch(stream, a, b, c, rows, groups); return 0;
    });
    cmd.Run();
}
}
TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("deepseek_v41_wo_a(Tensor x, Tensor weight, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("deepseek_v41_wo_a", &wo_a);
}
#endif
