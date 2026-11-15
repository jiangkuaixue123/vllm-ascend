// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>

void prefill_prefix_silu_launch(void*, void*, void*, void*, uint32_t, uint32_t, uint32_t);

namespace {
at::Tensor prefill_prefix_silu(const at::Tensor& x, const at::Tensor& groups) {
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kFloat && x.is_contiguous(),
                "Prefix SiLU expects a contiguous FP32 matrix");
    TORCH_CHECK(x.size(1) % 8 == 0 && x.numel() <= UINT32_MAX,
                "Prefix SiLU requires eight-element alignment and bounded rows");
    TORCH_CHECK(groups.dim() == 1 && groups.scalar_type() == at::kLong && groups.is_contiguous()
                && groups.device() == x.device(), "Prefix SiLU requires local contiguous int64 counts");
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto output = at::empty_like(x);
    auto stream = c10_npu::getCurrentNPUStream().stream();
    auto x_ptr = x.data_ptr();
    auto groups_ptr = groups.data_ptr();
    auto out_ptr = output.data_ptr();
    const uint32_t rows = x.size(0), width = x.size(1), count = groups.numel();
    at_npu::native::OpCommand cmd;
    cmd.Name("prefill_prefix_silu");
    // The command pool retains handlers; tensor captures would retain entire activations.
    cmd.SetCustomHandler([x_ptr, groups_ptr, out_ptr, rows, width, count, stream]() -> int {
        prefill_prefix_silu_launch(stream, x_ptr, groups_ptr, out_ptr, rows, width, count);
        return 0;
    });
    cmd.Run();
    return output;
}
}

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("prefill_prefix_silu(Tensor x, Tensor groups) -> Tensor");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("prefill_prefix_silu", &prefill_prefix_silu);
}
#endif
