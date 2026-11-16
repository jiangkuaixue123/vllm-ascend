// SPDX-License-Identifier: Apache-2.0
#ifdef VLLM_ENABLE_ATB_AND_DIRECT_KERNELS
#include <ATen/ATen.h>
#include <cmath>
#include <tuple>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "aclnn_torch_adapter/NPUBridge.h"

void prefill_prefix_silu_launch(void*, void*, void*, void*, uint32_t, uint32_t, uint32_t);
void prefill_prefix_route_quant_launch(void*, void*, void*, void*, void*, void*, uint32_t,
                                     uint32_t, uint32_t, float);

void prefill_ordered_combine_launch(void*, void*, void*, void*, void*, void*, void*,
                                    uint32_t, uint32_t, uint32_t);

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

std::tuple<at::Tensor, at::Tensor> prefix_route_quant_meta(
    const at::Tensor& x, const at::Tensor&, const at::Tensor&, double) {
    return {at::empty({x.size(0), x.size(1) / 2}, x.options().dtype(at::kChar)),
            at::empty({x.size(0)}, x.options().dtype(at::kFloat))};
}

std::tuple<at::Tensor, at::Tensor> prefix_route_quant(
    const at::Tensor& x, const at::Tensor& factor, const at::Tensor& groups, double limit) {
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16 && x.is_contiguous()
                && x.size(0) <= 49152 && x.size(1) >= 128 && x.size(1) <= 4608
                && x.size(1) % 128 == 0, "Prefix requant expects bounded BF16 gate/up rows");
    TORCH_CHECK(factor.dim() == 1 && factor.size(0) == x.size(0) && factor.scalar_type() == at::kFloat,
                "Prefix requant requires one FP32 route factor per row");
    TORCH_CHECK(groups.dim() == 1 && groups.numel() <= 24 && groups.scalar_type() == at::kLong,
                "Prefix requant requires at most 24 int64 expert counts");
    TORCH_CHECK(std::isfinite(limit) && limit >= 0 && limit <= 65536,
                "Prefix requant requires a finite nonnegative clipping limit");
    for (const auto* tensor : {&x, &factor, &groups}) {
        TORCH_CHECK(tensor->device() == x.device() && tensor->is_contiguous()
                    && !tensor->is_neg() && !tensor->is_conj(), "Prefix requant requires local plain tensors");
        TORCH_CHECK(vllm_ascend::NPUBridge::GetNpuStorageImplDesc(*tensor).npu_format_ == 2,
                    "Prefix requant requires ND storage");
    }
    const c10_npu::OptionalNPUGuard guard(x.device());
    auto [output, scales] = prefix_route_quant_meta(x, factor, groups, limit);
    if (!x.size(0)) return {output, scales};
    void* input = x.data_ptr(); void* routes = factor.data_ptr(); void* counts = groups.data_ptr();
    void* out = output.data_ptr(); void* scale = scales.data_ptr();
    const uint32_t rows = x.size(0), width = x.size(1) / 2, count = groups.numel();
    const float clip = limit;
    auto stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("prefill_prefix_route_quant");
    // The command pool retains handlers; retain pointers, not activation tensors.
    cmd.SetCustomHandler([stream, input, routes, counts, out, scale, rows, width, count, clip]() -> int {
        prefill_prefix_route_quant_launch(stream, input, routes, counts, out, scale, rows, width, count, clip);
        return 0;
    });
    cmd.Run();
    return {output, scales};
}
void ordered_combine(const at::Tensor& x, const at::Tensor& factor, const at::Tensor& ids,
                     const at::Tensor& owned, at::Tensor& out, const c10::optional<at::Tensor>& destinations) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 5120 && x.size(0) <= 49152
                && x.scalar_type() == at::kBFloat16, "Ordered combine requires bounded BF16 expert rows");
    TORCH_CHECK(ids.dim() == 2 && ids.size(1) == 6 && ids.size(0) <= 8192 && ids.scalar_type() == at::kInt,
                "Ordered combine requires bounded int32 top-six indices");
    TORCH_CHECK(owned.sizes() == ids.sizes() && owned.scalar_type() == at::kFloat
                && factor.dim() == 1 && factor.size(0) == x.size(0) && factor.scalar_type() == at::kFloat,
                "Ordered combine requires FP32 ownership and row factors");
    TORCH_CHECK(out.dim() == 2 && out.size(1) == 5120 && out.size(0) <= 32768
                && out.scalar_type() == at::kFloat, "Ordered combine requires bounded FP32 output rows");
    TORCH_CHECK(destinations ? destinations->sizes() == at::IntArrayRef({ids.size(0)})
                && destinations->scalar_type() == at::kInt : out.size(0) == ids.size(0),
                "Ordered combine requires matching output destinations");
    for (const auto* t : {&x, &factor, &ids, &owned, static_cast<const at::Tensor*>(&out)}) {
        TORCH_CHECK(t->device() == x.device() && t->is_contiguous() && !t->is_neg() && !t->is_conj()
                    && vllm_ascend::NPUBridge::GetNpuStorageImplDesc(*t).npu_format_ == 2,
                    "Ordered combine requires local plain contiguous ND tensors");
    }
    if (destinations) TORCH_CHECK(destinations->device() == x.device() && destinations->is_contiguous()
                    && !destinations->is_neg() && !destinations->is_conj()
                    && vllm_ascend::NPUBridge::GetNpuStorageImplDesc(*destinations).npu_format_ == 2,
                    "Ordered combine requires local contiguous ND destinations");
    const c10_npu::OptionalNPUGuard guard(x.device());
    if (!ids.size(0)) return;
    void* input = x.data_ptr(); void* routes = factor.data_ptr(); void* indices = ids.data_ptr();
    void* ownership = owned.data_ptr(); void* output = out.data_ptr();
    void* mapping = destinations ? destinations->data_ptr() : nullptr;
    const uint32_t rows = ids.size(0), expanded = x.size(0), outputRows = out.size(0);
    auto stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("prefill_ordered_combine");
    cmd.SetCustomHandler([stream, input, routes, indices, ownership, output, mapping, rows, expanded, outputRows]() -> int {
        prefill_ordered_combine_launch(stream, input, routes, indices, ownership, output, mapping, rows, expanded, outputRows);
        return 0;
    });
    cmd.Run();
}

}

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("prefill_ordered_combine(Tensor x, Tensor factor, Tensor ids, Tensor owned, Tensor(a!) out, Tensor? destinations) -> ()");
    m.def("prefill_prefix_silu(Tensor x, Tensor groups) -> Tensor");
    m.def("prefill_prefix_route_quant(Tensor x, Tensor factor, Tensor groups, float limit) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("prefill_ordered_combine", &ordered_combine);
    m.impl("prefill_prefix_silu", &prefill_prefix_silu);
    m.impl("prefill_prefix_route_quant", &prefix_route_quant);
}
TORCH_LIBRARY_IMPL(_C_ascend, Meta, m) {
    m.impl("prefill_prefix_route_quant", &prefix_route_quant_meta);
}
#endif
