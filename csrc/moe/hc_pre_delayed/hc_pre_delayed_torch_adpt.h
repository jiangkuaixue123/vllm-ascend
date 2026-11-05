// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_HC_PRE_DELAYED_TORCH_ADPT_H
#define VLLM_ASCEND_HC_PRE_DELAYED_TORCH_ADPT_H
#include <cmath>
#include <torch/extension.h>

namespace vllm_ascend {
constexpr int64_t HC_DELAYED_WIDTH = 4;
constexpr int64_t HC_DELAYED_HIDDEN = 5120;
constexpr int64_t HC_DELAYED_MIX = 24;
using HcPreDelayedOutputs = std::tuple<at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor>;

inline void check_hc_pre_delayed(const at::Tensor &x, const at::Tensor &fn,
    const at::Tensor &scale, const at::Tensor &base, const at::Tensor &previous,
    double normEps)
{
    TORCH_CHECK(x.dim() == 3, "HC delayed-pre requires x[T,4,5120].");
    TORCH_CHECK((x.size(0) == 6 || x.size(0) == 12) && x.size(1) == HC_DELAYED_WIDTH &&
        x.size(2) == HC_DELAYED_HIDDEN, "HC delayed-pre supports T=6/12, H=4, D=5120.");
    TORCH_CHECK(fn.dim() == 2 && fn.size(0) == HC_DELAYED_MIX &&
        fn.size(1) == HC_DELAYED_WIDTH * HC_DELAYED_HIDDEN, "HC fn must be [24,20480].");
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == 3, "HC scale must be [3].");
    TORCH_CHECK(base.dim() == 1 && base.size(0) == HC_DELAYED_MIX, "HC base must be [24].");
    TORCH_CHECK(previous.dim() == 2 && previous.size(0) == x.size(0) &&
        previous.size(1) == HC_DELAYED_WIDTH, "HC previous_pre must be [T,4].");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "HC x must retain BF16 precision.");
    TORCH_CHECK(x.is_contiguous(), "HC x must be contiguous.");
    TORCH_CHECK(!x.is_neg() && !x.is_conj(), "HC x must not have lazy negative/conjugate flags.");
    for (const at::Tensor *input : {&fn, &scale, &base, &previous}) {
        TORCH_CHECK(input->scalar_type() == at::kFloat, "HC gates and weights must retain FP32 precision.");
        TORCH_CHECK(input->is_contiguous(), "HC weights and gates must be contiguous.");
        TORCH_CHECK(!input->is_neg() && !input->is_conj(),
            "HC weights and gates must not have lazy negative/conjugate flags.");
        TORCH_CHECK(input->device() == x.device(), "HC inputs must share a device.");
    }
    const float eps = static_cast<float>(normEps);
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0f, "HC norm_eps must be positive finite FP32.");
}

inline HcPreDelayedOutputs make_hc_pre_delayed_outputs(const at::Tensor &x)
{
    auto t = x.sym_size(0);
    auto fp32 = x.options().dtype(at::kFloat);
    return {
        at::empty_symint({t, HC_DELAYED_HIDDEN}, x.options()),
        at::empty_symint({t, HC_DELAYED_WIDTH}, fp32),
        at::empty_symint({t, HC_DELAYED_WIDTH}, fp32),
        at::empty_symint({t, HC_DELAYED_WIDTH, HC_DELAYED_WIDTH}, fp32),
        at::empty_symint({t, HC_DELAYED_MIX}, fp32),
        at::empty_symint({t, 1}, fp32),
    };
}
}  // namespace vllm_ascend
#endif
