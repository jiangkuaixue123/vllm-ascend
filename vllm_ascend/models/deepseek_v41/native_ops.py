# SPDX-License-Identifier: Apache-2.0
"""A5 operators for the original DSV4.1 block-scaled checkpoint."""

from functools import cache
from importlib import import_module

import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.eager_kernels import act_quant, quantize_fp8_blocks


@cache
def initialize_native_ops():
    """Register the installed custom-op vendor before entering any compiled graph."""
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import_module("vllm_ascend.vllm_ascend_C")


def native_hc_ops(residual):
    if residual.device.type != "npu" or residual.dtype != torch.bfloat16 or residual.shape[-2:] != (4, 5120):
        return None
    # The standalone CPU oracle does not initialize the installed device package.
    from vllm_ascend.device.device_config import get_device_config

    if not get_device_config().supports_hc_pre_coefficients:
        return None
    initialize_native_ops()
    return torch.ops._C_ascend


def hash_route(logits, bias, *, top_k, temperature=1.0, route_scale=1.5):
    """Text-only selection: sqrt(softplus(logit/T)) + bias, then unbiased renorm."""
    # The image's norm_type=2 is sqrt-softplus with unbiased top-k renormalization.
    # Its renorm flag must be zero. The scaling factor is applied once by the kernel.
    weights, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        logits.float() / temperature,
        top_k,
        bias=bias.float(),
        k_group=1,
        group_count=1,
        routed_scaling_factor=route_scale,
        eps=1e-20,
        group_select_mode=0,
        renorm=0,
        norm_type=2,
    )
    return weights, ids


def rms_norm(x, weight, eps):
    """Preserve the checkpoint's FP32 normalization and final activation dtype."""
    return torch.ops.npu.npu_rms_norm(x.float(), weight.float(), eps)[0].to(x.dtype)


class NativeRMSNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.eps = source.eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


class NativeGroupedOutputProjection(nn.Module):
    """Block-diagonal wo_a with the checkpoint's E4M3 weights and per-32 UE8M0 scales."""

    def __init__(self, groups, rank, width):
        super().__init__()
        self.groups, self.rank, self.width = groups, rank, width
        self.weight = nn.Parameter(torch.empty(groups * rank, width, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(groups * rank // 32, width // 32, dtype=torch.uint8), requires_grad=False)
        self.register_buffer("paired_weight_scale", None, persistent=False)

    @torch.no_grad()
    def pack_after_loading(self):
        # The strict loader writes canonical [group * rank, K] rows each time.
        # Reuse that parameter's storage for the kernel's contiguous [group, K, rank].
        weight = self.weight.view(self.groups, self.rank, self.width).transpose(1, 2).contiguous()
        self.weight.copy_(weight.reshape_as(self.weight))
        scale = self.scale.repeat_interleave(32, dim=0).view(self.groups, self.rank, self.width // 64, 2)
        self.paired_weight_scale = scale.transpose(1, 2).contiguous().view(torch.float8_e8m0fnu)

    def forward(self, x):
        import torch_npu

        flat = x.reshape(-1, self.groups, self.width)
        quant, _, raw = quantize_fp8_blocks(flat)
        activation_scale = raw.reshape(flat.size(0), self.groups, self.width // 64, 2)
        out = torch_npu.npu_transpose_quant_batchmatmul(
            quant,
            self.weight.view(self.groups, self.width, self.rank),
            dtype=torch.bfloat16,
            x1_scale=activation_scale.view(torch.float8_e8m0fnu),
            x2_scale=self.paired_weight_scale,
            group_sizes=(0, 0, 32),
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        return out.reshape(*x.shape[:-1], self.rank)


def use_grouped_mx_projection(device, width, rank):
    if device.type != "npu" or width % 64 or rank % 32:
        return False
    from vllm_ascend.device.device_config import get_device_config

    return get_device_config().supports_grouped_mx_output_projection


def pack_grouped_output_weights(model):
    for layer in model.modules():
        if isinstance(layer, NativeGroupedOutputProjection):
            layer.pack_after_loading()


def grouped_output_projection(x, projection):
    if isinstance(projection, NativeGroupedOutputProjection):
        return projection(x)
    act_quant(x, inplace=True)
    weight = projection.weight.view(x.size(-2), -1, x.size(-1))
    return torch.einsum("...gk,grk->...gr", x, weight)


def quantize_activation(x):
    """Encode actual E4M3 bytes and paired UE8M0 bytes using the HF per-32 rule."""
    quant, _, raw = quantize_fp8_blocks(x)
    return quant.reshape(-1, x.shape[-1]), raw.reshape(-1, x.shape[-1] // 64, 2)


class NativeMXLinear(nn.Module):
    """Format post-load scale rows once; A5 consumes the original weight bytes directly."""

    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.scale = source.scale
        self.packed_fp4 = source.packed_fp4
        rows = self.weight.shape[0]
        scale = self.scale.view(torch.uint8)
        if not self.packed_fp4:
            scale = scale.repeat_interleave(32, dim=0)[:rows]
        self.register_buffer("paired_scale", scale.reshape(rows, -1, 2).transpose(0, 1), persistent=False)

    def forward(self, x):
        import torch_npu

        q, activation_scale = quantize_activation(x)
        extra = {"x2_dtype": torch_npu.float4_e2m1fn_x2} if self.packed_fp4 else {}
        result = torch_npu.npu_quant_matmul(
            q,
            self.weight.T,
            self.paired_scale,
            pertoken_scale=activation_scale,
            output_dtype=torch.bfloat16,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
            **extra,
        )
        return result.reshape(*x.shape[:-1], self.weight.shape[0])


def replace_native_leaves(module, *, prefix="", sites=None):
    """Replace only the stage-one compact Linear and weighted RMSNorm API."""
    from .backend import Linear, RMSNorm

    if sites is None:
        sites = {}
    for name, child in list(module.named_children()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, NativeGroupedOutputProjection):
            sites[path] = "npu::npu_transpose_quant_batchmatmul"
        elif isinstance(child, (NativeMXLinear, NativeRMSNorm)):
            continue
        elif isinstance(child, RMSNorm):
            setattr(module, name, NativeRMSNorm(child))
            sites[path] = "npu::npu_rms_norm"
        elif isinstance(child, Linear) and child.scale is not None:
            if child.weight.shape[-1] * (2 if child.packed_fp4 else 1) % 64:
                raise ValueError(f"Incompatible K for paired MX scales at {path}")
            setattr(module, name, NativeMXLinear(child))
            sites[path] = "npu::npu_quant_matmul"
        else:
            replace_native_leaves(child, prefix=path, sites=sites)
    return sites
