# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_kv import kv_norm_rope_qdq
from vllm_ascend.ops.triton.deepseek_v41 import fp8_quant_dequant, fp4_quant_dequant
from vllm_ascend.models.deepseek_v41.attention import _OUTPUT_QDQ_TILE_ROWS


def _quantize_projection_input(output):
    """Consume the attention output; E4M3 quantization groups only its last dimension."""
    if output.shape[0] <= _OUTPUT_QDQ_TILE_ROWS:
        return fp8_quant_dequant(output, 32)
    for tile in output.split(_OUTPUT_QDQ_TILE_ROWS):
        fp8_quant_dequant(tile, 32, inplace=True)
    return output


def project_kv(self, states, positions):
    kv = self.wkv(states)[0]
    phases = positions.reshape(-1).float().unsqueeze(-1) * self.frequencies
    return kv_norm_rope_qdq(kv, self.kv_norm.weight, phases.cos(), phases.sin(), self.kv_norm.eps)


def _project_output(self, output, dtype):
    if self.dsa_cp:
        # Exchange local tokens and complete heads for complete tokens and
        # local heads. Each receiver owns the matching wo_a / wo_b slice.
        send = output.reshape(-1, self.tp, self.local_heads, self.dim).transpose(0, 1).contiguous()
        send = send.reshape(-1, self.local_heads, self.dim)
        output = torch.empty_like(send)
        dist.all_to_all_single(output, send, group=self.sequence_group.device_group)
    output = output.view(-1, self.groups, self.local_heads * self.dim // self.groups)
    output = _quantize_projection_input(output)
    projection = self.wo_a.weight.view(self.groups, self.config.o_lora_rank, -1)
    output = torch.einsum("sgd,grd->sgr", output, projection).flatten(1)
    if self.sequence_group is not None:
        # The input is already the local WO column slice. Keep the BF16
        # linear partial and perform precisely one FP32 TP reduce-scatter.
        output = self.wo_b.quant_method.apply(self.wo_b, output, bias=None)
        output = self.sequence_group.reduce_scatter(output.float(), dim=0)
    else:
        output = self.wo_b(output)[0].float()
        if self.tp > 1:
            output = tensor_model_parallel_all_reduce(output)
    return output.to(dtype)
