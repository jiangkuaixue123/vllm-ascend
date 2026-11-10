# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

import torch
import torch.nn.functional as F
from vllm_ascend.ops.triton.deepseek_v41_router import router_weights


def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
    gate = self.custom_routing_function.__self__
    bias = gate.bias
    if gate.bias_vl is not None and router_logits.shape[-1] == bias.numel() + 1:
        router_logits, modality = router_logits.split((bias.numel(), 1), dim=-1)
        bias = torch.where(modality.bool(), gate.bias_vl, bias)
    # Preserve native scoring and native top-k, including their tie behavior.
    scores = F.softplus(router_logits.float() / gate.temperature).sqrt()
    indices = (scores + bias.float()).topk(self.top_k, dim=-1).indices
    weights = router_weights(scores, indices, self.renormalize, gate.scaling_factor)
    return weights, indices.to(torch.int32 if indices_type is None else indices_type)
