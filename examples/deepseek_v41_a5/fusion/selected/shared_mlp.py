# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_moe import clipped_swiglu_route


def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor | None = None) -> torch.Tensor:
    activation = clipped_swiglu_route(
        self.w1(hidden_states), self.w3(hidden_states), routing_weights, self.swiglu_limit, hidden_states.dtype
    )
    output = self.w2(activation)
    if self.reduce_shared:
        output = tensor_model_parallel_all_reduce(output.float())
    return output.to(hidden_states.dtype)
