# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_engram import engram_gate_residual
from vllm_ascend.models.deepseek_v41.engram import apply_engram_gate as reference_gate


def apply_engram_gate(
    x,
    embedding_values,
    wkv,
    q_weight,
    k_weight,
    *,
    eps,
    token_mask=None,
    projection_input_transform=None,
    query_transform=None,
    value_transform=None,
):
    if any(t is not None for t in (projection_input_transform, query_transform, value_transform)):
        return reference_gate(
            x,
            embedding_values,
            wkv,
            q_weight,
            k_weight,
            eps=eps,
            token_mask=token_mask,
            projection_input_transform=projection_input_transform,
            query_transform=query_transform,
            value_transform=value_transform,
        )
    projected = wkv(embedding_values.flatten(-2))
    return engram_gate_residual(x, projected, q_weight, k_weight, token_mask, eps)


def _apply_values(
    self, x: torch.Tensor, values: torch.Tensor, token_mask: torch.Tensor | None, *, sequence_parallel: bool = False
) -> torch.Tensor:
    if sequence_parallel:
        # The mmap tables are vocabulary-sharded, not token-sharded. Every
        # TP rank looked up the same hashes. Sum those contributions only
        # for the output token rows this rank owns.
        values = sp_reduce_scatter(values.flatten(1)).reshape(x.shape[0], *values.shape[1:])
    elif self.embedding.tp_size > 1:
        reduced = self.tp_reduce(values, self.embedding.tp_group)
        if reduced is not None:
            values = reduced
    return apply_engram_gate(
        x,
        values,
        self.wkv,
        self.q_weight,
        self.k_weight,
        eps=self.eps,
        token_mask=token_mask,
        projection_input_transform=self.projection_input_transform,
        query_transform=self.query_transform,
        value_transform=self.value_transform,
    )
