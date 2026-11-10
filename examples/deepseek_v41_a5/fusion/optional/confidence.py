# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.ops.triton.deepseek_v41_router import draft_confidence


def compute_confidence(self, hidden_states, markov_embed):
    return draft_confidence(hidden_states, markov_embed, self.mtp[-1].confidence_head.proj.weight)
