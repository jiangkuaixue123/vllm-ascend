# SPDX-License-Identifier: Apache-2.0
"""Small V4.1 DSpark state and Markov regressions, no checkpoints or NPU allocation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.dspark import DSparkDraft, greedy_block


def test_markov_depends_on_preceding_selected_token():
    # Each symbol deterministically links to the next. The second row follows
    # a different cycle from its own seed on every position.
    bias = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0], [5.0, 0.0, 0.0]])
    last = SimpleNamespace(
        norm=lambda x: x,
        markov_head=lambda ids: (bias[ids], ids[:, None].float()),
        confidence_head=lambda hidden, embed: embed.squeeze(-1),
    )
    h = torch.zeros(2, 4, 3)
    result = greedy_block(last, lambda x, full_logits: x, h, torch.tensor([0, 1]))
    assert result.token_ids.tolist() == [[1, 2, 0, 1], [2, 0, 1, 2]]
    assert torch.equal(result.logits.argmax(-1), result.token_ids)
    assert result.confidence.tolist() == [[0, 1, 2, 0], [1, 2, 0, 1]]


def test_draft_rejects_out_of_range_rows_and_positions():
    model = DSparkDraft.__new__(DSparkDraft)
    nn.Module.__init__(model)
    model.max_model_len, model.max_requests, model.vocab_size, model.k = 16, 2, 10, 5
    for seed, slot, position in [(0, -1, 2), (0, 2, 2), (0, 0, -1), (0, 0, 10), (10, 0, 2)]:
        with pytest.raises(ValueError):
            model.propose(torch.tensor([seed]), torch.tensor([slot]), torch.tensor([position]))
    with pytest.raises(ValueError):
        model.commit_target(torch.zeros(1, 4), torch.tensor([0]), torch.tensor([16]))
