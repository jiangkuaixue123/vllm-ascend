# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("tokens", [1, 6, 17, 48, 96, 192])
def test_prepared_feature_rows_use_packed_projection_across_batch_sizes(tokens):
    from vllm_ascend.models.hy_v4_flash.engram import FlashEngram

    hidden = SimpleNamespace(
        ndim=3,
        shape=(tokens, 4, 3072),
        device=SimpleNamespace(type="npu"),
        dtype=torch.bfloat16,
        is_contiguous=lambda: True,
    )
    x = torch.empty(1, dtype=torch.bfloat16)
    state = SimpleNamespace(channel_weights=x)
    qualify = FlashEngram._can_group_feature_projections
    assert qualify(state, hidden, x, object())
    assert qualify(state, hidden, x, None) == (tokens == 6)
    hidden.is_contiguous = lambda: False
    assert not qualify(state, hidden, x, object())
