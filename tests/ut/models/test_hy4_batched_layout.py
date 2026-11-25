# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.models.hy_v4_flash.reference import window_candidates


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("query_width", [1, 6])
def test_hy4_verification_layout_isolates_requests(batch_size, query_width):
    from vllm_ascend.models.hy_v4_flash.decode_layout import (
        VerificationLayout,
        decode_positions,
        dense_candidates,
        pool_update_positions,
        window_history,
    )

    lens = torch.arange(batch_size, dtype=torch.int32) * 47 + query_width
    slots = torch.arange(batch_size * query_width).view(batch_size, query_width)
    slots[1::4, -1] = -1
    slots[2::4] = -1
    meta = SimpleNamespace(decode_seq_lens=lens, slot_mapping=slots.flatten())
    for permutation in (torch.arange(batch_size), torch.arange(batch_size).flip(0)):
        meta.decode_seq_lens.copy_(lens[permutation].clone())
        meta.slot_mapping.copy_(slots[permutation].flatten().clone())
        context = SimpleNamespace(additional_kwargs={})
        layout = VerificationLayout.for_forward(context, meta, query_width)
        assert layout is VerificationLayout.for_forward(context, meta, query_width)
        for i in range(batch_size):
            seq, live, pos = decode_positions(
                meta.decode_seq_lens[i : i + 1], meta.slot_mapping[i * query_width : (i + 1) * query_width], query_width
            )
            torch.testing.assert_close(layout.live[i], live[0])
            torch.testing.assert_close(layout.logical[i], pos[0])
            torch.testing.assert_close(layout.get_window_history(16)[i], window_history(seq, pos, 16)[0])
            torch.testing.assert_close(layout.get_window_candidates(16, i), window_candidates(pos[0], 16))
            torch.testing.assert_close(layout.pool_update(4, i), pool_update_positions(seq[0], query_width, 4))
            dense, recall = layout.dense_prefix(8, 4, 2, i)
            torch.testing.assert_close(dense, dense_candidates(pos[0], 8))
            torch.testing.assert_close(recall, (pos[0] // 4 > 2).unsqueeze(-1))
