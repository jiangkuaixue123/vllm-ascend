# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.models.hy_v4_flash.local_vocabulary import LocalVocabulary


@pytest.mark.parametrize("requests", [1, 4, 8, 16])
def test_local_embedding_handles_decode_rows_and_padding(requests):
    embedding = torch.nn.Embedding(19, 4)
    embedding.shard_indices = type("Shard", (), {"org_vocab_start_index": 0})()
    vocab = LocalVocabulary(embedding, embedding, max_decode_tokens=requests * 6)
    assert not vocab.can_embed(torch.zeros(requests, dtype=torch.long))
    vocab.table = torch.arange(76).reshape(19, 4).bfloat16()
    for count in (requests, requests * 6):
        ids = (torch.arange(count) * 7) % 22 - 1
        assert vocab.can_embed(ids)
        expected = torch.stack([vocab.table[max(i, 0)] if i < 19 else torch.zeros_like(vocab.table[0]) for i in ids])
        torch.testing.assert_close(vocab.embed(ids), expected)
    for ids in (torch.empty(0), torch.empty(requests, 6), torch.empty(requests * 6 + 1)):
        assert not vocab.can_embed(ids)
