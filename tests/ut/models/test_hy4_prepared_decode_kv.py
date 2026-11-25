# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.hy_v4_flash import attention


@pytest.mark.parametrize("tokens", [6, 24, 48, 96])
def test_cache_writer_consumes_only_the_prepared_live_prefix(monkeypatch, tokens):
    capacity = tokens + 2
    cache = SimpleNamespace(device=SimpleNamespace(type="npu"), shape=(7, 656))
    meta = SimpleNamespace(slot_mapping=torch.arange(capacity))
    payload = torch.arange(capacity * 2).view(capacity, 1, 2)
    scales = torch.arange(capacity * 2).view(capacity, 1, 2) * 3
    rope = torch.arange(capacity * 4).view(capacity, 4)
    positions = torch.arange(capacity).expand(3, -1)
    reads = []
    monkeypatch.setattr(attention, "write_c8_rows", lambda *args: reads.append(args))
    layer = SimpleNamespace(indexer=None, fuse_c8_write=True, page_planes=True)
    result = attention.AscendHYV4FlashAttention._write_cache_rows(
        layer, cache, meta, None, None, rope, positions, tokens, full_graph=True, packed_kv=(payload, scales)
    )
    assert result is None and len(reads) == 1
    buffer, slots, data, key_rope, scaling, index_key, coords = reads[0]
    assert buffer is cache and index_key is None
    for found, want in zip(
        (slots, data, key_rope, scaling, coords),
        (meta.slot_mapping[:tokens], payload[:tokens], rope[:tokens], scales[:tokens], positions[..., :tokens]),
        strict=True,
    ):
        torch.testing.assert_close(found, want)
