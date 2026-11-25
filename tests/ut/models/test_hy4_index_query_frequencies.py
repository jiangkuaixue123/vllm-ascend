# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def test_rotary_cache_scope_matches_stream_and_input_view(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import index_pool_fp8 as pool

    calls = []

    def frequency(pos, inverse, sections, interleaved):
        calls.append(1)
        return pos.float().clone(), pos.float().clone()

    monkeypatch.setattr(pool, "index_rotary_frequencies", frequency)
    inverse = torch.ones(32)
    positions = torch.arange(144).reshape(3, 48)
    shared = {}

    def get(x, stream=0, sections=(12, 10, 10), theta=1.0e7, cache=shared):
        return pool.shared_query_frequencies(cache, x, inverse, sections, True, theta, stream)

    first = get(positions[:, :12])
    assert get(positions[:, :12]) is first
    assert get(positions[:, 1:13]) is not first
    assert get(positions[:, :24:2]) is not first
    assert get(positions[:, :12], stream=1) is not first
    assert get(positions[:, :12], sections=(10, 12, 10)) is not first
    assert get(positions[:, :12], theta=10000.0) is not first
    positions.add_(1)
    fresh = get(positions[:, :12], cache={})
    torch.testing.assert_close(fresh[0], positions[:, :12].float())
    assert len(calls) == 7
