# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-local rotary identity, mutation and configuration boundaries."""

import pytest
import torch

from vllm_ascend.models.common.ops.rotary_reuse import DecodeRotaryCache, RotaryFrequencyReference


def frequency_reference(values=(1.0, 0.1)):
    with torch.inference_mode(False):
        tensor = torch.tensor(values)
    return tensor, RotaryFrequencyReference(tensor, tuple(values), tensor._version)


@torch.inference_mode()
def test_equal_frequency_tables_share_exact_factors():
    cache = DecodeRotaryCache()
    positions = cache.positions(torch.tensor([0, 1, 8192, 153600]))
    first, a = frequency_reference()
    second, b = frequency_reference()
    factors = cache.factors(positions, first, a)
    again = cache.factors(positions, second, b)
    assert all(x is y for x, y in zip(factors, again))
    phases = positions.float().unsqueeze(-1) * first
    phases = torch.stack((phases, phases), -1).view(4, 1, 1, 4)
    assert torch.equal(factors[0], phases.cos())
    assert torch.equal(factors[1], phases.sin())
    assert torch.equal(factors[2], -phases.sin())


@torch.inference_mode()
def test_position_sources_shards_and_groups_are_distinct():
    cache = DecodeRotaryCache()
    lengths = torch.arange(24)
    positions = cache.positions(lengths)
    assert cache.positions(lengths) is positions
    other = cache.positions(lengths.clone())
    assert other is not positions and torch.equal(other, positions)
    for offset in (0, 6, 12, 18):
        shard = cache.shard(positions, offset, 6)
        assert cache.shard(positions, offset, 6) is shard
        assert torch.equal(shard, positions[offset : offset + 6])
    for ratio in (1, 2, 4, 128):
        group = cache.group_start(positions, ratio)
        assert cache.group_start(positions, ratio) is group
        assert torch.equal(group, (positions + 1 - ratio).clamp_min(0))


@pytest.mark.parametrize("change", ["mutation", "replacement", "dtype", "inference"])
@torch.inference_mode()
def test_changed_frequency_buffer_is_not_reused(change):
    cache = DecodeRotaryCache()
    tensor, reference = frequency_reference()
    positions = torch.arange(6)
    assert cache.factors(positions, tensor, reference) is not None
    if change == "mutation":
        tensor.add_(1)
    elif change == "replacement":
        tensor = tensor.clone()
    elif change == "dtype":
        tensor = tensor.to(torch.float64)
    else:
        tensor = torch.tensor([1.0, 0.1])
    assert cache.factors(positions, tensor, reference) is None


@torch.inference_mode()
def test_distinct_configuration_and_forward_do_not_share():
    positions = torch.arange(6)
    first, a = frequency_reference()
    second, b = frequency_reference((1.0, 0.2))
    cache = DecodeRotaryCache()
    original = cache.factors(positions, first, a)
    different = cache.factors(positions, second, b)
    assert not torch.equal(original[0], different[0])
    new_forward = DecodeRotaryCache().factors(positions, first, a)
    assert original[0] is not new_forward[0]
    assert torch.equal(original[0], new_forward[0])


@torch.inference_mode()
def test_reused_runtime_recomputes_positions_for_new_forward():
    cache = DecodeRotaryCache()
    lengths = torch.arange(6)
    cache.begin_layer(0)
    first = cache.positions(lengths)
    cache.begin_layer(1)
    assert cache.positions(lengths) is first
    lengths.add_(17)
    cache.begin_layer(0)
    updated = cache.positions(lengths)
    assert updated is not first
    assert torch.equal(updated, (lengths - 1).clamp_min(0))
