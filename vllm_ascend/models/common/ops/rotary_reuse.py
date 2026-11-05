# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse exact rotary factors for immutable inputs within one decode forward."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RotaryFrequencyReference:
    tensor: torch.Tensor
    values: tuple[float, ...]
    version: int

    def matches(self, tensor):
        # A replaced, converted or modified buffer must use the ordinary path.
        return tensor is self.tensor and not tensor.is_inference() and tensor._version == self.version


class DecodeRotaryCache:
    """Forward-local cache; captured tensor operations run again on every replay.

    Decode metadata is immutable while a model forward executes. Source tensor
    references remain alive here so object identities cannot be recycled.
    """

    def __init__(self):
        self._positions = {}
        self._shards = {}
        self._groups = {}
        self._factors = {}
        self._last_layer = -1

    def begin_layer(self, layer_id):
        # The model visits layers in increasing order. A repeated or earlier
        # layer starts another forward when callers reuse their runtime object.
        if layer_id <= self._last_layer:
            self._positions.clear()
            self._shards.clear()
            self._groups.clear()
            self._factors.clear()
        self._last_layer = layer_id

    def positions(self, sequence_lengths):
        key = id(sequence_lengths)
        if key not in self._positions:
            self._positions[key] = (sequence_lengths, (sequence_lengths - 1).clamp_min(0))
        return self._positions[key][1]

    def shard(self, positions, offset, rows):
        key = (id(positions), offset, rows)
        if key not in self._shards:
            self._shards[key] = (positions, positions[offset : offset + rows])
        return self._shards[key][1]

    def group_start(self, positions, ratio):
        key = (id(positions), ratio)
        if key not in self._groups:
            self._groups[key] = (positions, (positions + 1 - ratio).clamp_min(0))
        return self._groups[key][1]

    def factors(self, positions, frequencies, reference):
        if not reference.matches(frequencies):
            return None
        key = (id(positions), reference.values, frequencies.device, frequencies.dtype)
        if key not in self._factors:
            phases = positions.float().unsqueeze(-1) * frequencies
            phases = torch.stack((phases, phases), dim=-1).view(positions.shape[0], 1, 1, -1)
            sine = phases.sin()
            self._factors[key] = (positions, phases.cos(), sine, -sine)
        return self._factors[key][1:]
