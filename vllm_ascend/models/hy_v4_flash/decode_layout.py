# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request positions shared by target layers inside one captured decode forward."""

from dataclasses import dataclass, field

import torch
from vllm.models.hy_v4_flash.reference import window_candidates


def decode_positions(seq_lens, slots, num_queries):
    live = slots.view(-1, num_queries) >= 0
    seq_lens = seq_lens.long().masked_fill(~live[:, 0], 0)
    offsets = torch.arange(num_queries, device=slots.device)
    logical = (seq_lens[:, None] - num_queries + offsets).masked_fill(~live, -1)
    return seq_lens, live, logical


def window_history(seq_lens, logical, window):
    offsets = torch.arange(1 - window - logical.shape[1], 0, device=logical.device)
    shared = seq_lens[:, None] + offsets
    # Slot padding can occur at the tail of an otherwise live request.
    return shared.masked_fill_(shared > logical.amax(dim=1, keepdim=True), -1)


def pool_update_positions(seq_len, num_queries, ratio):
    first = (seq_len - num_queries).clamp_min(0) // ratio
    end = seq_len // ratio
    width = ((num_queries + ratio - 1) // ratio) * ratio
    history = first * ratio + torch.arange(width, device=seq_len.device)
    return first, end, history.masked_fill(history >= end * ratio, 0)


def dense_candidates(visible, width):
    history = torch.arange(width, device=visible.device).expand(visible.shape[0], -1)
    return history.masked_fill(history > visible[:, None], -1)


@dataclass
class VerificationLayout:
    """Single-request tensors derived solely from the group's mutable runner inputs."""

    seq_lens: torch.Tensor
    live: torch.Tensor
    logical: torch.Tensor
    window_rows: dict = field(default_factory=dict)
    window_candidates: dict = field(default_factory=dict)
    pool_updates: dict = field(default_factory=dict)
    visible_pool_pages: dict = field(default_factory=dict)
    dense_prefixes: dict = field(default_factory=dict)

    @classmethod
    def for_forward(cls, context, meta, num_queries):
        # The captured warm-up and capture use the same meta but separate
        # ForwardContexts. Construct the position ops again inside the graph.
        layouts = context.additional_kwargs.setdefault("hy4flash_verification_layouts", {})
        key = (id(meta), num_queries)
        if key not in layouts:
            layouts[key] = cls(*decode_positions(meta.decode_seq_lens, meta.slot_mapping[:num_queries], num_queries))
        return layouts[key]

    def get_window_history(self, window):
        if window not in self.window_rows:
            self.window_rows[window] = window_history(self.seq_lens, self.logical, window)
        return self.window_rows[window]

    def get_window_candidates(self, window):
        if window not in self.window_candidates:
            self.window_candidates[window] = window_candidates(self.logical[0], window)
        return self.window_candidates[window]

    def pool_update(self, ratio):
        if ratio not in self.pool_updates:
            self.pool_updates[ratio] = pool_update_positions(self.seq_lens[0], self.logical.shape[1], ratio)
        return self.pool_updates[ratio]

    def dense_prefix(self, width, ratio, budget):
        key = (width, ratio, budget)
        if key not in self.dense_prefixes:
            visible = self.logical[0]
            self.dense_prefixes[key] = dense_candidates(visible, width), (visible // ratio > budget)[:, None]
        return self.dense_prefixes[key]
