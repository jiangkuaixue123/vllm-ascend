# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request positions shared by target layers inside one captured decode forward."""

from dataclasses import dataclass, field

import torch
from vllm.models.hy_v4_flash.reference import window_candidates

from .native_pool_updates import plan_pool_refresh


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
    history = first[..., None] * ratio + torch.arange(width, device=seq_len.device)
    return first, end, history.masked_fill(history >= end[..., None] * ratio, 0)


def dense_candidates(visible, width):
    history = torch.arange(width, device=visible.device).expand(visible.shape[0], -1)
    return history.masked_fill(history > visible[:, None], -1)


@dataclass
class VerificationLayout:
    """Batched positions derived from the current forward's mutable runner inputs."""

    seq_lens: torch.Tensor
    live: torch.Tensor
    logical: torch.Tensor
    window_rows: dict = field(default_factory=dict)
    window_candidates: dict = field(default_factory=dict)
    pool_updates: dict = field(default_factory=dict)
    visible_pool_pages: dict = field(default_factory=dict)
    native_update_slots: tuple | None = None
    lightning_pool_plans: dict = field(default_factory=dict)
    dense_prefixes: dict = field(default_factory=dict)

    @classmethod
    def for_forward(cls, context, meta, num_queries):
        # The captured warm-up and capture use the same meta but separate
        # ForwardContexts. Construct the position ops again inside the graph.
        layouts = context.additional_kwargs.setdefault("hy4flash_verification_layouts", {})
        key = (id(meta), num_queries)
        if key not in layouts:
            num_tokens = meta.decode_seq_lens.numel() * num_queries
            layouts[key] = cls(*decode_positions(meta.decode_seq_lens, meta.slot_mapping[:num_tokens], num_queries))
        return layouts[key]

    def get_native_update_slots(self, table):
        # The layout belongs to one metadata group and graph forward. Retain the table
        # itself so another view or KV group always resolves its own physical addresses.
        if self.native_update_slots is None or self.native_update_slots[0] is not table:
            slots = plan_pool_refresh(self.seq_lens, table, self.logical.shape[1])
            self.native_update_slots = (table, slots)
        return self.native_update_slots[1]

    def get_window_history(self, window):
        if window not in self.window_rows:
            self.window_rows[window] = window_history(self.seq_lens, self.logical, window)
        return self.window_rows[window]

    def get_batched_window_candidates(self, window):
        if window not in self.window_candidates:
            candidates = window_candidates(self.logical.flatten(), window)
            self.window_candidates[window] = candidates.view(*self.logical.shape, window)
        return self.window_candidates[window]

    def get_window_candidates(self, window, request):
        return self.get_batched_window_candidates(window)[request]

    def pool_update(self, ratio, request):
        if ratio not in self.pool_updates:
            self.pool_updates[ratio] = pool_update_positions(self.seq_lens, self.logical.shape[1], ratio)
        first, end, history = self.pool_updates[ratio]
        return first[request], end[request], history[request]

    def dense_prefix(self, width, ratio, budget, request):
        key = (width, ratio, budget)
        if key not in self.dense_prefixes:
            dense = dense_candidates(self.logical.flatten(), width).view(*self.logical.shape, width)
            recall = (self.logical // ratio > budget).unsqueeze(-1)
            self.dense_prefixes[key] = dense, recall
        dense, recall = self.dense_prefixes[key]
        return dense[request], recall[request]
