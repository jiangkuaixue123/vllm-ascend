# SPDX-License-Identifier: Apache-2.0
"""Rank-local DSpark projections indexed by the target scheduler's physical KV slots."""

import numpy as np
import torch

from vllm_ascend.models.deepseek_v41.compact_kv import CacheEncoding, cache_row_bytes, store_compact_kv, unpack_window


class DraftPages:
    def __init__(self, text_config, block_size, num_blocks, device):
        self.block_size = block_size
        self.width = text_config.head_dim
        self.row_bytes = cache_row_bytes(self.width, 8, 32)
        self.layers = text_config.num_nextn_predict_layers
        self.window = text_config.sliding_window
        self.num_slots = 0
        # One permanently zero row serves every padded proposal and profile step.
        self.values = torch.zeros((self.layers, 1, self.row_bytes), dtype=torch.uint8, device=device)
        self.bind(num_blocks or 0)

    @property
    def bytes_per_block(self):
        return self.layers * self.block_size * self.row_bytes

    def bind(self, num_blocks):
        slots = num_blocks * self.block_size
        if slots != self.num_slots:
            if self.num_slots:
                raise RuntimeError("Cannot replace live DSpark physical pages")
            self.values = self.values.new_zeros((self.layers, slots + 1, self.row_bytes))
            self.num_slots = slots

    def write(self, layer, slots, kv):
        store_compact_kv(self.values[layer], kv, slots, CacheEncoding.WINDOW_MXFP8)

    def read(self, layer, slots):
        return unpack_window(self.values[layer][slots], self.width)

    def addresses(self, blocks, positions):
        positions = np.asarray(positions, dtype=np.int64)
        if positions.size == 0:
            return positions
        logical = positions // self.block_size
        first, last = int(logical.min()), int(logical.max())
        if first < 0 or last >= len(blocks):
            raise ValueError("The target block table does not cover the committed draft positions")
        pages = np.asarray(blocks[first : last + 1], dtype=np.int64)
        addresses = pages[logical - first] * self.block_size + positions % self.block_size
        if addresses.min() < 0 or addresses.max() >= self.num_slots:
            raise ValueError("The target block table exceeds the bound DSpark page pool")
        return addresses

    def window_addresses(self, blocks, last):
        # Frozen HF orders proposal keys in physical-ring order. The caller masks
        # entries beyond the live length in DSparkAttention.forward.
        absolute = last // self.window * self.window + np.arange(self.window)
        absolute = np.where(absolute > last, absolute - self.window, absolute)
        return self.addresses(blocks, absolute.clip(min=0))
