# SPDX-License-Identifier: Apache-2.0
"""Paged text history and the bounded host-to-device inputs for a single DP rank."""

import math
from dataclasses import dataclass

import numpy as np
import torch


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _field(obj, *names):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


@dataclass
class PagePlan:
    positions: np.ndarray
    slots: np.ndarray
    block_tables: np.ndarray
    active: np.ndarray
    prompt_lengths: np.ndarray
    request_rows: np.ndarray


class PagedHistory:
    """Only real scheduler pages own state. Stale speculative suffixes are never read."""

    def __init__(self, max_len, block_size, num_blocks, *, max_draft_tokens=0):
        if min(max_len, block_size) <= 0 or block_size % 2 or (num_blocks is not None and num_blocks <= 0):
            raise ValueError("Expected a positive, even KV page size")
        self.max_len = max_len
        self.block_size = block_size
        # Before vLLM profiles memory it has no physical pool. The dummy
        # forward may use scratch slots but cannot admit a scheduler address.
        self.num_blocks = 0 if num_blocks is None else num_blocks
        self.num_pages = math.ceil(max_len / block_size)
        self.num_slots = block_size * self.num_blocks
        self.max_draft_tokens = max_draft_tokens
        self.tail_groups = self.groups_per_block(block_size, max_draft_tokens)
        self.tail_tags = np.full(self.num_blocks * self.tail_groups, -1, dtype=np.int64)
        self.tokens = np.full(self.num_slots, -1, dtype=np.int64)

    @staticmethod
    def groups_per_block(block_size, max_draft_tokens):
        if max_draft_tokens < 0:
            raise ValueError("A speculative verification width cannot be negative")
        # The verifier always retains its seed. A k-draft rejection can put
        # the preceding even row at most floor(k/2) groups behind the last even.
        return min(block_size // 2, 1 + max_draft_tokens // 2)

    def compressor_tail(self, plan):
        """Plan paired reads and unique writes before the capture/replay boundary."""
        pos = plan.positions
        rows = np.arange(len(pos), dtype=np.int64)
        prior = rows.copy()
        dest = len(self.tail_tags) + rows
        even = plan.active & (pos % 2 == 0)
        odd = plan.active & ~even
        local = np.zeros(len(pos), dtype=np.bool_)
        local[1:] = even[:-1] & odd[1:] & (plan.request_rows[1:] == plan.request_rows[:-1]) & (pos[1:] == pos[:-1] + 1)
        prior[local] -= 1
        slot = plan.slots.clip(min=0)
        tail = slot // self.block_size * self.tail_groups + (slot % self.block_size // 2) % self.tail_groups
        external = odd & ~local
        # A scheduler preemption restarts at zero or an aligned cached block.
        # Unexpected deeper unaligned rewinds must never read overwritten FP32.
        if (self.tail_tags[tail[external]] != pos[external] - 1).any():
            raise RuntimeError("Unaligned V4.1 compressor rewind exceeds its committed verification tail")
        prior[external] = -1 - tail[external]
        writes = rows[even][::-1]
        if len(writes):
            # Earlier even rows in the same query are read directly from its
            # projections. Only the last writer for each persistent slot moves.
            _, latest = np.unique(tail[writes], return_index=True)
            keep = writes[latest]
            dest[keep] = tail[keep]
            # A writer at a page boundary starts a new logical page, including
            # physical ID recycling, before publishing the current query's tags.
            starts = slot[even & (pos % self.block_size == 0)] // self.block_size
            self.tail_tags.reshape(-1, self.tail_groups)[starts] = -1
            self.tail_tags[tail[keep]] = pos[keep]
        return np.stack((prior, dest), axis=1)

    def plan(self, positions, slots, block_tables, query_ends, prompt_lengths=None):
        positions = np.asarray(positions, dtype=np.int64).reshape(-1)
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        width = positions.size
        if slots.size < width:
            slots = np.pad(slots, (0, width - slots.size), constant_values=-1)
        slots = slots[:width]
        active = slots >= 0
        positions = np.where(active, positions, 0)
        if active.any() and (slots.max() >= self.num_slots or positions.min() < 0 or positions.max() >= self.max_len):
            raise ValueError("Scheduler token address is outside the configured page pool")
        live_tables = np.empty((0, self.num_pages), dtype=np.int64)
        token_prompts = np.zeros(width, dtype=np.int64)
        request_rows = np.full(width, -1, dtype=np.int64)
        if active.any():
            ends = np.asarray(query_ends, dtype=np.int64).reshape(-1)
            tables = np.asarray(block_tables, dtype=np.int64)
            if ends.size == 0 or tables.ndim != 2:
                raise ValueError("Missing scheduler row lengths or physical page table")
            # Padded request rows in some backends end in zero. Only query
            # offsets through the last real token participate in the mapping.
            last = np.flatnonzero(active)[-1]
            enough = np.flatnonzero(ends > last)
            if not enough.size:
                raise ValueError("Scheduler query lengths do not cover active token slots")
            ends = ends[: enough[0] + 1]
            if (np.diff(ends) < 0).any() or (ends < 0).any() or ends.size > len(tables):
                raise ValueError("Invalid scheduler query offsets")
            rows = np.searchsorted(ends, np.flatnonzero(active), side="right")
            request_rows[active] = rows
            if prompt_lengths is not None:
                prompts = np.asarray(prompt_lengths, dtype=np.int64).reshape(-1)
                if prompts.size <= rows.max() or (prompts < 0).any():
                    raise ValueError("Incomplete scheduler prompt lengths")
                token_prompts[active] = prompts[rows]
            lengths = positions[active] // self.block_size + 1
            if lengths.max() > tables.shape[1]:
                raise ValueError("Scheduler page table does not cover the live context")
            columns = min(tables.shape[1], self.num_pages)
            # The scheduler owns one table per request, even when thousands of
            # tokens from the same prompt arrive together. Validate live pages
            # once and keep the unused suffix at the null page.
            request_lengths = np.zeros(len(ends), dtype=np.int64)
            np.maximum.at(request_lengths, rows, lengths)
            needed = np.arange(columns) < request_lengths[:, None]
            pages = np.where(needed, tables[: len(ends), :columns], 0)
            if pages.min() < 0 or pages.max() >= self.num_blocks:
                raise ValueError("Physical block ID is outside the scheduler page pool")
            live_tables = np.zeros((len(ends), self.num_pages), dtype=np.int64)
            live_tables[:, :columns] = pages
            physical = pages[rows, lengths - 1] * self.block_size
            physical += positions[active] % self.block_size
            if not np.array_equal(physical, slots[active]) or np.unique(physical).size != physical.size:
                raise ValueError("Block table, positions and scheduler slots disagree")
        return PagePlan(positions, slots, live_tables, active, token_prompts, request_rows)

    def from_context(self, ctx, key, positions):
        width = positions.numel()
        metadata = ctx.attn_metadata
        mappings = getattr(ctx, "slot_mapping", None)
        data = metadata.get(key) if isinstance(metadata, dict) else None
        slots = mappings.get(key) if isinstance(mappings, dict) else None
        if slots is None and data is not None:
            slots = _field(data, "slot_mapping")
        if data is None:
            if slots is not None and (_numpy(slots)[:width] >= 0).any():
                raise RuntimeError("Active slots with no scheduler attention metadata")
            return self.plan(np.zeros(width), np.full(width, -1), [], [])
        if slots is None:
            raise RuntimeError(f"Missing slot mapping on {type(data).__name__}")
        slots = _numpy(slots).reshape(-1)[:width]
        if not (slots >= 0).any():
            return self.plan(np.zeros(width), slots, [], [])
        starts = _field(data, "query_start_loc_cpu", "query_start_loc")
        ends = _numpy(starts)[1:] if starts is not None else _field(data, "actual_seq_lengths_q")
        tables = _field(data, "block_tables", "block_table", "block_table_tensor")
        if ends is None or tables is None:
            raise RuntimeError(f"Missing logical query boundaries or physical block table on {type(data).__name__}")
        prompts = getattr(data, "v41_prompt_lengths", None)
        if prompts is None:
            raise RuntimeError("Active V4.1 attention rows require scheduler prompt lengths")
        return self.plan(_numpy(positions), slots, _numpy(tables), _numpy(ends), prompts)

    def hash_ids(self, reference, plan, input_ids):
        """Use exactly HF's compressed IDs, int64 products, XOR, primes and per-head offsets."""
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)[plan.active]
        active = np.flatnonzero(plan.active)
        pos = plan.positions[active]
        token_map = reference.token_map.numpy()
        if ids.size and (ids.min() < 0 or ids.max() >= token_map.size):
            raise ValueError("Input ID outside the checkpoint tokenizer")
        self.tokens[plan.slots[active]] = token_map[ids]
        tokens = []
        for shift in range(reference.layout.max_ngram_size):
            at = np.maximum(pos - shift, 0)
            addr = (
                plan.block_tables[plan.request_rows[active], at // self.block_size] * self.block_size
                + at % self.block_size
            )
            value = self.tokens[addr]
            if (value[pos >= shift] < 0).any():
                raise RuntimeError("Engram history was read before the scheduler filled its KV pages")
            tokens.append(np.where(pos >= shift, value, reference.pad_id))
        products = np.stack(tokens, axis=-1)[:, None, :] * reference.multipliers.numpy()
        rolling, hashes = products[..., 0], []
        for i in range(1, reference.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % reference.primes.numpy()[:, i - 1])
        return np.concatenate(hashes, axis=-1) + reference.offsets.numpy()


class DecodeInputs:
    """Storage addresses stay constant across capture, prefill, speculative verification and replay."""

    def __init__(self, history, max_tokens, device, engrams, *, max_requests=None, window_sizes=(128,)):
        self.history = history
        self.max_tokens = max_tokens
        self.device = device
        self.pos = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.prompt_lengths = torch.zeros_like(self.pos)
        self.raw_span_hint = 0
        self.raw_windows = {
            size: (
                torch.full((max_tokens, size), -1, dtype=torch.long, device=device),
                torch.full((max_tokens, 1), size, dtype=torch.long, device=device),
            )
            for size in sorted(set(window_sizes))
        }
        self.tail_plan = torch.zeros((max_tokens, 2), dtype=torch.long, device=device)
        self.active = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.slots = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.max_requests = max_tokens if max_requests is None else max_requests
        # The last row is a permanent null table for inactive graph lanes.
        self.pages = torch.zeros((self.max_requests + 1, history.num_pages), dtype=torch.long, device=device)
        self.page_rows = torch.full_like(self.pos, self.max_requests)
        self.serial = torch.arange(max_tokens, device=device)
        self.fp4_lut = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            device=device,
            dtype=torch.float32,
        )
        self.slots.copy_(self.serial + history.num_slots)
        self.engrams = {}
        for layer in engrams:
            cols = layer.wkv.weight.shape[1] // layer.embed.dim
            shape = (max_tokens, cols, layer.embed.dim)
            self.engrams[layer.layer_hash_index] = (
                torch.zeros(shape, device=device, dtype=torch.uint8),
                torch.zeros((*shape[:2], shape[2] // 32), device=device, dtype=torch.uint8),
            )

    def _stage_raw_windows(self, plan):
        # A captured target forward reuses these tables in every attention
        # layer, including across the compressed-key and top-k device kernels.
        width = len(plan.positions)
        prefill = plan.positions < plan.prompt_lengths
        block_size = self.history.block_size
        for size, (device_ids, device_spans) in self.raw_windows.items():
            spans = np.where(plan.active & prefill, np.minimum(plan.prompt_lengths, size), size)
            starts = plan.positions - size + 1
            starts = np.where(prefill, np.maximum(starts, 0), starts)
            columns = np.arange(size, dtype=np.int64)
            logical = starts[:, None] + columns
            valid = (
                plan.active[:, None]
                & (columns < spans[:, None])
                & (logical >= 0)
                & (logical <= plan.positions[:, None])
            )
            ids = np.full((width, size), -1, dtype=np.int64)
            rows, keys = np.nonzero(valid)
            pos = logical[rows, keys]
            pages = plan.block_tables[plan.request_rows[rows], pos // block_size]
            ids[rows, keys] = pages * block_size + pos % block_size
            device_ids[:width].copy_(torch.from_numpy(ids))
            device_spans[:width].copy_(torch.from_numpy(spans[:, None]))

    def stage_layout(self, plan):
        width = len(plan.positions)
        if not 0 < width <= self.max_tokens:
            raise ValueError("Scheduled width exceeds the static input allocation")
        requests = len(plan.block_tables)
        if requests > self.max_requests:
            raise ValueError("Scheduler request count exceeds the physical table allocation")
        if requests:
            self.pages[:requests].copy_(torch.from_numpy(plan.block_tables))
        live = np.flatnonzero(plan.active)
        prompts = plan.prompt_lengths[live]
        if len(live) and (plan.positions[live] >= prompts).any():
            self.raw_span_hint = None
        else:
            self.raw_span_hint = int(prompts.max(initial=0))
        slots = np.where(plan.active, plan.slots, self.history.num_slots + np.arange(width))
        for target, source in (
            (self.pos, plan.positions),
            (self.prompt_lengths, plan.prompt_lengths),
            (self.active, plan.active),
            (self.slots, slots),
            (self.page_rows, np.where(plan.active, plan.request_rows, self.max_requests)),
        ):
            target[:width].copy_(torch.from_numpy(source))
        self.tail_plan[:width].copy_(torch.from_numpy(self.history.compressor_tail(plan)))
        self._stage_raw_windows(plan)
        return live

    def stage(self, plan, ids, reference, engrams):
        live = self.stage_layout(plan)
        if len(live):
            hashes = self.history.hash_ids(reference, plan, ids)
            for layer in engrams:
                embed = layer.embed
                if embed._weight is None or embed._scale is None:
                    raise RuntimeError("Engram checkpoint must be bound before scheduling")
                raw_dest, scale_dest = self.engrams[layer.layer_hash_index]
                layer_ids = hashes[:, layer.layer_hash_index]
                # Keep file-backed random reads and transient host payload bounded.
                step = max(1, embed.transfer_rows // layer_ids.shape[1])
                for begin in range(0, len(live), step):
                    rows = layer_ids[begin : begin + step]
                    idx = live[begin : begin + step]
                    raw = torch.from_numpy(embed._weight[rows])
                    scales = torch.from_numpy(embed._scale[rows])
                    if idx[-1] - idx[0] + 1 == len(idx):
                        raw_dest[idx[0] : idx[-1] + 1].copy_(raw)
                        scale_dest[idx[0] : idx[-1] + 1].copy_(scales)
                    else:
                        loc = torch.from_numpy(idx).to(self.device)
                        raw_dest.index_copy_(0, loc, raw.to(self.device))
                        scale_dest.index_copy_(0, loc, scales.to(self.device))
        return int(plan.positions.max()) + 1 if len(live) else 1

    def pages_for_rows(self, rows):
        """Map a tile of query tokens to the corresponding scheduler tables."""
        return self.pages.index_select(0, self.page_rows[rows])

    def physical(self, positions, rows, ratio=1):
        """Positions are last-token absolute addresses of completed groups when ratio > 1."""
        page_cols = positions // self.history.block_size
        pages = self.pages[self.page_rows[rows, None], page_cols]
        # Some masked indexer lanes can query beyond this particular token.
        pages = torch.where(page_cols <= self.pos[rows, None] // self.history.block_size, pages, 0)
        return (pages * self.history.block_size + positions % self.history.block_size) // ratio

    def write_addresses(self, width, ratio=1):
        live = self.active[:width] & ((self.pos[:width] + 1) % ratio == 0)
        return torch.where(live, self.slots[:width] // ratio, self.history.num_slots // ratio + self.serial[:width])
