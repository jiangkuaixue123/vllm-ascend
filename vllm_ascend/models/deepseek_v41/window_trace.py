# SPDX-License-Identifier: Apache-2.0
"""Bounded layer-0/1 raw-window tracing, enabled per rank by an explicit marker."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def _digest(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.tobytes()).hexdigest()


def _rows(tensor):
    values = tensor.detach().cpu().contiguous()
    raw = values.view(torch.uint8).numpy().reshape(values.shape[0], -1)
    flat = values.float().reshape(values.shape[0], -1)
    return [
        {"sha256": _digest(b), "finite": int(torch.isfinite(v).sum()), "elements": v.numel()} for b, v in zip(raw, flat)
    ]


def _raw_summary(raw, dim):
    # E4M3FN has exactly two NaN codes. pack_window stores FP32 exponent
    # bytes separately, with 255 representing positive infinity.
    codes, scales = raw[..., :dim], raw[..., dim:]
    return {
        "nan_codes": int(np.sum((codes & 127) == 127)),
        "infinite_scales": int(np.sum(scales == 255)),
    }


class WindowTrace:
    def __init__(self, directory, rank, width, attention, device, limit):
        if not 1 <= limit <= 32:
            raise ValueError("Window trace marker must contain a receipt count between 1 and 32")
        self.directory, self.rank, self.width = Path(directory), rank, width
        self.window, self.dim = attention.window_size, attention.head_dim
        self.device, self.remaining = device, limit
        self.step = 0
        self.rows = None
        self.expected = None
        self.pending = {}
        self.writes = [OrderedDict(), OrderedDict()]
        self.host_writes = []
        self.sinks = []
        self.buffers = {}
        projected = attention.n_local_heads * self.dim
        row_bytes = self.dim + self.dim // 32
        for name, shape, dtype in (
            ("q", (projected,), torch.bfloat16),
            ("kv", (self.dim,), torch.bfloat16),
            ("packed", (row_bytes,), torch.uint8),
            ("slots", (self.window,), torch.long),
            ("gathered", (self.window, row_bytes), torch.uint8),
            ("sparse_output", (projected,), torch.bfloat16),
            ("wo_a_input", (projected,), torch.bfloat16),
            ("wo_a_output", (attention.n_local_groups * attention.o_lora_rank,), torch.bfloat16),
        ):
            self.buffers[name] = torch.empty((2, width, *shape), dtype=dtype, device=device)

    def plan(self, plan):
        self.rows = None
        self.expected = None
        if not self.remaining:
            return
        self._flush_writes()
        self.step += 1
        for slot in plan.slots[plan.active]:
            for history in self.writes:
                history.pop(int(slot), None)
        # The first verify begins at prompt or prompt+1. Only its raw window
        # and the one-token seed can contribute writes from an eager invocation.
        take = np.flatnonzero(
            plan.active
            & (plan.positions >= plan.prompt_lengths - self.window)
            & (plan.positions <= plan.prompt_lengths)
        )
        if len(take):
            self.rows = torch.as_tensor(take, device=self.device)
            self.host_writes = [(int(plan.slots[i]), int(plan.positions[i])) for i in take]

    def verify_plan(self, plan, history):
        if not self.remaining:
            return
        logical = plan.positions[:, None] + np.arange(1 - self.window, 1, dtype=np.int64)
        valid = logical >= 0
        logical = np.maximum(logical, 0)
        table = plan.block_tables[plan.request_rows[:, None], logical // history.block_size]
        slots = table * history.block_size + logical % history.block_size
        self.expected = np.where(valid, logical, -1), np.where(valid, slots, -1)

    def tap(self, layer, name, tensor):
        if tensor.shape[0] == self.width:
            dest = self.buffers[name][layer]
            dest.copy_(tensor.reshape(dest.shape))

    def projections(self, layer, q, kv, packed, slots, window):
        self.tap(layer, "q", q)
        self.tap(layer, "kv", kv)
        self.tap(layer, "packed", packed)
        if self.rows is None or (packed.device.type == "npu" and torch.npu.is_current_stream_capturing()):
            return
        # Hold these small tensors until the next host staging boundary. All
        # reads run in the same stream as the original index_copy_ write.
        actual = slots.index_select(0, self.rows)
        self.pending[layer] = packed.index_select(0, self.rows), window.index_select(0, actual)

    def gathered(self, layer, slots, raw):
        if slots.shape == (self.width, self.window):
            self.tap(layer, "slots", slots)
            self.tap(layer, "gathered", raw)

    def _flush_writes(self):
        for layer, (before, after) in self.pending.items():
            before = before.cpu().numpy()
            after = after.cpu().numpy()
            history = self.writes[layer]
            for (slot, pos), a, b in zip(self.host_writes, before, after):
                history[slot] = (pos, self.step, a.copy(), b.copy())
            while len(history) > 8 * self.window:
                history.popitem(last=False)
        self.pending.clear()

    def finish(self, host, replay):
        if self.expected is None or not self.remaining:
            return
        known_pos, known_slots = self.expected
        tensors = {name: data.cpu().contiguous() for name, data in self.buffers.items()}
        stem = f"window-rank-{self.rank}-replay-{replay:06d}"
        arrays = {
            "expected_positions": known_pos,
            "expected_slots": known_slots,
            "head_dim": np.int64(self.dim),
            "attn_sink": torch.stack([v.detach().cpu().float() for v in self.sinks]).numpy(),
        }
        for key, values in tensors.items():
            arrays[key] = (values.view(torch.uint16) if values.dtype == torch.bfloat16 else values).numpy()
        report = {"rank": self.rank, "replay": replay, "host": host, "archive": stem + ".npz", "layers": []}
        for layer in range(2):
            got_slots = arrays["slots"][layer]
            got_raw = arrays["gathered"][layer]
            packed = arrays["packed"][layer]
            history = dict(self.writes[layer])
            # These six packed rows were produced by the graph itself. Actual
            # bytes read by the attention gather are checked against them too.
            for slot, pos, raw in zip(host["slots"], host["positions"], packed):
                history[slot] = (pos, self.step, raw, None)
            evidence = []
            unknown = []
            mismatches = []
            after_mismatches = []
            referenced = set()
            valid = known_slots >= 0
            for i, j in np.argwhere(valid):
                slot, pos = int(known_slots[i, j]), int(known_pos[i, j])
                ref = history.get(slot)
                if ref is None or ref[0] != pos:
                    unknown.append((int(i), int(j), pos, slot))
                    continue
                referenced.add(slot)
                detail = (int(i), int(j), pos, slot, int(got_slots[i, j]), ref[1])
                if not np.array_equal(got_raw[i, j], ref[2]):
                    mismatches.append(detail)
                if ref[3] is not None and not np.array_equal(ref[2], ref[3]):
                    after_mismatches.append(detail)
            row_bytes = packed.shape[-1]
            for slot in sorted(referenced):
                pos, step, before, after = history[slot]
                evidence.append((slot, pos, step, int(after is not None)))
            refs = np.asarray(evidence, dtype=np.int64).reshape(-1, 4)
            arrays[f"source_slots_pos_step_has_readback_{layer}"] = refs
            for key, idx in (("write", 2), ("readback", 3)):
                values = [history[int(slot)][idx] for slot, *_ in refs]
                arrays[f"{key}_{layer}"] = (
                    np.stack([v if v is not None else np.zeros(row_bytes, dtype=np.uint8) for v in values])
                    if values
                    else np.empty((0, row_bytes), dtype=np.uint8)
                )
            report["layers"].append(
                {
                    "cuts": {k: _rows(t[layer]) for k, t in tensors.items() if k not in ("slots", "gathered")},
                    "valid_window_cells": int(valid.sum()),
                    "address_mismatches": int(np.sum(got_slots != known_slots)),
                    "unknown_write_cells": len(unknown),
                    "unknown_examples": unknown[:8],
                    "write_vs_readback_mismatches": len(after_mismatches),
                    "write_vs_readback_examples": after_mismatches[:8],
                    "write_vs_gather_mismatches": len(mismatches),
                    "write_vs_gather_examples": mismatches[:8],
                    "gathered_valid_raw": _raw_summary(got_raw[valid], self.dim),
                    "gathered_row_sha256": [_digest(x) for x in got_raw],
                }
            )
        archive = self.directory / report["archive"]
        with archive.with_suffix(".partial").open("wb") as handle:
            np.savez(handle, **arrays)
        archive.with_suffix(".partial").replace(archive)
        report["archive_sha256"] = _digest(archive.read_bytes())
        report_path = self.directory / (stem + ".json")
        report_path.with_suffix(".json.partial").write_text(json.dumps(report, allow_nan=False) + "\n")
        report_path.with_suffix(".json.partial").replace(report_path)
        self.remaining -= 1
