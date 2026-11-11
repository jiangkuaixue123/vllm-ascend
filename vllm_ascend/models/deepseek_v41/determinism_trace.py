# SPDX-License-Identifier: Apache-2.0
"""Diagnostic receipts for the first speculative target verification of a request."""

import hashlib
import json
import math

import numpy as np
import torch


def _signature(tensor):
    rows = tensor.detach().cpu().contiguous()
    raw = rows.view(torch.uint8).numpy().reshape(rows.shape[0], -1)
    float_rows = rows.float().reshape(rows.shape[0], -1)
    return [
        {
            "sha256": hashlib.sha256(b.tobytes()).hexdigest(),
            "finite": int(torch.isfinite(v).sum()),
            "elements": v.numel(),
        }
        for b, v in zip(raw, float_rows)
    ]


class FirstVerifyTrace:
    def __init__(self, directory, model, width):
        self.directory = directory
        self.rank = model.dp_rank
        self.width = width
        self.layers = len(model.backbone.layers)
        dim = model.backbone.dim
        dev = model.backbone.device
        # Rows retain their BF16 bytes. A single D2H copy outside capture extracts the values.
        self.boundaries = torch.empty(self.layers, 4, width, dim, dtype=torch.bfloat16, device=dev)
        self.final = torch.empty(width, dim, dtype=torch.bfloat16, device=dev)
        self.inputs = torch.empty(4, width, dtype=torch.int64, device=dev)
        self.host = None
        self.pending = None
        self.captured_pointers = {}
        self.pointers = None
        self.compressed_observer = None
        compressed_marker = directory / f"diagnose-compressed-rank-{self.rank}"
        if compressed_marker.is_file():
            from .compressed_verify_trace import CompressedVerifyTrace

            settings = json.loads(compressed_marker.read_text())
            layer = settings["layer"]
            if type(layer) is not int or not 0 <= layer < self.layers:
                raise ValueError("Compressed trace layer is outside the target")
            attention = model.backbone.layers[layer].attn
            if not (attention.is_kv_source and attention.is_index_source and attention.compress_ratio):
                raise ValueError("Compressed trace layer must own its KV and index")
            self.compressed_observer = CompressedVerifyTrace(directory, self.rank, width, attention, dev, **settings)
            attention._compressed_observer = self.compressed_observer
        self.window_observer = None
        marker = directory / f"diagnose-attention-window-rank-{self.rank}"
        if marker.is_file():
            from .window_trace import WindowTrace

            attention = model.backbone.layers[0].attn
            self.window_observer = WindowTrace(
                directory, self.rank, width, attention, dev, int(marker.read_text().strip() or 8)
            )
            for block in model.backbone.layers[:2]:
                block.attn._window_observer = self.window_observer
                self.window_observer.sinks.append(block.attn.attn_sink)

    def plan(self, plan, values, history):
        if self.window_observer is not None:
            self.window_observer.plan(plan)
        self.host = None
        if len(plan.positions) != self.width or not plan.active.all():
            return
        start, prompt = int(plan.positions[0]), int(plan.prompt_lengths[0])
        if start not in (prompt, prompt + 1) or not np.array_equal(
            plan.positions, np.arange(start, start + self.width)
        ):
            return
        self.host = {
            "start": start,
            "prompt": prompt,
            "ids": list(map(int, values)),
            "positions": plan.positions.tolist(),
            "slots": plan.slots.tolist(),
            "active": plan.active.astype(np.int64).tolist(),
        }
        tables = plan.block_tables[plan.request_rows[0]]
        self.host["first_page"] = int(tables[0])
        if self.compressed_observer is not None and prompt == self.compressed_observer.prompt:
            self.compressed_observer.set_tables(plan, history)
        digest = hashlib.sha256()
        for begin in range(0, prompt, 8192):
            logical = np.arange(begin, min(begin + 8192, prompt), dtype=np.int64)
            address = tables[logical // history.block_size] * history.block_size + logical % history.block_size
            digest.update(history.tokens[address].astype("<i8", copy=False).tobytes())
        self.host["mapped_host_prompt_sha256"] = digest.hexdigest()
        if self.window_observer is not None:
            self.window_observer.verify_plan(plan, history)

    def before_wrapper(self, ids, positions, descriptor, replay):
        if ids.numel() != self.width:
            return
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        mask = getattr(_EXTRA_CTX, "mc2_mask", None)
        addrs = ids.data_ptr(), positions.data_ptr(), None if mask is None else mask.data_ptr()
        if not replay:
            self.captured_pointers[descriptor] = addrs
        captured = self.captured_pointers[descriptor]
        self.pointers = {
            "id_pointer_matches_capture": addrs[0] == captured[0],
            "position_pointer_matches_capture": addrs[1] == captured[1],
            "mc2_mask_pointer_matches_capture": addrs[2] == captured[2],
            "moe_mode": str(getattr(_EXTRA_CTX, "moe_comm_type", None)),
        }

    def capture_inputs(self, ids, state):
        self.inputs[0].copy_(ids)
        self.inputs[1].copy_(state.pos[: self.width])
        self.inputs[2].copy_(state.slots[: self.width])
        self.inputs[3].copy_(state.active[: self.width])

    def tap(self, layer, site, tensor):
        self.boundaries[layer, site].copy_(tensor.reshape(-1, self.final.shape[-1])[: self.width])

    def after_wrapper(self, returned, replay_id):
        if self.host is None:
            return
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        output = returned[0] if isinstance(returned, tuple) else returned
        captured = self.inputs.cpu().tolist()
        record = {
            "rank": self.rank,
            "replay": replay_id,
            "host": self.host,
            "pointers": self.pointers,
            "graph_ids_pos_slots_active": captured,
            "final_in_graph": _signature(self.final),
            "returned_final": _signature(output[: self.width]),
            "layers": [],
        }
        mask = getattr(_EXTRA_CTX, "mc2_mask", None)
        record["mc2_mask_before_sampling"] = None if mask is None else mask.cpu().tolist()
        stages = self.boundaries.cpu()
        for layer in range(self.layers):
            record["layers"].append(
                {
                    name: _signature(stages[layer, i])
                    for i, name in enumerate(("attention_input", "attention_output", "ffn_input", "ffn_output"))
                }
            )
        self.pending = record
        self._write()
        if self.window_observer is not None:
            self.window_observer.finish(self.host, replay_id)
        if self.compressed_observer is not None:
            self.compressed_observer.save(self.host, replay_id)

    def logits(self, hidden, logits):
        if self.pending is None:
            return
        # Compute the diagnostic top-k and digests on CPU to leave the original NPU sampling order intact.
        scores = logits.detach().cpu().float()
        best = scores.topk(3, dim=-1)
        self.pending["head_input"] = _signature(hidden)
        self.pending["logits"] = _signature(scores)
        self.pending["top_ids"] = best.indices.tolist()
        self.pending["top_values"] = [[v if math.isfinite(v) else None for v in row] for row in best.values.tolist()]
        self._write()
        self.pending = None

    def _write(self):
        suffix = f"det-rank-{self.rank}-replay-{self.pending['replay']:06d}.json"
        target = self.directory / suffix
        temporary = target.with_suffix(".partial")
        temporary.write_text(json.dumps(self.pending, separators=(",", ":"), allow_nan=False) + "\n")
        temporary.replace(target)
