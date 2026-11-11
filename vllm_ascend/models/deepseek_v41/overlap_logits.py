# SPDX-License-Identifier: Apache-2.0
"""Bounded opt-in raw target-head observations during rank-local verification."""

import hashlib
import json

import numpy as np
import torch


class OverlapLogitsTrace:
    def __init__(self, marker, rank, verify_width, vocab_size):
        required = {
            "prompt_tokens",
            "prompt_positions",
            "generation_offsets",
            "score_token_ids",
            "top_k",
            "max_receipts",
        }
        try:
            cfg = json.loads(marker.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"{marker.name} requires a JSON object with {sorted(required)}") from exc
        if not isinstance(cfg, dict) or set(cfg) != required:
            raise ValueError(f"{marker.name} must contain exactly {sorted(required)}")
        for key in ("prompt_tokens", "top_k", "max_receipts"):
            if type(cfg[key]) is not int or cfg[key] <= 0:
                raise ValueError(f"{marker.name}: {key} must be a positive integer")
        self.prompt = cfg["prompt_tokens"]
        for key, lower, upper in (
            ("prompt_positions", 0, self.prompt),
            ("generation_offsets", -1, None),
            ("score_token_ids", 0, vocab_size),
        ):
            items = cfg[key]
            if not isinstance(items, list) or any(
                type(i) is not int or i < lower or (upper is not None and i >= upper) for i in items
            ):
                raise ValueError(f"{marker.name}: invalid {key} list")
        if not cfg["generation_offsets"] or cfg["top_k"] > vocab_size:
            raise ValueError(f"{marker.name}: set at least one generation offset and top_k within the vocabulary")
        self.offsets = set(cfg["generation_offsets"])
        self.prompt_positions = cfg["prompt_positions"]
        self.token_ids = cfg["score_token_ids"]
        self.top_k = cfg["top_k"]
        self.limit = cfg["max_receipts"]
        self.verify_width = verify_width
        self.path = marker.parent / f"overlap-head-rank-{rank}.jsonl"
        self.pending = None
        self.written = 0

    def plan(self, plan, ids, history):
        self.pending = None
        if self.written >= self.limit:
            return
        active = np.flatnonzero(plan.active)
        if not len(active) or not (plan.prompt_lengths[active] == self.prompt).all():
            return
        groups = []
        for rr in np.unique(plan.request_rows[active]):
            lanes = active[plan.request_rows[active] == rr]
            pos = plan.positions[lanes]
            if not np.array_equal(pos, np.arange(pos[0], pos[-1] + 1)):
                return
            end_of_prompt = pos[-1] == self.prompt - 1
            decode = pos[0] >= self.prompt and len(lanes) in (1, self.verify_width)
            if not (end_of_prompt or decode):
                return
            pick = int(pos[-1] if end_of_prompt else pos[0])
            if pick - self.prompt not in self.offsets:
                return
            n_pages = (max(self.prompt, int(pos[-1]) + 1) + history.block_size - 1) // history.block_size
            blocks = plan.block_tables[rr, :n_pages]
            digest = hashlib.sha256()
            for start in range(0, self.prompt, 8192):
                ii = np.arange(start, min(start + 8192, self.prompt))
                loc = blocks[ii // history.block_size] * history.block_size + ii % history.block_size
                digest.update(history.tokens[loc].astype("<i8", copy=False).tobytes())
            samples = {
                str(i): int(
                    history.tokens[blocks[i // history.block_size] * history.block_size + i % history.block_size]
                )
                for i in self.prompt_positions
            }
            groups.append(
                {
                    "request_row": int(rr),
                    "first_position": int(pos[0]),
                    "last_position": int(pos[-1]),
                    "pick_position": pick,
                    "query_tokens": len(pos),
                    "query_ids": ids[lanes].tolist() if len(pos) <= self.verify_width else None,
                    "query_sha256_i64": hashlib.sha256(np.asarray(ids[lanes], dtype="<i8").tobytes()).hexdigest(),
                    "host_blocks": blocks.tolist(),
                    "mapped_host_prompt_sha256": digest.hexdigest(),
                    "mapped_host_tokens": samples,
                    "write_first_slot": int(plan.slots[lanes[0]]),
                    "write_last_slot": int(plan.slots[lanes[-1]]),
                    "rows_for_sampling": 1 if end_of_prompt else len(lanes),
                }
            )
        self.pending = {"sequence": self.written + 1, "input_width": len(plan.positions), "groups": groups}

    def head(self, hidden, logits, state):
        event, self.pending = self.pending, None
        if event is None:
            return
        groups = event["groups"]
        expected = sum(g["rows_for_sampling"] for g in groups)
        event["logit_rows"] = len(logits)
        event["expected_logit_rows"] = expected
        event["device_block_tables"] = [
            state.pages[g["request_row"], : len(g["host_blocks"])].cpu().tolist() for g in groups
        ]
        event["matched_logit_geometry"] = len(logits) == len(hidden) == expected
        if event["matched_logit_geometry"]:
            start = 0
            for row in groups:
                # Stock sampling gathers the final prompt row or all verification
                # rows per request. The oldest verification query is at the start.
                h = hidden[start].detach().cpu().contiguous()
                v = logits[start].detach().cpu().float().contiguous()
                top = torch.topk(v, self.top_k).indices.tolist()
                row["logit_row"] = start
                row["hidden_bytes_sha256"] = hashlib.sha256(h.view(torch.uint8).numpy().tobytes()).hexdigest()
                row["logits_fp32_sha256"] = hashlib.sha256(v.numpy().tobytes()).hexdigest()
                row["finite_count"] = int(torch.isfinite(v).sum())
                row["nan_count"] = int(torch.isnan(v).sum())
                row["positive_infinity_count"] = int(torch.isposinf(v).sum())
                row["negative_infinity_count"] = int(torch.isneginf(v).sum())
                row["top_ids"] = top
                row["raw_scores"] = {
                    str(k): float(v[k]) if torch.isfinite(v[k]) else None
                    for k in dict.fromkeys([*top, *self.token_ids])
                }
                start += row["rows_for_sampling"]
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        self.written += 1
