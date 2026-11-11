# SPDX-License-Identifier: Apache-2.0
"""Diagnostic raw/compressed attention cuts for the first six-row verification."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class CompressedVerifyTrace:
    def __init__(self, output, rank, width, attention, device, *, layer, prompt_tokens, limit):
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ValueError("Expected between 1 and 16 captured verifications")
        if type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 1024 or attention.compress_ratio <= 0:
            raise ValueError("Compressed short-trace prompt must contain 1 to 1024 tokens")
        self.layer = layer
        self.device = torch.device(device)
        self.prompt = prompt_tokens
        self.tables = None
        self.ratio = attention.compress_ratio
        self.output, self.rank, self.width, self.remaining = Path(output), rank, width, limit
        heads, dim, win = attention.n_local_heads, attention.head_dim, attention.window_size
        top = attention.indexer.index_topk
        self.max_valid_keys = (self.prompt + width + 1) // self.ratio
        if self.max_valid_keys > top:
            raise ValueError("The short probe must fit every visible compressed key in the model top-k")
        index_dim = attention.indexer.index_head_dim
        self.sink = attention.attn_sink
        self.buffers = {}
        for key, tail, dtype in (
            ("query", (heads, dim), torch.bfloat16),
            ("kv", (dim,), torch.bfloat16),
            ("raw_ids", (win,), torch.long),
            ("raw_data", (win, dim + dim // 32), torch.uint8),
            ("selected_logical", (top,), torch.long),
            ("comp_ids", (top,), torch.long),
            ("comp_data", (self.max_valid_keys, dim // 2 + dim // 16), torch.uint8),
            ("compressed_write_slots", (), torch.long),
            ("compressed_write_input", (dim,), torch.bfloat16),
            ("compressed_write_data", (dim // 2 + dim // 16,), torch.uint8),
            ("index_write_input", (index_dim,), torch.bfloat16),
            ("index_write_data", (index_dim // 2 + index_dim // 32,), torch.uint8),
            ("raw_spans", (1,), torch.long),
            ("sparse_output", (heads, dim), torch.bfloat16),
            ("wo_a_input", (heads * dim,), torch.bfloat16),
            ("wo_a_output", (attention.n_local_groups * attention.o_lora_rank,), torch.bfloat16),
        ):
            self.buffers[key] = torch.empty((width, *tail), device=device, dtype=dtype)

    def set_tables(self, plan, history):
        end = (int(plan.positions.max()) // history.block_size) + 1
        self.tables = plan.block_tables[plan.request_rows, :end].copy()
        self.block_size = history.block_size

    def captures(self, count):
        if count != self.width:
            return False
        # Dynamic eager prefill may use different widths. Only graph capture
        # emits copies on the device. CPU exercises the identical observation.
        return self.device.type == "cpu" or torch.npu.is_current_stream_capturing()

    def tap(self, key, tensor):
        dst = self.buffers[key]
        if key in ("wo_a_input", "wo_a_output"):
            tensor = tensor.flatten(1)
        if key in ("selected_logical", "comp_ids", "comp_data"):
            if tensor.ndim != dst.ndim or tensor.shape[0] != self.width or tensor.shape[2:] != dst.shape[2:]:
                raise ValueError(f"Invalid compressed trace geometry for {key}: {tuple(tensor.shape)}")
            if key == "comp_data":
                tensor = tensor[:, : self.max_valid_keys]
            if tensor.shape[1] > dst.shape[1]:
                raise ValueError(f"Too many columns for {key}: {tensor.shape[1]}")
            if tensor.shape[1] < dst.shape[1]:
                dst[:, tensor.shape[1] :].fill_(0 if key == "comp_data" else -1)
                dst = dst[:, : tensor.shape[1]]
        if tensor.shape != dst.shape:
            raise ValueError(f"Expected {tuple(dst.shape)} for {key}, got {tuple(tensor.shape)}")
        dst.copy_(tensor)

    def save(self, host, replay):
        if self.remaining == 0 or host["prompt"] != self.prompt or self.tables is None:
            return
        arrays = {}
        for k, v in self.buffers.items():
            data = v.cpu().contiguous()
            arrays[k] = (data.view(torch.uint16) if data.dtype == torch.bfloat16 else data).numpy()
        arrays["sink"] = self.sink.detach().float().cpu().numpy()
        arrays["host_tables"] = self.tables
        arrays["block_size"] = np.int64(self.block_size)
        arrays["ratio"] = np.int64(self.ratio)
        name = f"compressed{self.layer}-rank-{self.rank}-replay-{replay:06d}"
        path = self.output / (name + ".npz")
        with path.with_suffix(".partial").open("wb") as f:
            np.savez(f, **arrays)
        path.with_suffix(".partial").replace(path)
        receipt = {
            "layer": self.layer,
            "replay": replay,
            "host": host,
            "archive": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        tmp = path.with_suffix(".json.partial")
        tmp.write_text(json.dumps(receipt) + "\n")
        tmp.replace(path.with_suffix(".json"))
        self.remaining -= 1
