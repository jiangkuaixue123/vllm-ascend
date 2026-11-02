# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023 DeepSeek
"""DeepSeek V4.1 hash state, indexed host tables, and gated residual update.

Hashing and gating follow the permitted DeepSeek-V4.1-Flash reference at
314113f8a7dadb388215396ba5fbee2c610a04d6, inference/engram.py and model.py.
The INT8/E8M0 storage path is for the separately supplied local checkpoint.

MIT permission notice for the adapted reference:
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO

import numpy as np
import torch
from sympy import isprime
from tokenizers import Regex, normalizers
from torch import nn

HASH_SEED_MULTIPLIER = 10007
DEAD_TOKEN = -1
E8M0_BLOCK_SIZE = 32
FP32_EXPONENT_SHIFT = 23
FP32_E8M0_ZERO_BITS = 0x00400000
FP32_QUIET_NAN_BITS = 0x7FC00000
FP32_SIGN_MASK = -(1 << 31)
FP32_MAGNITUDE_MASK = 0x7FFFFFFF
GATE_CLAMP_MIN = 1e-6
MAX_HEADER_BYTES = 16 * 1024 * 1024
DEFAULT_LOOKUP_CHUNK_ROWS = 4096
MAX_DENSE_TENSOR_BYTES = 512 * 1024 * 1024
STORAGE_TYPES = MappingProxyType(
    {
        "I8": (1, torch.int8),
        "U8": (1, torch.uint8),
        "BF16": (2, torch.bfloat16),
        "F16": (2, torch.float16),
        "F32": (4, torch.float32),
        "I32": (4, torch.int32),
        "I64": (8, torch.int64),
        "F8_E8M0": (1, None),
        "F8_E4M3": (1, None),
        "F8_E5M2": (1, None),
    }
)


def build_compressed_token_map(tokenizer: Any) -> tuple[tuple[int, ...], int]:
    """Use the reference Rust tokenizer decoding and normalization sequence."""
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys: dict[str, int] = {}
    lookup = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    return tuple(lookup), len(keys)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @property
    def hash_columns(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads

    @classmethod
    def from_config(cls, config: Any) -> EngramLayout | None:
        layer_ids = tuple(config.engram_layer_ids)
        if not layer_ids:
            return None
        size, heads = config.engram_max_ngram_size, config.engram_n_heads
        if (
            size < 2
            or heads < 1
            or config.engram_head_dim < 1
            or config.engram_vocab_size < 1
            or len(set(layer_ids)) != len(layer_ids)
        ):
            raise ValueError("Invalid Engram layout")
        primes = []
        seen: set[int] = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(size - 1):
                current = config.engram_vocab_size - 1
                per_head = []
                for _ in range(heads):
                    current += 1
                    while current in seen or not isprime(current):
                        current += 1
                    seen.add(current)
                    per_head.append(current)
                per_ngram.append(tuple(per_head))
            primes.append(tuple(per_ngram))
        rows = tuple(config.engram_num_embeddings)
        if len(rows) != len(layer_ids) or any(
            count < sum(sum(group) for group in layer) for count, layer in zip(rows, primes)
        ):
            raise ValueError("Engram table cannot contain its prime bucket ranges")
        return cls(size, layer_ids, rows, tuple(primes), heads, config.engram_head_dim)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, compressed_vocab_size: int
) -> tuple[tuple[int, ...], ...]:
    if compressed_vocab_size < 1 or max_ngram_size < 2:
        raise ValueError("Invalid compressed vocabulary or n-gram size")
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    return tuple(
        tuple(
            int(value)
            for value in (
                np.random.default_rng(HASH_SEED_MULTIPLIER * layer_id).integers(
                    0, bound, size=max_ngram_size, dtype=np.int64
                )
                * 2
                + 1
            )
        )
        for layer_id in layer_ids
    )


@dataclass
class EngramRequestState:
    """An explicitly owned request cursor and bounded compressed lookback."""

    position: int = 0
    tail: tuple[int, ...] = ()
    vocabulary_signature: str | None = None

    def fork(self) -> EngramRequestState:
        return replace(self)

    def reset(self) -> None:
        self.position = 0
        self.tail = ()


def _cpu_ids(values: Sequence[int] | torch.Tensor) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError("Hash/table indices must be a 1-D CPU integer tensor")
        return values.tolist()
    result = list(values)
    if any(type(value) is not int for value in result):
        raise ValueError("Token indices must be integers")
    return result


class EngramHasher:
    """Compute reference hashes on CPU from scheduler-owned token histories."""

    def __init__(
        self,
        layout: EngramLayout,
        token_map: Sequence[int],
        *,
        compressed_vocab_size: int,
        pad_token_id: int,
    ):
        self.layout = layout
        self.token_map = tuple(token_map)
        if (
            not self.token_map
            or not 0 <= pad_token_id < len(self.token_map)
            or set(self.token_map) != set(range(compressed_vocab_size))
        ):
            raise ValueError("Compressed token map does not match model configuration")
        self.pad_id = self.token_map[pad_token_id]
        self.multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab_size)
        self.offsets = []
        for layer in layout.primes:
            offset, offsets = 0, []
            for group in layer:
                for prime in group:
                    offsets.append(offset)
                    offset += prime
            self.offsets.append(tuple(offsets))
        identity = np.asarray(self.token_map, dtype="<i8").tobytes()
        self.signature = hashlib.sha256(identity).hexdigest()

    @classmethod
    def from_tokenizer(
        cls,
        layout: EngramLayout,
        tokenizer: Any,
        *,
        compressed_vocab_size: int,
        pad_token_id: int,
    ) -> EngramHasher:
        token_map, actual_size = build_compressed_token_map(tokenizer)
        if actual_size != compressed_vocab_size:
            raise ValueError(f"Compressed vocabulary mismatch: {actual_size} != {compressed_vocab_size}")
        return cls(
            layout,
            token_map,
            compressed_vocab_size=actual_size,
            pad_token_id=pad_token_id,
        )

    def _compress(self, token_ids: Sequence[int] | torch.Tensor, token_mask: Sequence[bool] | None) -> list[int]:
        ids = _cpu_ids(token_ids)
        if any(not 0 <= value < len(self.token_map) for value in ids):
            raise ValueError("Token ID outside compressed vocabulary mapping")
        if token_mask is not None:
            mask = list(token_mask)
            if len(mask) != len(ids) or any(type(value) is not bool for value in mask):
                raise ValueError("token_mask must contain one bool per token")
        else:
            mask = [True] * len(ids)
        return [self.token_map[value] if live else DEAD_TOKEN for value, live in zip(ids, mask)]

    def new_state(
        self,
        history_ids: Sequence[int] | torch.Tensor = (),
        *,
        position: int = 0,
        history_mask: Sequence[bool] | None = None,
    ) -> EngramRequestState:
        """Restore only the last min(position, max_ngram-1) raw token IDs."""
        history = self._compress(history_ids, history_mask)
        if type(position) is not int or position < 0 or len(history) != min(position, self.layout.max_ngram_size - 1):
            raise ValueError("Missing or excessive request lookback history")
        return EngramRequestState(position, tuple(history), self.signature)

    def hash_tokens(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        state: EngramRequestState,
        *,
        start_position: int,
        token_mask: Sequence[bool] | None = None,
    ) -> torch.Tensor:
        """Return [tokens, layers, hash_columns], then advance only this state."""
        lookback = self.layout.max_ngram_size - 1
        if (
            start_position != state.position
            or state.position < 0
            or len(state.tail) != min(state.position, lookback)
            or state.vocabulary_signature not in (None, self.signature)
        ):
            raise ValueError("Request cursor/history does not match Engram hashing")
        compressed = self._compress(token_ids, token_mask)
        rows = []
        for index in range(len(compressed)):
            position = state.position + index
            tokens, blocked = [], False
            for shift in range(self.layout.max_ngram_size):
                source_position = position - shift
                if source_position < 0:
                    source = self.pad_id
                    blocked = True
                elif source_position < state.position:
                    source = state.tail[source_position - state.position]
                else:
                    source = compressed[source_position - state.position]
                blocked = blocked or source == DEAD_TOKEN
                tokens.append(self.pad_id if blocked else source)
            layers = []
            for layer_index, multipliers in enumerate(self.multipliers):
                products = [token * factor for token, factor in zip(tokens, multipliers)]
                rolling, hashes = products[0], []
                for size_index in range(lookback):
                    rolling ^= products[size_index + 1]
                    hashes.extend(rolling % prime for prime in self.layout.primes[layer_index][size_index])
                layers.append([value + offset for value, offset in zip(hashes, self.offsets[layer_index])])
            rows.append(layers)
        result = torch.tensor(rows, dtype=torch.int64, device="cpu").reshape(
            len(rows), len(self.layout.layer_ids), self.layout.hash_columns
        )
        state.tail = (state.tail + tuple(compressed))[-lookback:]
        state.position += len(compressed)
        state.vocabulary_signature = self.signature
        return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class IndexedTensor:
    name: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    metadata: Mapping[str, str]


class IndexedMmapStore:
    """Read only index-selected tensors; retain no full tensor copies."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        index_name: str = "quant_model_weights.safetensors.index.json",
        max_rows_per_read: int = DEFAULT_LOOKUP_CHUNK_ROWS,
    ):
        self.root = Path(model_path).resolve()
        index = json.loads((self.root / index_name).read_text(), object_pairs_hook=_unique_object)
        if not isinstance(index.get("weight_map"), dict) or max_rows_per_read < 1:
            raise ValueError("Invalid tensor index or row read bound")
        self.weight_map = MappingProxyType(dict(index["weight_map"]))
        self.index_metadata = MappingProxyType(dict(index.get("metadata", {})))
        self.max_rows_per_read = max_rows_per_read
        self._files: dict[Path, tuple[BinaryIO, mmap.mmap]] = {}
        self._headers: dict[Path, dict[str, IndexedTensor]] = {}
        self._closed = False

    def _open_shard(self, name: str) -> Path:
        if self._closed:
            raise RuntimeError("Indexed mmap store is closed")
        relative = self.weight_map.get(name)
        if not isinstance(relative, str):
            raise KeyError(f"Tensor is not selected by checkpoint index: {name}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Index shard path escapes checkpoint directory")
        if path in self._files:
            return path
        stream = path.open("rb")
        try:
            size = path.stat().st_size
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors header length")
            length = struct.unpack("<Q", raw)[0]
            if not 0 < length <= min(MAX_HEADER_BYTES, size - 8):
                raise ValueError("Invalid safetensors header length")
            header = json.loads(stream.read(length), object_pairs_hook=_unique_object)
            metadata = header.get("__metadata__", {})
            if not isinstance(metadata, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
            ):
                raise ValueError("Invalid safetensors metadata")
            descriptions = {}
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                dtype, shape = value.get("dtype"), value.get("shape")
                offsets = value.get("data_offsets")
                if (
                    dtype not in STORAGE_TYPES
                    or not isinstance(shape, list)
                    or any(type(dim) is not int or dim < 0 for dim in shape)
                    or not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)
                ):
                    raise ValueError(f"Invalid tensor header: {key}")
                lo, hi = offsets
                nbytes = math.prod(shape) * STORAGE_TYPES[dtype][0]
                if not 0 <= lo <= hi <= size - length - 8 or hi - lo != nbytes:
                    raise ValueError(f"Tensor payload bounds/shape mismatch: {key}")
                descriptions[key] = IndexedTensor(
                    key,
                    path,
                    dtype,
                    tuple(shape),
                    8 + length + lo,
                    nbytes,
                    MappingProxyType(dict(metadata)),
                )
            intervals = sorted(
                (item.offset, item.offset + item.nbytes) for item in descriptions.values() if item.nbytes
            )
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError("Safetensors payload intervals overlap")
            mapped = mmap.mmap(stream.fileno(), length=0, access=mmap.ACCESS_READ)
        except BaseException:
            stream.close()
            raise
        self._files[path] = (stream, mapped)
        self._headers[path] = descriptions
        return path

    def describe(self, name: str) -> IndexedTensor:
        path = self._open_shard(name)
        if name not in self._headers[path]:
            raise ValueError(f"Index-selected shard does not contain tensor: {name}")
        return self._headers[path][name]

    def read_rows(self, name: str, rows: Sequence[int]) -> bytes:
        item = self.describe(name)
        if len(item.shape) != 2 or len(rows) > self.max_rows_per_read:
            raise ValueError("Expected a bounded lookup into a 2-D tensor")
        if any(type(row) is not int or not 0 <= row < item.shape[0] for row in rows):
            raise ValueError("Row outside indexed tensor")
        width = item.shape[1] * STORAGE_TYPES[item.dtype][0]
        mapped = self._files[item.shard][1]
        return b"".join(mapped[item.offset + row * width : item.offset + (row + 1) * width] for row in rows)

    def read_tensor(self, name: str, *, max_bytes: int, device: torch.device | str = "cpu") -> torch.Tensor:
        item = self.describe(name)
        dtype = STORAGE_TYPES[item.dtype][1]
        if dtype is None or item.nbytes > max_bytes:
            raise ValueError("Full tensor read is unsupported or exceeds explicit bound")
        if not item.nbytes:
            return torch.empty(item.shape, dtype=dtype, device=device)
        mapped = self._files[item.shard][1]
        selected = bytearray(mapped[item.offset : item.offset + item.nbytes])
        return torch.frombuffer(selected, dtype=dtype).reshape(item.shape).to(device)

    def close(self) -> None:
        for stream, mapped in self._files.values():
            mapped.close()
            stream.close()
        self._files.clear()
        self._headers.clear()
        self._closed = True

    def __enter__(self) -> IndexedMmapStore:
        if self._closed:
            raise RuntimeError("Indexed mmap store is closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def decode_int8_e8m0_rows(
    weights: torch.Tensor,
    scales: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode selected rows with the exact bit path verified on the target NPU."""
    if (
        weights.ndim != 2
        or scales.ndim != 2
        or weights.dtype != torch.int8
        or scales.dtype != torch.uint8
        or weights.device != scales.device
        or weights.shape[1] % E8M0_BLOCK_SIZE
        or scales.shape != (weights.shape[0], weights.shape[1] // E8M0_BLOCK_SIZE)
    ):
        raise ValueError("Invalid INT8/E8M0 row layout")
    raw = scales.to(torch.int32)
    bits = torch.bitwise_left_shift(raw, FP32_EXPONENT_SHIFT)
    bits = bits.masked_fill(raw == 0, FP32_E8M0_ZERO_BITS)
    bits = bits.masked_fill(raw == 255, FP32_QUIET_NAN_BITS)
    decoded = bits.view(torch.float32)
    result = weights.float().unflatten(-1, (-1, E8M0_BLOCK_SIZE))
    return (result * decoded.unsqueeze(-1)).flatten(-2).to(output_dtype)


class MmapEngramEmbedding(nn.Module):
    """TP row shard backed by a shared host mmap store, not device parameters.

    Forward returns only this TP rank's contribution. The caller must reduce
    within its injected TP group before using the complete embedding values.
    No DP or EP collectives are inferred here. The model owner closes the store.
    """

    def __init__(
        self,
        store: IndexedMmapStore,
        prefix: str,
        *,
        tp_rank: int,
        tp_size: int,
        tp_group: Any = None,
        chunk_rows: int = DEFAULT_LOOKUP_CHUNK_ROWS,
    ):
        super().__init__()
        if tp_size < 1 or not 0 <= tp_rank < tp_size or not 0 < chunk_rows <= store.max_rows_per_read:
            raise ValueError("Invalid TP shard or lookup chunk size")
        weight = store.describe(prefix + ".weight")
        scale = store.describe(prefix + ".scale")
        if (
            weight.dtype != "I8"
            or scale.dtype != "F8_E8M0"
            or len(weight.shape) != 2
            or weight.shape[0] < 1
            or weight.shape[1] < 1
            or weight.shape[1] % E8M0_BLOCK_SIZE
            or scale.shape != (weight.shape[0], weight.shape[1] // E8M0_BLOCK_SIZE)
        ):
            raise ValueError("Indexed table is not row-major INT8/E8M0 group32")
        self.store, self.prefix = store, prefix
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group
        self.num_embeddings, self.embedding_dim = weight.shape
        self.shard_rows = (self.num_embeddings + tp_size - 1) // tp_size
        self.start_row = tp_rank * self.shard_rows
        self.end_row = min(self.start_row + self.shard_rows, self.num_embeddings)
        self.chunk_rows = chunk_rows

    def forward(self, indices: torch.Tensor, *, device: torch.device | str) -> torch.Tensor:
        if indices.device.type != "cpu" or indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Mmap lookup needs explicit CPU hash IDs")
        rows = _cpu_ids(indices.reshape(-1))
        if any(not 0 <= row < self.num_embeddings for row in rows):
            raise ValueError("Hash ID outside Engram table")
        output = torch.zeros((len(rows), self.embedding_dim), dtype=torch.bfloat16, device=device)
        selected = [(index, row) for index, row in enumerate(rows) if self.start_row <= row < self.end_row]
        for start in range(0, len(selected), self.chunk_rows):
            chunk = selected[start : start + self.chunk_rows]
            positions, row_ids = zip(*chunk)
            weight_bytes = self.store.read_rows(self.prefix + ".weight", row_ids)
            scale_bytes = self.store.read_rows(self.prefix + ".scale", row_ids)
            weights = torch.frombuffer(bytearray(weight_bytes), dtype=torch.int8)
            scales = torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint8)
            values = decode_int8_e8m0_rows(
                weights.reshape(len(chunk), self.embedding_dim).to(device),
                scales.reshape(len(chunk), self.embedding_dim // E8M0_BLOCK_SIZE).to(device),
            )
            destination = torch.tensor(positions, dtype=torch.int64, device=device)
            output.index_copy_(0, destination, values)
        return output.reshape(*indices.shape, self.embedding_dim)


def engram_gate_activation(dot: torch.Tensor) -> torch.Tensor:
    """Reference signed square root, without NPU copysign's CPU fallback."""
    if dot.dtype != torch.float32:
        raise ValueError("Engram gate scores must be FP32")
    magnitude = dot.abs().clamp_min(GATE_CLAMP_MIN).sqrt()
    bits = magnitude.view(torch.int32) & FP32_MAGNITUDE_MASK
    bits = bits | (dot.view(torch.int32) & FP32_SIGN_MASK)
    return torch.sigmoid(bits.view(torch.float32))


def apply_engram_gate(
    x: torch.Tensor,
    embedding_values: torch.Tensor,
    wkv: nn.Module,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
    token_mask: torch.Tensor | None = None,
    projection_input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    query_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    value_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply the reference gate with explicit, externally verified basis changes.

    Query transforms affect gate computation only; value transforms map the
    projected update into the original x basis. No rotation is guessed from
    quantization metadata or folded silently into weights.
    """
    if x.ndim < 2 or q_weight.shape != x.shape[-2:] or k_weight.shape != q_weight.shape:
        raise ValueError("Engram HC dimensions disagree")
    projection_input = embedding_values.flatten(-2)
    if projection_input_transform is not None:
        projection_input = projection_input_transform(projection_input)
    kv = wkv(projection_input)
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    query = x if query_transform is None else query_transform(x)
    if query.shape != x.shape:
        raise ValueError("Query basis transform changed the residual shape")
    h = query.float()
    weight = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps)
    rstd = rstd * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = engram_gate_activation(dot)
    if token_mask is not None:
        if token_mask.shape != x.shape[:-2] or token_mask.dtype != torch.bool:
            raise ValueError("Invalid Engram token mask")
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    if value_transform is not None:
        value = value_transform(value)
    if value.shape != x.shape[:-2] + (dim,):
        raise ValueError("Value basis transform changed the update shape")
    return (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


@torch.no_grad()
def load_engram_dense_weights(
    store: IndexedMmapStore,
    prefix: str,
    *,
    wkv: nn.Linear,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> None:
    """Load BF16 dense weights from exactly the shards chosen by the index."""
    for suffix, target in (
        ("wkv.weight", wkv.weight),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
    ):
        key = prefix + "." + suffix
        description = store.describe(key)
        if description.dtype != "BF16" or description.shape != tuple(target.shape):
            raise ValueError(f"Dense Engram tensor dtype/shape mismatch: {key}")
        value = store.read_tensor(key, max_bytes=MAX_DENSE_TENSOR_BYTES, device=target.device)
        target.copy_(value)


class DeepseekV41Engram(nn.Module):
    """Compose indexed lookup, caller-owned TP reduction and the reference gate."""

    def __init__(
        self,
        embedding: MmapEngramEmbedding,
        wkv: nn.Module,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        *,
        eps: float,
        tp_reduce: (Callable[[torch.Tensor, Any], torch.Tensor | None] | None) = None,
        rotation_contract: str = "unresolved",
        projection_input_transform: nn.Module | None = None,
        query_transform: nn.Module | None = None,
        value_transform: nn.Module | None = None,
    ):
        super().__init__()
        if embedding.tp_size > 1 and tp_reduce is None:
            raise ValueError("TP-sharded Engram requires an explicit TP reduction")
        self.embedding, self.wkv = embedding, wkv
        self.q_weight = nn.Parameter(q_weight, requires_grad=False)
        self.k_weight = nn.Parameter(k_weight, requires_grad=False)
        self.eps, self.tp_reduce = eps, tp_reduce
        self.projection_input_transform = projection_input_transform
        self.query_transform, self.value_transform = query_transform, value_transform
        self.set_rotation_contract(rotation_contract)

    def set_rotation_contract(self, contract: str) -> None:
        if contract not in ("unresolved", "unrotated", "external"):
            raise ValueError("Unknown Engram rotation contract")
        self.rotation_contract = contract

    def prepare_decode(
        self,
        hash_ids: torch.Tensor,
        *,
        bucket_size: int,
        device: torch.device | str,
        slots: Sequence[int] | torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage local rows before capture and each replay into the same tensor.

        ``hash_ids`` contains only active CPU request rows. ``slots`` maps them
        to graph rows; removed or idle rows are cleared on every preparation.
        The caller retains the first returned buffer and supplies it as ``out``
        whenever the corresponding graph is reused.
        """
        if (
            hash_ids.ndim != 2
            or hash_ids.device.type != "cpu"
            or hash_ids.dtype not in (torch.int32, torch.int64)
            or hash_ids.shape[1] < 1
            or bucket_size < 1
            or hash_ids.shape[0] > bucket_size
        ):
            raise ValueError("Expected active CPU hash IDs [tokens, columns] in a positive bucket")
        active = hash_ids.shape[0]
        positions = list(range(active)) if slots is None else _cpu_ids(slots)
        if (
            len(positions) != active
            or len(set(positions)) != active
            or any(not 0 <= slot < bucket_size for slot in positions)
        ):
            raise ValueError("Decode slots must be unique, in range and match active hash rows")
        shape = (bucket_size, hash_ids.shape[1], self.embedding.embedding_dim)
        target = torch.device(device)
        if out is None:
            out = torch.empty(shape, device=target, dtype=torch.bfloat16)
        elif (
            tuple(out.shape) != shape
            or out.dtype != torch.bfloat16
            or out.device.type != target.type
            or (target.index is not None and out.device.index != target.index)
            or not out.is_contiguous()
        ):
            raise ValueError("Decode embedding buffer shape, dtype or device mismatch")
        # Gather before mutating a reusable buffer so a failed lookup can be retried.
        values = self.embedding(hash_ids, device=out.device) if active else None
        out.zero_()
        if values is not None:
            destination = torch.tensor(positions, dtype=torch.int64, device=out.device)
            out.index_copy_(0, destination, values)
        return out

    def forward_decode(self, x: torch.Tensor, values: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Use only device tensors in the captured TP reduction and gate."""
        if self.rotation_contract == "unresolved":
            raise RuntimeError("Verify Engram rotation and configure its explicit transforms")
        if (
            values.ndim != 3
            or values.shape[0] != x.shape[0]
            or values.shape[-1] != self.embedding.embedding_dim
            or values.dtype != torch.bfloat16
            or values.device != x.device
            or token_mask.shape != values.shape[:1]
            or token_mask.dtype != torch.bool
            or token_mask.device != x.device
        ):
            raise ValueError("Invalid staged Engram decode values or slot mask")
        # A TP implementation may reduce in place. The graph input stays rank-local.
        return self._apply_values(x, values.clone() if self.embedding.tp_size > 1 else values, token_mask)

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.rotation_contract == "unresolved":
            raise RuntimeError("Verify Engram rotation and configure its explicit transforms")
        values = self.embedding(hash_ids, device=x.device)
        return self._apply_values(x, values, token_mask)

    def _apply_values(self, x: torch.Tensor, values: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if self.embedding.tp_size > 1:
            reduced = self.tp_reduce(values, self.embedding.tp_group)
            if reduced is not None:
                values = reduced
        return apply_engram_gate(
            x,
            values,
            self.wkv,
            self.q_weight,
            self.k_weight,
            eps=self.eps,
            token_mask=token_mask,
            projection_input_transform=self.projection_input_transform,
            query_transform=self.query_transform,
            value_transform=self.value_transform,
        )
