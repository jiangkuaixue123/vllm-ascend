# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for request histories, bounded indexed tables and Engram gates.

These tests target errors that can silently change tokens or load stale duplicate
weights. Tiny synthetic safetensors exercise I/O without a model or accelerator.
"""

import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v41.engram import (
    DeepseekV41Engram,
    EngramHasher,
    EngramLayout,
    IndexedMmapStore,
    MmapEngramEmbedding,
    apply_engram_gate,
    build_compressed_token_map,
    decode_int8_e8m0_rows,
    engram_gate_activation,
    load_engram_dense_weights,
)


def layout_config(**updates):
    values = dict(
        engram_layer_ids=(1,),
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_vocab_size=5,
        engram_head_dim=32,
        engram_num_embeddings=(36,),
    )
    values.update(updates)
    return SimpleNamespace(**values)


def make_hasher():
    layout = EngramLayout.from_config(layout_config())
    hasher = EngramHasher(layout, (0, 1, 1, 2, 3), compressed_vocab_size=4, pad_token_id=0)
    hasher.multipliers = ((1, 3, 5),)
    return hasher


def write_shard(path: Path, tensors, metadata=None):
    header = {"__metadata__": metadata or {"format": "pt"}}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def write_index(root, weight_map):
    (root / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )


@pytest.fixture
def table_checkpoint(tmp_path):
    prefix = "layers.1.engram.embed"
    weights = bytes(value for row in (1, 2, 3, 255, 254) for value in [row] * 32)
    scales = bytes((127, 128, 126, 255, 0))
    write_shard(
        tmp_path / "table.safetensors",
        {
            prefix + ".weight": ("I8", (5, 32), weights),
            prefix + ".scale": ("F8_E8M0", (5, 1), scales),
        },
        {"embedding_storage": "row_major_group32"},
    )
    write_index(
        tmp_path,
        {
            prefix + ".weight": "table.safetensors",
            prefix + ".scale": "table.safetensors",
        },
    )
    return tmp_path, prefix


def test_normalization_keeps_space_and_partial_byte_tokens_distinct():
    texts = [" The", "the", "THÉ", " ", "\t", "", "\ufffd", "\ufffd"]

    class Backend:
        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is False
            return texts[ids[0]]

        def id_to_token(self, token_id):
            return "raw-byte-" + str(token_id)

    class Tokenizer:
        backend_tokenizer = Backend()

        def __len__(self):
            return len(texts)

    lookup, count = build_compressed_token_map(Tokenizer())
    assert lookup[:3] == (0, 0, 0)
    assert lookup[3] == lookup[4]
    assert len({lookup[3], lookup[5], lookup[6], lookup[7]}) == 4
    assert count == 5


def test_prime_buckets_do_not_overlap_across_layers_or_ngram_sizes():
    layout = EngramLayout.from_config(layout_config(engram_layer_ids=(1, 14), engram_num_embeddings=(36, 90)))
    assert layout.primes == (((5, 7), (11, 13)), ((17, 19), (23, 29)))
    with pytest.raises(ValueError, match="bucket ranges"):
        EngramLayout.from_config(layout_config(engram_num_embeddings=(35,)))


def test_known_hash_order_and_prefill_decode_chunk_equivalence():
    hasher = make_hasher()
    full = hasher.hash_tokens([1, 3, 4], hasher.new_state(), start_position=0)
    assert full.tolist() == [[[1, 6, 13, 24]], [[1, 6, 13, 24]], [[0, 10, 12, 23]]]
    state = hasher.new_state()
    first = hasher.hash_tokens([1, 3], state, start_position=0)
    last = hasher.hash_tokens([4], state, start_position=2)
    assert torch.equal(full, torch.cat((first, last)))
    assert state.position == 3 and state.tail == (2, 3)


def test_dead_token_blocks_all_older_lookbacks():
    hasher = make_hasher()
    result = hasher.hash_tokens([3, 4, 3], hasher.new_state(), start_position=0, token_mask=[True, False, True])
    assert result[1, 0].tolist() == [0, 5, 12, 23]
    assert result[2, 0].tolist() == [2, 7, 14, 25]


def test_request_forks_and_restored_lookback_do_not_share_mutable_state():
    hasher = make_hasher()
    state = hasher.new_state()
    hasher.hash_tokens([1, 3], state, start_position=0)
    branch = state.fork()
    expected = hasher.hash_tokens([4], branch, start_position=2)
    assert state.position == 2
    restored = hasher.new_state([1, 3], position=2)
    assert torch.equal(expected, hasher.hash_tokens([4], restored, start_position=2))
    state.reset()
    assert state.position == 0 and state.tail == ()
    with pytest.raises(ValueError, match="lookback"):
        hasher.new_state([1], position=7)


def test_invalid_chunk_does_not_advance_request_state():
    hasher = make_hasher()
    state = hasher.new_state()
    with pytest.raises(ValueError, match="cursor"):
        hasher.hash_tokens([1], state, start_position=1)
    with pytest.raises(ValueError, match="outside"):
        hasher.hash_tokens([99], state, start_position=0)
    assert state.position == 0 and state.tail == ()
    empty = hasher.hash_tokens([], state, start_position=0)
    assert empty.shape == (0, 1, 4) and state.position == 0


def test_compressed_vocab_mismatch_fails_before_hashing():
    layout = EngramLayout.from_config(layout_config())
    with pytest.raises(ValueError, match="Compressed token map"):
        EngramHasher(layout, (0, 1), compressed_vocab_size=3, pad_token_id=0)


def test_e8m0_exact_fp32_bits_include_subnormal_nan_and_signed_weights():
    weights = torch.tensor(([1, -1, 127, -128] * 8) * 8, dtype=torch.int8)
    weights = weights.reshape(1, 256)
    raw = torch.tensor([[0, 1, 126, 127, 128, 253, 254, 255]], dtype=torch.uint8)
    result = decode_int8_e8m0_rows(weights, raw, output_dtype=torch.float32)
    assert result[0, 0].item() == math.ldexp(1, -127)
    assert result[0, 1].item() == -math.ldexp(1, -127)
    assert result[0, 96:100].tolist() == [1, -1, 127, -128]
    assert torch.isinf(result[0, 194])
    assert torch.isnan(result[0, -32:]).all()
    with pytest.raises(ValueError, match="layout"):
        decode_int8_e8m0_rows(weights, raw[:, :-1])


def test_tp_row_shards_sum_to_full_lookup_including_repeats_and_empty_shards(
    table_checkpoint,
):
    root, prefix = table_checkpoint
    ids = torch.tensor([[0, 2, 4], [2, 3, 1]])
    with IndexedMmapStore(root, max_rows_per_read=2) as store:
        full = MmapEngramEmbedding(store, prefix, tp_rank=0, tp_size=1, chunk_rows=2)(ids, device="cpu")
        partials = [
            MmapEngramEmbedding(store, prefix, tp_rank=rank, tp_size=8, chunk_rows=2)(ids, device="cpu")
            for rank in range(8)
        ]
        torch.testing.assert_close(sum(partials), full, rtol=0, atol=0, equal_nan=True)
        assert all(torch.count_nonzero(item) == 0 for item in partials[5:])
        assert torch.equal(full[0, 1], full[1, 0])


def test_mmap_storage_is_not_moved_as_a_registered_device_parameter(table_checkpoint):
    root, prefix = table_checkpoint
    with IndexedMmapStore(root) as store:
        embedding = MmapEngramEmbedding(store, prefix, tp_rank=0, tp_size=1)
        assert list(embedding.parameters()) == []
        assert embedding.state_dict() == {}
        embedding.to(dtype=torch.float32)
        assert store.describe(prefix + ".weight").dtype == "I8"
        metadata = store.describe(prefix + ".weight").metadata
        assert metadata["embedding_storage"] == "row_major_group32"
        with pytest.raises(ValueError, match="outside"):
            embedding(torch.tensor([5]), device="cpu")


def test_row_read_bound_and_explicit_close_are_enforced(table_checkpoint):
    root, prefix = table_checkpoint
    store = IndexedMmapStore(root, max_rows_per_read=2)
    with store:
        with pytest.raises(ValueError, match="bounded"):
            store.read_rows(prefix + ".weight", [0, 1, 2])
        with pytest.raises(ValueError, match="explicit bound"):
            store.read_tensor(prefix + ".weight", max_bytes=1)
    with pytest.raises(RuntimeError, match="closed"):
        store.describe(prefix + ".weight")


def test_index_selected_extra_dense_weights_override_unindexed_old_entries(tmp_path):
    prefix = "layers.1.engram"
    shapes = {"q_weight": (2, 2), "k_weight": (2, 2), "wkv.weight": (6, 32)}
    for filename, bits in (
        ("old.safetensors", 0x3F80),
        ("engram-extra.safetensors", 0x4000),
    ):
        tensors = {
            prefix + "." + key: (
                "BF16",
                shape,
                struct.pack("<H", bits) * math.prod(shape),
            )
            for key, shape in shapes.items()
        }
        write_shard(tmp_path / filename, tensors)
    write_index(tmp_path, {prefix + "." + key: "engram-extra.safetensors" for key in shapes})
    with IndexedMmapStore(tmp_path) as store:
        wkv = nn.Linear(32, 6, bias=False, dtype=torch.bfloat16)
        q_weight = torch.empty((2, 2), dtype=torch.bfloat16)
        k_weight = torch.empty_like(q_weight)
        load_engram_dense_weights(store, prefix, wkv=wkv, q_weight=q_weight, k_weight=k_weight)
        assert torch.all(wkv.weight == 2) and torch.all(q_weight == 2)
        assert torch.all(k_weight == 2)
        shard = store.describe(prefix + ".wkv.weight").shard
        assert shard.name == "engram-extra.safetensors"


def test_index_and_payload_corruption_fail_instead_of_guessing_a_shard(tmp_path):
    write_index(tmp_path, {"weight": "../outside.safetensors"})
    with (
        IndexedMmapStore(tmp_path) as store,
        pytest.raises(ValueError, match="escapes"),
    ):
        store.describe("weight")
    write_index(tmp_path, {"weight": "bad.safetensors"})
    write_shard(tmp_path / "bad.safetensors", {"weight": ("I8", (3, 32), bytes(32))})
    with (
        IndexedMmapStore(tmp_path) as store,
        pytest.raises(ValueError, match="bounds/shape"),
    ):
        store.describe("weight")


def test_gate_normalizes_each_hc_copy_and_preserves_masked_tokens():
    x = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
    embedding = torch.zeros((2, 1, 32))
    embedding[:, :, 0] = 1
    wkv = nn.Linear(32, 6, bias=False)
    with torch.no_grad():
        wkv.weight.zero_()
        wkv.weight[:, 0] = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0, 2.0])
    weights = torch.ones((2, 2))
    result = apply_engram_gate(
        x,
        embedding,
        wkv,
        weights,
        weights,
        eps=0,
        token_mask=torch.tensor([True, False]),
    )
    positive = 1 / (1 + math.exp(-(2**0.25)))
    expected = x.clone()
    expected[0, 0] += positive * torch.tensor([1.0, 2.0])
    expected[0, 1] += (1 - positive) * torch.tensor([1.0, 2.0])
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(result[1], x[1])


def test_gate_activation_preserves_negative_zero_without_copysign():
    scores = torch.tensor([0.0, -0.0, 1e-12, -1e-12, 4.0, -4.0])
    expected = torch.sigmoid(torch.copysign(scores.abs().clamp_min(1e-6).sqrt(), scores))
    actual = engram_gate_activation(scores)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0] > actual[1]


def test_rotation_query_hook_does_not_change_the_residual_addition_basis():
    torch.manual_seed(0)
    x, embedding = torch.randn(3, 2, 2), torch.randn(3, 1, 32)
    wkv, weights = nn.Linear(32, 6, bias=False), torch.ones(2, 2)
    canonical = apply_engram_gate(-x, embedding, wkv, weights, weights, eps=1e-6)
    actual = apply_engram_gate(
        x,
        embedding,
        wkv,
        weights,
        weights,
        eps=1e-6,
        query_transform=lambda value: -value,
    )
    torch.testing.assert_close(actual, x + (canonical + x), rtol=1e-6, atol=1e-6)


def test_module_requires_explicit_tp_reduction_and_resolvable_rotation_contract(
    table_checkpoint,
):
    root, prefix = table_checkpoint
    with IndexedMmapStore(root) as store:
        embedding = MmapEngramEmbedding(store, prefix, tp_rank=0, tp_size=2)
        wkv = nn.Linear(32, 6, bias=False, dtype=torch.bfloat16)
        weights = torch.ones(2, 2, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="TP reduction"):
            DeepseekV41Engram(embedding, wkv, weights, weights, eps=1e-6)
        embedding = MmapEngramEmbedding(store, prefix, tp_rank=0, tp_size=1)
        module = DeepseekV41Engram(embedding, wkv, weights, weights, eps=1e-6)
        x = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
        ids = torch.tensor([[0]])
        with pytest.raises(RuntimeError, match="rotation"):
            module(x, ids)
        module.set_rotation_contract("unrotated")
        assert module(x, ids).shape == x.shape


def test_decode_staging_reuses_address_and_clears_retired_request_rows(
    table_checkpoint,
):
    root, prefix = table_checkpoint
    with IndexedMmapStore(root) as store:
        module = DeepseekV41Engram(
            MmapEngramEmbedding(store, prefix, tp_rank=0, tp_size=1),
            nn.Linear(64, 6, bias=False, dtype=torch.bfloat16),
            torch.ones(2, 2),
            torch.ones(2, 2),
            eps=1e-6,
            rotation_contract="unrotated",
        )
        states = torch.arange(12, dtype=torch.bfloat16).reshape(3, 2, 2)
        staged = None
        addresses = []
        for ids, slots in (([[0, 1], [2, 1]], [2, 0]), ([[1, 2]], [1]), ([], [])):
            hash_ids = torch.tensor(ids, dtype=torch.long).reshape(-1, 2)
            slot_ids = torch.tensor(slots, dtype=torch.long)
            staged = module.prepare_decode(hash_ids, bucket_size=3, device="cpu", slots=slot_ids, out=staged)
            addresses.append(staged.data_ptr())
            valid = torch.zeros(3, dtype=torch.bool)
            valid[slot_ids] = True
            expected = states.clone()
            if slots:
                expected[slot_ids] = module(states[slot_ids], hash_ids)
            torch.testing.assert_close(module.forward_decode(states, staged, valid), expected, rtol=0, atol=0)
            assert torch.count_nonzero(staged[~valid]) == 0
        assert len(set(addresses)) == 1


def test_decode_reduction_never_mutates_rank_local_graph_input(table_checkpoint):
    root, prefix = table_checkpoint
    group = object()
    calls = []

    def reduce(values, supplied_group):
        assert supplied_group is group
        calls.append(values.shape)
        values.add_(1)

    with IndexedMmapStore(root) as store:
        module = DeepseekV41Engram(
            MmapEngramEmbedding(store, prefix, tp_rank=0, tp_size=2, tp_group=group),
            nn.Linear(32, 6, bias=False, dtype=torch.bfloat16),
            torch.ones(2, 2),
            torch.ones(2, 2),
            eps=1e-6,
            tp_reduce=reduce,
            rotation_contract="unrotated",
        )
        staged = module.prepare_decode(torch.tensor([[0], [4]]), bucket_size=2, device="cpu")
        local = staged.clone()
        x = torch.ones(2, 2, 2, dtype=torch.bfloat16)
        mask = torch.tensor([True, True])
        first = module.forward_decode(x, staged, mask)
        second = module.forward_decode(x, staged, mask)
        assert torch.equal(staged, local)
        assert torch.equal(first, second)
        assert calls == [(2, 1, 32), (2, 1, 32)]
        with pytest.raises(ValueError, match="unique"):
            module.prepare_decode(
                torch.tensor([[0], [1]]),
                bucket_size=2,
                device="cpu",
                slots=[1, 1],
                out=staged,
            )
        assert torch.equal(staged, local)
