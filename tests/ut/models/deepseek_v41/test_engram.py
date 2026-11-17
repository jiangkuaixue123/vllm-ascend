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


def test_request_vocabulary_identity_preserves_equal_maps_and_rejects_different_maps():
    left, right = make_hasher(), make_hasher()
    state = left.new_state()
    expected = left.hash_tokens([2, 3], state.fork(), start_position=0)
    assert torch.equal(right.hash_tokens([2, 3], state.fork(), start_position=0), expected)
    different = EngramHasher(left.layout, (0, 2, 1, 2, 3), compressed_vocab_size=4, pad_token_id=0)
    with pytest.raises(ValueError, match="cursor/history"):
        different.hash_tokens([2], state, start_position=0)
    assert state.position == 0 and state.tail == ()


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


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("decode", [False, True])
def test_sequence_sharded_gate_reduces_table_rows_to_their_token_owner(table_checkpoint, monkeypatch, decode, rank):
    root, prefix = table_checkpoint
    # Alternate between the two vocabulary owners. Padding is a token row,
    # not an extra lookup in the vocabulary.
    ids = torch.tensor([[0], [4], [1]])
    states = torch.arange(12, dtype=torch.bfloat16).view(3, 2, 2) / 7
    padded = torch.cat((states, states.new_zeros(1, 2, 2)))
    projection = nn.Linear(32, 6, bias=False, dtype=torch.bfloat16)
    weights = torch.ones(2, 2, dtype=torch.bfloat16)
    with IndexedMmapStore(root) as store:
        full = DeepseekV41Engram(
            MmapEngramEmbedding(store, prefix, tp_rank=0, tp_size=1),
            projection,
            weights,
            weights,
            eps=1e-6,
            rotation_contract="unrotated",
        )
        oracle = full(states, ids)
        expected = torch.cat((oracle, oracle.new_zeros(1, 2, 2)))
        begin = 2 * rank
        part = DeepseekV41Engram(
            MmapEngramEmbedding(store, prefix, tp_rank=rank, tp_size=2),
            projection,
            weights,
            weights,
            eps=1e-6,
            rotation_contract="unrotated",
            tp_reduce=lambda *args: pytest.fail("SP must reduce-scatter to token owners"),
        )
        raw = part.embedding(ids, device="cpu")
        full_vals = full.embedding(ids, device="cpu")
        calls = []

        def reduce_scatter(values):
            calls.append(tuple(values.shape))
            torch.testing.assert_close(values[:3], raw.flatten(1))
            if values.shape[0] == 4:
                assert not values[3].count_nonzero()
            summed = torch.cat((full_vals, full_vals.new_zeros(1, 1, 32))).flatten(1)
            return summed[begin : begin + 2]

        monkeypatch.setattr("vllm_ascend.models.deepseek_v41.engram.sp_reduce_scatter", reduce_scatter)
        if decode:
            staged = part.prepare_decode(ids, bucket_size=4, device="cpu")
            before = staged.clone()
            mask = torch.tensor([True, True, True, False])[begin : begin + 2]
            result = part.forward_decode(padded[begin : begin + 2], staged, mask, sequence_parallel=True)
            torch.testing.assert_close(staged, before)
        else:
            result = part(padded[begin : begin + 2], ids, sequence_parallel=True)
        torch.testing.assert_close(result, expected[begin : begin + 2])
        assert calls == [(4 if decode else 3, 32)]


@pytest.mark.parametrize(
    "dtype,width",
    [
        ("I8", 1),
        ("U8", 1),
        ("BF16", 2),
        ("F16", 2),
        ("F32", 4),
        ("I32", 4),
        ("I64", 8),
        ("F8_E8M0", 1),
        ("F8_E4M3", 1),
        ("F8_E5M2", 1),
    ],
)
def test_batch_row_read_preserves_raw_bytes_order_duplicates_and_lifetime(tmp_path, dtype, width):
    data = bytes(range(4 * 5 * width))
    write_shard(
        tmp_path / "rows.safetensors",
        {"prefix": ("U8", (1, 3), b"abc"), "rows": (dtype, (4, 5), data)},
    )
    write_index(tmp_path, {"rows": "rows.safetensors"})
    with IndexedMmapStore(tmp_path, max_rows_per_read=4) as store:
        selected = store.read_rows("rows", [3, 0, 3, 1])
        assert (
            selected
            == data[15 * width : 20 * width]
            + data[: 5 * width]
            + data[15 * width : 20 * width]
            + data[5 * width : 10 * width]
        )
        assert store.read_rows("rows", []) == b""
        assert store.read_rows("rows", [0, 1, 2, 3]) == data
        for bad in ([True], [-1], [4], [1.0], [None]):
            with pytest.raises(ValueError, match="Row outside"):
                store.read_rows("rows", bad)
    # Owned bytes must survive close without retaining an mmap view.
    assert len(selected) == 4 * 5 * width


def test_batch_row_read_accepts_zero_width_after_nonempty_payload(tmp_path):
    write_shard(
        tmp_path / "rows.safetensors",
        {"prefix": ("U8", (1, 3), b"abc"), "rows": ("BF16", (3, 0), b"")},
    )
    write_index(tmp_path, {"rows": "rows.safetensors"})
    with IndexedMmapStore(tmp_path) as store:
        assert store.read_rows("rows", [2, 0, 2]) == b""


@torch.inference_mode()
def test_batched_upload_matches_native_device_for_all_int8_e8m0_codes(tmp_path, monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    weights = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).repeat(256, 1)
    scales = torch.arange(256, dtype=torch.uint8).unsqueeze(1).expand(-1, 8).contiguous()
    expected = decode_int8_e8m0_rows(weights.npu(), scales.npu()).cpu()
    write_shard(
        tmp_path / "all-codes.safetensors",
        {
            "table.weight": ("I8", (256, 256), weights.numpy().tobytes()),
            "table.scale": ("F8_E8M0", (256, 8), scales.numpy().tobytes()),
        },
    )
    write_index(tmp_path, {"table.weight": "all-codes.safetensors", "table.scale": "all-codes.safetensors"})
    from vllm_ascend.ops.triton import deepseek_v41_engram_upload as upload

    observed = []
    native_upload = upload.try_upload_engram_rows

    def record_upload(*args):
        used = native_upload(*args)
        observed.append(used)
        return used

    monkeypatch.setattr(upload, "try_upload_engram_rows", record_upload)
    # One code per chunk also covers the native packet's single-row tail.
    with IndexedMmapStore(tmp_path, max_rows_per_read=1) as store:
        embedding = MmapEngramEmbedding(store, "table", tp_rank=0, tp_size=1, chunk_rows=1)
        actual = embedding(torch.arange(256), device="npu").cpu()
    different = actual.view(torch.int16) != expected.view(torch.int16)
    assert not different.any(), torch.nonzero(different)[:16].tolist()
    assert len(observed) == 256 and all(observed)


@pytest.mark.parametrize("tp,rank", [(1, 0), (4, 0), (4, 3), (8, 0), (8, 7)])
@pytest.mark.parametrize("bucket", [24, 48])
def test_mmap_decode_upload_refreshes_stable_graph_input(table_checkpoint, tp, rank, bucket):
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")

    def unexpected_reduce(*args):
        raise AssertionError("Staging local rows must not communicate")

    root, prefix = table_checkpoint
    raw_weights = torch.tensor([1, 2, 3, -1, -2], dtype=torch.int8).unsqueeze(1).expand(-1, 32).contiguous()
    raw_scales = torch.tensor([127, 128, 126, 255, 0], dtype=torch.uint8).unsqueeze(1)
    table = decode_int8_e8m0_rows(raw_weights.npu(), raw_scales.npu()).cpu()
    with IndexedMmapStore(root, max_rows_per_read=2) as store:
        embedding = MmapEngramEmbedding(store, prefix, tp_rank=rank, tp_size=tp, chunk_rows=2)
        module = DeepseekV41Engram(
            embedding,
            nn.Linear(64, 6, bias=False, dtype=torch.bfloat16),
            torch.ones(2, 2),
            torch.ones(2, 2),
            eps=1e-6,
            tp_reduce=unexpected_reduce,
            rotation_contract="unrotated",
        )
        out = None
        graph = None
        for step in range(64):
            count = step % 3
            ids = ((torch.arange(count * 2) + step) % 5).reshape(count, 2)
            slots = [bucket - 1, 0][:count]
            out = module.prepare_decode(ids, bucket_size=bucket, device="npu", slots=slots, out=out)
            expected = torch.zeros(bucket, 2, 32, dtype=torch.bfloat16)
            if count:
                values = table[ids].clone()
                local = (ids >= embedding.start_row) & (ids < embedding.end_row)
                values[~local] = 0
                expected[slots] = values
            assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16))
            if graph is None:
                address = out.data_ptr()
                _ = out.clone()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = out.clone()
            assert out.data_ptr() == address
            graph.replay()
            assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("tp,rank", [(1, 0), (4, 0), (4, 3), (8, 0), (8, 7)])
@pytest.mark.parametrize("bucket", [24, 48])
@torch.inference_mode()
def test_packet_mmap_upload_keeps_shards_holes_and_graph_buffers(tmp_path, tp, rank, bucket):
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    weights = torch.arange(16 * 256, dtype=torch.int16).to(torch.int8).reshape(16, 256)
    scales = torch.arange(16 * 8, dtype=torch.int16).to(torch.uint8).reshape(16, 8) + 65
    scales[3].fill_(255)
    scales[4].zero_()
    write_shard(
        tmp_path / "packet.safetensors",
        {
            "table.weight": ("I8", (16, 256), weights.numpy().tobytes()),
            "table.scale": ("F8_E8M0", (16, 8), scales.numpy().tobytes()),
        },
    )
    write_index(tmp_path, {"table.weight": "packet.safetensors", "table.scale": "packet.safetensors"})
    table = decode_int8_e8m0_rows(weights.npu(), scales.npu()).cpu()

    def unexpected_reduce(*args):
        raise AssertionError("Staging local rows must not communicate")

    with IndexedMmapStore(tmp_path, max_rows_per_read=3) as store:
        embedding = MmapEngramEmbedding(store, "table", tp_rank=rank, tp_size=tp, chunk_rows=3)
        module = DeepseekV41Engram(
            embedding,
            nn.Linear(64, 6, bias=False, dtype=torch.bfloat16),
            torch.ones(2, 2),
            torch.ones(2, 2),
            eps=1e-6,
            tp_reduce=unexpected_reduce,
            rotation_contract="unrotated",
        )
        out = None
        graph = None
        for step in range(64):
            count = step % 4
            ids = ((torch.arange(count * 2) + step) % 16).reshape(count, 2)
            slots = [bucket - 1, 0, bucket // 2][:count]
            out = module.prepare_decode(ids, bucket_size=bucket, device="npu", slots=slots, out=out)
            expected = torch.zeros(bucket, 2, 256, dtype=torch.bfloat16)
            if count:
                values = table[ids].clone()
                local = (ids >= embedding.start_row) & (ids < embedding.end_row)
                values[~local] = 0
                expected[slots] = values
            assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16))
            if graph is None:
                address = out.data_ptr()
                _ = out.clone()
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = out.clone()
            assert out.data_ptr() == address
            graph.replay()
            torch.npu.synchronize()
            assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_large_packet_keeps_permuted_unique_destinations():
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from vllm_ascend.ops.triton.deepseek_v41_engram_upload import try_upload_engram_rows

    weights = torch.arange(4096 * 256, dtype=torch.int32).to(torch.int8).reshape(4096, 256)
    scales = torch.arange(4096 * 8, dtype=torch.int32).to(torch.uint8).reshape(4096, 8)
    positions = list(range(8190, -1, -2))
    output = torch.zeros(8192, 256, dtype=torch.bfloat16, device="npu")
    assert try_upload_engram_rows(weights.numpy().tobytes(), scales.numpy().tobytes(), positions, output)
    expected = torch.zeros_like(output)
    expected[torch.tensor(positions, device="npu")] = decode_int8_e8m0_rows(weights.npu(), scales.npu())
    assert torch.equal(output.cpu().view(torch.int16), expected.cpu().view(torch.int16))


@pytest.mark.parametrize("count", [127, 128, 129, 512, 8192])
@pytest.mark.parametrize("position", [0, 7])
@pytest.mark.parametrize("masked", [False, True])
def test_bulk_hash_matches_chunked_integer_reference(count, position, masked):
    layout = EngramLayout.from_config(
        layout_config(
            engram_layer_ids=(1, 14),
            engram_max_ngram_size=4,
            engram_n_heads=8,
            engram_vocab_size=16000000,
            engram_num_embeddings=(1000000000, 1000000000),
        )
    )
    hasher = EngramHasher(layout, tuple(range(257)), compressed_vocab_size=257, pad_token_id=0)
    tokens = (torch.arange(count * 2, dtype=torch.long) % 257)[::2]
    mask = [not masked or i % 17 != 0 for i in range(count)]
    history = [3, 0, 255][-min(position, 3) :] if position else []
    state = hasher.new_state(
        history, position=position, history_mask=[False] + [True] * (len(history) - 1) if masked and history else None
    )
    reference = state.fork()
    expected = []
    for start in range(0, count, 127):
        expected.append(
            hasher.hash_tokens(
                tokens[start : start + 127],
                reference,
                start_position=reference.position,
                token_mask=mask[start : start + 127],
            )
        )
    actual = hasher.hash_tokens(tokens, state, start_position=position, token_mask=mask)
    assert actual.dtype == torch.int64 and actual.device.type == "cpu"
    assert torch.equal(actual, torch.cat(expected))
    assert state == reference
    snapshot = tuple(state.tail)
    actual.zero_()
    assert state.tail == snapshot


def test_bulk_hash_malformed_layout_keeps_reference_error_and_state():
    layout = EngramLayout(3, (1,), (36,), (((5, 7), (11,)),), 2, 32)
    hasher = EngramHasher(layout, (0, 1, 2, 3), compressed_vocab_size=4, pad_token_id=0)
    state = hasher.new_state()
    with pytest.raises(RuntimeError):
        hasher.hash_tokens([1] * 128, state, start_position=0)
    assert state.position == 0 and state.tail == ()


@torch.inference_mode()
def test_runtime_packet_rows_share_one_compiled_kernel():
    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")
    from vllm_ascend.ops.triton.deepseek_v41_engram_upload import _unpack_engram_rows, try_upload_engram_rows

    observed = []
    for count in (1, 2, 3, 5, 34, 35, 36, 37, 38, 71, 72, 73, 143, 144, 145, 4095, 4096):
        weights = torch.arange(count * 256, dtype=torch.int32).to(torch.int8).reshape(count, 256)
        scales = torch.full((count, 8), 127, dtype=torch.uint8)
        output = torch.zeros(count + 2, 256, dtype=torch.bfloat16, device="npu")
        assert try_upload_engram_rows(weights.numpy().tobytes(), scales.numpy().tobytes(), list(range(count)), output)
        torch.npu.synchronize()
        assert torch.equal(output[:count].cpu(), weights.bfloat16())
        assert not output[count:].count_nonzero()
        observed.append(sum(len(cache) for cache in _unpack_engram_rows.cache.values()))
    assert len(set(observed)) == 1, observed
    print("PACKET_RUNTIME_ROW_CACHE_STABLE", observed[0], flush=True)
