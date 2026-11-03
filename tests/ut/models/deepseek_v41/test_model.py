# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from tests.ut.models.deepseek_v41 import paged_cache
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.models.deepseek_v41.attention import (
    DeepseekV41Cache,
    DeepseekV41DecodeMetadata,
    DeepseekV41Metadata,
    DeepseekV41Runtime,
    RequestSpan,
)
from vllm_ascend.models.deepseek_v41.engram import EngramHasher, EngramLayout


class TensorSlice:
    def __init__(self, value):
        self.value = value

    def get_shape(self):
        return self.value.shape

    def __getitem__(self, indices):
        return self.value[indices]


class TensorReader:
    def __init__(self, value):
        self.value = value

    def get_slice(self, key):
        return TensorSlice(self.value)


def unloaded_module(module_class):
    module = object.__new__(module_class)
    nn.Module.__init__(module)
    return module


@pytest.mark.parametrize(
    "module_class,key,shape,target_shape,expected_slice",
    [
        (ColumnParallelLinear, "w1.weight", (6, 2), (2, 2), (slice(2, 4), slice(None))),
        (
            ColumnParallelLinear,
            "w1.weight_scale",
            (6, 1),
            (2, 1),
            (slice(2, 4), slice(None)),
        ),
        (RowParallelLinear, "w2.weight", (2, 6), (2, 2), (slice(None), slice(2, 4))),
        (
            RowParallelLinear,
            "w2.weight_scale",
            (2, 1),
            (2, 1),
            (slice(None), slice(None)),
        ),
        (ReplicatedLinear, "w2.weight", (2, 6), (2, 6), (slice(None), slice(None))),
    ],
)
def test_tp_load_shards_correct_axis_but_keeps_ep_expert_full(module_class, key, shape, target_shape, expected_slice):
    source = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    target = torch.zeros(target_shape)
    model_module.read_parameter_slice(TensorReader(source), key, target, unloaded_module(module_class), 1, 3)
    torch.testing.assert_close(target, source[expected_slice])


def test_vocab_last_tp_rank_zeroes_padding_without_loading_extra_rows():
    module = unloaded_module(VocabParallelEmbedding)
    module.shard_indices = SimpleNamespace(org_vocab_start_index=3, org_vocab_end_index=5)
    source = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    target = torch.full((3, 2), float("nan"))
    model_module.read_parameter_slice(TensorReader(source), "embed.weight", target, module, 1, 2)
    torch.testing.assert_close(target[:2], source[3:])
    torch.testing.assert_close(target[2], torch.zeros(2))


@pytest.mark.parametrize("shape", [(5, 2), (6, 3)])
def test_invalid_tp_shape_cannot_silently_drop_or_reshape_weights(shape):
    with pytest.raises(ValueError, match="TP partition|shape mismatch"):
        model_module.read_parameter_slice(
            TensorReader(torch.zeros(shape)),
            "w1.weight",
            torch.zeros(2, 2),
            unloaded_module(ColumnParallelLinear),
            1,
            3,
        )


@pytest.fixture
def indexed_model(tmp_path, monkeypatch):
    model = unloaded_module(model_module.DeepseekV41ForCausalLM)
    model.sequence_parallel = False
    model.model_path = tmp_path
    model.engram_store = None
    model.engram_layer_ids = ()
    model.proj = nn.Linear(2, 2, bias=False)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 1)
    return model


def unexpected_weight_iterator():
    raise AssertionError("The generic shard iterator must not be materialized")
    yield


def write_index(root, weight_map):
    (root / "quant_model_weights.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def test_authoritative_index_wins_over_stale_duplicate_shard(indexed_model):
    root = indexed_model.model_path
    selected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    save_file({"proj.weight": selected}, str(root / "selected.safetensors"))
    save_file({"proj.weight": -selected}, str(root / "stale.safetensors"))
    write_index(root, {"proj.weight": "selected.safetensors"})
    loaded = indexed_model.load_weights(unexpected_weight_iterator())
    assert loaded == {"proj.weight"}
    torch.testing.assert_close(indexed_model.proj.weight.cpu(), selected)


def test_missing_required_parameter_fails_before_loading(indexed_model):
    write_index(indexed_model.model_path, {})
    with pytest.raises(KeyError, match="Required V4.1 parameter absent.*proj.weight"):
        indexed_model.load_weights(unexpected_weight_iterator())


def test_duplicate_key_in_authoritative_index_is_rejected(indexed_model):
    index = indexed_model.model_path / "quant_model_weights.safetensors.index.json"
    index.write_text('{"weight_map":{"proj.weight":"first.safetensors","proj.weight":"second.safetensors"}}')
    with pytest.raises(ValueError, match="Duplicate"):
        indexed_model.load_weights(unexpected_weight_iterator())


class IdleRecordingBlock(nn.Module):
    def __init__(self, cache):
        super().__init__()
        self.attn = nn.Module()
        self.attn.cache = cache
        self.engram = None
        self.local_token_counts = []

    def forward(self, hidden_states, pre_mix, runtime):
        self.local_token_counts.append(hidden_states.shape[0])
        return hidden_states, pre_mix


@pytest.fixture
def paged_hash_model():
    config = SimpleNamespace(
        engram_max_ngram_size=4,
        engram_layer_ids=[0, 1],
        engram_n_heads=2,
        engram_head_dim=32,
        engram_vocab_size=5,
        engram_num_embeddings=[200, 400],
        head_dim=32,
        index_head_dim=32,
        hidden_size=4,
        hc_mult=2,
    )
    # Non-identity raw->compressed mapping catches accidental double compression
    # when the model restores raw token history from scheduler-owned pages.
    token_map = tuple((index * 3 + 1) % 7 for index in range(20))
    hasher = EngramHasher(
        EngramLayout.from_config(config),
        token_map,
        compressed_vocab_size=7,
        pad_token_id=2,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=2),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache = DeepseekV41Cache(config, vllm_config, "hash-test.cache", token_history=True)
    paged_cache.allocate(cache, 20, 2)
    model = unloaded_module(model_module.DeepseekV41ForCausalLM)
    model.sequence_parallel = False
    model.config = config
    model.engram_hasher = hasher
    model.layers = nn.ModuleList([IdleRecordingBlock(cache)])
    model.embed = nn.Embedding(20, config.hidden_size)
    model.norm = nn.Identity()
    model.engram_layer_ids = tuple(config.engram_layer_ids)
    return model


def hash_batch(model, ids, spans, block_tables, image_mask=None):
    cache = model.layers[0].attn.cache
    tables = torch.tensor(block_tables, dtype=torch.int32)
    slots = []
    for request_idx, span in enumerate(spans):
        positions = torch.arange(span.start_pos, span.seq_len)
        slots.append(
            tables[request_idx][positions // cache.block_size] * cache.block_size + positions % cache.block_size
        )
    metadata = DeepseekV41Metadata(spans, tables, torch.cat(slots), ids.numel())
    runtime = DeepseekV41Runtime(spans)
    return model._hash_tokens(ids, runtime, metadata, image_mask), metadata


def test_model_hash_restores_bounded_raw_history_across_pages_and_reordered_batch(paged_hash_model, monkeypatch):
    model = paged_hash_model
    cache = model.layers[0].attn.cache
    hasher = model.engram_hasher
    a = torch.tensor([8, 3, 11, 6, 1, 9, 10, 5])
    b = torch.tensor([4, 10, 2, 12, 7, 0])
    expected_a = hasher.hash_tokens(a, hasher.new_state(), start_position=0)
    expected_b = hasher.hash_tokens(b, hasher.new_state(), start_position=0)
    reads = []
    original_read = cache.read

    def observe(metadata, request_idx, positions, bank, offset, width):
        reads.append((request_idx, positions.tolist(), bank, offset, width))
        return original_read(metadata, request_idx, positions, bank, offset, width)

    monkeypatch.setattr(cache, "read", observe)
    first, _ = hash_batch(
        model,
        torch.cat((a[:3], b[:2])),
        [RequestSpan(0, 3, 3), RequestSpan(3, 5, 2)],
        [[5, 2, 7, 9], [0, 4, 8, 11]],
    )
    torch.testing.assert_close(first, torch.cat((expected_a[:3], expected_b[:2])))
    assert reads == []
    second, _ = hash_batch(
        model,
        torch.cat((b[2:5], a[3:6])),
        [RequestSpan(0, 3, 5), RequestSpan(3, 6, 6)],
        [[0, 4, 8, 11], [5, 2, 7, 9]],
    )
    torch.testing.assert_close(second, torch.cat((expected_b[2:5], expected_a[3:6])))
    history_column = cache.dim + cache.index_dim
    assert reads == [
        (0, [0, 1], 1, history_column, 1),
        (1, [0, 1, 2], 1, history_column, 1),
    ]
    # Physically move both requests' pages, then poison obsolete slots. The
    # restored state must use the current request table rather than batch index.
    for source, destination in ((5, 13), (2, 10), (7, 15), (0, 12), (4, 14), (8, 16)):
        for bank in cache.kv_cache:
            bank[destination].copy_(bank[source])
            bank[source].fill_(999)
    # Earlier tokens are deliberately unreadable: only the last three raw IDs
    # may be needed to resume 4-gram hashing.
    cache.field(1, history_column, 1).reshape(-1, 2)[13] = 999
    cache.field(1, history_column, 1).reshape(-1, 2)[10, 0] = 999
    cache.field(1, history_column, 1).reshape(-1, 2)[12] = 999
    third, metadata = hash_batch(
        model,
        torch.cat((a[6:], b[5:])),
        [RequestSpan(0, 2, 8), RequestSpan(2, 3, 6)],
        [[13, 10, 15, 17], [12, 14, 16, 18]],
    )
    torch.testing.assert_close(third, torch.cat((expected_a[6:], expected_b[5:])))
    assert reads[2:] == [
        (0, [3, 4, 5], 1, history_column, 1),
        (1, [2, 3, 4], 1, history_column, 1),
    ]
    # The metadata slot mapping stores raw IDs, not compressed vocabulary IDs.
    stored = cache.field(1, history_column, 1)[metadata.slot_mapping.long(), 0]
    torch.testing.assert_close(stored, torch.cat((a[6:], b[5:])).float())


def test_idle_dp_hash_has_valid_empty_shape_and_still_visits_model_blocks(
    paged_hash_model,
    monkeypatch,
):
    model = paged_hash_model
    metadata = DeepseekV41Metadata(
        [],
        torch.empty((0, 0), dtype=torch.int32),
        torch.empty(0, dtype=torch.int64),
        0,
    )
    model.layers[0].attn.cache.metadata = lambda: metadata
    hashes = model._hash_tokens(torch.empty(0, dtype=torch.int64), DeepseekV41Runtime([]), metadata)
    assert hashes.shape == (0, 2, 6)
    assert hashes.dtype == torch.int64
    assert hashes.device.type == "cpu"

    def reject_empty_dense(*args, **kwargs):
        raise AssertionError("Idle local-DP inputs must not enter dense kernels")

    monkeypatch.setattr(model.embed, "forward", reject_empty_dense)
    monkeypatch.setattr(model.norm, "forward", reject_empty_dense)
    # A padded idle-DP input must still enter each block: peer DP ranks may have
    # real tokens and their MoE collectives require this rank's participation.
    output = model(torch.tensor([0]), torch.tensor([0]))
    assert model.layers[0].local_token_counts == [0]
    torch.testing.assert_close(output, torch.zeros(1, model.config.hidden_size))


def test_idle_block_still_calls_moe_without_entering_attention_or_hc():
    block = unloaded_module(model_module.DeepseekV41Block)
    block.config = SimpleNamespace(hidden_size=4)
    calls = []

    class RecordFFN(nn.Module):
        def forward(self, value):
            calls.append(value.shape)
            return value

    block.ffn = RecordFFN()
    # No attention/HC parameters are installed: touching those paths must fail.
    hidden = torch.empty(0, 2, 4, dtype=torch.bfloat16)
    pre = torch.empty(0, 2, dtype=torch.float32)
    result, next_pre = block(hidden, pre, DeepseekV41Runtime([]))
    assert calls == [torch.Size([0, 4])]
    assert result is hidden and next_pre is pre


def test_engram_query_unrotation_uses_transpose_and_returns_fp32():
    # Non-symmetric orthogonal Q distinguishes x@Q.T from x@Q.
    matrix = torch.tensor([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
    original = torch.tensor([[[1.125, 2.25, -3.5], [5, -6, 7.75]]])
    rotated = (original @ matrix).bfloat16()
    transform = model_module.EngramQueryUnrotation(matrix)
    restored = transform(rotated)
    assert restored.dtype == torch.float32
    assert transform.matrix.dtype == torch.float32
    assert transform.state_dict() == {}
    torch.testing.assert_close(restored, original, rtol=0, atol=0)
    assert not torch.equal(restored, rotated.float() @ matrix)


def test_engram_query_unrotation_rejects_low_precision_matrix_at_construction():
    with pytest.raises(ValueError, match="FP32"):
        model_module.EngramQueryUnrotation(torch.eye(3, dtype=torch.bfloat16))


def test_engram_query_unrotation_rejects_later_module_downcast():
    transform = model_module.EngramQueryUnrotation(torch.eye(3))
    transform.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="retain FP32"):
        transform(torch.ones(2, 3, dtype=torch.bfloat16))


@pytest.fixture
def staged_hash_model(paged_hash_model):
    model = paged_hash_model
    cache = model.layers[0].attn.cache

    class Rows(nn.Module):
        def prepare_decode(self, ids, *, bucket_size, slots, device, out):
            if out is None:
                out = torch.zeros(
                    bucket_size,
                    model.engram_hasher.layout.hash_columns,
                    dtype=torch.long,
                )
            out.zero_()
            if ids.shape[0]:
                out[slots] = ids
            return out

    model.layers.append(IdleRecordingBlock(cache))
    for layer in model.layers:
        layer.engram = Rows()
    model._decode_capture_sizes = frozenset({1, 2, 4, 8})
    model._decode_engram_values = {}
    return model


def test_graph_engram_refreshes_paged_history_and_preserves_idle_cache(staged_hash_model):
    model = staged_hash_model
    cache = model.layers[0].attn.cache
    history = torch.tensor([3, 4, 7])
    _, old = hash_batch(model, history, [RequestSpan(0, 3, 3)], [[4, 7, 2]])
    md = DeepseekV41Metadata([], old.block_table_tensor, torch.tensor([15]), 1)
    md.decode = DeepseekV41DecodeMetadata(
        torch.tensor([4]), md.slot_mapping, md.block_table_tensor, torch.tensor([True]), 6
    )
    cache.metadata = lambda: md
    for position, token, slot in [(3, 9, 15), (4, 2, 4)]:
        md.slot_mapping.fill_(slot)
        md.decode.seq_lens.fill_(position + 1)
        assert model.prepare_decode_graph_inputs(torch.tensor([token]), torch.tensor([position])) == 1
        history = torch.cat([history, torch.tensor([token])])
        expected = model.engram_hasher.hash_tokens(history, model.engram_hasher.new_state(), start_position=0)
        for (_, layer), rows in model._decode_engram_values.items():
            torch.testing.assert_close(rows, expected[-1:, layer])
        if position == 3:
            address = model._decode_engram_values[1, 0].data_ptr()
        assert address == model._decode_engram_values[1, 0].data_ptr()
    before = paged_cache.snapshot(cache)
    md.decode.active.zero_()
    md.slot_mapping.fill_(-1)
    assert model.prepare_decode_graph_inputs(torch.tensor([0]), torch.tensor([0])) == 0
    assert model._decode_engram_values[1, 0].count_nonzero() == 0
    torch.testing.assert_close(paged_cache.bits(cache), before)


@pytest.mark.parametrize("sizes", [[1], [1, 2, 4, 8], list(range(1, 9)), [3, 7]])
def test_decode_capture_accepts_configured_subsets(sizes):
    config = SimpleNamespace(cudagraph_mode=model_module.CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=sizes)
    assert model_module._decode_capture_sizes(config) == frozenset(sizes)


@pytest.mark.parametrize("sizes", [[], [0], [1, 9], [1, 1], [True]])
def test_decode_capture_rejects_unsupported_buckets(sizes):
    config = SimpleNamespace(cudagraph_mode=model_module.CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=sizes)
    with pytest.raises(ValueError, match="capture sizes"):
        model_module._decode_capture_sizes(config)


def test_multirow_engram_keeps_per_bucket_storage_and_request_owned_history(staged_hash_model):
    model = staged_hash_model
    cache = model.layers[0].attn.cache
    histories = [[8, 3, 11], [4, 10], [9]]
    tables = [[5, 2, 7, 9], [0, 4, 8, 11], [1, 3, 6, 10]]
    for ids, table in zip(histories, tables):
        hash_batch(model, torch.tensor(ids), [RequestSpan(0, len(ids), len(ids))], [table])
    addresses = {}
    # Switch physical row order and bucket sizes. Inactive rows deliberately
    # contain invalid IDs/positions and slot -1; they must touch no cache row.
    batches = [
        ([0], [6]),
        ([1, None, 0, 2], [2, 99, 1, 15]),
        ([None, 2], [99, 4]),
        ([2, 1, None, 0], [7, 12, 99, 9]),
        ([0], [10]),
        ([None, 1, None, None, None, None, 2, None], [99, 7, 99, 99, 99, 99, 3, 99]),
        ([None] * 4, [99] * 4),
    ]
    for assignments, tokens in batches:
        bucket = len(assignments)
        previous_buckets = {
            key: value.clone() for key, value in model._decode_engram_values.items() if key[0] != bucket
        }
        positions = [len(histories[i]) if i is not None else 99 for i in assignments]
        block_table = torch.tensor([tables[i] if i is not None else [0] * 4 for i in assignments], dtype=torch.int32)
        live = torch.tensor([i is not None for i in assignments])
        slots = torch.tensor(
            [tables[i][pos // 2] * 2 + pos % 2 if i is not None else -1 for i, pos in zip(assignments, positions)]
        )
        lengths = torch.where(live, torch.tensor(positions) + 1, 0)
        decode = DeepseekV41DecodeMetadata(lengths, slots, block_table, live, 8)
        metadata = DeepseekV41Metadata([], block_table, slots, int(live.sum()), decode)
        cache.metadata = lambda current=metadata: current
        expected_cache = paged_cache.snapshot(cache)
        assert model.prepare_decode_graph_inputs(torch.tensor(tokens), torch.tensor(positions)) == int(live.sum())
        for row, (request, token) in enumerate(zip(assignments, tokens)):
            if request is None:
                for layer in model.engram_layer_ids:
                    assert model._decode_engram_values[bucket, layer][row].count_nonzero() == 0
                continue
            histories[request].append(token)
            expected = model.engram_hasher.hash_tokens(
                histories[request], model.engram_hasher.new_state(), start_position=0
            )[-1]
            expected_cache[1].view(torch.float32).reshape(-1)[slots[row]] = token
            for layer in model.engram_layer_ids:
                torch.testing.assert_close(model._decode_engram_values[bucket, layer][row], expected[layer])
        torch.testing.assert_close(paged_cache.bits(cache), expected_cache)
        for key, value in model._decode_engram_values.items():
            assert addresses.setdefault(key, value.data_ptr()) == value.data_ptr()
        for key, old in previous_buckets.items():
            torch.testing.assert_close(model._decode_engram_values[key], old)


def test_graph_reports_live_rows_without_engram(staged_hash_model):
    model = staged_hash_model
    model.engram_hasher = None
    cache = model.layers[0].attn.cache
    slots = torch.tensor([5, -1, 7, -1])
    table = torch.zeros(4, 2, dtype=torch.int32)
    live = torch.tensor([True, False, True, False])
    decode = DeepseekV41DecodeMetadata(torch.tensor([1, 0, 1, 0]), slots, table, live, 4)
    cache.metadata = lambda: DeepseekV41Metadata([], table, slots, 2, decode)
    before = paged_cache.snapshot(cache)
    assert model.prepare_decode_graph_inputs(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)) == 2
    torch.testing.assert_close(paged_cache.bits(cache), before)


def test_sequence_parallel_hc_keeps_local_rows_and_gathers_only_unpadded_tokens(paged_hash_model, monkeypatch):
    model = paged_hash_model
    model.sequence_parallel = True
    rows = torch.arange(5 * model.config.hidden_size, dtype=torch.float32).view(5, -1)
    padded = torch.cat((rows, rows.new_zeros(1, rows.shape[1])))
    # Emulate the second contiguous shard: two real rows and one pad row.
    monkeypatch.setattr(model_module, "sp_shard", lambda x: padded[3:])
    observed = []

    def gather(local):
        torch.testing.assert_close(local, padded[3:])
        observed.append(local.shape)
        return padded

    monkeypatch.setattr(model_module, "sp_all_gather", gather)
    local, mix = model._start_hc(rows)
    assert local.shape == (3, model.config.hc_mult, model.config.hidden_size)
    torch.testing.assert_close(model._finish_hc(local, mix, len(rows)), rows)
    assert observed == [(3, model.config.hidden_size)]


def test_dspark_graph_buckets_and_aux_layer_input(paged_hash_model):
    config = SimpleNamespace(
        cudagraph_mode=model_module.CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[24, 48]
    )
    assert model_module._decode_capture_sizes(config, 6) == {24, 48}
    config.cudagraph_capture_sizes = [8]
    with pytest.raises(ValueError, match="width 6"):
        model_module._decode_capture_sizes(config, 6)
    model = paged_hash_model
    model.config.dspark_target_layer_ids = [0, 1]
    model.set_aux_hidden_state_layers((1, 2))
    with pytest.raises(ValueError, match="projection order"):
        model.set_aux_hidden_state_layers((2, 1))
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
    auxiliary = []
    for layer_id in range(3):
        model._capture_aux(auxiliary, layer_id, hidden + layer_id, 3)
    assert len(auxiliary) == 2
    for layer_id, captured in enumerate(auxiliary):
        torch.testing.assert_close(captured, hidden.mean(dim=1) + layer_id)


def test_dspark_graph_hash_discards_rejected_suffix(staged_hash_model):
    model = staged_hash_model
    model._decode_capture_sizes = {6}
    cache = model.layers[0].attn.cache
    table = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
    previous = []
    for start, tokens in [(0, [2, 6, 8, 9, 12, 3]), (2, [3, 1, 4, 7, 18, 9])]:
        positions = torch.arange(start, start + 6)
        decode = DeepseekV41DecodeMetadata(
            positions + 1, positions, table.expand(6, -1), torch.ones(6, dtype=torch.bool), 12, 6
        )
        md = DeepseekV41Metadata([], table, positions, 6, decode)
        cache.metadata = lambda md=md: md
        assert model.prepare_decode_graph_inputs(torch.tensor(tokens), positions) == 6
        previous = previous[:start] + tokens
        expected = model.engram_hasher.hash_tokens(previous, model.engram_hasher.new_state(), start_position=0)[-6:]
        for layer_id in model.engram_layer_ids:
            torch.testing.assert_close(model._decode_engram_values[6, layer_id], expected[:, layer_id])


def test_image_prefix_keeps_engram_boundary_after_request_eviction(paged_hash_model):
    model = paged_hash_model
    prefix = torch.tensor([1, 4, 4, 4])
    flags = torch.tensor([False, True, True, True])
    _, metadata = hash_batch(model, prefix, [RequestSpan(0, 4, 4)], [[3, 7]], flags)
    cache = model.layers[0].attn.cache
    history_column = cache.dim + cache.index_dim
    stored = cache.read(metadata, 0, torch.arange(4), 1, history_column, 1).flatten().long()
    assert stored.tolist() == [1, -1, -1, -1]
    # A fresh request reuses the full physical prefix, then adds text whose
    # token id happens to equal the one used for the previous image.
    suffix = torch.tensor([4, 2, 3])
    actual, _ = hash_batch(model, suffix, [RequestSpan(0, 3, 7)], [[3, 7, 1, 2]])
    expected = model.engram_hasher.hash_tokens(
        torch.cat((prefix, suffix)),
        model.engram_hasher.new_state(),
        start_position=0,
        token_mask=[True, False, False, False, True, True, True],
    )
    torch.testing.assert_close(actual, expected[-3:])


def test_embedding_merge_uses_scheduler_span_not_vocabulary_id(paged_hash_model):
    model = paged_hash_model
    ids = torch.tensor([3, 3, 3, 3])
    original = model.embed_input_ids(ids).clone()
    features = torch.arange(8).float().reshape(2, 4)
    mask = torch.tensor([False, True, True, False])
    merged = model.embed_input_ids(ids, [features], is_multimodal=mask)
    torch.testing.assert_close(merged[mask], features)
    torch.testing.assert_close(merged[~mask], original[~mask])
    with pytest.raises(ValueError, match="span mask"):
        model.embed_input_ids(ids, [features])
