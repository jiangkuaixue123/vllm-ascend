# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.models.hy_v4_flash.configs import SLIDING_ATTENTION, SPARSE_ATTENTION, HYV4FlashConfig, HYV4FlashVLConfig
from vllm.models.hy_v4_flash.reference import (
    decode_e4m3fn,
    hash_ngrams,
    indexpool_fp8,
    mla_attention_probs,
    sparse_candidates,
    window_candidates,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES, dequantize_c8_rows, pack_c8_components
from vllm_ascend.models.hy_v4_flash import greedy_sampling
from vllm_ascend.models.hy_v4_flash.aclgraph import FlashAclGraphManager
from vllm_ascend.models.hy_v4_flash.attention import AscendHYV4FlashAttention, FlashMetadataBuilder
from vllm_ascend.models.hy_v4_flash.cache_pages import PAGE_BYTES, PAGE_TOKENS
from vllm_ascend.models.hy_v4_flash.dfly_speculator import build_reuse_indices
from vllm_ascend.models.hy_v4_flash.layer_recall import paired_recall_layer, recall_rows_for_forward
from vllm_ascend.models.hy_v4_flash.prefill_c8 import apply_attention_sink, prefill_window_indices


def test_gpu_c8_scale_packs_per_tile_fp8_and_preserves_rope():
    attention = AscendHYV4FlashAttention.__new__(AscendHYV4FlashAttention)
    attention.use_fp8_cache = True
    attention.use_gpu_c8_scale = True
    latent = torch.zeros((1, 512), dtype=torch.bfloat16)
    latent[0, :2] = torch.tensor([3, 1])
    latent[0, 128:131] = torch.tensor([-7, 2, -1])
    latent[0, 384] = 2**-20
    rope = torch.arange(64, dtype=torch.bfloat16).unsqueeze(0)

    packed = attention._pack_kv(latent, rope)
    _, _, scales = packed.split((512, 128, 16), dim=-1)
    expected_scales = torch.tensor([[2**-7, 2**-6, 2**-13, 2**-13]])
    torch.testing.assert_close(scales.view(torch.float32), expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(dequantize_c8_rows(packed), torch.cat((latent, rope), dim=-1).float(), rtol=0, atol=0)


def test_flat_vl_config_resolves_architecture_and_per_layer_geometry():
    config = HYV4FlashConfig.from_dict(
        {
            "model_type": "hy_v4_flash",
            "architectures": ["HYV4FlashForConditionalGeneration"],
            "hidden_size": 3072,
            "num_hidden_layers": 2,
            "layer_types": ["sparse", "sliding_attention"],
            "num_attention_heads": 32,
            "swa_num_attention_heads": 64,
            "vision_config": {"hidden_size": 1152},
            "rope_parameters": {
                "sparse": {"rope_theta": 10000},
                "sliding_attention": {"rope_theta": 20000},
            },
        },
    )
    assert isinstance(config, HYV4FlashVLConfig)
    assert config.architectures == ["HYV4FlashVLForConditionalGeneration"]
    text = config.get_text_config()
    assert text.layer_types == [SPARSE_ATTENTION, SLIDING_ATTENTION]
    assert text.rope_for_layer(1)["rope_theta"] == 20000
    assert text.rope_for_layer(0)["mrope_section"] == [12, 10, 10]
    assert text.swa_num_attention_heads == 64
    assert config.vision_config.out_hidden_size == 3072


def test_indexpool_fp8_matches_cast_after_hadamard_transform():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(8, 128, generator=generator).bfloat16()
    matrix = torch.ones((1, 1))
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    transformed = x.float() @ matrix * 128**-0.5
    step = torch.exp2(torch.ceil(torch.log2(transformed.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    expected = (transformed / step).to(torch.float8_e4m3fn).float() * step
    torch.testing.assert_close(indexpool_fp8(x), expected, atol=0, rtol=0)


def test_indexpool_scores_relu_before_weighted_reduction_and_masks_future_groups():
    queries = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]).expand(2, -1, -1)
    keys = torch.tensor([[2.0, -100.0], [1.0, 0.0], [500.0, 500.0]])
    result = sparse_candidates(queries, torch.ones(2, 2), keys, torch.tensor([0, 9]), 4, 1)
    assert result.tolist() == [[-1, -1, -1, -1, 0, -1, -1, -1], [0, 1, 2, 3, 8, 9, -1, -1]]


def test_indexpool_decode_batch_matches_per_request_and_masks_padding():
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(3, 4, 8, generator=generator)
    weights = torch.randn(3, 4, generator=generator)
    keys = torch.randn(3, 5, 8, generator=generator)
    positions = torch.tensor([19, 6, -1])
    selected = sparse_candidates(query, weights, keys, positions, 4, 2)
    expected = torch.cat(
        [sparse_candidates(query[i : i + 1], weights[i : i + 1], keys[i], positions[i : i + 1], 4, 2) for i in range(3)]
    )
    torch.testing.assert_close(selected, expected)
    assert (selected[-1] < 0).all()


def test_flash_decode_graph_dispatch_pads_uniform_decode_batches_only():
    manager = FlashAclGraphManager.__new__(FlashAclGraphManager)
    manager._graphs_captured = True
    graph = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=3, num_reqs=3, uniform_token_count=1)
    verify = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=6, num_reqs=1, uniform_token_count=6)
    manager._candidates = {}
    manager._decode_descs = {(1, 0, 1): [graph] * 4, (6, 0, 1): [verify] * 2}
    manager.graphs = {graph: object(), verify: object()}

    assert manager.dispatch(3, 3, 1, 0) is graph
    assert manager.dispatch(1, 6, 6, 0) is verify
    assert manager.dispatch(2, 6, 3, 0).cg_mode == CUDAGraphMode.NONE
    assert manager.dispatch(2, 2, 1, 0) is graph
    assert manager.dispatch(3, 3, None, 0).cg_mode == CUDAGraphMode.NONE
    assert manager.dispatch(1, 3, None, 0).num_reqs == 1
    del manager.graphs[verify]
    assert manager.dispatch(1, 6, 6, 0).cg_mode == CUDAGraphMode.NONE


def test_dfly_reuses_anchor_topk_and_masks_discarded_target_and_future_draft_tokens():
    target_rows = torch.tensor([[1, 4, 6, 7, 8, -1], [0, 1, 2, 3, -1, -1]])
    selected = build_reuse_indices(
        target_rows, torch.tensor([7, 4]), width=3, window=2, active=torch.tensor([True, False])
    )
    assert selected[0].tolist() == [
        [1, 4, -1, -1, -1, -1, 5, 6, 7, -1, -1],
        [1, 4, -1, -1, -1, -1, 5, 6, 7, 8, -1],
        [1, 4, -1, -1, -1, -1, 5, 6, 7, 8, 9],
    ]
    assert selected[1].eq(-1).all()


def test_precise_decode_metadata_excludes_short_prefills():
    builder = FlashMetadataBuilder.__new__(FlashMetadataBuilder)
    builder.precise_decode = True
    common = SimpleNamespace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 3, 9, 10]),
        seq_lens_cpu=torch.tensor([64, 134, 22]),
        is_prefilling=torch.tensor([True, False, False]),
        block_table_tensor=torch.empty(0),
        slot_mapping=torch.empty(0),
    )
    assert builder.build(0, common).precise_decode_reqs == (False, True, True)
    builder.precise_decode = False
    assert builder.build(0, common).precise_decode_reqs == ()


def test_full_decode_metadata_uses_live_device_lengths():
    builder = FlashMetadataBuilder.__new__(FlashMetadataBuilder)
    builder.precise_decode = False
    seq_lens = torch.tensor([6, 6])
    common = SimpleNamespace(
        num_reqs=2,
        query_start_loc_cpu=torch.tensor([0, 6, 12]),
        seq_lens_cpu=seq_lens.clone(),
        seq_lens=seq_lens,
        block_table_tensor=torch.empty(0),
        slot_mapping=torch.empty(0),
    )
    assert builder.build(0, common).decode_seq_lens is None
    captured = builder.build_for_cudagraph_capture(common)
    seq_lens.copy_(torch.tensor([511, 2052]))
    assert captured.decode_seq_lens is seq_lens
    assert captured.decode_seq_lens.tolist() == [511, 2052]


def test_window_boundary_and_zero_value_sink():
    indices = window_candidates(torch.tensor([0, 511, 512]), 512)
    assert (indices[:, 0].tolist(), indices[:, -1].tolist()) == ([-1, 0, 1], [0, 511, 512])
    probs = mla_attention_probs(
        torch.tensor([[[0.0]]]),
        torch.tensor([[[7.0], [100.0]]]),
        torch.tensor([[0, -1]]),
        torch.tensor([0.0]),
        scale=1.0,
    )
    assert probs.tolist() == [[[0.5, 0.0]]]


def test_full_swa_shared_read_preserves_windows_and_ignores_padding_pages():
    queries, heads, window, page_size = 6, 8, 512, 64
    width = 576
    seq_lens = torch.tensor([64, 514, 4096])
    slot_mapping = torch.tensor([[0] * queries, [0] * 4 + [-1] * 2, [-1] * queries]).flatten()
    # The last two scheduled positions of request 1 cross into an unallocated page.
    # A fully padded request also has stale sequence metadata.
    tables = torch.arange(8).flip(0).repeat(3, 1).int()
    meta = SimpleNamespace(decode_seq_lens=seq_lens, slot_mapping=slot_mapping, block_table=tables)
    rope = torch.zeros(8 * page_size, 64, dtype=torch.bfloat16)
    latent = (torch.arange(8 * page_size * 512).reshape(-1, 512) % 17 - 8).to(torch.float8_e4m3fn)
    cache = pack_c8_components(latent, rope, torch.full((8 * page_size, 4), 0.125))
    query = torch.zeros((seq_lens.numel() * queries, heads, width), dtype=torch.bfloat16)
    wv = torch.eye(512, 2, dtype=torch.bfloat16).expand(heads, -1, -1).contiguous()

    attention = AscendHYV4FlashAttention.__new__(AscendHYV4FlashAttention)
    torch.nn.Module.__init__(attention)
    attention.window = window
    attention.max_model_len = 4096
    attention.local_heads = heads
    attention.latent_dim = 512
    attention.scale = width**-0.5
    attention.indexer = None
    attention.use_fp8_cache = True
    attention.kv_storage_width = C8_ROW_BYTES
    attention.use_triton_mla_logits = False
    attention.batch_verify_attention = True
    attention.learnable_sink_param = torch.zeros(heads)
    attention.topk_snapshot = torch.empty(query.shape[0], window, dtype=torch.long)

    def run(share):
        attention.share_swa_c8_reads = share
        out = query.new_empty(query.shape[0], heads * 2)
        attention._forward_decode_graph(meta, cache, page_size, None, None, None, query, wv, out)
        return out, attention.topk_snapshot.clone()

    baseline, baseline_indices = run(False)
    shared, shared_indices = run(True)
    assert shared_indices.equal(baseline_indices)
    torch.testing.assert_close(shared, baseline, atol=0, rtol=0)
    assert shared[queries + 4 :].eq(0).all()


def test_engram_hash_matches_across_prefill_chunk_boundary():
    tokens = torch.tensor([2, 6, 9, 4, 3])
    tok_map = torch.arange(16, dtype=torch.int32)
    multipliers = torch.tensor([[7, 11, 13], [3, 5, 7]])
    primes = torch.tensor([[[11, 13], [17, 19]], [[5, 7], [11, 13]]])
    offsets = torch.tensor([[0, 11, 24, 41], [0, 5, 12, 23]])
    args = (tok_map, multipliers, primes, offsets, 0, 0, (9,))
    whole = hash_ngrams(tokens, torch.tensor([-1, -1]), *args)
    tail = hash_ngrams(tokens[2:], torch.tensor([6, 2]), *args)
    assert torch.equal(whole[:, 2:], tail)
    assert whole[:, 2].eq(-1).all()
    ngram = 4 * 7 ^ 9 * 11 ^ 6 * 13
    assert whole[0, 3, 2] == ngram % 17 + 24


def test_engram_fp8_row_bytes_match_native_decode():
    bits = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    native = bits.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(decode_e4m3fn(bits), native, atol=0, rtol=0, equal_nan=True)


def test_fp8_cache_reads_selected_rows_with_per_tile_scales_and_bf16_rope():
    payload = (torch.arange(3 * 512).reshape(3, 512) % 57 - 28).to(torch.float8_e4m3fn)
    rope = (torch.arange(3 * 64).reshape(3, 64) / 32).bfloat16()
    scales = torch.tensor([[0.25, 0.5, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    packed = pack_c8_components(payload, rope, scales)
    cache = torch.zeros((8, C8_ROW_BYTES + 8), dtype=torch.uint8)
    slots = torch.tensor([5, 1, 4])
    cache[:, :C8_ROW_BYTES].index_copy_(0, slots, packed)
    order = torch.tensor([[1, 2], [0, 1]])
    rows = cache[:, :C8_ROW_BYTES][slots[order]]
    expected = torch.cat((payload.float() * scales.repeat_interleave(128, dim=-1), rope.float()), dim=-1)
    torch.testing.assert_close(dequantize_c8_rows(rows), expected[order], atol=0, rtol=0)


def test_paged_cache_gathers_fields_in_logical_order():
    block_table = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]], dtype=torch.int32)
    positions = torch.tensor([[0, 5, 5], [3, 6, 1]])
    slots = torch.tensor([[4, 7, 7], [7, 4, 3]])
    for dtype, kv_width, sidecar_width in ((torch.bfloat16, 576, 140), (torch.uint8, C8_ROW_BYTES, 280)):
        width = kv_width + sidecar_width
        cache = (torch.arange(8 * width).reshape(8, width) % 127).to(dtype)
        for columns in (slice(kv_width), slice(kv_width, None)):
            expected = cache[slots][..., columns]
            rows = AscendHYV4FlashAttention._gather(cache, block_table, 2, positions, columns)
            torch.testing.assert_close(rows, expected, atol=0, rtol=0)
            assert rows.is_contiguous()
            single = AscendHYV4FlashAttention._gather(cache, block_table[0], 2, positions[0], columns)
            torch.testing.assert_close(single, expected[0], atol=0, rtol=0)


def test_sparse_c8_page_copy_preserves_token_addresses():
    pages = torch.full((4, PAGE_BYTES), 255, dtype=torch.uint8)
    rows = torch.arange(2 * PAGE_TOKENS * C8_ROW_BYTES).view(2, PAGE_TOKENS, C8_ROW_BYTES)
    expected = (rows % 251).to(torch.uint8)
    table = torch.tensor([2, 0])
    for page, logical in zip(table, expected, strict=True):
        pages[page, : PAGE_TOKENS * C8_ROW_BYTES].copy_(logical.flatten())

    positions = torch.tensor([0, PAGE_TOKENS - 1, PAGE_TOKENS, 2 * PAGE_TOKENS - 1])
    for physical_page in (2, 3):
        pages[physical_page].copy_(pages[2])
        table[0] = physical_page
        selected = AscendHYV4FlashAttention._gather(pages, table, PAGE_TOKENS, positions, slice(C8_ROW_BYTES))
        torch.testing.assert_close(selected, expected.flatten(0, 1)[positions], atol=0, rtol=0)


def test_ihc_rms_uses_global_query_boundaries_on_sequence_shards(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import ihc

    metadata = SimpleNamespace(query_start_loc=(0, 7, 11), precise_decode_reqs=(False, True))
    monkeypatch.setattr(ihc, "get_flash_attention_metadata", lambda: metadata)
    rows = torch.arange(16 * 33, dtype=torch.float32).view(16, 33) / 7
    eps = 1e-5
    means = [rows[:7].square().mean(-1, keepdim=True)]
    means.extend(row.square().mean(-1, keepdim=True) for row in rows[7:11].split(1))
    means.append(rows[11:].square().mean(-1, keepdim=True))
    expected = (torch.cat(means) + eps).rsqrt()
    shards = [ihc._inverse_rms(shard, eps, rank * 2) for rank, shard in enumerate(rows.split(2))]
    torch.testing.assert_close(torch.cat(shards), expected, rtol=0, atol=0)


def test_prefill_window_indices_keep_causal_rows_in_order():
    logical = torch.tensor([0, 1, 3, 4, 9], dtype=torch.int32)
    offsets = torch.arange(4, dtype=torch.int32)
    actual = prefill_window_indices(logical, offsets, 4)
    expected = torch.tensor([[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, 3], [1, 2, 3, 4], [6, 7, 8, 9]])
    torch.testing.assert_close(actual, expected.to(torch.int32))


def test_c8_prefill_renormalization_matches_explicit_sink():
    logits = torch.tensor([[[7.0, 0.0, -3.0], [1.0, 3.0, 6.0]], [[-3.0, -1.0, 2.0], [1.0, 2.0, 1.0]]])
    values = torch.tensor([[0.1, 2.0], [1.9, 3.0], [-3.2, 4.0]])
    sink = torch.tensor([7.0, -1.0])
    maximum = logits.amax(-1)
    total = (logits - maximum.unsqueeze(-1)).exp().sum(-1)
    context = logits.softmax(-1) @ values
    logits_with_sink = torch.cat((logits, sink.expand(logits.shape[0], -1).unsqueeze(-1)), dim=-1)
    expected = logits_with_sink.softmax(-1)[..., :-1] @ values
    torch.testing.assert_close(apply_attention_sink(context, maximum, total, sink), expected)


def test_neighbor_recall_is_limited_to_one_forward_and_matching_metadata():
    layer_types = [SPARSE_ATTENTION if i in (0, 1, 5, 9, 13, 17) else SLIDING_ATTENTION for i in range(18)]
    assert paired_recall_layer(layer_types, 0, SPARSE_ATTENTION) is None
    assert paired_recall_layer(layer_types, 17, SPARSE_ATTENTION) is None
    donor = paired_recall_layer(layer_types, 1, SPARSE_ATTENTION)
    consumer = paired_recall_layer(layer_types, 5, SPARSE_ATTENTION)
    assert donor == (1, True) and consumer == (1, False)
    context = SimpleNamespace(additional_kwargs={})
    metadata = object()
    shape = (2, 6, 4, 511)
    _, pending = recall_rows_for_forward(context, metadata, donor, *shape)
    pending.append(torch.tensor([0, 1, 2, -1]))
    assert recall_rows_for_forward(context, metadata, consumer, *shape) == (None, None)
    pending.append(torch.tensor([8, 9, -1, -1]))
    rows, writer = recall_rows_for_forward(context, metadata, consumer, *shape)
    assert rows is pending and writer is None
    assert recall_rows_for_forward(context, object(), consumer, *shape) == (None, None)
    assert recall_rows_for_forward(context, metadata, consumer, 2, 1, 4, 511) == (None, None)
    fresh = SimpleNamespace(additional_kwargs={})
    assert recall_rows_for_forward(fresh, metadata, consumer, *shape) == (None, None)


def test_plain_greedy_verifier_defers_to_transformed_or_sampled_requests():
    slots = np.array([1, 3])
    temp = np.zeros(4)
    needs_processing = np.zeros(4, dtype=bool)
    sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(temperature=SimpleNamespace(np=temp)),
        needs_logits_processing=needs_processing,
        get_logprobs_dims=lambda _: None,
        return_sampling_mask=False,
        compute_nans=False,
        trace_replay_state=None,
    )
    rejection = SimpleNamespace(
        sampler=sampler,
        watermark_key=None,
        synthetic_conditional_rates=None,
        enable_adaptive_verification=False,
        use_block_verification=False,
        fly_window_size=0,
    )
    runner = SimpleNamespace(
        sampler=sampler, rejection_sampler=rejection, batch_sharder=None, speculator=SimpleNamespace(draft_logits=None)
    )
    batch = SimpleNamespace(idx_mapping_np=slots, num_draft_tokens=5, num_reqs=2)
    assert greedy_sampling.plain_greedy_batch(runner, batch, None)
    temp[0] = 0.75
    assert greedy_sampling.plain_greedy_batch(runner, batch, None)
    temp[3] = 0.75
    assert not greedy_sampling.plain_greedy_batch(runner, batch, None)
    temp[3] = 0
    needs_processing[1] = True
    assert not greedy_sampling.plain_greedy_batch(runner, batch, None)
    needs_processing[1] = False
    assert not greedy_sampling.plain_greedy_batch(runner, batch, object())
    rejection.enable_adaptive_verification = True
    assert not greedy_sampling.plain_greedy_batch(runner, batch, None)
    rejection.enable_adaptive_verification = False
    runner.sampler.get_logprobs_dims = lambda _: (5, 0)
    assert not greedy_sampling.plain_greedy_batch(runner, batch, None)


def test_native_index_pages_select_physical_block_size():
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    layer = SimpleNamespace(page_planes=True, cache_width=1025, use_fp8_cache=True)
    backend = AscendHYV4FlashAttention.get_attn_backend(layer)
    assert backend.get_preferred_block_size(128) == 64
    assert backend.get_supported_kernel_block_sizes() == [64]
    assert backend.supports_block_size(64)
    assert not backend.supports_block_size(128)
    spec = AscendHYV4FlashAttention.get_kv_cache_spec(layer, config)
    assert spec.num_states == 64
    assert spec.page_size_bytes == PAGE_BYTES
    layer.page_planes = False
    assert AscendHYV4FlashAttention.get_attn_backend(layer).supports_block_size(128)


def test_copy_only_group_preserves_rank_order_and_fallback(monkeypatch):
    from unittest.mock import Mock

    from vllm_ascend.models.hy_v4_flash import model

    group = Mock()
    group.size.return_value = 8
    gathers = []

    def collect(output, local, group):
        gathers.append(group)
        for rank in range(8):
            output.narrow(0, rank * local.shape[0], local.shape[0]).copy_(local + rank)

    monkeypatch.setattr(model.dist, "all_gather_into_tensor", collect)
    fallback = Mock(side_effect=lambda x: x)
    monkeypatch.setattr(model, "sp_all_gather", fallback)
    for shape in ((1, 12), (1, 4, 12), (3, 3072), (6, 4, 3072), (12, 4, 3072)):
        x = torch.arange(torch.Size(shape).numel(), dtype=torch.bfloat16).reshape(shape)
        torch.testing.assert_close(model.gather_verify_rows(x, group), torch.cat([x + rank for rank in range(8)]))
        assert model.gather_verify_rows(x, None) is x
    wide = torch.zeros(13, 4, 3072, dtype=torch.bfloat16)
    assert model.gather_verify_rows(wide, group) is wide
    other = wide[:1].float()
    assert model.gather_verify_rows(other, group) is other
    assert gathers == [group] * 5
    assert fallback.call_count == 7


def test_ordered_scatter_uses_only_validated_decode_shapes(monkeypatch):
    from types import SimpleNamespace

    from vllm_ascend.models.hy_v4_flash import ordered_verification_scatter as scatter

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(decode_cudagraph_capture_request_sizes=[1, 4, 8, 16]),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=96),
    )
    probed = []

    def check(tp, group, local_rows):
        probed.append(local_rows)
        return local_rows != 3

    monkeypatch.setattr(scatter, "matches_native_scatter", check)
    shards = scatter.checked_verification_shards(config, None, None)
    assert probed == [1, 2, 3, 6, 12]
    assert shards == {1, 2, 6, 12}
    for size in (5, 6, 13, 16, 45, 48, 91, 96):
        assert scatter.supports_ordered_scatter(torch.zeros(size, 3072, dtype=torch.bfloat16), shards)
    assert not scatter.supports_ordered_scatter(torch.zeros(24, 3072, dtype=torch.bfloat16), shards)
    assert not scatter.supports_ordered_scatter(torch.zeros(6, 3072), shards)
    assert not scatter.supports_ordered_scatter(torch.empty(3072, 6, dtype=torch.bfloat16).T, shards)


def test_resident_pool_invalidation_preserves_captured_storage():
    from vllm_ascend.models.hy_v4_flash.resident_pool import ResidentPool

    pool = ResidentPool(131, 128, torch.device("cpu"))
    addresses = (pool.physical.data_ptr(), pool.keys.data_ptr())
    assert pool.physical.shape == (2,) and pool.keys.shape == (32, 128)
    assert pool.state_dict() == {}
    pool.physical.copy_(torch.tensor([3, 1]))
    pool.invalidate()
    torch.testing.assert_close(pool.physical, torch.full_like(pool.physical, -1))
    assert addresses == (pool.physical.data_ptr(), pool.keys.data_ptr())


def test_draft_projection_sum_preserves_wide_and_non_bf16_reductions():
    from types import SimpleNamespace
    from unittest.mock import Mock, PropertyMock, patch

    from vllm_ascend.models.hy_v4_flash.draft_projection_reduce import DraftRowParallelOp

    transport = Mock()
    transport.all_reduce.side_effect = lambda x: x + 2
    op = object.__new__(DraftRowParallelOp)
    op.layer = None
    op.quant_method = SimpleNamespace(apply=lambda layer, x, bias: x)
    op.reducers = {n: SimpleNamespace(reduce=lambda x: x + 1) for n in (6, 24)}
    with patch.object(DraftRowParallelOp, "comm_group", new_callable=PropertyMock, return_value=transport):
        for shape, dtype, increment in [
            ((6, 3072), torch.bfloat16, 1),
            ((12, 3072), torch.bfloat16, 2),
            ((24, 3072), torch.bfloat16, 1),
            ((6, 3072), torch.float32, 2),
            ((6, 128), torch.bfloat16, 2),
        ]:
            x = torch.zeros(shape, dtype=dtype)
            out, bias = op.apply_impl(x)
            assert bias is None
            torch.testing.assert_close(out, x + increment)


def test_shared_fp32_draft_history_requires_matching_request_shapes_and_native_pages():
    from types import SimpleNamespace

    from vllm_ascend.hisparse.c8 import C8_ROW_BYTES
    from vllm_ascend.models.hy_v4_flash.shared_precision_history import supports_shared_history

    pages = SimpleNamespace(
        shape=(1, 1, 64),
        device=torch.device("meta"),
        dtype=torch.uint8,
        is_contiguous=lambda: True,
        numel=lambda: 1 << 20,
    )
    source = SimpleNamespace(kv_cache=pages, page_planes=True, kv_storage_width=C8_ROW_BYTES)
    table = torch.empty(1, 10, dtype=torch.int32)
    snapshot = torch.empty(1, 2048, dtype=torch.int64)
    positions = torch.empty(1, 6, dtype=torch.int64)
    assert not supports_shared_history(source, table, snapshot, positions, 0)
    pages.device = SimpleNamespace(type="npu")
    assert supports_shared_history(source, table, snapshot, positions, 0)
    assert not supports_shared_history(source, table, snapshot, positions.expand(2, -1), 0)
    assert not supports_shared_history(source, table, snapshot, positions, 128)
    source.page_planes = False
    assert not supports_shared_history(source, table, snapshot, positions, 0)


def test_shared_fp32_draft_history_routes_the_six_queries_to_their_own_selected_rows(monkeypatch):
    from vllm_ascend.models.hy_v4_flash import dfly
    from vllm_ascend.models.hy_v4_flash.shared_precision_history import SharedPrecisionHistory

    query = torch.randn(6, 4, 5)
    context = torch.randn(6, 4, 3)
    weights = torch.randn(4, 3, 2)
    page_table = torch.arange(8)[None, :]
    logical = torch.arange(9, 15)[None, :]
    recall = SharedPrecisionHistory(torch.arange(2048)[None, :], torch.tensor([True]))
    source = SimpleNamespace(cache_rows=lambda: page_table)
    draft = torch.zeros(6, 8, dtype=torch.uint8)
    sink = torch.zeros(4)
    layer = SimpleNamespace(
        context=(64, page_table, recall, logical),
        target_attention=lambda: source,
        _pack_kv=lambda a, b: draft,
        learnable_sink_param=sink,
        scale=0.0625,
        absorption_weights=SimpleNamespace(value=weights),
    )

    def attention(q, cache, table, private, selected, anchor, learned_sink, scale, page):
        assert cache is page_table and learned_sink is sink
        torch.testing.assert_close(q, query)
        torch.testing.assert_close(selected.target_rows, recall.target_rows)
        torch.testing.assert_close(selected.active, recall.active)
        assert scale == 0.0625 and page == 64
        torch.testing.assert_close(table, page_table)
        torch.testing.assert_close(private, draft.unsqueeze(0))
        torch.testing.assert_close(anchor, logical[:, 0])
        return context

    monkeypatch.setattr(dfly, "attend_shared_fp32", attention)
    out = torch.empty(6, 8)
    dfly.DFlyAttention._forward_paged(layer, None, None, None, None, None, None, None, out, ready_query=query)
    expected = torch.einsum("qhd,hdk->qhk", context, weights).reshape_as(out)
    torch.testing.assert_close(out, expected)
