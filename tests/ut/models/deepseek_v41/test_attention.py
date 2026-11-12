# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny paged-attention scenarios independent of a serving engine or checkpoint."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tests.ut.models.deepseek_v41 import paged_cache
from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.models.deepseek_v41.attention_ops import sparse_attn


@pytest.fixture(params=["cpu", "npu"], autouse=True)
def tensor_device(request):
    if request.param == "npu":
        if importlib.util.find_spec("torch_npu") is None:
            pytest.skip("torch_npu is not installed")
        importlib.import_module("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("No NPU is available")
    previous = torch.get_default_device()
    torch.set_default_device(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_device(previous)


class TinyLinear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.reduce_results = kwargs.get("reduce_results", True)
        dtype = kwargs.get("params_dtype", torch.bfloat16)
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=dtype))
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x, bias: torch.nn.functional.linear(x, layer.weight, bias)
        )

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


@pytest.fixture
def make_layer(monkeypatch):
    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 1)
    for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(attention, name, TinyLinear)

    def create(ratio=2, layer_id=0, layers=1, kv_sources=None, index_sources=None, groups=1):
        config = SimpleNamespace(
            hidden_size=32,
            head_dim=32,
            index_head_dim=32,
            num_attention_heads=2,
            qk_rope_head_dim=4,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=groups,
            index_n_heads=2,
            compress_ratios=[ratio] * layers,
            rms_norm_eps=1e-6,
            kv_source_layer_ids=(kv_sources if kv_sources is not None else ([0] if ratio else [])),
            index_source_layer_ids=(index_sources if index_sources is not None else ([0] if ratio else [])),
            rope_theta=10000,
            compress_rope_theta=160000,
            rope_scaling=None,
            sliding_window=3,
            index_topk=2,
            candidate_source_layer_id=-1,
            candidate_topk_blocks=2,
            candidate_block_size=2,
        )
        vllm_config = SimpleNamespace(
            quant_config=None,
            cache_config=SimpleNamespace(block_size=4),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = attention.DeepseekV41Attention(config, vllm_config, layer_id, "tiny.{}".format(layer_id))
        finally:
            torch.set_default_dtype(previous_dtype)
        generator = torch.Generator(device="cpu").manual_seed(100 + layer_id)
        # This 32-dimensional fixture exercises model math and page handling.
        # The native kernel has a separate full-width test on the A3.
        layer.attention_kernel = lambda q, kv, sink, indices, scale, **kwargs: sparse_attn(q, kv, sink, indices, scale)
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                if name.endswith("norm.weight"):
                    parameter.fill_(1)
                elif name == "attn_sink":
                    parameter.zero_()
                else:
                    values = torch.randn(parameter.shape, generator=generator, device="cpu") * 0.15
                    parameter.copy_(values.to(parameter.device))
        paged_cache.allocate(layer.cache, 12, 4, layer.wq_a.weight.device)
        return layer

    return create


def run_batch(layer, hidden, spans, tables):
    tables = torch.tensor(tables, dtype=torch.int32, device=hidden.device)
    slots = []
    for request_idx, span in enumerate(spans):
        positions = torch.arange(span.start_pos, span.seq_len, device=hidden.device)
        slots.append(tables[request_idx][positions // 4] * 4 + positions % 4)
    metadata = attention.DeepseekV41Metadata(spans, tables, torch.cat(slots), hidden.shape[0])
    layer.cache.metadata = lambda: metadata
    runtime = attention.DeepseekV41Runtime(spans)
    with torch.no_grad():
        output = layer(hidden, runtime)
    return output, runtime


def hidden_tokens(count, seed=21):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(count, 32, generator=generator, device="cpu").bfloat16().to(torch.get_default_device())


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_odd_prefill_then_decode_matches_complete_prefill(make_layer, ratio):
    x = hidden_tokens(7)
    full_layer = make_layer(ratio)
    chunked_layer = make_layer(ratio)
    expected, full_runtime = run_batch(full_layer, x, [attention.RequestSpan(0, 7, 7)], [[3, 1]])
    outputs = []
    begin = 0
    for end in (3, 4, 5, 7):
        actual, runtime = run_batch(
            chunked_layer,
            x[begin:end],
            [attention.RequestSpan(0, end - begin, end)],
            [[3, 1]],
        )
        outputs.append(actual)
        if ratio:
            assert runtime.shared[0].compressed.shape[0] == end // ratio
        begin = end
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)
    if ratio:
        torch.testing.assert_close(
            runtime.shared[0].compressed,
            full_runtime.shared[0].compressed,
            rtol=0,
            atol=0,
        )
    if ratio == 1:
        assert chunked_layer.compressor.wgate is None
        assert chunked_layer.compressor.wkv.weight.dtype == torch.bfloat16


def test_pending_compression_survives_page_boundary(make_layer):
    x = hidden_tokens(6)
    whole = make_layer(2)
    split = make_layer(2)
    expected, _ = run_batch(whole, x, [attention.RequestSpan(0, 6, 6)], [[7, 2]])
    before, _ = run_batch(split, x[:5], [attention.RequestSpan(0, 5, 5)], [[7, 2]])
    after, _ = run_batch(split, x[5:], [attention.RequestSpan(0, 1, 6)], [[7, 2]])
    torch.testing.assert_close(torch.cat((before, after)), expected, rtol=0, atol=0)


def test_request_reorder_and_physical_page_relocation_preserve_history(make_layer):
    a, b = hidden_tokens(4, 1), hidden_tokens(6, 2)
    layer = make_layer(2)
    expected_a, _ = run_batch(make_layer(2), a, [attention.RequestSpan(0, 4, 4)], [[3, 1]])
    expected_b, _ = run_batch(make_layer(2), b, [attention.RequestSpan(0, 6, 6)], [[0, 2]])
    initial, runtime = run_batch(
        layer,
        torch.cat((a[:3], b[:5])),
        [attention.RequestSpan(0, 3, 3), attention.RequestSpan(3, 8, 5)],
        [[3, 1], [0, 2]],
    )
    assert runtime.shared[0] is not runtime.shared[1]
    torch.testing.assert_close(initial[:3], expected_a[:3], rtol=0, atol=0)
    torch.testing.assert_close(initial[3:], expected_b[:5], rtol=0, atol=0)
    for source, destination in ((3, 6), (1, 7), (0, 8), (2, 9)):
        for bank in layer.cache.kv_cache:
            bank[destination].copy_(bank[source])
            bank[source].fill_(999)
    later, runtime = run_batch(
        layer,
        torch.cat((b[5:], a[3:])),
        [attention.RequestSpan(0, 1, 6), attention.RequestSpan(1, 2, 4)],
        [[8, 9], [6, 7]],
    )
    torch.testing.assert_close(later[0], expected_b[5], rtol=0, atol=0)
    torch.testing.assert_close(later[1], expected_a[3], rtol=0, atol=0)
    assert runtime.shared[0].compressed.shape[0] == 3
    assert runtime.shared[1].compressed.shape[0] == 2


def test_candidate_blocks_pin_latest_block_without_making_future_keys_causal():
    logits = torch.tensor(
        [
            [-torch.inf] * 5,
            [1, -torch.inf, -torch.inf, -torch.inf, -torch.inf],
            [9, 8, 1, -torch.inf, -torch.inf],
            [9, 8, 7, 6, 1],
        ]
    )
    lengths = torch.tensor([[0], [1], [3], [5]])
    keep = attention.select_candidate_blocks(logits, lengths, 1, 2)
    expected = torch.tensor(
        [
            [False] * 5,
            [True, True, False, False, False],
            [False, False, True, True, False],
            [False, False, False, False, True],
        ]
    )
    torch.testing.assert_close(keep, expected)
    masked = logits.masked_fill(~keep, -torch.inf)
    for row, length in enumerate((0, 1, 3, 5)):
        assert torch.isneginf(masked[row, length:]).all()


def test_indexer_masks_future_compression_groups(make_layer):
    layer = make_layer(2)
    x = hidden_tokens(7)
    _, runtime = run_batch(layer, x, [attention.RequestSpan(0, 7, 7)], [[3, 1]])
    selected = runtime.shared[0].topk
    lengths = (torch.arange(7) + 1) // 2
    assert ((selected == -1) | (selected < lengths[:, None])).all()


def test_query_projection_does_not_add_per_head_rmsnorm(make_layer, monkeypatch):
    layer = make_layer(0)
    x = hidden_tokens(1)
    captured = []
    original = layer.attention_kernel

    def capture(q, *args, **kwargs):
        captured.append(q.clone())
        return original(q, *args, **kwargs)

    monkeypatch.setattr(layer, "attention_kernel", capture)
    with torch.no_grad():
        layer.wq_b.weight.mul_(7)
        expected = layer.wq_b(layer.q_norm(layer.wq_a(x)[0]))[0].reshape(1, 1, 2, 32)
    run_batch(layer, x, [attention.RequestSpan(0, 1, 1)], [[3]])
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


def test_following_layer_uses_request_local_shared_sources(make_layer):
    source = make_layer(2, 0, 2, [0], [0, 1])
    following = make_layer(2, 1, 2, [0], [0, 1])
    x = hidden_tokens(5)
    spans = [attention.RequestSpan(0, 2, 2), attention.RequestSpan(2, 5, 3)]
    _, runtime = run_batch(source, x, spans, [[3], [1]])
    metadata = source.cache.metadata()
    following.cache.metadata = lambda: metadata
    compressed = [shared.compressed.clone() for shared in runtime.shared]
    with torch.no_grad():
        output = following(x, runtime)
    assert output.shape == x.shape
    for expected, shared in zip(compressed, runtime.shared):
        torch.testing.assert_close(shared.compressed, expected, rtol=0, atol=0)
    assert runtime.shared[0].topk.shape[0] == 2
    assert runtime.shared[1].topk.shape[0] == 3


def test_uncompressed_rotary_frequencies_ignore_compressed_yarn_config():
    config = SimpleNamespace(
        qk_rope_head_dim=4,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling={
            "original_max_position_embeddings": 64,
            "factor": 4,
            "beta_fast": 4,
            "beta_slow": 1,
        },
    )
    torch.testing.assert_close(
        attention.rotary_frequencies(config, False),
        torch.tensor([1.0, 0.01], device="cpu"),
        rtol=0,
        atol=0,
    )


def graph_metadata(length, table, max_len, *, active=True):
    blocks = torch.tensor([table], dtype=torch.int32)
    slot = table[(length - 1) // 4] * 4 + (length - 1) % 4 if active else -1
    decode = attention.DeepseekV41DecodeMetadata(
        torch.tensor([length], dtype=torch.long),
        torch.tensor([slot], dtype=torch.long),
        blocks,
        torch.tensor([active]),
        max_len,
    )
    return attention.DeepseekV41Metadata([], blocks, decode.slot_mapping, int(active), decode)


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("defer_mask", [False, True])
def test_static_decode_matches_eager_over_pages_and_pending_groups(make_layer, ratio, defer_mask):
    eager, graph = make_layer(ratio), make_layer(ratio)
    x, table = hidden_tokens(7), [3, 7, 6, 1]
    for layer in (eager, graph):
        run_batch(layer, x[:3], [attention.RequestSpan(0, 3, 3)], [table])
    for length in range(4, 8):
        token = x[length - 1 : length]
        expected, _ = run_batch(eager, token, [attention.RequestSpan(0, 1, length)], [table])
        md = graph_metadata(length, table, max_len=16)
        graph.cache.metadata = lambda md=md: md
        with torch.no_grad():
            actual = graph(token, attention.DeepseekV41Runtime.for_decode(md), defer_output_mask=defer_mask)
            if defer_mask:
                actual = torch.where(md.decode.active[:, None], actual, 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(paged_cache.bits(graph.cache), paged_cache.bits(eager.cache), rtol=0, atol=0)


@pytest.mark.parametrize("defer_mask", [False, True])
def test_static_decode_dummy_preserves_all_cache_values(make_layer, defer_mask):
    layer = make_layer(2)
    for bank in layer.cache.kv_cache:
        bank.uniform_(-1, 1)
    previous = paged_cache.snapshot(layer.cache)
    md = graph_metadata(1, [0, 0, 0, 0], 16, active=False)
    layer.cache.metadata = lambda: md
    with torch.no_grad():
        output = layer(hidden_tokens(1), attention.DeepseekV41Runtime.for_decode(md), defer_output_mask=defer_mask)
        if defer_mask:
            output = torch.where(md.decode.active[:, None], output, 0)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    torch.testing.assert_close(paged_cache.bits(layer.cache), previous, rtol=0, atol=0)


def test_npu_graph_replays_changed_position_page_and_idle_mask(make_layer, tensor_device):
    if tensor_device != "npu":
        pytest.skip("This test captures a real NPUGraph on the NPU")
    eager, captured_layer = make_layer(2), make_layer(2)
    states, table = hidden_tokens(5), [3, 7, 6, 1]
    for layer in (eager, captured_layer):
        run_batch(layer, states[:3], [attention.RequestSpan(0, 3, 3)], [table])

    md = graph_metadata(0, table, max_len=16, active=False)
    captured_layer.cache.metadata = lambda: md
    runtime = attention.DeepseekV41Runtime.for_decode(md)
    descriptor = md.decode
    token = torch.zeros_like(states[:1])
    with torch.no_grad():
        # Resolve lazy kernels and allocation before capturing the inactive row.
        for _ in range(2):
            captured_layer(token, runtime)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = captured_layer(token, runtime)

        def replay(length, active, blocks):
            updated = graph_metadata(length, blocks, max_len=16, active=active).decode
            descriptor.seq_lens.copy_(updated.seq_lens)
            descriptor.slot_mapping.copy_(updated.slot_mapping)
            descriptor.block_table_tensor.copy_(updated.block_table_tensor)
            descriptor.active.copy_(updated.active)
            if active:
                token.copy_(states[length - 1 : length])
            graph.replay()
            torch.npu.synchronize()

        replay(4, True, table)
        expected, _ = run_batch(eager, states[3:4], [attention.RequestSpan(0, 1, 4)], [table])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            paged_cache.bits(captured_layer.cache), paged_cache.bits(eager.cache), rtol=0, atol=0
        )

        before_idle = paged_cache.snapshot(captured_layer.cache)
        replay(0, False, [0, 0, 0, 0])
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
        torch.testing.assert_close(paged_cache.bits(captured_layer.cache), before_idle, rtol=0, atol=0)

        # Relocate the first logical page and append an incomplete ratio-two group.
        moved_table = [8, 6, 7, 1]
        for layer in (eager, captured_layer):
            for bank in layer.cache.kv_cache:
                bank[8].copy_(bank[3])
                bank[3].fill_(99)
        replay(5, True, moved_table)
        expected, _ = run_batch(eager, states[4:5], [attention.RequestSpan(0, 1, 5)], [moved_table])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            paged_cache.bits(captured_layer.cache), paged_cache.bits(eager.cache), rtol=0, atol=0
        )


def test_decode_metadata_replay_changes_values_but_retains_capture_addresses():
    builder = object.__new__(attention.DeepseekV41MetadataBuilder)
    builder.decode_buffers = {}
    builder.native_layers = []
    builder.native_decode_plans = {}
    builder.capture_sizes = {1, 2, 4, 8}
    builder.decode_graph_enabled = True
    builder.max_model_len = 16
    builder.block_size = 4
    common = SimpleNamespace(
        num_input_tokens=1,
        num_actual_tokens=1,
        num_reqs=1,
        max_query_len=1,
        max_seq_len=5,
        seq_lens=torch.tensor([5], dtype=torch.int32),
        slot_mapping=torch.tensor([8]),
        block_table_tensor=torch.tensor([[1, 2, 0, 0]]),
        query_start_loc_cpu=torch.tensor([0, 1], device="cpu"),
        seq_lens_cpu_upper_bound=torch.tensor([5], device="cpu"),
    )
    buf = builder.build_for_cudagraph_capture(common).decode
    tensors = (buf.seq_lens, buf.slot_mapping, buf.block_table_tensor, buf.active)
    pointers = [tensor.data_ptr() for tensor in tensors]
    assert not buf.active.cpu().tolist()[0]
    assert builder.build(0, common).decode is buf
    assert buf.active.cpu().tolist() == [True] and buf.slot_mapping.cpu().tolist() == [8]
    common.block_table_tensor = torch.tensor([[7, 6, 0, 0]])
    common.slot_mapping = torch.tensor([24])
    builder.build(0, common)
    assert buf.block_table_tensor.cpu().tolist() == [[7, 6, 0, 0]]
    assert buf.slot_mapping.cpu().tolist() == [24]
    common.slot_mapping.fill_(-1)
    builder.build(0, common)
    assert buf.active.cpu().tolist() == [False]
    assert [tensor.data_ptr() for tensor in tensors] == pointers


def test_decode_metadata_buckets_keep_distinct_buffers_and_mask_padding():
    builder = object.__new__(attention.DeepseekV41MetadataBuilder)
    builder.decode_buffers = {}
    builder.native_layers = []
    builder.native_decode_plans = {}
    builder.capture_sizes = {1, 2, 4, 8}
    builder.decode_graph_enabled = True
    builder.max_model_len = 16
    builder.block_size = 4

    def common(rows, actual):
        return SimpleNamespace(
            num_input_tokens=rows,
            num_actual_tokens=actual,
            num_reqs=rows,
            max_query_len=1,
            max_seq_len=5,
            seq_lens=torch.full((rows,), 5),
            slot_mapping=torch.arange(rows),
            block_table_tensor=torch.ones(rows, 4, dtype=torch.int32),
        )

    buffers = {rows: builder._update_decode(common(rows, rows), capture=True) for rows in builder.capture_sizes}
    assert len({buf.active.data_ptr() for buf in buffers.values()}) == len(buffers)
    for rows in sorted(builder.capture_sizes):
        assert not buffers[rows].active.any()
        updated = builder._update_decode(common(rows, rows))
        assert updated is buffers[rows]
        assert updated.active.all()
    partial = common(8, 6)
    partial.slot_mapping[1] = -1
    updated = builder._update_decode(partial)
    assert updated.active.tolist() == [True, False, True, True, True, True, False, False]
    assert updated.slot_mapping.tolist()[-2:] == [-1, -1]
    assert updated.seq_lens.tolist()[-2:] == [0, 0]
    assert not updated.block_table_tensor[-2:].any()
    assert buffers[4].active.all()
    assert builder._update_decode(common(3, 3)) is None
    partial.max_query_len = 2
    assert builder._update_decode(partial) is None


def multirow_metadata(lengths, tables):
    slots = [
        table[(length - 1) // 4] * 4 + (length - 1) % 4 if length else -1 for length, table in zip(lengths, tables)
    ]
    blocks = torch.tensor(tables, dtype=torch.int32)
    decode = attention.DeepseekV41DecodeMetadata(
        torch.tensor(lengths), torch.tensor(slots), blocks, torch.tensor([length > 0 for length in lengths]), 16
    )
    return attention.DeepseekV41Metadata([], blocks, decode.slot_mapping, sum(length > 0 for length in lengths), decode)


def test_tied_cutoff_prefers_lower_column_without_changing_distinct_scores():
    # The third row differs by one FP32 ULP. It must outrank column 0.
    logits = torch.tensor(
        [[2.0, 0, 0, 0, -1], [-1.0, -0.0, 0.0, -1, -1], [1.0, 1.0 + 2**-23, 2.0, 0, -1]], dtype=torch.float32
    )
    expected = torch.tensor([[0, 1], [1, 2], [1, 2]])
    for width in (5, 16):
        padded = torch.nn.functional.pad(logits, (0, width - logits.shape[1]), value=-torch.inf)
        torch.testing.assert_close(attention.topk_columns(padded, 2).sort(-1).values, expected)


@pytest.mark.parametrize("rows", [2, 4, 8])
def test_multirow_decode_replays_live_idle_and_relocated_pages(make_layer, tensor_device, rows):
    """Compare packed eager rows against fixed-bucket replay, including a live physical slot 0."""
    eager, captured = make_layer(2), make_layer(2)
    tables = [[i, rows + i, 0, 0] for i in range(rows)]
    # Start one row on each side of a page / compression-group boundary.
    lengths = [0, 4, 3, 1, 2, 0, 4, 3][:rows]
    states = [hidden_tokens(8, 35 + i) for i in range(rows)]
    for layer in (eager, captured):
        paged_cache.allocate(layer.cache, 2 * rows + 1, 4)
        for i, length in enumerate(lengths):
            if length:
                run_batch(layer, states[i][:length], [attention.RequestSpan(0, length, length)], [tables[i]])
    idle = [0] * rows
    md = multirow_metadata(idle, tables)
    captured.cache.metadata = lambda: md
    runtime = attention.DeepseekV41Runtime.for_decode(md)
    inputs = torch.zeros(rows, 32, dtype=torch.bfloat16)
    graph = None
    with torch.no_grad():
        if tensor_device == "npu":
            for _ in range(2):
                captured(inputs, runtime)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = captured(inputs, runtime)

        def replay(live_rows):
            current = [length + 1 if i in live_rows else 0 for i, length in enumerate(lengths)]
            update = multirow_metadata(current, tables).decode
            for name in ("seq_lens", "slot_mapping", "block_table_tensor", "active"):
                getattr(md.decode, name).copy_(getattr(update, name))
            for i in live_rows:
                inputs[i].copy_(states[i][lengths[i]])
            if graph is None:
                output = captured(inputs, runtime)
            else:
                graph.replay()
                torch.npu.synchronize()
                output = result
            expected = torch.zeros_like(output)
            if live_rows:
                spans = [attention.RequestSpan(j, j + 1, current[i]) for j, i in enumerate(live_rows)]
                actual, _ = run_batch(eager, inputs[live_rows], spans, [tables[i] for i in live_rows])
                expected[live_rows] = actual
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(paged_cache.bits(captured.cache), paged_cache.bits(eager.cache), rtol=0, atol=0)
            for i in live_rows:
                lengths[i] += 1

        replay(list(range(0, rows, 2)))
        replay([])
        moved = 2 * rows
        for layer in (eager, captured):
            for bank in layer.cache.kv_cache:
                bank[moved].copy_(bank[tables[1][0]])
                bank[tables[1][0]].fill_(99)
        tables[1][0] = moved
        replay(list(range(rows)))


@pytest.mark.parametrize("rows", [1, 2])
def test_static_decode_reuses_source_pages_and_candidate_selection(make_layer, rows):
    def layers():
        result = [make_layer(1, n, 3, [0], [0, 1]) for n in range(3)]
        for layer in result:
            layer.config.candidate_source_layer_id = 0
        return result

    eager, graph = layers(), layers()
    lengths = [6, 9][:rows]
    states = [hidden_tokens(length, 21 + i) for i, length in enumerate(lengths)]
    tables = [[4, 7, 3, 2], [1, 9, 5, 11]][:rows]
    prefill = torch.cat([x[:-1] for x in states])
    spans = []
    begin = 0
    for length in lengths:
        spans.append(attention.RequestSpan(begin, begin + length - 1, length - 1))
        begin += length - 1
    for sequence in (eager, graph):
        _, runtime = run_batch(sequence[0], prefill, spans, tables)
        for layer in sequence[1:]:
            layer.cache.metadata = sequence[0].cache.metadata
            with torch.no_grad():
                layer(prefill, runtime)
    tokens = torch.stack([x[-1] for x in states])
    queries = [attention.RequestSpan(i, i + 1, length) for i, length in enumerate(lengths)]
    first, eager_runtime = run_batch(eager[0], tokens, queries, tables)
    md = multirow_metadata(lengths, tables)
    graph_runtime = attention.DeepseekV41Runtime.for_decode(md)
    for index, layer in enumerate(graph):
        layer.cache.metadata = lambda: md
        eager[index].cache.metadata = eager[0].cache.metadata
        with torch.no_grad():
            observed = layer(tokens, graph_runtime)
            expected = first if index == 0 else eager[index](tokens, eager_runtime)
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        torch.testing.assert_close(paged_cache.bits(layer.cache), paged_cache.bits(eager[index].cache), atol=0, rtol=0)
        assert graph_runtime.source_cache is graph[0].cache


def test_cache_spec_accounts_for_mixed_precision_source(make_layer):
    layer = make_layer(2)
    spec = layer.cache.get_kv_cache_spec(None)
    assert spec.dtype == torch.bfloat16
    assert spec.head_size >= layer.dim + layer.config.index_head_dim
    assert spec.head_size_v == 4 * layer.dim
    assert layer.cache.pages()[0].shape == (48, spec.head_size)
    assert layer.cache.field(0, layer.dim, layer.dim).dtype == torch.float32
    assert layer.cache.field(1, layer.dim, layer.config.index_head_dim).dtype == torch.bfloat16


class CPUSequenceGroup:
    """Use real Gloo collectives for the tiny TP2 layout comparison."""

    def __init__(self, rank):
        self.rank_in_group = rank
        self.world_size = 2
        self.device_group = dist.group.WORLD
        self.gathers = self.scatters = 0

    def all_gather(self, x, dim):
        assert dim == 0
        self.gathers += 1
        out = x.new_empty(2 * x.shape[0], *x.shape[1:])
        dist.all_gather_into_tensor(out, x.contiguous())
        return out

    def reduce_scatter(self, x, dim):
        assert dim == 0 and x.dtype == torch.float32
        self.scatters += 1
        out = x.new_empty(x.shape[0] // 2, *x.shape[1:])
        dist.reduce_scatter_tensor(out, x.contiguous())
        return out


def sequence_attention_worker(rank, init_uri):
    dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=2)
    torch.set_num_threads(1)
    try:
        with pytest.MonkeyPatch.context() as patch, torch.no_grad():
            make = make_layer.__wrapped__(patch)
            full = make(groups=2)
            group = CPUSequenceGroup(rank)

            class Column(TinyLinear):
                def __init__(self, input_size, output_size, **kwargs):
                    super().__init__(input_size, output_size // 2, **kwargs)

            class Row(TinyLinear):
                def __init__(self, input_size, output_size, **kwargs):
                    super().__init__(input_size // 2, output_size, **kwargs)

            def reduce(tensor):
                dist.all_reduce(tensor)
                return tensor

            patch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: 2)
            patch.setattr(attention, "get_tp_group", lambda: group)
            patch.setattr(attention, "ColumnParallelLinear", Column)
            patch.setattr(attention, "RowParallelLinear", Row)
            patch.setattr(attention, "tensor_model_parallel_all_reduce", reduce)

            def build(sp, cp, layer_id):
                patch.setattr(attention, "enable_sp", lambda _: sp)
                patch.setattr(attention, "enable_dsa_cp", lambda: cp)
                layer = make(2, layer_id, 2, [0], [0, 1], groups=2)
                layer.config.candidate_source_layer_id = 0
                weights = dict(full.named_parameters())
                for name, value in layer.named_parameters():
                    source = weights[name]
                    if source.shape != value.shape:
                        axis = 1 if name == "wo_b.weight" else 0
                        source = source.chunk(2, dim=axis)[rank]
                    value.copy_(source)
                return layer

            for cp in (False, True):
                reference, sequence = build(False, False, 0), build(True, cp, 0)
                following_ref, following_seq = build(False, False, 1), build(True, cp, 1)
                assert sequence.wq_b.weight.shape[0] == (2 if cp else 1) * full.dim
                q_rows = []
                sequence.wq_b.register_forward_pre_hook(
                    lambda mod, args, q_rows=q_rows: q_rows.append(args[0].shape[0])
                )
                packed = hidden_tokens(5)
                padded = torch.nn.functional.pad(packed, (0, 0, 0, 1))
                spans = [attention.RequestSpan(0, 4, 4), attention.RequestSpan(4, 5, 1)]
                tables = [[3, 7, 0, 0], [1, 8, 0, 0]]
                expected, ref_runtime = run_batch(reference, packed, spans, tables)
                expected = torch.nn.functional.pad(expected, (0, 0, 0, 1))
                sequence.cache.metadata = reference.cache.metadata
                start = rank * 3
                seq_runtime = attention.DeepseekV41Runtime(spans)
                observed = sequence(padded[start : start + 3], seq_runtime)
                torch.testing.assert_close(observed, expected[start : start + 3], rtol=0, atol=0)
                torch.testing.assert_close(
                    paged_cache.bits(sequence.cache), paged_cache.bits(reference.cache), rtol=0, atol=0
                )
                following_ref.cache.metadata = following_seq.cache.metadata = reference.cache.metadata
                expected = torch.nn.functional.pad(following_ref(packed, ref_runtime), (0, 0, 0, 1))
                observed = following_seq(padded[start : start + 3], seq_runtime)
                torch.testing.assert_close(observed, expected[start : start + 3], rtol=0, atol=0)
                # Start with one completely idle TP shard, then change both
                # requests, including a page boundary on the first request.
                for live_lengths in ([5, 0, 0, 0], [6, 0, 2, 0], [0, 0, 0, 0]):
                    md = multirow_metadata(live_lengths, [tables[0], [0] * 4, tables[1], [0] * 4])
                    reference.cache.metadata = sequence.cache.metadata = lambda md=md: md
                    tokens = hidden_tokens(4, 85)
                    ref_runtime = attention.DeepseekV41Runtime.for_decode(md)
                    seq_runtime = attention.DeepseekV41Runtime.for_decode(md)
                    expected = reference(tokens, ref_runtime)
                    start = 2 * rank
                    observed = sequence(tokens[start : start + 2], seq_runtime)
                    torch.testing.assert_close(observed, expected[start : start + 2], rtol=0, atol=0)
                    torch.testing.assert_close(
                        paged_cache.bits(sequence.cache), paged_cache.bits(reference.cache), rtol=0, atol=0
                    )
                    following_seq.cache.metadata = following_ref.cache.metadata = reference.cache.metadata
                    expected = following_ref(tokens, ref_runtime)
                    observed = following_seq(tokens[start : start + 2], seq_runtime)
                    torch.testing.assert_close(observed, expected[start : start + 2], rtol=0, atol=0)
                assert q_rows == ([3, 2, 2, 2] if cp else [6, 4, 4, 4])
            assert group.gathers == group.scatters == 16
    finally:
        dist.destroy_process_group()


def test_sequence_parallel_prefill_and_decode_match_tp_reference(tmp_path, tensor_device):
    if tensor_device != "cpu":
        pytest.skip("This two-rank reference test uses CPU Gloo")
    mp.spawn(sequence_attention_worker, args=((Path(tmp_path) / "store").as_uri(),), nprocs=2, join=True)


def test_output_projection_reduces_fp32_after_local_bf16_gemm(make_layer, monkeypatch):
    layer = make_layer(0)
    assert layer.wo_b.reduce_results is False
    x = hidden_tokens(2)
    local, _ = run_batch(layer, x, [attention.RequestSpan(0, 2, 2)], [[3]])
    reductions = []

    def reduce(value):
        reductions.append(value.dtype)
        return value * 2

    monkeypatch.setattr(attention, "tensor_model_parallel_all_reduce", reduce)
    layer.tp = 2
    actual, _ = run_batch(layer, x, [attention.RequestSpan(0, 2, 2)], [[3]])
    assert reductions == [torch.float32]
    torch.testing.assert_close(actual, (local.float() * 2).to(x.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0, 2])
@torch.no_grad()
def test_dspark_six_token_verify_and_rejection_use_causal_pages(make_layer, ratio, tensor_device):
    """The replacement query starts at the accepted prefix, even with stale future KV."""
    eager, verifier = make_layer(ratio), make_layer(ratio)
    tables = [[0, 2, 4, 6], [1, 3, 5, 7]]
    builder = object.__new__(attention.DeepseekV41MetadataBuilder)
    builder.decode_buffers = {}
    builder.native_layers = []
    builder.native_decode_plans = {}
    builder.capture_sizes = {12}
    builder.decode_graph_enabled = True
    builder.query_width = 6
    builder.max_model_len = 16
    builder.block_size = 4
    graph = None
    graph_inputs = hidden_tokens(12, 19)
    for starts, seed in (([0, 0], 21), ([2, 5], 39)):
        spans = [attention.RequestSpan(i * 6, (i + 1) * 6, start + 6) for i, start in enumerate(starts)]
        slots = [tables[i][p // 4] * 4 + p % 4 for i, start in enumerate(starts) for p in range(start, start + 6)]
        common = SimpleNamespace(
            num_input_tokens=12,
            num_actual_tokens=12,
            num_reqs=2,
            max_query_len=6,
            max_seq_len=max(starts) + 6,
            seq_lens=torch.tensor(starts) + 6,
            # Simulate an optimistic scheduler estimate after rejection.
            seq_lens_cpu_upper_bound=torch.tensor([15, 15]),
            query_start_loc_cpu=torch.tensor([0, 6, 12]),
            slot_mapping=torch.tensor(slots),
            block_table_tensor=torch.tensor(tables, dtype=torch.int32),
        )
        md = builder.build(0, common)
        assert md.requests == spans
        assert md.decode.seq_lens.tolist() == [p + 1 for start in starts for p in range(start, start + 6)]
        verifier.cache.metadata = lambda md=md: md
        states = hidden_tokens(12, seed)
        expected, _ = run_batch(eager, states, spans, tables)
        runtime = attention.DeepseekV41Runtime.for_decode(md)
        if tensor_device == "cpu":
            actual = verifier(states, runtime)
        else:
            if graph is None:
                builder.build_for_cudagraph_capture(common)
                verifier(graph_inputs, runtime)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    graph_output = verifier(graph_inputs, runtime)
                builder.build(0, common)
            graph_inputs.copy_(states)
            graph.replay()
            torch.npu.synchronize()
            actual = graph_output
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_verify_cache_scatter_keeps_slot_zero_with_padding(make_layer):
    cache = make_layer(0).cache
    slots = torch.tensor([-1, 0, 7, -1, 4, -1])
    live = slots >= 0
    md = attention.DeepseekV41DecodeMetadata(
        torch.ones(6, dtype=torch.int64), slots, torch.zeros(6, 4, dtype=torch.int32), live, 16, 6
    )
    values = torch.arange(24, dtype=torch.float32).view(6, 4)
    expected = cache.pages()[0].clone()
    expected[0, 1:5] = values[1]
    expected[7, 1:5] = values[2]
    expected[4, 1:5] = values[4]
    cache.decode_write(md, values, 0, 1, live)
    torch.testing.assert_close(cache.pages()[0], expected, rtol=0, atol=0)
    cache.decode_write(md, values * -1, 0, 1, torch.zeros_like(live))
    torch.testing.assert_close(cache.pages()[0], expected, rtol=0, atol=0)
