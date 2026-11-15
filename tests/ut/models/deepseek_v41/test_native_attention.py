# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical-slot layout and real A3 shared-KV kernel checks."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import native_attention as native
from vllm_ascend.models.deepseek_v41.attention_ops import sparse_attn


def inputs(batch, rows, length, heads=8):
    generator = torch.Generator(device="cpu").manual_seed(191)
    q = torch.randn(batch, rows, heads, native.LATENT_DIM, generator=generator).bfloat16()
    kv = torch.randn(batch, length, native.LATENT_DIM, generator=generator).bfloat16()
    sink = torch.linspace(-1.0, 2.0, q.size(2), dtype=torch.float32)
    return q, kv, sink


def fake_sharedkv(q, *, ori_kv, ori_sparse_indices, ori_block_table, sinks, metadata, softmax_scale, **common):
    rows, heads, dim = q.shape
    pages = ori_kv.flatten(0, 1).squeeze(1)
    if ori_sparse_indices is not None:
        indices = ori_sparse_indices.reshape(1, rows, -1)
    else:
        batch = ori_block_table.shape[0]
        length = int(common["seqused_kv"][0])
        query_width = rows // batch
        window = common["ori_win_left"] + 1
        pos = length - query_width + torch.arange(query_width).view(1, -1, 1)
        relative = pos + torch.arange(1 - window, 1).view(1, 1, -1)
        blocks = ori_block_table[:, :1].view(-1, 1, 1)
        indices = torch.where(relative >= 0, blocks * native.PAGE_SIZE + relative, -1).reshape(1, rows, window)
    return [sparse_attn(q.view(1, rows, heads, dim), pages.unsqueeze(0), sinks, indices, softmax_scale)[0]]


@pytest.fixture
def backend(monkeypatch):
    plans = []
    calls = []

    def metadata(*args, **kwargs):
        plans.append(kwargs)
        return torch.zeros(1024, dtype=torch.int32)

    def launch(*args, **kwargs):
        calls.append(kwargs)
        return fake_sharedkv(*args, **kwargs)

    monkeypatch.setattr(
        torch.ops,
        "_C_ascend",
        SimpleNamespace(npu_sparse_attn_sharedkv_metadata=metadata, npu_sparse_attn_sharedkv=launch),
    )
    return native.DeepseekV41NativeAttention(), plans, calls


def test_slot_compaction_keeps_batch_stride_duplicates_and_masked_middle_rows():
    logical = torch.tensor([[[4, -1, 0, 4], [-1, -1, -1, -1]], [[-1, 2, -1, 1], [3, -1, -1, 3]]])
    slots = native.physical_indices(logical, 16, 5, 128)
    assert slots[:, 0, :5].tolist() == [[4, 0, 4, -1, -1], [-1] * 5, [18, 17, -1, -1, -1], [19, 19, -1, -1, -1]]
    assert torch.all(slots[:, :, 5:] == -1)


def test_prefill_slots_follow_source_identity_and_geometry(monkeypatch):
    prepared = native.PrefillAttentionInputs()
    positions = torch.arange(11, 14)
    topk = torch.tensor([[2, -1, 0, 2], [-1, 2, -1, 1], [3, -1, -1, 3]])
    compact = native.physical_indices
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return compact(*args, **kwargs)

    monkeypatch.setattr(native, "physical_indices", counted)

    def slots(source, query_start=11, raw_length=6, compressed_length=5):
        pos = positions + query_start - 11
        actual = prepared.prepare_slots(source, pos, query_start, 14, 8, raw_length, compressed_length, 4)
        ids = native.prefill_indices(pos, 4, 14, 8, raw_length, source)
        length = raw_length + compressed_length
        assert torch.equal(actual, compact(ids, native.aligned_length(length, 16), length, 128))
        return actual

    first = slots(topk)
    assert slots(topk) is first and len(calls) == 1
    replacement = topk.flip(-1).contiguous()
    assert slots(replacement) is not first and len(calls) == 2
    slots(replacement, query_start=10)
    slots(replacement, raw_length=7)
    slots(replacement, compressed_length=4)
    assert len(calls) == 5
    prepared.slots = None  # The producer invalidates before publishing another top-k.
    slots(replacement, compressed_length=4)
    assert len(calls) == 6


def test_prefill_workspace_refreshes_raw_and_compressed_and_bounds_residency():
    caches = [native.PrefillAttentionInputs(), native.PrefillAttentionInputs()]
    raw = torch.arange(5 * 512).reshape(5, 512).bfloat16()
    compressed = torch.full((4, 512), 7, dtype=raw.dtype)
    limit = 16 * 512 * raw.element_size()
    workspace = caches[0].update_workspace(raw, compressed, limit)
    assert torch.equal(workspace[0, :9], torch.cat((raw, compressed)))
    assert torch.count_nonzero(workspace[0, 9:]) == 0
    raw.add_(1)
    assert caches[0].update_workspace(raw, compressed, limit) is workspace
    assert torch.equal(workspace[0, :5], raw)
    replacement = compressed + 3
    caches[0].update_workspace(raw, replacement, limit)
    assert torch.equal(workspace[0, 5:9], replacement)
    assert caches[1].update_workspace(raw, compressed, limit - caches[0].workspace_bytes) is None
    assert sum(c.workspace_bytes for c in caches) <= limit
    # Ratio/length changes within the same aligned page must move the tail too.
    caches[0].update_workspace(raw[:3], replacement.repeat(2, 1), limit)
    assert torch.equal(caches[0].workspace[0, :11], torch.cat((raw[:3], replacement.repeat(2, 1))))
    assert torch.count_nonzero(caches[0].workspace[0, 11:]) == 0
    assert caches[0].update_workspace(raw.repeat(4, 1), compressed, limit) is None
    assert caches[0].workspace_bytes == 0 and caches[0].compressed is None


def test_compact_lengths_and_slots_share_the_same_generation():
    logical = torch.tensor([[[4, -1, 0, 4, 99], [-1, -1, -1, -1, -1], [2, 1, -1, 0, 3]]])
    table = torch.zeros((3, 8), dtype=torch.int32)
    table[:, 0] = native.PREFILL_LENGTH_TAG
    slots = native.physical_indices(logical, 16, 5, 128, lengths=table[:, 1])
    assert table[:, 1].tolist() == [3, 0, 4]
    assert torch.equal((slots[:, 0] >= 0).sum(-1), table[:, 1])
    assert torch.count_nonzero(table[:, 2:]) == 0


@pytest.mark.parametrize("width", [128, 512, 640, 2048])
@torch.inference_mode()
def test_native_prepared_prefix_contract_on_npu(width):
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Requires Ascend A3")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend A3")
    runner = native.DeepseekV41NativeAttention()
    q, kv, sink = (t.npu() for t in inputs(1, 6, width + 1, 64))
    ids = torch.arange(width, device=q.device).expand(1, 6, -1)
    slots = native.physical_indices(ids, native.aligned_length(width + 1, 16), width + 1, width)
    table = torch.zeros((6, width // 16), dtype=torch.int32, device=q.device)
    table[:, 0] = native.PREFILL_LENGTH_TAG
    counts = torch.tensor([0, 1, 2, 4, 7, width], device=q.device, dtype=torch.int32)
    table[:, 1].copy_(counts)
    short = torch.where(torch.arange(width, device=q.device)[None, None] < counts[None, :, None], ids, -1)
    expected = runner(q, kv, sink, short, 512**-0.5, mode="declared prefix reference")
    actual = runner._native(
        q,
        kv,
        sink,
        slots,
        512**-0.5,
        "prepared prefix witness",
        None,
        physical_slots=True,
        slot_table=table,
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.count_nonzero(actual[:, 0]) == 0  # Positive slots with declared length zero prove native dispatch.
    full = runner(q, kv, sink, ids, 512**-0.5, mode="unmarked table reference")
    for invalid_tag, invalid_count in [
        (native.PREFILL_LENGTH_TAG + 1, 1),
        (native.PREFILL_LENGTH_TAG, -1),
        (native.PREFILL_LENGTH_TAG, width + 1),
    ]:
        table[:, 0] = invalid_tag
        table[:, 1] = invalid_count
        actual = runner._native(
            q,
            kv,
            sink,
            slots,
            512**-0.5,
            "fallback table",
            None,
            physical_slots=True,
            slot_table=table,
        )
        assert torch.equal(actual.view(torch.int16), full.view(torch.int16))
    if width == 128:
        metadata, old_table, common = runner._plan(q.device, 64, 1, 6, width, width, width - 1)
        multi_query_table = old_table.clone()
        multi_query_table[:, 0] = native.PREFILL_LENGTH_TAG
        multi_query_table[:, 1] = 1
        padded = native.F.pad(kv, (0, 0, 0, native.aligned_length(width + 1, 16) - width - 1))

        def multi_query(block_table):
            return torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q.reshape(-1, 64, 512),
                ori_kv=padded.view(-1, 16, 1, 512),
                ori_sparse_indices=slots,
                ori_block_table=block_table,
                sinks=sink,
                metadata=metadata,
                softmax_scale=512**-0.5,
                **common,
            )[0]

        assert torch.equal(multi_query(old_table).view(torch.int16), multi_query(multi_query_table).view(torch.int16))


def test_prepared_prefill_uses_logical_length_and_existing_pages(backend, monkeypatch):
    runner, _, calls = backend
    q, kv, sink = inputs(1, 3, 13)
    topk = torch.tensor([[2, -1, 0], [3, 1, 1], [-1, -1, 0]])
    positions = torch.arange(6, 9)
    indices = native.prefill_indices(positions, 4, 9, 0, 9, topk)
    expected = runner._native(q, kv, sink, indices, 0.125, "reference", None)
    prepared = native.PrefillAttentionInputs()
    slots = prepared.prepare_slots(topk, positions, 6, 9, 0, 9, 4, 4)
    pages = prepared.update_workspace(kv[0, :9], kv[0, 9:], 16 * 512 * 2)

    def no_padding(*args, **kwargs):
        raise AssertionError("Prepared pages must not be copied/padded again")

    monkeypatch.setattr(native.F, "pad", no_padding)
    actual = runner.prefill_with_physical_slots(
        q,
        pages[:, :13],
        sink,
        slots,
        0.125,
        padded_kv=pages,
        slot_table=prepared.slot_table,
    )
    assert torch.equal(actual, expected)
    assert calls[-1]["ori_kv"].data_ptr() == pages.data_ptr()
    assert calls[-1]["ori_block_table"] is prepared.slot_table
    with pytest.raises(ValueError, match="aligned contiguous"):
        runner.prefill_with_physical_slots(q, pages[:, :13], sink, slots, 0.125, padded_kv=pages[:, :13])


@pytest.mark.parametrize("rows,raw_length,compressed_length", [(3, 17, 0), (257, 1027, 513), (2040, 8160, 8160)])
@torch.inference_mode()
def test_prefill_native_pages_and_consumer_reuse_on_npu(rows, raw_length, compressed_length):
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Requires Ascend A3")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend A3")
    runner = native.DeepseekV41NativeAttention()
    q, kv, sink = (t.npu() for t in inputs(1, rows, raw_length + compressed_length, 64))
    positions = torch.arange(raw_length - rows, raw_length, device=q.device)
    width = min(512, compressed_length)
    topk = torch.arange(width, device=q.device).flip(0).expand(rows, -1).clone()
    topk[:, 1::3] = -1
    prepared = native.PrefillAttentionInputs()
    raw, compressed = kv[0, :raw_length].clone(), kv[0, raw_length:].clone()
    slots = prepared.prepare_slots(
        topk, positions, raw_length - rows, raw_length, 0, raw_length, compressed_length, 128
    )
    logical = native.prefill_indices(positions, 128, raw_length, 0, raw_length, topk)
    for layer in range(3):
        raw.add_(0.125)
        if layer == 2:
            compressed = compressed + 0.25
        joint = torch.cat((raw, compressed)).unsqueeze(0)
        expected = runner(q, joint, sink, logical, 512**-0.5, mode="retained prefill")
        pages = prepared.update_workspace(raw, compressed, 32 * 1024 * 1024)
        actual = runner.prefill_with_physical_slots(
            q,
            pages[:, : joint.shape[1]],
            sink,
            slots,
            512**-0.5,
            padded_kv=pages,
            slot_table=prepared.slot_table,
        )
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(pages[:, : joint.shape[1]], joint)
    # Budget fallback still reuses slots and the original native padding path.
    actual = runner.prefill_with_physical_slots(q, joint, sink, slots, 512**-0.5, slot_table=prepared.slot_table)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_indexed_slot_math_and_plan_reuse(backend):
    runner, plans, calls = backend
    q, kv, sink = inputs(2, 6, 17)
    ids = torch.tensor([0, 3, -1, 3, 16, -1]).expand(2, 6, -1).clone()
    ids[0, 3] = -1
    for _ in range(2):
        output = runner._native(q, kv, sink, ids, 0.125, "DSA target indexed decode", None)
        # The packed token batch changes the CPU matmul reduction order; one
        # last-place BF16 difference can occur at its rounding boundary.
        torch.testing.assert_close(output, sparse_attn(q, kv, sink, ids, 0.125), atol=0, rtol=torch.finfo(q.dtype).eps)
        ids[1, 4, 0] = 7
    assert len(plans) == 1 and len(calls) == 2
    assert plans[0]["batch_size"] == 12 and plans[0]["max_seqlen_q"] == 1
    assert plans[0]["cmp_ratio"] == 1 and not plans[0]["has_cmp_kv"]


def test_implicit_window_alignment(backend):
    runner, plans, calls = backend
    q, kv, sink = inputs(2, 3, 7)
    indices = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], dtype=torch.int32).expand(2, -1, -1)
    output = runner._native(q, kv, sink, indices, 0.125, "SWA target causal prefill", 4)
    torch.testing.assert_close(output, sparse_attn(q, kv, sink, indices, 0.125), atol=0, rtol=0)
    assert calls[0]["ori_sparse_indices"] is None and plans[0]["ori_win_left"] == 3


def test_graph_plan_survives_eager_eviction_and_unprepared_buckets_fail(backend):
    runner, _, calls = backend
    q, kv, sink = inputs(2, 1, 8)
    ids = torch.tensor([[[2, 3, 5]], [[1, 7, 3]]])
    runner.pin_decode_plan(q.device, 8, 2, ids.shape[-1])
    key = (q.device, 8, 2, 128)
    plan = runner.decode_plans[key]
    for length in range(2, 43):
        runner.plan(q.device, 8, 1, 1, length, 0, 127)
    assert runner.plan.cache_info().currsize == 32
    runner._native(q, kv, sink, ids, 0.125, "target", None, uniform_decode=True)
    assert runner.decode_plans[key] is plan
    assert calls[-1]["metadata"] is plan[0] and calls[-1]["ori_block_table"] is plan[1]
    assert calls[-1]["cu_seqlens_q"] is plan[2]["cu_seqlens_q"]
    assert calls[-1]["seqused_kv"] is plan[2]["seqused_kv"]
    with pytest.raises(RuntimeError, match="not prepared"):
        runner._native(q[:1], kv[:1], sink, ids[:1], 0.125, "target", None, uniform_decode=True)


def test_builder_reuses_pinned_aicpu_tensors_across_layers(backend):
    runner, plans, _ = backend
    next_layer = native.DeepseekV41NativeAttention()
    shared = {}
    for instance in (runner, next_layer):
        instance.pin_decode_plan(torch.device("cpu"), 64, 3, 384, shared)
    assert len(plans) == len(shared) == 1
    key = next(iter(shared))
    assert runner.decode_plans[key] is next_layer.decode_plans[key] is shared[key]


@pytest.mark.parametrize("rows", [3, 6])
@pytest.mark.parametrize("width", [128, 384, 640])
def test_real_tp8_verifier_plan_on_npu(rows, width):
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Requires Ascend A3")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend A3")
    assert native.enable_custom_op()
    device = torch.device("npu:0")
    runner = native.DeepseekV41NativeAttention()
    runner.pin_decode_plan(device, 64, rows, width)
    meta, _, common = runner.decode_plans[(device, 64, rows, width)]
    assert meta.shape == (1024,)
    assert common["cu_seqlens_q"].cpu().tolist() == list(range(rows + 1))
    assert common["seqused_kv"].cpu().tolist() == [width] * rows


def test_kernel_failure_propagates_and_unsupported_geometry_is_rejected(backend, monkeypatch):
    runner, _, _ = backend
    q, kv, sink = inputs(1, 1, 2)
    ids = torch.tensor([[[1]]])
    with pytest.raises(ValueError, match="multiple of four"):
        runner._native(q[:, :, :3], kv, sink[:3], ids, 1.0, "invalid", None)
    with pytest.raises(ValueError, match="2048"):
        runner._native(q, kv, sink, ids.expand(1, 1, 2049), 1.0, "invalid", None)

    def fail(*args, **kwargs):
        raise RuntimeError("native execution error")

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_attn_sharedkv", fail)
    with pytest.raises(RuntimeError, match="native execution error"):
        runner._native(q, kv, sink, ids, 1.0, "failure", None)


@pytest.mark.parametrize(
    "mode,heads",
    [
        ("SWA target causal prefill", 64),
        ("DSA target indexed decode", 64),
        ("DSA target indexed decode", 8),
        ("SWA draft bidirectional", 8),
    ],
)
def test_native_attention_on_npu(mode, heads):
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Requires Ascend A3")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend A3")
    batch, rows, length = 2, 3, 17
    q, kv, sink = inputs(batch, rows, length, heads=heads)
    q, kv, sink = q.npu(), kv.npu(), sink.npu()
    window = 4 if "causal" in mode else None
    if window:
        ids = torch.arange(length - rows, length).view(1, rows, 1) + torch.arange(1 - window, 1)
        ids = ids.expand(batch, -1, -1).npu()
    else:
        ids = torch.tensor([0, 8, -1, 8, 16]).expand(batch, rows, -1).clone().npu()
        ids[0, 1] = -1
    runner = native.DeepseekV41NativeAttention()
    uniform_decode = mode == "DSA target indexed decode"
    if uniform_decode:
        assert native.enable_custom_op()
        runner.pin_decode_plan(q.device, heads, batch * rows, ids.shape[-1])

    def execute():
        return runner(
            q, kv, sink, ids, q.shape[-1] ** -0.5, mode=mode, swa_window=window, uniform_decode=uniform_decode
        )

    def verify(actual):
        expected = sparse_attn(q.cpu(), kv.cpu(), sink.cpu(), ids.cpu(), q.shape[-1] ** -0.5)
        torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=0.025, atol=0.02)

    verify(execute())
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = execute()
    q.mul_(0.625)
    if not window:
        ids[1, 2, 0] = 4
    graph.replay()
    verify(output)


@pytest.mark.parametrize("heads", [8, 64])
@torch.inference_mode()
def test_native_contiguous_slots_preserve_page_boundaries_and_order(heads):
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("Requires Ascend A3")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend A3")
    q, kv, sink = [value.npu() for value in inputs(1, 3, 257, heads=heads)]
    # Runs cross physical pages and coexist with duplicates, reverse order and holes.
    ids = torch.arange(7, 135).expand(1, 3, -1).clone().npu()
    ids[0, 1, :8] = torch.tensor([15, 16, 17, -1, 256, 0, 0, 3]).npu()
    ids[0, 2] = -1
    op = native.DeepseekV41NativeAttention()

    def run():
        return op(q, kv, sink, ids, 512**-0.5, mode="DSA slot run correctness")

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    for shift in (0, 16):
        ids[0, 0] = torch.arange(7 + shift, 135 + shift, device=q.device)
        graph.replay()
        expected = sparse_attn(q.cpu(), kv.cpu(), sink.cpu(), ids.cpu(), 512**-0.5)
        torch.testing.assert_close(output.float().cpu(), expected.float(), rtol=0.025, atol=0.02)
