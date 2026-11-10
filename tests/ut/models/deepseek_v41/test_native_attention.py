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
    if native.is_950():
        pytest.skip("A5 SFA does not use the A3 metadata operator")
    assert native.enable_native_attention()
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
        ("DSA target indexed decode", 16),
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
        assert native.enable_native_attention()
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


def test_a5_duplicate_slots_can_exceed_kv_length_on_npu():
    if not torch.npu.is_available() or not native.is_950():
        pytest.skip("Requires A5")
    assert native.enable_native_attention()
    q, kv, sink = inputs(1, 1, 2, heads=16)
    indices = torch.zeros(1, 1, 129, dtype=torch.int32)
    expected = sparse_attn(q, kv, sink, indices, 512**-0.5)
    runner = native.DeepseekV41NativeAttention()
    actual = runner(q.npu(), kv.npu(), sink.npu(), indices.npu(), 512**-0.5, mode="duplicate regression")
    torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=0.025, atol=0.02)
