# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache final representation, metadata lifetime and independent HC masks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.hc_post import hc_post_fused
from vllm_ascend.models.deepseek_v41 import native_attention
from vllm_ascend.models.deepseek_v41.attention import DecodeSlotArena
from vllm_ascend.models.deepseek_v41.core import hc_post
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Block
from vllm_ascend.ops.triton import deepseek_v41_slot_arena as kernels
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op


def exact(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.fixture(autouse=True)
def device():
    torch.npu.set_device(0)
    assert enable_custom_op()
    init_device_properties_triton()


def raw_reference(field, md, rows, dtype):
    group_rows = min(6, rows)
    groups = rows // group_rows
    first = md.seq_lens.view(groups, group_rows)[:, 0]
    limit = md.seq_lens.view(groups, group_rows)[:, -1]
    pos = (first - 128).clamp_min(0)[:, None] + torch.arange(133, device=field.device)
    safe = pos.clamp(0, md.max_model_len - 1)
    page = md.block_table_tensor[::group_rows].gather(1, safe // 128).long()
    physical = page * 128 + safe % 128
    valid = (pos < limit[:, None]) & (physical >= 0) & (physical < field.shape[0])
    valid &= md.active.view(groups, group_rows).any(-1)[:, None]
    words = field.view(torch.int16).index_select(0, physical.clamp(0, field.shape[0] - 1).flatten())
    words = words.reshape(groups, 133, 512)
    value = (words.int() << 16).view(torch.float32) if dtype == torch.float32 else words.view(dtype)
    return torch.where(valid[:, :, None], value, 0.0).bfloat16()


@pytest.mark.parametrize("rows", [3, 6, 12, 18])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("stride", [512, 1024])
@torch.inference_mode()
def test_raw_final_form_all_bf16_words_and_dynamic_graph(rows, dtype, stride):
    group_rows = min(6, rows)
    groups = rows // group_rows
    words = torch.arange(65536, dtype=torch.int32).short().view(128, 512).npu()
    storage = torch.empty(256, stride, dtype=torch.int16, device="npu")
    storage[:, :512].copy_(words.repeat(2, 1))
    field = storage[:, :512].view(torch.bfloat16)
    md = SimpleNamespace(
        seq_lens=torch.arange(129 - group_rows, 129, device="npu").repeat(groups).long(),
        active=torch.ones(rows, dtype=torch.bool, device="npu"),
        block_table_tensor=torch.zeros(rows, 128, dtype=torch.int32, device="npu"),
        max_model_len=16384,
    )
    ids = torch.zeros(rows, 512, dtype=torch.int64, device="npu")
    output = torch.empty(groups, 133, 512, dtype=torch.bfloat16, device="npu")
    kernels.read(field, md, ids, 2 if dtype == torch.float32 else 1, True, dtype, output)
    exact(output, raw_reference(field, md, rows, dtype))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        kernels.read(field, md, ids, 2 if dtype == torch.float32 else 1, True, dtype, output)
    for step, end in enumerate([0, 1, 3, 127, 128, 129, 1029, 16384]):
        md.seq_lens.copy_(torch.arange(end + 1 - group_rows, end + 1, device="npu").repeat(groups))
        md.active.copy_(torch.arange(rows, device="npu") % 3 != step % 3)
        if step in (0, 4):
            md.active.fill_(bool(step))
        md.block_table_tensor.fill_(step % 2)
        if step == 2:
            md.block_table_tensor[:, 0] = -1
            md.block_table_tensor[:, 1] = 2
        field.view(torch.int16).copy_(words.roll(step, dims=1).repeat(2, 1))
        graph.replay()
        exact(output, raw_reference(field, md, rows, dtype))


def test_slot_metadata_keeps_alias_views_and_invalidates_replacement():
    md = SimpleNamespace(seq_lens=torch.arange(6), active=torch.ones(6, dtype=torch.bool))
    topk = torch.zeros(6, 512, dtype=torch.int64)
    arena = DecodeSlotArena()
    assert arena.slots_for(md, topk, True) is None
    arena.slots = torch.empty(6, 1, 640, dtype=torch.int32)
    slots = arena.slots
    same_view = SimpleNamespace(seq_lens=md.seq_lens[:], active=md.active[:])
    assert arena.slots_for(same_view, topk.view_as(topk), False) is slots
    assert arena.slot_inputs[0] is same_view.seq_lens
    assert arena.slots_for(md, topk, True) is None
    assert arena.slots is arena.slot_live is None
    arena.slots = slots
    assert arena.slots_for(md, topk.clone(), False) is None
    assert arena.slots_for(md, topk, False) is None
    arena.slots = slots
    assert arena.slots_for(md, topk, False) is slots
    assert arena.slots_for(md, topk.t().contiguous().t(), False) is None


@pytest.mark.parametrize("rows", [3, 6, 12])
@torch.inference_mode()
def test_hc_distinguishes_empty_sublayer_from_inactive_residual(rows):
    sub = torch.randn(rows, 5120, dtype=torch.bfloat16, device="npu")
    residual = torch.randn(rows, 4, 5120, dtype=sub.dtype, device=sub.device)
    post, comb = torch.rand(rows, 4, device="npu"), torch.rand(rows, 4, 4, device="npu")
    live = torch.ones(rows, dtype=torch.bool, device="npu")
    input_live = live.clone()
    runtime = SimpleNamespace(decode=object(), block_row_mask=live, attention_output_mask=input_live)
    owner = SimpleNamespace()
    hc_post_fused(sub, residual, post, comb, row_mask=live, input_mask=input_live)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        runtime.attention_output_mask = input_live
        actual = DeepseekV41Block.post_residual(owner, sub, residual, post, comb, runtime)
        second = DeepseekV41Block.post_residual(owner, sub, residual, post, comb, runtime)
    assert runtime.attention_output_mask is None
    for step in range(16):
        live.copy_(torch.arange(rows, device="npu") % 3 != step % 3)
        input_live.copy_(torch.arange(rows, device="npu") % 2 == step % 2)
        sub.copy_(torch.randn_like(sub))
        sub[~input_live] = float("nan")
        graph.replay()
        masked = torch.where(input_live[:, None], sub, 0)
        exact(actual, torch.where(live[:, None, None], hc_post(masked, residual, post, comb), 0))
        torch.testing.assert_close(
            second,
            torch.where(live[:, None, None], hc_post(sub, residual, post, comb), 0),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    assert hc_post_fused(sub, residual, post, comb, input_mask=input_live.float()) is None
    runtime.attention_output_mask = input_live
    with patch("vllm_ascend.models.deepseek_v41.model.hc_post_fused", return_value=None):
        exact(DeepseekV41Block.post_residual(owner, sub, residual, post, comb, runtime), actual)


@pytest.mark.parametrize("rows", [3, 6, 12])
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fallback", [False, True])
@torch.inference_mode()
def test_selected_and_native_live_rows_with_deferred_empty_mask(rows, source_dtype, fallback):
    q = torch.randn(rows, 1, 64, 512, dtype=torch.bfloat16, device="npu")
    kv = torch.randn(rows, 640, 512, dtype=q.dtype, device=q.device)
    slots = torch.arange(rows * 640, dtype=torch.int32, device=q.device).view(rows, 1, 640)
    sink = torch.linspace(-1, 1, 64, device=q.device).bfloat16().float()
    runner = native_attention.DeepseekV41NativeAttention()
    runner.sink_source_dtype = source_dtype
    runner.pin_decode_plan(q.device, 64, rows, 640)
    original = native_attention.try_selected_fia
    with patch.object(native_attention, "try_selected_fia", (lambda *a, **kw: None) if fallback else original):
        runner.decode_with_physical_slots(q, kv, sink, slots, 512**-0.5)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            expected = runner.decode_with_physical_slots(q, kv, sink, slots, 512**-0.5)
            actual = runner.decode_with_physical_slots(q, kv, sink, slots, 512**-0.5, mask_empty_rows=False)
        for step in range(8):
            slots.copy_(torch.arange(rows * 640, dtype=torch.int32, device=q.device).view(rows, 1, 640))
            slots[torch.arange(rows, device=q.device) % 3 == step % 3] = -1
            if step == 7:
                slots.fill_(-1)
            q.copy_(torch.randn_like(q))
            graph.replay()
            live = slots[:, 0, 0] >= 0
            exact(actual[live], expected[live])
            exact(torch.where(live[:, None, None, None], actual, 0), expected)
