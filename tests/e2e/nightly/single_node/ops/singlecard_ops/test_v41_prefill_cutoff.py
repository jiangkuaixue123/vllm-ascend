# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve native TopK ties and the selected-attention consumer exactly."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention as model
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention, PrefillAttentionInputs
from vllm_ascend.ops.triton import deepseek_v41_index_ties as kernels
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op


def stock(scores):
    columns = torch.arange(scores.shape[1], dtype=torch.float32, device=scores.device)
    top = scores.topk(512, dim=-1)
    cutoff = top.values[:, -1:]
    first = torch.where(scores == cutoff, columns, scores.shape[1]).topk(512, largest=False).values
    tied = top.values == cutoff
    ordinal = tied.long().cumsum(-1) - 1
    expected = torch.where(tied, first.gather(-1, ordinal.clamp_min(0)).long(), top.indices)
    return expected, top, first


def context():
    return SimpleNamespace(cudagraph_runtime_mode=model.CUDAGraphMode.NONE, is_draft_model=False, in_profile_run=False)


@pytest.mark.parametrize(
    "rows,width,dtype",
    [
        (32, 1020, torch.bfloat16),
        (64, 8160, torch.bfloat16),
        (128, 16384, torch.bfloat16),
        (64, 1020, torch.float32),
        (128, 8160, torch.float32),
        (64, 640, torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_cutoff_bits_nan_inf_and_dynamic_replay(rows, width, dtype):
    torch.npu.set_device(0)
    init_device_properties_triton()
    scores = torch.randint(-8, 9, (rows, width), device="npu").to(dtype)
    scores[0].fill_(torch.nan)
    scores[1].fill_(-torch.inf)
    scores[2].fill_(torch.inf)
    scores[3, ::7] = torch.nan
    scores[4].zero_()
    expected, top, first = stock(scores)
    actual = kernels.finalize_prefill_ties(top.values, top.indices, first)
    assert actual is not None and torch.equal(actual, expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = kernels.finalize_prefill_ties(top.values, top.indices, first)
    for step in range(3):
        scores.copy_(scores.roll(step + 1, 1))
        expected, updated, new_first = stock(scores)
        top.values.copy_(updated.values)
        top.indices.copy_(updated.indices)
        first.copy_(new_first)
        graph.replay()
        assert torch.equal(replayed, expected)
        with (
            patch.object(model, "is_forward_context_available", return_value=True),
            patch.object(model, "get_forward_context", return_value=context()),
            patch.object(kernels, "finalize_prefill_ties", wraps=kernels.finalize_prefill_ties) as spy,
        ):
            assert torch.equal(model.topk_columns(scores, 512), expected)
            assert spy.call_count == 1


@torch.inference_mode()
def test_cutoff_gate_keeps_draft_capture_and_profile():
    torch.npu.set_device(0)
    scores = torch.randn(64, 1020, device="npu")
    expected, _, _ = stock(scores)
    for attribute, value in [
        ("is_draft_model", True),
        ("in_profile_run", True),
        ("cudagraph_runtime_mode", model.CUDAGraphMode.FULL),
    ]:
        ctx = context()
        setattr(ctx, attribute, value)
        with (
            patch.object(model, "is_forward_context_available", return_value=True),
            patch.object(model, "get_forward_context", return_value=ctx),
            patch.object(kernels, "finalize_prefill_ties", wraps=kernels.finalize_prefill_ties) as spy,
        ):
            assert torch.equal(model.topk_columns(scores, 512), expected)
            assert spy.call_count == 0


@pytest.mark.parametrize("rows", [128, 257])
@torch.inference_mode()
def test_cutoff_to_new_sparse_consumer(rows):
    torch.npu.set_device(0)
    assert enable_custom_op()
    init_device_properties_triton()
    scores = torch.randint(-4, 5, (rows, 1020), device="npu").bfloat16()
    expected, _, _ = stock(scores)
    choices = []
    for start in range(0, rows, 64):
        with (
            patch.object(model, "is_forward_context_available", return_value=True),
            patch.object(model, "get_forward_context", return_value=context()),
        ):
            choices.append(model.topk_columns(scores[start : start + 64], 512))
    actual = torch.cat(choices)
    assert torch.equal(actual, expected)
    expected = expected.sort(-1).values
    actual = actual.sort(-1).values
    positions = torch.arange(rows, device="npu") + 1024
    old, new = PrefillAttentionInputs(), PrefillAttentionInputs()
    old_slots = old.prepare_slots(expected, positions, 0, 2048, 0, 2048, 1020, 128)
    new_slots = new.prepare_slots(actual, positions, 0, 2048, 0, 2048, 1020, 128, encode_runs=True)
    query = torch.randn(1, rows, 64, 512, device="npu", dtype=torch.bfloat16) / 8
    kv = torch.randn(1, 3072, 512, device="npu", dtype=torch.bfloat16) / 8
    sink = torch.linspace(-1, 1, 64, device="npu")
    op = DeepseekV41NativeAttention()
    op.plan(query.device, 64, rows, 1, 640, 640, 639)
    reference = op._native(
        query, kv, sink, old_slots, 512**-0.5, "cutoff consumer", None, physical_slots=True, slot_table=old.slot_table
    )
    result = op._native(
        query, kv, sink, new_slots, 512**-0.5, "cutoff consumer", None, physical_slots=True, slot_table=new.slot_table
    )
    assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
