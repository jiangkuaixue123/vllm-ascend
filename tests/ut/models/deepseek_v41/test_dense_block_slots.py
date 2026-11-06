# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry cache and full-block dispatch without initializing the accelerator."""

import ast
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/native_attention.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
    scope = {
        "torch": torch,
        "F": F,
        "os": os,
        "lru_cache": lru_cache,
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "enable_custom_op": lambda: True,
        "sparse_attn": lambda q, kv, sink, indices, scale: indices,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**scope)


@pytest.mark.parametrize("batch,rows", [(1, 5), (2, 3)])
@pytest.mark.parametrize("length", [5, 6, 127, 128, 133])
def test_full_block_has_exact_batch_offsets_and_negative_tail(module, batch, rows, length):
    runner = module.DeepseekV41NativeAttention()
    logical, slots = runner.full_block_indices(torch.device("cpu"), batch, rows, length)
    page_width = ((length + 15) // 16) * 16
    width = ((length + 127) // 128) * 128
    expected = []
    for b in range(batch):
        for _ in range(rows):
            expected.append([b * page_width + i for i in range(length)] + [-1] * (width - length))
    assert torch.equal(slots[:, 0], torch.tensor(expected, dtype=torch.int32))
    assert slots.shape == (batch * rows, 1, width) and slots.is_contiguous()
    assert torch.equal(logical, torch.arange(length).expand(batch, rows, -1))
    again = runner.full_block_indices(torch.device("cpu"), batch, rows, length)
    assert again[0] is logical and again[1] is slots


def test_cache_is_bounded_and_separate_for_each_attention_instance(module):
    first, second = module.DeepseekV41NativeAttention(), module.DeepseekV41NativeAttention()
    original = first.full_block_indices(torch.device("cpu"), 1, 5, 5)
    other = second.full_block_indices(torch.device("cpu"), 1, 5, 5)
    assert original[1] is not other[1]
    for length in range(6, 45):
        first.full_block_indices(torch.device("cpu"), 1, 5, length)
    assert first.full_block_indices.cache_info().currsize == 32
    assert second.full_block_indices.cache_info().currsize == 1
    recreated = first.full_block_indices(torch.device("cpu"), 1, 5, 5)
    assert recreated[1] is not original[1] and torch.equal(recreated[1], original[1])


def test_cpu_full_block_and_mutating_sparse_input_dispatch(module):
    runner = module.DeepseekV41NativeAttention()
    q, kv, sink = torch.zeros(1, 5, 4, 512), torch.zeros(1, 8, 512), torch.zeros(4)
    output = runner(q, kv, sink, None, 1.0, mode="draft")
    assert torch.equal(output, torch.arange(8).expand(1, 5, -1))
    ids = torch.tensor([7, -1, 2, 2, 0]).expand(1, 5, -1).clone()
    assert runner(q, kv, sink, ids, 1.0, mode="sparse") is ids
    ids[0, 0, 0] = 3
    assert runner(q, kv, sink, ids, 1.0, mode="sparse")[0, 0, 0] == 3
    assert runner.full_block_indices.cache_info().currsize == 1


@pytest.mark.parametrize("batch,rows,length", [(0, 5, 8), (1, 0, 8), (1, 5, 0), (1, 5, 2049)])
def test_invalid_full_block_geometry_is_rejected(module, batch, rows, length):
    with pytest.raises(ValueError, match="nonempty rows"):
        module.DeepseekV41NativeAttention().full_block_indices(torch.device("cpu"), batch, rows, length)


@pytest.mark.parametrize("options", [{"swa_window": 4}, {"uniform_decode": True}])
def test_full_block_cannot_replace_causal_or_target_decode_indices(module, options):
    runner = module.DeepseekV41NativeAttention()
    with pytest.raises(ValueError, match="bidirectional"):
        runner(torch.zeros(1, 5, 4, 512), torch.zeros(1, 8, 512), torch.zeros(4), None, 1.0, mode="invalid", **options)
