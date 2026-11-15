# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_ascend.models.deepseek_v41 import fused_weights
from vllm_ascend.ops import vocab_parallel_embedding as ops


def head_and_processor(monkeypatch, bias=False):
    monkeypatch.setattr(ops, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0))
    monkeypatch.setattr(ops, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(ops, "embedding_tp_enable", lambda: False)
    with torch.device("npu"):
        head = ops.AscendParallelLMHead(64, 5120, bias=bias, params_dtype=torch.float32, prefix="head")
    head.weight_loader(head.weight, torch.randn(64, 5120))
    head._cache_weight_cast = True
    if bias:
        head.bias.copy_(torch.randn(64).npu())
    processor = object.__new__(ops.AscendLogitsProcessor)
    nn.Module.__init__(processor)
    processor.head_dtype = torch.bfloat16
    return head, processor


def check_head(processor, head, x, result):
    reference = LogitsProcessor._apply_head(processor, head, x, head.bias)
    torch.testing.assert_close(result.view(torch.int16), reference.view(torch.int16), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 5, 6, 12])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_head_cast_cache_reload_and_graph(monkeypatch, rows, bias):
    head, processor = head_and_processor(monkeypatch, bias)
    x = torch.randn(rows, 5120).bfloat16().npu().float()
    master = head.weight.clone()
    result = processor._apply_head(head, x, head.bias)
    check_head(processor, head, x, result)
    assert head.weight.dtype == torch.float32
    torch.testing.assert_close(head.weight, master, rtol=0, atol=0)
    assert head._head_cast_weight.dtype == torch.bfloat16
    assert "_head_cast_weight" not in head.state_dict()
    address = head._head_cast_weight.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = processor._apply_head(head, x, head.bias)
    for state_dict in (False, True):
        new_weight = torch.randn(64, 5120)
        if state_dict:
            state = {"weight": new_weight}
            if bias:
                state["bias"] = torch.randn(64)
            head.load_state_dict(state)
        else:
            head.weight_loader(head.weight, new_weight)
        assert head._head_cast_weight.data_ptr() == address
        x.neg_()
        graph.replay()
        torch.npu.synchronize()
        check_head(processor, head, x, out)


@torch.inference_mode()
def test_indexed_head_load_refreshes_captured_cache(monkeypatch, tmp_path):
    head, processor = head_and_processor(monkeypatch)
    model = nn.Module()
    model.head = head
    model.model_path = tmp_path
    model.config = SimpleNamespace()
    x = torch.randn(6, 5120).npu()
    processor._apply_head(head, x, None)
    address = head._head_cast_weight.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = processor._apply_head(head, x, None)
    replacement = torch.randn_like(head.weight)

    @contextmanager
    def reader(*args, **kwargs):
        yield SimpleNamespace()

    def read_parameter_slice(reader, source, param, module, tp_rank, tp_size):
        assert source == "head.weight" and module is head and (tp_rank, tp_size) == (0, 1)
        param.copy_(replacement)

    monkeypatch.setattr(fused_weights, "safe_open", reader)
    loaded = fused_weights.load_indexed_v41_weights(
        model, {"head.weight": "model.safetensors"}, 0, 1, read_parameter_slice
    )
    assert loaded == {"head.weight"}
    assert head._head_cast_weight.data_ptr() == address
    graph.replay()
    torch.npu.synchronize()
    check_head(processor, head, x, out)


@torch.inference_mode()
def test_head_cast_cache_requires_opt_in_and_inference(monkeypatch):
    head, processor = head_and_processor(monkeypatch)
    x = torch.randn(5, 5120).npu()
    head._cache_weight_cast = False
    check_head(processor, head, x, processor._apply_head(head, x, None))
    assert head._head_cast_weight is None
    head._cache_weight_cast = True
    with torch.inference_mode(False):
        check_head(processor, head, x, processor._apply_head(head, x, None))
    assert head._head_cast_weight is None


@torch.inference_mode()
def test_head_cast_cache_rebuilds_after_module_dtype_change(monkeypatch):
    head, processor = head_and_processor(monkeypatch)
    x = torch.randn(6, 5120).npu()
    processor._apply_head(head, x, None)
    head.float()
    assert head._head_cast_weight.dtype == torch.float32
    check_head(processor, head, x, processor._apply_head(head, x, None))
    assert head._head_cast_weight.dtype == torch.bfloat16


@torch.inference_mode()
def test_loader_prepares_cache_before_first_captured_head(monkeypatch):
    head, processor = head_and_processor(monkeypatch)
    assert head._head_cast_weight is None
    head.weight_loader(head.weight, torch.randn(64, 5120))
    assert head._head_cast_weight is not None
    address = head._head_cast_weight.data_ptr()
    x = torch.randn(6, 5120, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = processor._apply_head(head, x, None)
    for _ in range(4):
        head.weight_loader(head.weight, torch.randn(64, 5120))
        x.neg_()
        graph.replay()
        assert head._head_cast_weight.data_ptr() == address
        check_head(processor, head, x, out)
