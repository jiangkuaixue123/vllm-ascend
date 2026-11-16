# SPDX-License-Identifier: Apache-2.0
"""Compare complete cache storage and live graph ownership with the stock consumer."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.ops.triton import deepseek_v41_context_store as terminal
from vllm_ascend.utils import enable_custom_op


def quantize(values, kind):
    if kind == "fp8":
        return attention.fp8_quant_dequant(values)
    e4 = kind == "fp4_e4m3"
    return attention.fp4_quant_dequant(values, 16 if e4 else 32, scale_format="e4m3" if e4 else "e8m0")


@pytest.mark.parametrize("rows", [6, 24, 48])
@pytest.mark.parametrize("ratio", [0, 1, 2])
@torch.inference_mode()
def test_quantized_cache_full_storage_graph(rows, ratio):
    torch_npu.npu.set_device(0)
    assert enable_custom_op()
    physical = 257
    generator = torch.Generator().manual_seed(519)
    widths = (1024, 2048) if ratio == 2 else (512, 1024) if ratio == 1 else (512, 512)
    initial = [
        torch.randint(-32768, 32768, (physical + 2, width), dtype=torch.int16, generator=generator).npu()
        for width in widths
    ]
    fields = [(0, 0, 512, False, "fp8")]
    if ratio:
        fields += [(0, 512, 512, True, "fp4_e4m3"), (1, 512, 128, True, "fp4_e8m0")]
    slots = torch.zeros(rows, dtype=torch.int64, device="npu")
    active = torch.zeros(rows, dtype=torch.bool, device="npu")
    complete = torch.zeros_like(active)
    md = attention.DeepseekV41DecodeMetadata(None, slots, None, active, 16384, 6, 0)
    inputs = [torch.zeros((rows, w), dtype=torch.bfloat16, device="npu") for _, _, w, _, _ in fields]
    context = SimpleNamespace(cudagraph_runtime_mode=attention.CUDAGraphMode.FULL, is_draft_model=False)
    arms = []
    for _ in range(2):
        backing = [t.clone() for t in initial]
        cache = attention.DeepseekV41Cache.__new__(attention.DeepseekV41Cache)
        torch.nn.Module.__init__(cache)
        cache.dim, cache.index_dim, cache.source_ratio = 512, 128, ratio
        cache.kv_cache = tuple(t[1:-1].view(torch.bfloat16) for t in backing)
        cache.prefix = "model.layers.2.attn.cache"
        arms.append((cache, backing))

    def forward(cache, candidate):
        for (bank, offset, _, gated, kind), values in zip(fields, inputs):
            enabled = complete if gated else active
            if candidate:
                cache.decode_write_quantized(md, values, bank, offset, enabled, kind)
            else:
                cache.decode_write(md, quantize(values, kind), bank, offset, enabled)

    original_store = terminal.store_quantized_cache_rows

    def checked_store(*args):
        accepted = original_store(*args)
        assert accepted, "candidate used the stock fallback"
        return accepted

    with (
        patch.object(attention, "is_forward_context_available", return_value=True),
        patch.object(attention, "get_forward_context", return_value=context),
    ):
        graphs = []
        for i, (cache, _) in enumerate(arms):
            forward(cache, bool(i))
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with (
                patch.object(terminal, "store_quantized_cache_rows", wraps=checked_store) as store,
                torch.npu.graph(graph),
            ):
                forward(cache, bool(i))
            assert store.call_count == (len(fields) if i else 0)
            graphs.append(graph)
        for _, backing in arms:
            for target, original in zip(backing, initial):
                target.copy_(original)
        addresses = [x.data_ptr() for x in (slots, active, complete, *inputs)]
        for step in range(7):
            positions = (torch.arange(rows) * 5 + step) % physical
            positions[:6] = torch.tensor([0, 0, physical - 1, physical - 1, -1, physical])
            if step == 2:
                positions[-1] = 2**40
            live = torch.arange(rows) % 4 != 3
            live[:4] = True
            if step in (0, 6):
                live.zero_()
            if step == 3:
                live[:2] = False
            enabled = (torch.arange(rows) + step) % 2 == 0
            enabled[2:4] = True
            slots.copy_(positions)
            active.copy_(live)
            complete.copy_(enabled)
            originals = []
            for values in inputs:
                host = torch.randint(-32768, 32768, values.shape, dtype=torch.int16, generator=generator)
                if step % 2:
                    host = (torch.randn(values.shape, generator=generator) * 3).bfloat16().view(torch.int16)
                host[:, :10] = torch.tensor(
                    [0, -32768, 1, -32767, 0x7F80, -128, 0x7F81, 0x7FC2, -127, -1], dtype=torch.int16
                )
                host[3].copy_(host[2])
                values.view(torch.int16).copy_(host)
                originals.append(host)
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            for bank, original in enumerate(initial):
                assert torch.equal(arms[0][1][bank], arms[1][1][bank]), (rows, ratio, step, bank)
                assert torch.equal(arms[1][1][bank][0], original[0])
                assert torch.equal(arms[1][1][bank][-1], original[-1])
            assert [x.data_ptr() for x in (slots, active, complete, *inputs)] == addresses
            assert all(torch.equal(v.view(torch.int16).cpu(), ref) for v, ref in zip(inputs, originals))


@torch.inference_mode()
def test_unqualified_terminal_has_no_side_effect():
    torch_npu.npu.set_device(0)
    field = torch.zeros(257, 512, dtype=torch.bfloat16, device="npu")
    slots = torch.zeros(20, dtype=torch.int64, device="npu")
    active = torch.ones(20, dtype=torch.bool, device="npu")
    values = torch.ones(20, 512, dtype=torch.bfloat16, device="npu")
    assert not terminal.store_quantized_cache_rows(field, slots, active, active, values)
    assert not field.count_nonzero()


def test_all_positive_slots_version_and_frame_fallback():
    torch_npu.npu.set_device(0)
    field = torch.zeros(257, 512, dtype=torch.bfloat16, device="npu")
    field.view(torch.int16)[0, 0] = 0x7FC1
    slots = torch.arange(1, 25, dtype=torch.int64, device="npu")
    active = torch.ones(24, dtype=torch.bool, device="npu")
    values = torch.ones(24, 512, dtype=torch.bfloat16, device="npu")
    expected = field.clone()
    expected.index_copy_(0, slots, values)
    version = field._version
    with torch.inference_mode():
        assert terminal.store_quantized_cache_rows(field, slots, active, active, values)
    torch.npu.synchronize()
    assert field._version > version
    assert torch.equal(field.view(torch.int16), expected.view(torch.int16))
    cache = attention.DeepseekV41Cache.__new__(attention.DeepseekV41Cache)
    torch.nn.Module.__init__(cache)
    cache.dim, cache.index_dim, cache.source_ratio = 512, 128, 0
    cache.kv_cache = (field, field.clone())
    md = attention.DeepseekV41DecodeMetadata(None, slots, None, active, 16384, 6, 0)
    context = SimpleNamespace(cudagraph_runtime_mode=attention.CUDAGraphMode.NONE, is_draft_model=False)
    with (
        torch.inference_mode(),
        patch.object(attention, "is_forward_context_available", return_value=True),
        patch.object(attention, "get_forward_context", return_value=context),
        patch.object(terminal, "store_quantized_cache_rows", side_effect=AssertionError("non-FULL launch")),
    ):
        cache.decode_write_quantized(md, values, 0, 0, active)
    assert torch.equal(field.view(torch.int16), expected.view(torch.int16))
