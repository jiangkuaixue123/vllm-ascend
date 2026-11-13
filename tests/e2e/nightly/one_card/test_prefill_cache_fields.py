# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41 import attention


def words(value):
    return value.view(torch.int16 if value.dtype == torch.bfloat16 else torch.int32)


def cache(backing, ratio):
    result = attention.DeepseekV41Cache.__new__(attention.DeepseekV41Cache)
    torch.nn.Module.__init__(result)
    result.dim, result.index_dim, result.source_ratio, result.token_history = (
        512,
        128,
        ratio,
        False,
    )
    result.kv_cache = tuple(t[1:-1].view(torch.bfloat16) for t in backing)
    return result


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("physical,rows", [(257, 1), (257, 41), (4097, 2048), (8193, 8192), (184704, 41)])
@torch.inference_mode()
def test_prefill_fields_match_stock_bits_and_dynamic_graph(ratio, physical, rows):
    widths = (1024, 2048) if ratio == 2 else (512, 1024)
    gen = torch.Generator().manual_seed(609)
    initial = [
        torch.randint(-32768, 32768, (physical + 2, width), dtype=torch.int16, generator=gen).npu() for width in widths
    ]
    arms = [(cache([t.clone() for t in initial], ratio)) for _ in range(2)]
    slots = torch.arange(rows, dtype=torch.int64, device="npu")
    for arm in arms:
        arm.slots = lambda *args: slots
    fields = [(0, 0, 512), (0, 512, 512), (1, 512, 128)]
    if ratio == 2:
        fields.insert(2, (1, 0, 512))
    values = [
        torch.zeros(rows, width, dtype=arms[0].field(bank, offset, width).dtype, device="npu")
        for bank, offset, width in fields
    ]

    def forward(arm):
        outputs = []
        for (bank, offset, width), value in zip(fields, values):
            arm.write_slots(slots, value, bank, offset)
            outputs.append(arm.read(None, 0, None, bank, offset, width))
        return outputs

    graphs, outputs = [], []
    for index, arm in enumerate(arms):
        with (
            patch.object(attention, "_supports_eager_field_io", return_value=False)
            if index == 0
            else patch.object(attention, "_read_eager_field", wraps=attention._read_eager_field) as reader,
            patch.object(attention, "_scatter_decode_field", wraps=attention._scatter_decode_field) as writer,
        ):
            forward(arm)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                outputs.append(forward(arm))
            if index:
                assert reader.call_count == 2 * (4 if ratio == 2 else 2)
                assert writer.call_count == reader.call_count
        graphs.append(graph)
    for arm in arms:
        for target, seed in zip(arm.kv_cache, initial):
            target.view(torch.int16).copy_(seed[1:-1])
    for step in range(3):
        slots.copy_((torch.arange(rows, device="npu") * 7 + step) % physical)
        for value in values:
            dst = words(value)
            low, high = (-32768, 32768) if dst.dtype == torch.int16 else (-(2**31), 2**31)
            host = torch.randint(low, high, dst.shape, dtype=dst.dtype, generator=gen)
            special = (
                [0, -32768, 1, 0x7F80, -128, 0x7F81, 0x7FC2, -127, -1]
                if dst.dtype == torch.int16
                else [
                    0,
                    -(2**31),
                    1,
                    0x7F800000,
                    -0x800000,
                    0x7F800001,
                    0x7FC12345,
                    -0x7FFFFF,
                    -1,
                ]
            )
            host[:, : len(special)] = torch.tensor(special, dtype=dst.dtype)
            dst.copy_(host)
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        for reference, actual in zip(outputs[0], outputs[1]):
            assert torch.equal(words(actual), words(reference))
            assert actual.is_contiguous()
        for before, after in zip(arms[0].kv_cache, arms[1].kv_cache):
            assert torch.equal(before.view(torch.int16), after.view(torch.int16))
        for arm in arms:
            for bank, seed in zip(arm.kv_cache, initial):
                full = bank.view(torch.int16).as_strided(seed.shape, seed.stride(), storage_offset=0)
                assert torch.equal(full[0], seed[0]) and torch.equal(full[-1], seed[-1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_eager_field_read_keeps_negative_and_repeated_slots(dtype):
    base = torch.randint(-32768, 32768, (259, 1024), dtype=torch.int16).npu()
    storage = base.view(dtype)
    if dtype == torch.float32:
        storage = torch.cat((storage, storage), 1)
    field = storage[1:-1, 128:640]
    slots = torch.tensor([-1, 0, 0, 1, 256], dtype=torch.int64, device="npu")
    assert attention._supports_eager_field_io(field, slots)
    expected = field[slots]
    actual = attention._read_eager_field(field, slots)
    assert torch.equal(words(actual), words(expected))
    assert not attention._supports_eager_field_io(field, slots, field[:5])
    alias = field.as_strided((5, 512), (512, 1))
    assert alias.is_contiguous() and not attention._supports_eager_field_io(field, slots, alias)


def test_eager_field_cpu_fallback_and_alias_guard():
    page = torch.zeros(257, 1024)[:, :512]
    slots = torch.tensor([0, 1])
    assert not attention._supports_eager_field_io(page, slots, page[:2])
