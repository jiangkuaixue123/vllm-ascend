# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.ut.models.deepseek_v41 import paged_cache
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Cache, DeepseekV41DecodeMetadata
from vllm_ascend.ops.triton import deepseek_v41_cache as fused


def make_cache(ratio, device, blocks=4, block_size=16):
    config = SimpleNamespace(head_dim=512, index_head_dim=128)
    vc = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache = DeepseekV41Cache(config, vc, "fusion-test", source_ratio=ratio, token_history=True)
    paged_cache.allocate(cache, blocks, block_size, device=device)
    return cache


def to_device(md, device):
    return DeepseekV41DecodeMetadata(
        md.seq_lens.to(device),
        md.slot_mapping.to(device),
        md.block_table_tensor.to(device),
        md.active.to(device),
        md.max_model_len,
        md.query_width,
    )


@pytest.mark.parametrize("ratio", [0, 1, 2])
@pytest.mark.parametrize("query_width", [1, 6])
def test_npu_triton_cache_fields_read_write_graph(ratio, query_width):
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    torch.manual_seed(731)
    reference, actual = make_cache(ratio, "cpu"), make_cache(ratio, "npu")
    for bank, target in zip(reference.kv_cache, actual.kv_cache):
        bank.copy_(torch.randn_like(bank))
        target.copy_(bank)
    slots = torch.tensor([-1, 0, 23, -1, 47, 100, 15, -1])
    active = torch.tensor([False, True, True, False, True, True, True, False])
    md = DeepseekV41DecodeMetadata(
        torch.tensor([0, 30, 15, 10, 64, 40, 60, 0]),
        slots,
        torch.tensor(
            [
                [0, 2, 1, 3],
                [0, 2, 1, 3],
                [1, 0, -1, 3],
                [0, 1, 2, 3],
                [2, 1, 99, 0],
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [0, 1, 2, 3],
            ],
            dtype=torch.int32,
        ),
        active,
        64,
        query_width,
    )
    device_md = to_device(md, "npu")
    fields = [(0, 0, 512), (0, 3, 17), (1, 640, 1)]
    if ratio:
        fields += [(0, 512, 512), (1, 512, 128)]
    if ratio > 1:
        fields += [(1, 0, 512)]
    positions = torch.tensor([-3, 0, 7, 15, 20, 39, 63, 80]).expand(8, -1)
    device_positions = positions[0].to("npu").expand(8, -1)
    for bank, offset, width in fields:
        values = torch.randn(8, width * 2)[:, ::2]
        enabled = active.clone()
        enabled[4] = False
        reference.decode_write(md, values, bank, offset, enabled)
        device_values = torch.empty((8, width * 2), device="npu")
        device_values[:, ::2].copy_(values)
        fused.decode_write(actual, device_md, device_values[:, ::2], bank, offset, enabled.npu())
        # Compare bank bits, including untouched fields and invalid slots.
        for expected, result in zip(paged_cache.bits(reference), paged_cache.bits(actual)):
            assert torch.equal(result.cpu(), expected)
        expected = reference.decode_read(md, positions, bank, offset, width)
        result = fused.decode_read(actual, device_md, device_positions, bank, offset, width)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = fused.decode_read(actual, device_md, device_positions, bank, offset, width)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    # Read/write addresses and masks must be refreshed by graph replay.
    values = torch.randn(8, 512).npu()
    enabled = torch.ones(8, dtype=torch.bool, device="npu")
    fused.decode_write(actual, device_md, values, 0, 0, enabled)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fused.decode_write(actual, device_md, values, 0, 0, enabled)
    for bank, target in zip(reference.kv_cache, actual.kv_cache):
        target.copy_(bank)
    md.slot_mapping = torch.tensor([-1, 4, 25, -1, 0, 100, 3, -1])
    device_md.slot_mapping.copy_(md.slot_mapping)
    reference.decode_write(md, values.cpu(), 0, 0, torch.ones(8, dtype=torch.bool))
    graph.replay()
    torch.npu.synchronize()
    for expected, result in zip(paged_cache.bits(reference), paged_cache.bits(actual)):
        assert torch.equal(result.cpu(), expected)


def test_npu_triton_cache_duplicate_live_slots_keep_last_writer():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Requires a physical NPU")
    reference, actual = make_cache(0, "cpu"), make_cache(0, "npu")
    md = DeepseekV41DecodeMetadata(
        torch.ones(6, dtype=torch.int64),
        torch.tensor([0, 3, 0, 3, -1, 0]),
        torch.zeros(6, 4, dtype=torch.int32),
        torch.ones(6, dtype=torch.bool),
        64,
        1,
    )
    enabled = torch.tensor([True, True, True, False, True, False])
    values = torch.arange(6 * 512, dtype=torch.float32).reshape(6, 512)
    reference.decode_write(md, values, 0, 0, enabled)
    fused.decode_write(actual, to_device(md, "npu"), values.npu(), 0, 0, enabled.npu())
    for expected, result in zip(paged_cache.bits(reference), paged_cache.bits(actual)):
        assert torch.equal(result.cpu(), expected)
