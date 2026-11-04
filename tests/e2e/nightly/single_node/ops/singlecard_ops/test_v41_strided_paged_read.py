# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise equivalence of partial fields before/after real ACLGraph replay."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import envs
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

if HAS_TRITON and not envs.VLLM_BATCH_INVARIANT:
    from vllm_ascend.ops.triton.deepseek_v41_paged_read import read_strided_cache_field


@pytest.mark.skipif(not HAS_TRITON or envs.VLLM_BATCH_INVARIANT, reason="Requires non-batch-invariant Triton-Ascend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width, stride, offset", [(128, 1, 512), (512, 1, 0), (512, 1, 512), (512, 2, 1)])
def test_strided_cache_field_graph_replay(dtype, width, stride, offset):
    init_device_properties_triton()
    bits_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    limit = torch.iinfo(bits_dtype)
    gen = torch.Generator().manual_seed(23)
    raw = torch.randint(limit.min, limit.max, (35, 1024), dtype=bits_dtype, generator=gen)
    raw[2, offset] = limit.min  # Valid negative zero must not be canonicalized.
    raw[3, offset] = limit.max  # The reference mask also governs NaN payloads on the A3.
    raw = raw.to("npu")
    field = raw[2:, offset : offset + width * stride : stride].view(dtype)
    assert field.storage_offset() > 0 and field.shape == (33, width)
    slots = torch.tensor([[0, 1, -1, 33, 32], [32, 0, 0, 8, 16]], dtype=torch.int64, device="npu")
    active = torch.zeros(slots.shape[0], dtype=torch.bool, device="npu")
    # Logical slot numbers match physical slots. This permits the same non-unit-column
    # field and invalid inputs as the original regression through real decode_read dispatch.
    length = field.shape[0]
    table = torch.arange(length, dtype=torch.int32, device="npu").expand(slots.shape[0], -1).contiguous()
    md = SimpleNamespace(
        max_model_len=length,
        seq_lens=torch.full_like(active, length, dtype=torch.long),
        active=active,
        block_table_tensor=table,
    )
    cache = SimpleNamespace(block_size=1, field=lambda bank, start, size: field)
    before = raw.clone()
    with patch.object(attention, "read_strided_cache_field", wraps=read_strided_cache_field) as fused:
        attention.DeepseekV41Cache.decode_read(cache, md, slots, 0, 0, width)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = attention.DeepseekV41Cache.decode_read(cache, md, slots, 0, 0, width)
        assert fused.call_count == (2 if stride == 1 else 0)
    for step in range(3):
        if step == 1:
            active.fill_(True)
        elif step == 2:
            slots.copy_(slots.flip([1]))
            active[1].fill_(False)
            # Both zero and the last physical slot are real even after same-address input updates.
        graph.replay()
        torch.npu.synchronize()
        valid = active[:, None] & (slots >= 0) & (slots < field.shape[0])
        safe = torch.where(valid, slots, 0)
        expected = field.index_select(0, safe.flatten()).reshape(*slots.shape, width)
        expected = torch.where(valid[..., None], expected, 0)
        torch.testing.assert_close(result.view(bits_dtype), expected.view(bits_dtype), rtol=0, atol=0)
    torch.testing.assert_close(raw, before, rtol=0, atol=0)


@pytest.mark.skipif(not HAS_TRITON or envs.VLLM_BATCH_INVARIANT, reason="Requires non-batch-invariant Triton-Ascend")
@pytest.mark.parametrize("shape", [(0, 17), (6, 0)])
def test_strided_cache_field_empty(shape):
    field = torch.empty((11, 1024), dtype=torch.float32, device="npu")[:, 512:]
    slots = torch.empty(shape, dtype=torch.int64, device="npu")
    active = torch.empty_like(slots, dtype=torch.bool)
    result = read_strided_cache_field(field, slots, active)
    assert result.shape == (*shape, 512) and result.dtype == field.dtype and result.device == field.device
