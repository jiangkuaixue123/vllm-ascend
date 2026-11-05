# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned A3 correctness entry; this test does not certify service performance."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.dspark import block_attention, ring_positions, try_stage_block_inputs
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("short", [False, True])
@torch.inference_mode()
def test_staging_preserves_single_request_native_words(batch, short):
    generator = torch.Generator().manual_seed(185)
    histories = [128, 129, 255, 256, 8192, 8193, 153593, 153594][:batch]
    if short:
        histories[0] = 127
    backing = torch.randn(12 * 128, 1024, generator=generator).bfloat16().to("npu")
    field = backing[:, :512]
    table = torch.randint(12, (batch, 1200), dtype=torch.int32, generator=generator).to("npu")
    table[:, 0] = 0
    q = torch.randn(batch * 5, 16, 512, generator=generator).bfloat16().to("npu")
    query = torch.randn(batch * 5, 512, generator=generator).bfloat16().to("npu")
    sink = torch.zeros(16, device="npu", dtype=torch.float32)
    metadata = SimpleNamespace(
        requests=[SimpleNamespace(begin=i * 5, end=(i + 1) * 5, start_pos=h) for i, h in enumerate(histories)],
        num_actual_tokens=batch * 5,
        block_table_tensor=table,
    )
    cache = SimpleNamespace(block_size=128, field=lambda *args: field)
    before = backing.contiguous().view(torch.int16).cpu()
    staged = try_stage_block_inputs(q, query, metadata, cache, window_size=128, block_size=5, dtype=torch.bfloat16)
    assert (staged is not None) == (batch > 1 and not short)
    parent_kernel, candidate_kernel = DeepseekV41NativeAttention(), DeepseekV41NativeAttention()
    for row, history in enumerate(histories):
        positions = ring_positions(history, 128, device=q.device)
        slots = table[row][positions // 128].long() * 128 + positions % 128
        context = field[slots]
        rows = slice(row * 5, (row + 1) * 5)
        reference = block_attention(q[rows], context, query[rows], sink, 512**-0.5, parent_kernel)
        repeat = block_attention(q[rows], context, query[rows], sink, 512**-0.5, parent_kernel)
        assert torch.equal(reference.contiguous().view(torch.int16), repeat.contiguous().view(torch.int16))
        if staged is None:
            actual = block_attention(q[rows], context, query[rows], sink, 512**-0.5, candidate_kernel)
        else:
            kv, indices = staged
            actual = candidate_kernel(
                q[rows].unsqueeze(0), kv[row : row + 1], sink, indices, 512**-0.5, mode="SWA draft bidirectional"
            )[0]
        assert torch.equal(reference.contiguous().view(torch.int16), actual.contiguous().view(torch.int16))
    torch.npu.synchronize()
    assert torch.equal(backing.contiguous().view(torch.int16).cpu(), before)
