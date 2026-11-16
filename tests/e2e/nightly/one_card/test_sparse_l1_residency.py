# SPDX-License-Identifier: Apache-2.0
"""Compare resident QK/PV with the original per-row slot protocol on real A3."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.native_attention import (
    PREFILL_LENGTH_TAG,
    DeepseekV41NativeAttention,
    physical_indices,
)
from vllm_ascend.ops.triton.deepseek_v41_prefill_slots import encode_prefill_runs
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize(
    "rows,width,mixed", [(128, 128, False), (129, 512, False), (257, 640, False), (257, 640, True)]
)
@torch.inference_mode()
def test_resident_tiles_with_legacy_rows_and_replay(rows, width, mixed):
    torch.npu.set_device(0)
    assert enable_custom_op()
    torch.manual_seed(2417)
    pool = 2048
    q = torch.randn(1, rows, 64, 512, device="npu", dtype=torch.bfloat16) / 8
    kv = torch.randn(1, pool, 512, device=q.device, dtype=q.dtype) / 8
    sink = torch.linspace(-1, 2, 64, device=q.device)
    ids = torch.randint(0, pool, (1, rows, width), device=q.device, dtype=torch.int32)
    ids[0, :, :128] = torch.arange(rows, device=q.device)[:, None] + torch.arange(128, device=q.device)[None, :]
    if width > 128:
        ids[0, 1::3, 128:256] = 777
        ids[0, 2::3, 128:256] = torch.arange(128, device=q.device).flip(0)
    table = torch.zeros(rows, width // 16, dtype=torch.int32, device=q.device)
    table[:, 0] = PREFILL_LENGTH_TAG
    slots = torch.empty((rows, 1, width), dtype=ids.dtype, device=ids.device)
    lengths = torch.tensor([0, 1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513, 639, 640], device=q.device)
    positions = torch.arange(width, device=q.device)[None, None, :]
    row = torch.arange(rows, device=q.device)
    op = DeepseekV41NativeAttention()

    def update(step):
        q.copy_(torch.randn_like(q) / (step + 8))
        kv.copy_(torch.randn_like(kv) / (step + 8))
        live = lengths[(row + step) % lengths.numel()].clamp_max(width)
        selected = torch.where(positions < live[None, :, None], ids, -1)
        slots.copy_(physical_indices(selected, pool, pool, width, lengths=table[:, 1]))

    def run():
        encoded_table = table.clone()
        encoded = encode_prefill_runs(slots, encoded_table, pool)
        if mixed:
            encoded = torch.where((row % 2 == 0)[:, None, None], encoded, slots)
            encoded_table[:, 0] = torch.where(row % 2 == 0, encoded_table[:, 0], PREFILL_LENGTH_TAG)

        def attention(selected, header):
            return op._native(
                q,
                kv,
                sink,
                selected,
                512**-0.5,
                "L1 consumer regression",
                None,
                physical_slots=True,
                slot_table=header,
            )

        return attention(slots, table), attention(encoded, encoded_table)

    update(0)
    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        expected, actual = run()
    for step in range(4):
        update(step)
        graph.replay()
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))
