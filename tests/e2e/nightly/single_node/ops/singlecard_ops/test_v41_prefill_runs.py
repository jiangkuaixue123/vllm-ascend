# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run descriptors preserve slot order, boundaries and request-cache ownership."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.native_attention import PREFILL_LENGTH_TAG, PrefillAttentionInputs
from vllm_ascend.ops.triton.deepseek_v41_prefill_slots import (
    PREFILL_RUNS_TAG,
    SLOT_LIMIT,
    encode_prefill_runs,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture(autouse=True)
def device():
    torch.npu.set_device(0)
    init_device_properties_triton()


@pytest.mark.parametrize("width", [512, 640])
@torch.inference_mode()
def test_runs_keep_random_duplicate_descending_and_cross_page_slots(width):
    gen = torch.Generator().manual_seed(376)
    slots = torch.randint(0, 8192, (128, 1, width), generator=gen, dtype=torch.int32)
    slots[:32, 0] = torch.arange(width) + 15
    slots[32:64, 0] = torch.arange(width).flip(0)
    slots[64:80] = 33
    slots[80:96, :, 117:] = -1
    slots[96:] = -1
    slots[95, 0, :4] = torch.tensor([SLOT_LIMIT - 5, SLOT_LIMIT - 4, SLOT_LIMIT - 3, SLOT_LIMIT - 2])
    expected = torch.ones_like(slots)
    for col in range(width - 2, -1, -1):
        continued = (slots[..., col] >= 0) & (slots[..., col + 1] == slots[..., col] + 1)
        expected[..., col] = torch.where(continued, expected[..., col + 1] + 1, 1)
    expected.clamp_(max=128)
    source = slots.npu()
    table = torch.zeros((128, width // 16), dtype=torch.int32, device="npu")
    table[:, 0] = PREFILL_LENGTH_TAG
    table[:, 1] = (source >= 0).sum(-1).view(-1)
    lengths = table[:, 1].clone()
    encoded = encode_prefill_runs(source, table, SLOT_LIMIT - 1)
    words = encoded.cpu()
    assert torch.equal(torch.where(words >= 0, words & 0xFFFFFF, -1), slots)
    assert torch.equal((words[words >= 0] >> 24) + 1, expected[words >= 0])
    assert torch.equal(table[:, 1], lengths)
    assert (table[:, 0] == PREFILL_RUNS_TAG).all().item()
    assert torch.equal(source.cpu(), slots)


@pytest.mark.parametrize(
    "rows,width,length,noncontiguous",
    [(127, 640, 8192, False), (128, 128, 8192, False), (128, 640, SLOT_LIMIT, False), (128, 640, 8192, True)],
)
def test_unsupported_layout_keeps_original_protocol(rows, width, length, noncontiguous):
    slots = torch.zeros((rows, 1, width * (2 if noncontiguous else 1)), dtype=torch.int32, device="npu")
    if noncontiguous:
        slots = slots[..., ::2]
    table = torch.full((rows, width // 16), PREFILL_LENGTH_TAG, dtype=torch.int32, device="npu")
    assert encode_prefill_runs(slots, table, length) is slots
    assert (table == PREFILL_LENGTH_TAG).all().item()


def test_prepared_protocol_cache_reuses_only_same_source_and_geometry():
    prepared = PrefillAttentionInputs()
    positions = torch.arange(128, 256, device="npu")
    topk = torch.arange(512, device="npu", dtype=torch.int32).expand(128, -1).contiguous()
    args = (positions, 128, 256, 0, 256, 512, 128)
    canonical = prepared.prepare_slots(topk, *args)
    encoded = prepared.prepare_slots(topk, *args, encode_runs=True)
    assert torch.equal(encoded & 0xFFFFFF, canonical)
    assert prepared.prepare_slots(topk, *args, encode_runs=True) is encoded
    assert (prepared.slot_table[:, 0] == PREFILL_RUNS_TAG).all().item()
    changed = topk.clone()
    changed[:, :32] = -1
    updated = prepared.prepare_slots(changed, *args, encode_runs=True)
    assert updated is not encoded
    assert (prepared.slot_table[:, 1] == 608).all().item()
    ordinary = prepared.prepare_slots(changed, *args)
    assert torch.equal(torch.where(updated >= 0, updated & 0xFFFFFF, -1), ordinary)
    assert (prepared.slot_table[:, 0] == PREFILL_LENGTH_TAG).all().item()
