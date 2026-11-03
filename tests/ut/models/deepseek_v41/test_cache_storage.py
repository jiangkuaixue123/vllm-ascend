# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from tests.ut.models.deepseek_v41 import paged_cache
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Cache
from vllm_ascend.utils import calc_split_factor


def config(block):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block, num_gpu_blocks_override=None),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        model_config=SimpleNamespace(max_model_len=150000),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1),
    )


@pytest.mark.parametrize("block", [64, 128])
def test_heterogeneous_full_pages_account_for_150k(block):
    vc = config(block)
    model = SimpleNamespace(head_dim=512, index_head_dim=128)
    specs = {}
    for idx in range(43):
        name = f"layers.{idx}.attn.cache" if idx < 40 else f"mtp.{idx - 40}.attn.cache"
        ratio = {2: 2, 8: 2, 14: 2, 20: 1}.get(idx, 0)
        layer = DeepseekV41Cache(model, vc, name, source_ratio=ratio, token_history=idx == 0)
        specs[name] = layer.get_kv_cache_spec(vc)
    groups = get_kv_cache_groups(vc, specs)
    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)
    per_page = 61596 * block
    assert groups[0].kv_cache_spec.page_size_bytes == per_page
    # Include the null block required by the scheduler in addition to actual pages.
    blocks = (150000 + block - 1) // block + 1
    kv = get_kv_cache_config_from_groups(vc, groups, blocks * per_page)
    assert kv.num_blocks == blocks
    assert sum(t.size for t in kv.kv_cache_tensors) == blocks * per_page < 9 * 1024**3
    for allocation in kv.kv_cache_tensors:
        (name,) = allocation.shared_by
        spec = specs[name]
        assert allocation.size == blocks * spec.page_size_bytes
        splits = calc_split_factor([spec.head_size, spec.head_size_v])
        assert [int(allocation.size // f) for f in splits] == [
            blocks * block * dim * 2 for dim in (spec.head_size, spec.head_size_v)
        ]


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_source_fields_and_history_preserve_distinct_precision_and_slots(device):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("No NPU is available")
    vc = config(4)
    layer = DeepseekV41Cache(
        SimpleNamespace(head_dim=32, index_head_dim=16), vc, "source", source_ratio=2, token_history=True
    )
    paged_cache.allocate(layer, 4, 4, device=device)
    slots = torch.tensor([9, 2], device=device)
    raw = torch.linspace(0.25001, 0.25111, 64, dtype=torch.float32, device="cpu").view(2, 32)
    score = raw + 0.00007
    window = torch.arange(64, device="cpu").view(2, 32).bfloat16() / 4
    index = window[:, :16] / 2
    for bank, offset, values in (
        (0, 0, window),
        (0, 32, raw),
        (1, 0, score),
        (1, 32, index),
        (1, 48, torch.tensor([[129279], [-1]], device="cpu")),
    ):
        layer.write_slots(slots, values.to(device), bank, offset)
    for bank, offset, expected in (
        (0, 0, window),
        (0, 32, raw),
        (1, 0, score),
        (1, 32, index),
        (1, 48, torch.tensor([[129279.0], [-1.0]], dtype=torch.float32, device="cpu")),
    ):
        field = layer.field(bank, offset, expected.shape[-1])
        torch.testing.assert_close(field.index_select(0, slots).cpu(), expected, rtol=0, atol=0)
        assert torch.count_nonzero(field[1].cpu()) == 0
    # Independently inspect the underlying BF16 banks at two noncontiguous
    # physical slots, including all FP32 mantissa bits and signed token IDs.
    key, value = layer.pages()
    assert key.dtype == value.dtype == torch.bfloat16
    key_words = key.index_select(0, slots).cpu()
    value_bits = value.index_select(0, slots).cpu().view(torch.int32)
    assert torch.equal(value_bits[:, :32], raw.contiguous().view(torch.int32))
    assert torch.equal(value_bits[:, 32:64], score.contiguous().view(torch.int32))
    assert torch.equal(key_words[:, 32:48], index)
    assert key_words[:, 48:50].view(torch.float32).flatten().tolist() == [129279.0, -1.0]
    with pytest.raises(ValueError, match="Unallocated"):
        layer.field(1, 15, 8)
