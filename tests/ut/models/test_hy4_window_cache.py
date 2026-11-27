# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec, KVCacheLayout
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.models.hy_v4_flash.attention import AscendHYV4FlashAttention
from vllm_ascend.models.hy_v4_flash.kv_cache import Hy4SlidingWindowCacheSpec, group_hy4_cache_specs
from vllm_ascend.worker.v2.attn_utils import allocate_shared_attention_kv_cache


def test_hy4_window_pages_share_the_sparse_pool():
    page = dict(block_size=64, num_kv_heads=1, head_size=1025, dtype=torch.uint8)
    sparse = HiddenStateCacheSpec(**page)
    window = Hy4SlidingWindowCacheSpec(**page, sliding_window=512)
    sparse_names = [f"sparse_{index}" for index in range(12)]
    window_names = [f"window_{index}" for index in range(30)]
    specs = dict.fromkeys(sparse_names, sparse) | dict.fromkeys(window_names, window)
    assert KVCacheSpecRegistry.get_manager_class(window) is SlidingWindowManager
    assert group_hy4_cache_specs(dict.fromkeys(sparse_names, sparse), disable_hybrid=False) is None
    with pytest.raises(ValueError, match="requires the hybrid KV cache manager"):
        group_hy4_cache_specs(specs, disable_hybrid=True)
    groups = group_hy4_cache_specs(specs, disable_hybrid=False)
    assert [len(group.layer_names) for group in groups] == [12, 10, 10, 10]
    assert {name for group in groups for name in group.layer_names} == specs.keys()

    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="hy_v4_flash")),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
    )
    num_blocks = 4
    memory = len(sparse_names) * sparse.page_size_bytes * num_blocks
    cache = get_kv_cache_config_from_groups(config, groups, memory)
    assert cache.num_blocks == num_blocks
    allocation_sizes = []

    def allocate(size):
        allocation_sizes.append(size)
        return torch.zeros(size, dtype=torch.int8)

    raw = allocate_shared_attention_kv_cache(cache, allocate, split_kv=False)
    assert allocation_sizes == [num_blocks * sparse.page_size_bytes] * len(sparse_names)
    first_sparse = raw[sparse_names[0]].reshape(num_blocks, sparse.page_size_bytes)
    first_window = raw[window_names[0]].reshape(num_blocks, window.page_size_bytes)
    second_window = raw[window_names[3]]
    assert first_sparse.data_ptr() == first_window.data_ptr()
    assert first_sparse.data_ptr() != second_window.data_ptr()
    # The allocator gives live groups distinct block IDs even when they share a layer slot.
    first_sparse[1].fill_(20)
    first_window[2].fill_(31)
    assert first_sparse[1, 0] == 20
    assert first_window[2, 0] == 31


def test_hy4_kv_producer_keeps_existing_page_layout():
    layer = SimpleNamespace(cache_width=1025, use_fp8_cache=True, is_window_layer=True, page_planes=True, window=512)
    config = SimpleNamespace(
        additional_config={"hy4flash_bounded_window_cache": True},
        cache_config=SimpleNamespace(block_size=64),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
    )
    producer_spec = AscendHYV4FlashAttention.get_kv_cache_spec(layer, config)
    config.kv_transfer_config.kv_role = "kv_consumer"
    consumer_spec = AscendHYV4FlashAttention.get_kv_cache_spec(layer, config)
    assert isinstance(producer_spec, HiddenStateCacheSpec)
    assert isinstance(consumer_spec, Hy4SlidingWindowCacheSpec)
    assert producer_spec.page_size_bytes == consumer_spec.page_size_bytes
