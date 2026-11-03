# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch
from vllm.config.cache import CacheConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Backend, DeepseekV41Cache, DeepseekV41MetadataBuilder
from vllm_ascend.utils import refresh_block_size

MAX_MODEL_LEN = 150_000


@pytest.mark.parametrize(
    "requested,apc,chunked,expected",
    [
        (128, True, True, 128),
        (None, True, True, 128),
        (64, True, False, 128),
        (64, False, True, 128),
        (64, False, False, 64),
    ],
)
def test_v41_platform_page_size_policy(requested, apc, chunked, expected):
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=requested, enable_prefix_caching=apc),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v41"), is_hybrid=False),
        speculative_config=None,
    )
    refresh_block_size(config)
    assert config.cache_config.block_size == expected


@pytest.mark.parametrize("block_size", [64, 128])
def test_reused_prefix_and_dspark_graph_tables_at_page_boundary(block_size):
    init_none_hash(sha256)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN, enforce_eager=False),
        num_speculative_tokens=5,
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[24], static_forward_context={}),
    )
    cache = DeepseekV41Cache(SimpleNamespace(head_dim=32, index_head_dim=32), config, "target")
    spec = cache.get_kv_cache_spec(config)
    assert spec.block_size == block_size and DeepseekV41Backend.supports_block_size(block_size)
    manager = KVCacheManager(
        KVCacheConfig(
            num_blocks=12, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["target", "draft"], spec)]
        ),
        max_model_len=block_size * 2,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
    )

    def request(name, suffix):
        return Request(
            request_id=name,
            prompt_token_ids=[1] * block_size + suffix,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=get_request_block_hasher(block_size, sha256),
        )

    cold = request("cold", [2])
    manager.allocate_slots(cold, block_size + 1)
    cold.num_computed_tokens = block_size + 1
    manager.free(cold)
    blocks, reused, _ = manager.get_computed_blocks(request("warm", [3, 4]))
    assert reused == block_size and len(blocks.blocks[0]) == 1

    builder = DeepseekV41MetadataBuilder(spec, ["target"], config, torch.device("cpu"))
    # The six verification tokens straddle the block boundary on a noncontiguous page table.
    positions = torch.arange(block_size - 3, block_size + 3)
    table = torch.tensor([[4, 2]], dtype=torch.int32)
    slots = torch.cat(
        (torch.arange(5 * block_size - 3, 5 * block_size), torch.arange(2 * block_size, 2 * block_size + 3))
    )
    common = SimpleNamespace(
        num_input_tokens=24,
        num_actual_tokens=6,
        num_reqs=1,
        max_query_len=6,
        max_seq_len=block_size + 3,
        seq_lens=torch.tensor([block_size + 3]),
        block_table_tensor=table,
        slot_mapping=slots,
    )
    capture = builder._update_decode(common, capture=True)
    replay = builder._update_decode(common, bounds=[0, 6])
    assert replay is capture and replay.block_table_tensor.shape == (24, math.ceil(MAX_MODEL_LEN / block_size))
    assert replay.block_table_tensor[:6, :2].tolist() == [[4, 2]] * 6
    torch.testing.assert_close(replay.seq_lens[:6], positions + 1)
    torch.testing.assert_close(replay.slot_mapping[:6], slots)
    assert replay.active[:6].all() and not replay.active[6:].any()
    torch.testing.assert_close(cache.slots(common, 0, positions), slots)
    common.block_table_tensor = table[:, :1]
    with pytest.raises(ValueError, match="block table does not span"):
        builder._update_decode(common, bounds=[0, 6])

    # The final DSpark verifier window also fits the partially occupied last page at 150k.
    common.max_seq_len = MAX_MODEL_LEN
    common.seq_lens.fill_(MAX_MODEL_LEN)
    common.block_table_tensor = torch.arange(replay.block_table_tensor.shape[1], dtype=torch.int32).unsqueeze(0)
    tail = torch.arange(MAX_MODEL_LEN - 6, MAX_MODEL_LEN)
    common.slot_mapping = tail
    assert builder._update_decode(common, bounds=[0, 6]) is capture
    torch.testing.assert_close(replay.seq_lens[:6], tail + 1)
    torch.testing.assert_close(replay.slot_mapping[:6], tail)
    torch.testing.assert_close(cache.slots(common, 0, tail), tail)
