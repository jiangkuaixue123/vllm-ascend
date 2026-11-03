# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from tests.ut.models.deepseek_v41 import paged_cache, test_attention, test_dspark, test_model
from tests.ut.models.deepseek_v41.test_attention import hidden_tokens, run_batch
from tests.ut.models.deepseek_v41.test_dspark import run_block
from tests.ut.models.deepseek_v41.test_model import hash_batch
from vllm_ascend.models.deepseek_v41.attention import RequestSpan
from vllm_ascend.spec_decode.dspark_proposer import AscendDeepseekV41DSparkProposer

# Import only the fixtures these CPU tests use, without the attention suite's
# module-wide NPU parametrization.
make_layer = test_attention.make_layer
draft = test_dspark.draft
paged_hash_model = test_model.paged_hash_model


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_two_suffixes_share_read_only_prefix_pages(make_layer, ratio):
    prefix, a, b = hidden_tokens(4), hidden_tokens(3, 45), hidden_tokens(2, 78)
    layer = make_layer(ratio)
    run_batch(layer, prefix, [RequestSpan(0, 4, 4)], [[5, 1]])
    shared = paged_cache.snapshot(layer.cache, 5)
    cold = []
    for suffix in (b, a):
        combined = torch.cat((prefix, suffix))
        expected, _ = run_batch(make_layer(ratio), combined, [RequestSpan(0, len(combined), len(combined))], [[1, 4]])
        cold.append(expected[len(prefix) :])
    resumed, _ = run_batch(
        layer,
        torch.cat((b, a)),
        [RequestSpan(0, 2, 6), RequestSpan(2, 5, 7)],
        [[5, 3], [5, 8]],
    )
    torch.testing.assert_close(resumed, torch.cat(cold), atol=0, rtol=0)
    torch.testing.assert_close(paged_cache.bits(layer.cache, 5), shared, atol=0, rtol=0)


def test_engram_branches_restore_raw_history_from_the_same_prefix(paged_hash_model):
    model = paged_hash_model
    prefix, branches = [2, 4, 6, 8], ([3, 5], [9, 1, 7])
    hash_batch(model, torch.tensor(prefix), [RequestSpan(0, 4, 4)], [[4, 7]])
    cache = model.layers[0].attn.cache
    pages = paged_cache.snapshot(cache, [4, 7])
    actual, _ = hash_batch(
        model,
        torch.tensor(branches[0] + branches[1]),
        [RequestSpan(0, 2, 6), RequestSpan(2, 5, 7)],
        [[4, 7, 2, 0], [4, 7, 5, 6]],
    )
    expected = torch.cat(
        [
            model.engram_hasher.hash_tokens(prefix + s, model.engram_hasher.new_state(), start_position=0)[
                len(prefix) :
            ]
            for s in branches
        ]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(paged_cache.bits(cache, [4, 7]), pages)


@torch.inference_mode()
def test_skipped_draft_populates_reusable_context_and_preserves_old_pages(draft):
    """A batch can skip drafting near max length while other rows still extend reusable prefixes."""
    proposer = object.__new__(AscendDeepseekV41DSparkProposer)
    proposer.model = draft
    proposer._layer_group_idx = [0, 1, 0]
    aux = torch.randn((8, 96), generator=torch.Generator().manual_seed(472)).bfloat16()
    positions = torch.tensor([0, 1, 2, 3, 4, 5])
    # First four rows completed one shareable block. Another request writes
    # its suffix elsewhere. The final scheduled row is a masked draft slot.
    groups = {0: torch.tensor([4, 5, 6, 7, 20, -1, 11]), 1: torch.tensor([8, 9, 10, 11, 16, -1, 7])}
    proposer._per_group_slot_mappings = groups
    proposer.store_skipped_target_context(aux.chunk(3, dim=-1), positions)
    context = draft.combine_hidden_states(aux[:6])
    for stage, gid in zip(draft.mtp, proposer._layer_group_idx):
        page = stage.attn.cache.pages()[0]
        expected = stage.attn.project_kv(context, positions)
        torch.testing.assert_close(page.index_select(0, groups[gid][:5])[:, :32], expected[:5], rtol=0, atol=0)
        assert torch.count_nonzero(page[0]) == 0 and torch.count_nonzero(page[-1]) == 0

    # Recompute into the same stage tables from scratch as an independent
    # cold reference; the warm proposal consumes no target states for its hit.
    saved = [paged_cache.snapshot(s.attn.cache) for s in draft.mtp]
    # Runner can use a different group mapping at each stage. Use one table
    # for this cold/warm block comparison after verifying grouped writes above.
    for stage, gid in zip(draft.mtp, proposer._layer_group_idx):
        block = int(groups[gid][0]) // 4
        for bank in stage.attn.cache.kv_cache:
            bank[3].copy_(bank[block])
    ids = [2, 15, 15, 15, 15]
    warm = run_block(draft, ids, [4], [[3, 6, 0]])
    for stage, previous in zip(draft.mtp, saved):
        for bank, old in zip(paged_cache.bits(stage.attn.cache), previous):
            bank.copy_(old)
    draft.precompute_and_store_context_kv(context[:4], positions[:4], [torch.arange(12, 16)] * 3)
    cold = run_block(draft, ids, [4], [[3, 6, 0]])
    torch.testing.assert_close(warm, cold, rtol=0, atol=0)


def image_request(colour, uid=None):
    image = Image.new("RGB", (64, 64), colour)
    inputs = ProcessorInputs(
        prompt=[10, 11, 12, 13] + [77] * 8 + [21, 22, 23, 24, 25],
        mm_data_items=MultiModalDataParser().parse_mm_data({"image": image}),
        mm_uuid_items=None if uid is None else {"image": [uid]},
    )
    image_hash = inputs.get_mm_hashes("deepseek_v41", "sha256")["image"][0]
    # V4.1 embeds the complete image interval. Text tokens on either side do not consume feature rows.
    feature = MultiModalFeatureSpec(
        data=None, modality="image", identifier=image_hash, mm_position=PlaceholderRange(offset=4, length=8)
    )
    return Request(
        request_id=f"{colour}-{uid}",
        prompt_token_ids=inputs.prompt,
        mm_features=[feature],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(4, sha256),
    )


def test_upstream_cache_identity_uses_image_and_survives_chunk_boundaries(make_layer):
    init_none_hash(sha256)
    layer = make_layer(2)
    group = KVCacheGroupSpec(["target", "draft"], layer.cache.get_kv_cache_spec(None))
    cfg = KVCacheConfig(num_blocks=32, kv_cache_tensors=[], kv_cache_groups=[group])
    manager = KVCacheManager(cfg, max_model_len=32, scheduler_block_size=4, hash_block_size=4)
    first = image_request("blue")
    # Populate two chunks, with their split inside the visual interval.
    manager.allocate_slots(first, 8)
    first.num_computed_tokens = 8
    manager.allocate_slots(first, 9)
    first.num_computed_tokens = 17
    manager.free(first)
    assert manager.get_computed_blocks(image_request("blue"))[1] == 16
    assert manager.get_computed_blocks(image_request("red"))[1] == 4
    assert manager.get_computed_blocks(image_request("red", "other-media"))[1] == 4
    # Upstream treats an explicit media UUID as the caller's content identity.
    assert image_request("blue", "stable-media").block_hashes == image_request("red", "stable-media").block_hashes
