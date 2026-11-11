# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend.models.deepseek_v41.decode_state import PagedHistory
from vllm_ascend.models.deepseek_v41.prefix_cache import DraftPages
from vllm_ascend.models.deepseek_v41.prefix_scheduler import reuse_projected_blocks
from vllm_ascend.models.deepseek_v41.spec_runner import _make_proposer_type


def manager(block, num_blocks, max_len):
    init_none_hash(sha256)
    spec = FullAttentionSpec(block_size=block, num_kv_heads=1, head_size=32, dtype=torch.bfloat16)
    config = KVCacheConfig(num_blocks, [], [KVCacheGroupSpec(["target"], spec)])
    return KVCacheManager(config, max_len, block, block, enable_caching=True, use_eagle=True)


def test_stock_prefix_hash_hits_survive_release_and_eviction():
    block = 128
    mgr = manager(block, 7, 512)
    hasher = get_request_block_hasher(block, sha256)

    def request(name, ids):
        return Request(name, ids, SamplingParams(max_tokens=1), None, block_hasher=hasher)

    def admit(req, expected):
        hits, n, _ = mgr.get_computed_blocks(req)
        assert n == expected
        assert mgr.allocate_slots(req, req.num_tokens - n, n, hits) is not None
        return mgr.get_block_ids(req.request_id)[0]

    first, second = [1] * block, [2] * block
    owner = request("owner", first + second + [3])
    owner_blocks = admit(owner, 0)
    warm = request("warm", first + second + [4])
    # When projections are absent stock EAGLE drops an additional whole block.
    assert mgr.get_computed_blocks(warm)[1] == block
    reuse_projected_blocks(mgr)
    warm_blocks = admit(warm, 2 * block)
    assert warm_blocks[:2] == owner_blocks[:2] and warm_blocks[2] != owner_blocks[2]
    mgr.free(owner)
    assert all(mgr.block_pool.blocks[p].ref_cnt == 1 for p in warm_blocks[:2])
    low = request("low", first + [9] * block + [8])
    low_blocks = admit(low, block)
    assert low_blocks[0] == warm_blocks[0] and low_blocks[1] != warm_blocks[1]
    mgr.free(warm)
    mgr.free(low)
    for i in range(4):
        cold = request(f"cold-{i}", [20 + i] * (2 * block + 1))
        admit(cold, 0)
        mgr.free(cold)
    # The allocator really evicted the old hash and recycled its physical pages.
    assert mgr.get_computed_blocks(request("again", first + second + [5]))[1] == 0


def test_fresh_shared_page_ngrams_match_after_owner_release_and_reclaim():
    state = PagedHistory(16, 4, 9)
    ref = SimpleNamespace(
        layout=SimpleNamespace(max_ngram_size=4),
        token_map=torch.arange(128),
        pad_id=0,
        multipliers=torch.tensor([[3, 5, 7, 11]]),
        primes=torch.tensor([[[13], [17], [19]]]),
        offsets=torch.tensor([[0, 13, 30]]),
    )

    def write(tokens, start, blocks):
        pos = np.arange(start, start + len(tokens))
        slots = np.asarray(blocks)[pos // 4] * 4 + pos % 4
        plan = state.plan(pos, slots, [blocks], [len(pos)])
        return state.hash_ids(ref, plan, tokens)

    common = list(range(10, 18))
    write(common[:3], 0, [5, 3, 4])
    write(common[3:] + [91], 3, [5, 3, 4])
    a = write([31, 32, 33, 34], 8, [5, 3, 7])
    b = write([60, 61, 62, 63], 8, [5, 3, 8])
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, write(common + [31, 32, 33, 34], 0, [1, 2, 6])[-4:])
    np.testing.assert_array_equal(b, write(common + [60, 61, 62, 63], 0, [1, 2, 6])[-4:])
    # Reuse the released physical IDs for different content; the previous
    # compressed-token history must not affect its first suffix.
    np.testing.assert_array_equal(write([77] * 9, 0, [5, 3, 4]), write([77] * 9, 0, [1, 2, 6]))


def test_physical_draft_commits_only_accepted_and_uses_shared_window_after_reorder():
    pages = DraftPages(SimpleNamespace(head_dim=32, num_nextn_predict_layers=3, sliding_window=4), 4, 0, "cpu")
    assert pages.values.shape == (3, 1, 33) and pages.values.dtype == torch.uint8
    pages.bind(9)
    ptr = pages.values.data_ptr()
    pages.bind(9)
    assert pages.values.data_ptr() == ptr and pages.bytes_per_block == 3 * 4 * 33
    with pytest.raises(RuntimeError, match="live"):
        pages.bind(8)

    # Encode the fixture's integer identity labels as exact FP8 zeros/ones in each real physical row.
    bits = torch.arange(32)

    class Draft:
        k = 2
        mtp = [SimpleNamespace(main_proj=SimpleNamespace(weight=torch.empty(4, 4)))]

        def commit_target(self, aux, slots, positions):
            for i in range(pages.layers):
                label = aux[:, :1].long() + 100 * i
                pages.write(i, slots, ((label >> bits) & 1).bfloat16())

    cls = _make_proposer_type(object)
    proposer = cls.__new__(cls)
    proposer.ready, proposer.capacity, proposer.device, proposer.receipt = True, 2, "cpu", None
    proposer.kv_cache_gid, proposer.draft_pages, proposer.model = 0, pages, Draft()
    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=18))
    batch = SimpleNamespace(req_ids=[])
    states = {}
    proposer.v41_runner = SimpleNamespace(input_batch=batch, requests=states)

    def block(seeds, slots, positions, role):
        proposer.window = torch.stack([(pages.read(i, slots).long() << bits).sum(-1) for i in range(pages.layers)])
        assert proposer.window.shape == (3, 2, 4)
        return SimpleNamespace(token_ids=torch.zeros((2, 2), dtype=torch.long))

    proposer._block = block

    def step(rows):
        batch.req_ids = [r[0] for r in rows]
        count, drafts, sampled, pos, features = {}, {}, [], [], []
        for req, tables, at, values, proposals, emitted in rows:
            states[req] = SimpleNamespace(prompt_token_ids=[0] * 5, num_computed_tokens=at, block_ids=(tables,))
            count[req], drafts[req] = len(values), proposals
            pos.extend(range(at, at + len(values)))
            features.extend(values)
            sampled.append(emitted)
        scheduler = SimpleNamespace(num_scheduled_tokens=count, scheduled_spec_decode_tokens=drafts)
        values = torch.tensor(features, dtype=torch.float32)[:, None].expand(-1, 4)
        proposer.worker_step(sampled, scheduler, torch.tensor(pos), values)

    a, b = [4, 2, 8, 7], [4, 2, 6, 3]
    step([("a", a, 0, [10, 11, 12], [], [])])
    step([("a", a, 3, [13, 14, 15, 16, 17, 18], [], [30])])
    # First b step skips both cached pages. Same batch commits its private
    # suffix and rejects both proposed tokens for a physical neighbor.
    step([("b", b, 8, [28], [], [31]), ("a", a, 9, [19, 91, 92], [91, 92], [23])])
    assert proposer.window[0].tolist() == [[28, 15, 16, 17], [18, 19, 16, 17]]
    assert not pages.read(0, torch.from_numpy(pages.addresses(a, [10, 11]))).any()
    assert (proposer.window[2] - proposer.window[0]).eq(200).all()
    del states["a"]
    step([("b", b, 9, [29], [], [4])])
    assert proposer.window[0, 0].tolist() == [28, 29, 16, 17]
    assert not proposer.window[:, 1].any()
    # Physical IDs are reused after eviction; staged rejections never enter
    # the cache and the immutable padding slot stays zero.
    step([("c", a, 0, list(range(41, 51)), [], [6])])
    assert proposer.window[0, 0].tolist() == [49, 50, 47, 48]
    assert not pages.values[:, pages.num_slots].any()
    # A provider can be later in the same batch than a request which hit
    # its scheduled prefix. All accepted rows are committed before proposing.
    step(
        [
            ("new-consumer", [8, 7, 5], 8, [228], [], [1]),
            ("new-provider", [8, 7, 3], 0, list(range(101, 110)), [], [2]),
        ]
    )
    assert proposer.window[0, 0].tolist() == [228, 106, 107, 108]
    with pytest.raises(ValueError, match="cover"):
        pages.addresses(a[:1], [8])


def test_draft_pages_pack_once_and_account_for_three_layers_and_padding():
    from vllm_ascend.models.deepseek_v41.eager_kernels import act_quant

    pages = DraftPages(SimpleNamespace(head_dim=512, num_nextn_predict_layers=3, sliding_window=128), 128, 2, "cpu")
    slots = torch.tensor([137, 19, 8, 21])
    assert pages.values.shape == (3, 257, 528)
    assert pages.bytes_per_block == 3 * 128 * 528
    assert pages.values.nbytes == 2 * pages.bytes_per_block + 3 * 528
    values = torch.randn((3, 4, 512), generator=torch.Generator().manual_seed(19)).bfloat16()
    values[0, 0, :2] = torch.tensor([0.0, -0.0])
    values[1, 1].mul_(2.0**-27)
    values[2, 2].mul_(2.0**14)
    for layer, raw in enumerate(values):
        expected = act_quant(raw.clone(), inplace=True)
        pages.write(layer, slots[:1], raw[:1])
        pages.write(layer, slots[1:], raw[1:])
        assert torch.equal(pages.read(layer, slots).view(torch.int16), expected.view(torch.int16))
        assert not pages.read(layer, torch.tensor([pages.num_slots])).view(torch.int16).any()


def test_three_real_draft_attention_projections_and_outputs_match_uncached_request():
    from torch import nn

    from vllm_ascend.models.deepseek_v41.dspark import DSparkAttention, DSparkDraft

    cfg = SimpleNamespace(head_dim=32, num_nextn_predict_layers=3, sliding_window=4)
    pages = DraftPages(cfg, 4, 5, "cpu")

    def build(store):
        torch.manual_seed(41)
        model = DSparkDraft.__new__(DSparkDraft)
        nn.Module.__init__(model)
        model.draft_pages = store
        model.max_model_len, model.max_requests = 16, 2
        model.mtp = nn.ModuleList()
        for layer_id in range(3):
            block = nn.Module()
            if layer_id == 0:
                block.main_proj = nn.Linear(96, 32, bias=False, dtype=torch.bfloat16)
                block.main_norm = nn.RMSNorm(32, dtype=torch.bfloat16)
            attn = DSparkAttention.__new__(DSparkAttention)
            nn.Module.__init__(attn)
            attn.pages, attn.page_layer = store, layer_id
            attn.head_dim = attn.o_lora_rank = 32
            attn.rope_head_dim = 8
            attn.n_local_heads = attn.n_local_groups = 1
            attn.window_size = 4
            attn.softmax_scale = 32**-0.5
            attn.attn_sink = nn.Parameter(torch.tensor([0.3], dtype=torch.float32))
            ang = torch.outer(torch.arange(24).float(), torch.tensor([1.0, 0.1, 0.01, 0.001]))
            attn.freqs_cis = torch.stack((ang.cos(), ang.sin()), -1)
            for name in ("wkv", "wq_a", "wq_b", "wo_a", "wo_b"):
                setattr(attn, name, nn.Linear(32, 32, bias=False, dtype=torch.bfloat16))
            attn.q_norm, attn.kv_norm = nn.RMSNorm(32, dtype=torch.bfloat16), nn.RMSNorm(32, dtype=torch.bfloat16)
            if store is None:
                attn.window_kv_cache = torch.zeros(2, 16, 32, dtype=torch.bfloat16)
            block.attn = attn
            model.mtp.append(block)
        return model

    old, paged = build(None), build(pages)
    inputs = torch.randn(12, 96, dtype=torch.bfloat16)
    pos = torch.arange(12)
    owner, consumer = [3, 1, 2], [3, 1, 4]
    old.commit_target(inputs, torch.ones_like(pos), pos)
    # A new request receives only the suffix. The owner finished after
    # populating two full blocks; each cached projected key is bit-identical.
    paged.commit_target(inputs[:8], torch.from_numpy(pages.addresses(owner, pos[:8])), pos[:8])
    paged.commit_target(inputs[8:], torch.from_numpy(pages.addresses(consumer, pos[8:])), pos[8:])
    proposal_hidden = torch.randn(1, 2, 32, dtype=torch.bfloat16)
    for i, (baseline, actual) in enumerate(zip(old.mtp, paged.mtp, strict=True)):
        restored = pages.read(i, torch.from_numpy(pages.addresses(consumer, pos)))
        assert torch.equal(restored.view(torch.int16), baseline.attn.window_kv_cache[1, :12].view(torch.int16))
        for last in (3, 8, 10, 11):
            expected = baseline.attn(proposal_hidden, torch.tensor([last]), torch.tensor([1]))
            rows = torch.from_numpy(pages.window_addresses(consumer, last)[None])
            observed = actual.attn(proposal_hidden, torch.tensor([last]), rows)
            assert torch.equal(observed, expected)


def test_short_generated_prefix_is_separate_from_long_prompt_and_user_salts():
    from vllm_ascend.models.deepseek_v41.prefix_scheduler import scope_prompt_hashes

    mgr = manager(4, 12, 16)
    reuse_projected_blocks(mgr)
    hasher = get_request_block_hasher(4, sha256)

    def make(name, prompt, salt=None):
        req = Request(name, prompt, SamplingParams(max_tokens=12), None, cache_salt=salt, block_hasher=hasher)
        scope_prompt_hashes(req, 4)
        return req

    tiny = make("short", [1, 2], "salt")
    tiny.append_output_token_ids([3, 4, 5])
    assert mgr.allocate_slots(tiny, 5) is not None
    mgr.free(tiny)
    equal = make("same", [1, 2], "salt")
    equal.append_output_token_ids([3, 4, 6])
    assert mgr.get_computed_blocks(equal)[1] == 4
    for prompt, salt in (([1, 2, 3, 4, 6], "salt"), ([1, 2], None), ([1, 2], '[2,"salt"]')):
        req = make("different", prompt, salt)
        if len(prompt) == 2:
            req.append_output_token_ids([3, 4, 6])
        assert mgr.get_computed_blocks(req)[1] == 0


@pytest.mark.parametrize("ratio", [1, 2])
def test_paged_target_reads_same_window_compressor_and_index_on_a_cache_hit(ratio):
    from torch import nn

    from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs
    from vllm_ascend.models.deepseek_v41.graph_ops import LayerPages, paged_attention

    torch.manual_seed(41)
    dim = 32

    def linear(a=dim, b=dim, dtype=torch.bfloat16):
        return nn.Linear(a, b, dtype=dtype, bias=False)

    def norm():
        return nn.RMSNorm(dim, dtype=torch.bfloat16)

    angle = torch.outer(torch.arange(16).float(), torch.tensor([1.0, 0.1, 0.01, 0.001]))
    idx = SimpleNamespace(
        index_head_dim=dim,
        k_norm=norm(),
        wk=linear(),
        rope_head_dim=8,
        index_topk=3,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        is_candidate_source=True,
        uses_candidates=False,
        n_local_heads=1,
        n_heads=1,
        softmax_scale=dim**-0.5,
        wq_b=linear(),
        weights_proj=linear(dim, 1),
    )
    comp = SimpleNamespace(
        norm=norm(),
        wkv=linear(dtype=torch.float32 if ratio == 2 else torch.bfloat16),
        wgate=linear(dtype=torch.float32),
    )
    attn = SimpleNamespace(
        head_dim=dim,
        n_local_heads=1,
        n_local_groups=1,
        o_lora_rank=dim,
        rope_head_dim=8,
        window_size=4,
        softmax_scale=dim**-0.5,
        freqs_cis=torch.stack((angle.cos(), angle.sin()), -1),
        wq_a=linear(),
        q_norm=norm(),
        wq_b=linear(),
        wkv=linear(),
        kv_norm=norm(),
        wo_a=linear(),
        wo_b=linear(),
        attn_sink=torch.tensor([0.1]),
        is_kv_source=True,
        is_index_source=True,
        compress_ratio=ratio,
        indexer=idx,
        compressor=comp,
    )
    reference = SimpleNamespace(
        layout=SimpleNamespace(max_ngram_size=2),
        token_map=torch.arange(32),
        pad_id=0,
        multipliers=torch.tensor([[3, 5]]),
        primes=torch.tensor([[[17]]]),
        offsets=torch.tensor([[0]]),
    )
    state = DecodeInputs(PagedHistory(16, 4, 9), 12, "cpu", [], window_sizes=(attn.window_size,))
    pages = LayerPages(attn, state)
    prefix = torch.randn(8, dim, dtype=torch.bfloat16)
    suffix = torch.randn(4, dim, dtype=torch.bfloat16)
    poison = torch.randn(4, dim, dtype=torch.bfloat16) * 10

    def run(start, x, table):
        p = np.arange(start, start + len(x))
        slots = np.asarray(table)[p // 4] * 4 + p % 4
        plan = state.history.plan(p, slots, [table], [len(x)], [12])
        state.stage(plan, p, reference, [])
        shared = {}
        out = paged_attention(attn, x[None], state, pages, shared, int(p[-1]) + 1)
        return out.clone(), shared["selected"].clone()

    owner, warm, cold = [3, 1, 8], [3, 1, 7], [2, 4, 6]
    run(0, prefix[:3], owner)
    run(3, prefix[3:], owner)
    run(8, poison, owner)
    run(0, prefix[:3], cold)
    run(3, prefix[3:], cold)
    expected, expected_ids = run(8, suffix, cold)
    observed, ids = run(8, suffix, warm)
    assert torch.equal(expected, observed) and torch.equal(expected_ids, ids)
    for name, row_ratio, _, _ in LayerPages.layout(attn):
        table = getattr(pages, name)
        rows = LayerPages.rows_per_block(name, row_ratio, 4, state.history.tail_groups)
        assert torch.equal(table[warm[2] * rows : (warm[2] + 1) * rows], table[cold[2] * rows : (cold[2] + 1) * rows])


def test_prefix_policy_follows_the_current_platform_scheduler(monkeypatch):
    from vllm.v1.core.sched import scheduler

    from vllm_ascend.models.deepseek_v41 import prefix_scheduler

    early = prefix_scheduler.V41PrefixScheduler

    class InstalledScheduler(scheduler.Scheduler):
        pass

    monkeypatch.setattr(scheduler, "Scheduler", InstalledScheduler)
    resolved = prefix_scheduler.V41PrefixScheduler
    assert issubclass(resolved, InstalledScheduler) and resolved is not early
    assert prefix_scheduler.V41PrefixScheduler is resolved
