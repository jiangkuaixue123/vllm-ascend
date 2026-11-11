# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs, PagedHistory
from vllm_ascend.models.deepseek_v41.graph_ops import LayerPages, _store_source


def test_rejection_rewind_page_reuse_and_chunked_pairs_match_full_fp32_tables():
    torch.manual_seed(3041)
    b, n, width, d = 16, 18, 64, 32
    history = PagedHistory(128, b, n, max_draft_tokens=5)
    state = DecodeInputs(history, width, "cpu", [])
    assert history.tail_groups == 3

    class Capture(nn.Module):
        def forward(self, x):
            self.seen = x.clone()
            return x.clone()

    norm = Capture()
    c = SimpleNamespace(wkv=nn.Linear(d, d, dtype=torch.float32), wgate=nn.Linear(d, d, dtype=torch.float32), norm=norm)
    angle = torch.outer(torch.arange(128).float(), torch.tensor([1.0, 0.1, 0.01, 0.001]))
    attn = SimpleNamespace(
        head_dim=d,
        compress_ratio=2,
        is_kv_source=True,
        compressor=c,
        rope_head_dim=8,
        freqs_cis=torch.stack((angle.cos(), angle.sin()), -1),
        indexer=SimpleNamespace(
            index_head_dim=d, rope_head_dim=8, k_norm=nn.Identity(), wk=nn.Linear(d, d, dtype=torch.bfloat16)
        ),
    )
    pages = LayerPages(attn, state)
    full_v = torch.zeros(n * b // 2 + width, d)
    full_s = torch.zeros_like(full_v)
    a = [4, 8, 2, 11, 5, 13, 10, 15]
    other = [3, 7, 9, 12, 16, 1, 14, 17]
    cached = [4, 8, 6, 12, 16, 1, 14, 17]
    initial_ptrs = (state.tail_plan.data_ptr(), pages.first_kv.data_ptr(), pages.first_score.data_ptr())
    ref = SimpleNamespace(
        layout=SimpleNamespace(max_ngram_size=2),
        token_map=torch.arange(256),
        pad_id=0,
        multipliers=torch.tensor([[3, 5]]),
        primes=torch.tensor([[[17]]]),
        offsets=torch.tensor([[0]]),
    )

    def step(*queries, padding=1):
        positions, slots, tables, ends = [], [], [], []
        for table, start, count in queries:
            pos = np.arange(start, start + count)
            positions.extend(pos.tolist())
            slots.extend((np.asarray(table)[pos // b] * b + pos % b).tolist())
            tables.append(table)
            ends.append(len(slots))
        active = len(positions)
        positions += [0] * padding
        slots += [-1] * padding
        plan = history.plan(positions, slots, tables, ends)
        ids = np.asarray(positions) % 256
        state.stage(plan, ids, ref, [])
        x = torch.randn(len(slots), d, dtype=torch.bfloat16)
        with torch.inference_mode():
            vals, scores = c.wkv(x.float()), c.wgate(x.float())
            # Independent full FP32 reference: no ring, no per-query shortcut.
            # It can materialize the pair for an arbitrary position forever.
            for i in range(active):
                if positions[i] % 2 == 0:
                    full_v[slots[i] // 2] = vals[i]
                    full_s[slots[i] // 2] = scores[i]
            previous_v, previous_s = vals.clone(), scores.clone()
            for i in range(active):
                previous_v[i] = full_v[slots[i] // 2]
                previous_s[i] = full_s[slots[i] // 2]
            pair = torch.stack((previous_v, vals), dim=1)
            weights = torch.stack((previous_s, scores), dim=1).softmax(1)
            expected = (pair * weights).sum(1).bfloat16()
            _store_source(attn, x, state, pages)
            assert torch.equal(norm.seen, expected)
            for j in np.flatnonzero(history.tail_tags >= 0):
                physical = (j // history.tail_groups * b + int(history.tail_tags[j]) % b) // 2
                assert torch.equal(pages.first_kv[j], full_v[physical])
                assert torch.equal(pages.first_score[j], full_s[physical])
            assert state.tail_plan[: len(slots), 1].unique().numel() == len(slots)

    step((a, 0, 15), (other, 0, 9))
    # First odd reads its old ring entry; this same call wraps that page
    # several times, crosses a boundary, interleaves two request rows.
    step((other, 9, 19), (a, 15, 31))
    step((a, 46, 6), (other, 28, 6))
    step((other, 29, 6), (a, 47, 6))  # both discarded all five drafts
    step((a, 48, 6))  # all rejected again, fresh even on next page
    step((a, 49, 6))  # odd needs 48, even though 50 and 52 were written
    step((a, 53, 3), (other, 31, 3))
    step((a, 56, 6))
    # Provider leaves; a second request resumes from two complete cached
    # pages and consumes a fresh private page whose first position is even.
    step((cached, 32, 15))
    step((cached, 47, 9))
    # Preemption discards request-local work. Restart on full cache boundary.
    step((cached, 32, 6))
    step((cached, 33, 4))
    step((a, 0, 13))  # old physical pages recycled for unrelated token input
    step((a, 13, 11))
    step(padding=5)
    assert initial_ptrs == (state.tail_plan.data_ptr(), pages.first_kv.data_ptr(), pages.first_score.data_ptr())
    # An external odd from an unsupported arbitrarily deep rewind is detected.
    plan = history.plan([1], [a[0] * b + 1], [a], [1])
    with pytest.raises(RuntimeError, match="rewind"):
        history.compressor_tail(plan)


def test_tail_geometry_and_one_million_token_meta_allocation():
    attn = SimpleNamespace(
        head_dim=512, compress_ratio=2, is_kv_source=True, indexer=SimpleNamespace(index_head_dim=128)
    )
    assert PagedHistory.groups_per_block(128, 0) == 1
    assert PagedHistory.groups_per_block(128, 5) == 3
    assert PagedHistory.groups_per_block(2, 5) == 1
    blocks = (1 << 20) // 128
    hist = SimpleNamespace(block_size=128, num_slots=blocks * 128, tail_groups=3)
    state = SimpleNamespace(history=hist, device="meta", max_tokens=4096)
    pages = LayerPages(attn, state)
    assert pages.first_kv.shape == pages.first_score.shape == (blocks * 3 + 4096, 512)
    scratch = sum(4096 * width * dtype.itemsize for _, _, width, dtype in LayerPages.layout(attn))
    size = sum(tensor.numel() * tensor.element_size() for tensor in vars(pages).values())
    assert size - scratch == blocks * LayerPages.bytes_per_block(attn, 128, 5)
    delta = (64 - 3) * 2 * 512 * 4
    assert 3 * blocks * delta == 6140461056


@pytest.mark.parametrize("draft_count", range(6))
def test_all_acceptance_offsets_and_page_alignments(draft_count):
    block_size = 128
    table = np.asarray([3, 1, 2])
    for start in range(2 * block_size):
        history = PagedHistory(384, block_size, 4, max_draft_tokens=draft_count)

        def build(at, width):
            pos = np.arange(at, at + width)
            slots = table[pos // block_size] * block_size + pos % block_size
            return history.plan(pos, slots, [table], [width])

        if start:
            history.compressor_tail(build(0, start))
        history.compressor_tail(build(start, 1 + draft_count))
        after_verify = history.tail_tags.copy()
        for accepted in range(draft_count + 1):
            last = start + accepted
            history.tail_tags[:] = after_verify
            reads_and_writes = history.compressor_tail(build(last + 1, 1))
            if last % 2 == 0:
                addr = -1 - reads_and_writes[0, 0]
                assert addr >= 0 and after_verify[addr] == last
