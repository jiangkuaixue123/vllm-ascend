# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import torch

from vllm_ascend.models.deepseek_v41.decode_state import DecodeInputs, PagedHistory
from vllm_ascend.models.deepseek_v41.graph_ops import _raw_window, _sparse


def stage(positions, prompts, *, padding=0):
    size = 128
    history = PagedHistory(512, size, len(positions) * 4 + 1)
    history.tokens[:] = 0
    state = DecodeInputs(history, len(positions) + padding, "cpu", [])
    tables = np.arange(1, 1 + len(positions) * 4).reshape(-1, 4)
    slots = [int(table[pos // size] * size + pos % size) for table, pos in zip(tables, positions, strict=True)]
    for table, pos, slot in zip(tables, positions, slots, strict=True):
        if pos % 2:
            history.compressor_tail(history.plan([pos - 1], [slot - 1], [table], [1]))
    data = SimpleNamespace(
        query_start_loc=torch.arange(len(positions) + 1),
        block_tables=tables,
        v41_prompt_lengths=np.array(prompts),
    )
    ctx = SimpleNamespace(attn_metadata={"target": data}, slot_mapping={"target": torch.tensor(slots + [-1] * padding)})
    plan = history.from_context(ctx, "target", torch.tensor(positions + [0] * padding))
    hash_state = SimpleNamespace(
        token_map=torch.arange(4),
        pad_id=0,
        layout=SimpleNamespace(max_ngram_size=2),
        multipliers=torch.tensor([[1, 3]]),
        primes=torch.tensor([[[7]]]),
        offsets=torch.tensor([[0]]),
    )
    context = state.stage(plan, [1] * len(plan.active), hash_state, [])
    return state, tables, context


def reference(q, keys, mask, sink, scale):
    maximum = torch.full((q.size(0), 1), -1e30)
    denom, numer = torch.zeros_like(maximum), torch.zeros_like(q, dtype=torch.float32)
    for tile, valid in zip(keys.split(64), mask.split(64), strict=True):
        scores = (q.float() @ tile.float().T) * scale
        scores.masked_fill_(~valid, -torch.inf)
        new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        prior = (maximum - new_max).exp()
        prob = (scores - new_max).exp()
        numer = numer * prior + prob.bfloat16().float() @ tile.float()
        denom = denom * prior + prob.sum(-1, keepdim=True)
        maximum = new_max
    return (numer / (denom + (sink[:, None] - maximum).exp())).bfloat16()


def test_mixed_rows_match_independent_prompt_and_decode_attention():
    poses = [14, 20, 15, 63, 64, 69, 95, 126, 127, 137, 201]
    prompts = [15, 26, 15, 16, 16, 16, 1000, 15, 15, 16, 15]
    state, tables, context = stage(poses, prompts, padding=1)
    count = len(poses) + 1
    raw, spans = _raw_window(state, count, context, 128)
    assert raw.shape == (count, 128)
    assert spans.flatten().tolist() == [min(n, 128) if p < n else 128 for p, n in zip(poses, prompts, strict=True)] + [
        128
    ]
    rng = torch.Generator().manual_seed(2015)

    def values(*shape):
        return torch.randn(shape, generator=rng).bfloat16()

    window, comp = values(state.history.num_slots, 32), values(state.history.num_slots // 2, 32)
    query = values(count, 3, 32)
    sink = torch.tensor([0.1, -0.7, 0.5])
    attn = SimpleNamespace(softmax_scale=32**-0.5, attn_sink=sink)
    comp_ids = torch.full((count, 128), -1)
    expected = []
    for i, (p, n, table) in enumerate(zip(poses, prompts, tables, strict=True)):

        def address(j, table=table):
            return int(table[j // 128] * 128 + j % 128)

        if p < n:
            begin = max(0, p - 127)
            raw_ids = [address(j) if j <= p else -1 for j in range(begin, begin + min(n, 128))]
        else:
            raw_ids = [address(j) if j >= 0 else -1 for j in range(p - 127, p + 1)]
        assert raw[i, : len(raw_ids)].tolist() == raw_ids
        assert (raw[i, len(raw_ids) :] == -1).all()
        cs = [address(j) // 2 for j in range(1, p + 1, 2)]
        comp_ids[i, : len(cs)] = torch.tensor(cs)
        w = torch.tensor(raw_ids)
        keys = torch.cat((window[w.clamp_min(0)], comp[cs]))
        valid = torch.cat((w >= 0, torch.ones(len(cs), dtype=torch.bool)))
        expected.append(reference(query[i], keys, valid, sink, attn.softmax_scale))
    out = _sparse(query, window, comp, raw, comp_ids, attn, spans)
    assert torch.equal(out[:-1], torch.stack(expected))
    assert torch.count_nonzero(out[-1]) == 0


def test_single_token_continuation_preserves_ring_and_short_prefill_stays_short():
    for p, n, w in ((14, 15, 15), (15, 15, 128), (95, 200, 128)):
        state, _, context = stage([p], [n])
        raw, span = _raw_window(state, 1, context, 128)
        assert raw.shape[1] == span.item() == w
        if p == 15:
            assert (raw[0, :112] == -1).all()
            assert raw[0, 112:].tolist() == list(range(128, 144))


def test_raw_schedule_uses_fixed_addresses_across_page_reuse_and_padding():
    history = PagedHistory(32, 4, 16)
    state = DecodeInputs(history, 8, "cpu", [], max_requests=2, window_sizes=(3, 9))
    positions = np.array([6, 7, 8, 9, 2, 3, 0, 0])
    prompts = np.array([5] * 4 + [4] * 2)
    views = {size: _raw_window(state, len(positions), history.max_len, size) for size in (3, 9)}
    for base in (1, 8, 1):
        tables = np.array([[base, base + 2, base + 3], [base + 1, 0, 0]])
        slots = [int(tables[0 if i < 4 else 1, p // 4] * 4 + p % 4) for i, p in enumerate(positions[:6])]
        plan = history.plan(positions, slots + [-1, -1], tables, [4, 6], [5, 4])
        state.stage_layout(plan)
        for size, (ids, spans) in views.items():
            attached_ids, attached_spans = _raw_window(state, len(positions), history.max_len, size)
            assert attached_ids.data_ptr() == ids.data_ptr()
            assert attached_spans.data_ptr() == spans.data_ptr()
            for i, (p, n) in enumerate(zip(positions[:6], prompts, strict=True)):
                request = 0 if i < 4 else 1
                begin = max(0, p - size + 1) if p < n else p - size + 1
                span = min(size, n) if p < n else size
                expected = [
                    int(tables[request, key // 4] * 4 + key % 4) if 0 <= key <= p and j < span else -1
                    for j, key in enumerate(range(begin, begin + size))
                ]
                assert ids[i].tolist() == expected
                assert spans[i].item() == span
            assert (ids[6:] == -1).all()
            assert (spans[6:] == size).all()
    empty = history.plan(np.zeros(8, dtype=np.int64), [-1] * 8, [], [], [])
    state.stage_layout(empty)
    for size, (ids, spans) in views.items():
        assert (ids == -1).all() and (spans == size).all()
        short_ids, short_spans = _raw_window(state, 8, 1, size)
        assert short_ids.shape == (8, 1) and (short_spans == 1).all()
