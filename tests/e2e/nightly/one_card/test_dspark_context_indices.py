# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real context-write dispatch with independent cache fields and a controlled producer."""

import statistics
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.models.deepseek_v41 import dspark
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Cache
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention

WIDTH = 512
HIDDEN = 5120
STAGES = 3
CALLS = 12
REPEATS = 5


class Cache:
    write_slots = DeepseekV41Cache.write_slots

    def __init__(self, backing):
        self.backing = backing

    def field(self, bank, offset, width):
        assert bank == offset == 0 and width == WIDTH
        return self.backing[:, 8 : 8 + width]


class Attention:
    store_context = dspark.DSparkAttention.store_context

    def __init__(self, backing, kv, weight):
        self.cache, self.kv, self.weight = Cache(backing), kv, weight

    def project_kv(self, states, positions):
        # Timings include a controlled BF16 projection producer, not a loaded model.
        return self.kv if self.weight is None else F.linear(states, self.weight)


def setup(rows, shared, *, projection=False, device="npu"):
    if device == "npu":
        torch.npu.set_device(0)
    torch.manual_seed(20261106)
    states = torch.randn(rows, HIDDEN, device=device, dtype=torch.bfloat16)
    positions = torch.arange(rows, device=device)
    mappings, parents, candidates = [], [], []
    slot_backing = None
    for stage in range(STAGES):
        if stage == 0 or not shared:
            slot_backing = torch.arange(rows + 1, device=device, dtype=torch.int64) + stage
            slot_backing[1] = 0
            slot_backing[2::3] = -1
        mappings.append(slot_backing[1:])
        backing = torch.randn(rows * 2 + 8, WIDTH + 16, device=device, dtype=torch.bfloat16)
        kv = torch.randn(rows, WIDTH, device=device, dtype=torch.bfloat16)
        kv[:, :4] = torch.tensor([0.0, -0.0, float("inf"), float("nan")], device=device, dtype=kv.dtype)
        weight = torch.randn(WIDTH, HIDDEN, device=device, dtype=torch.bfloat16) if projection else None
        parents.append(SimpleNamespace(attn=Attention(backing, kv, weight)))
        candidates.append(SimpleNamespace(attn=Attention(backing.clone(), kv, weight)))
    return SimpleNamespace(mtp=parents), SimpleNamespace(mtp=candidates), states, positions, mappings


def parent(model, states, positions, mappings):
    for stage, slots in zip(model.mtp, mappings):
        stage.attn.store_context(states, positions, slots)


def candidate(model, states, positions, mappings):
    dspark.DeepseekV41DSparkForCausalLM.precompute_and_store_context_kv(model, states, positions, mappings)


def assert_cache_bits(first, second):
    for a, b in zip(first.mtp, second.mtp):
        assert torch.equal(a.attn.cache.backing.cpu().view(torch.int16), b.attn.cache.backing.cpu().view(torch.int16))


@pytest.mark.parametrize("rows", [6, 24, 48])
@pytest.mark.parametrize("shared", [True, False])
@torch.inference_mode()
def test_reused_context_indices_feed_dense_attention_exactly(monkeypatch, rows, shared):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    old, new, states, positions, mappings = setup(rows, shared)
    window, block, heads = 128, 5, 64
    q = torch.randn(block, heads, WIDTH, device="npu", dtype=torch.bfloat16)
    sink = torch.zeros(heads, device="npu", dtype=torch.float32)
    old_runners = [DeepseekV41NativeAttention() for _ in range(STAGES)]
    new_runners = [DeepseekV41NativeAttention() for _ in range(STAGES)]
    for first, second in zip(old.mtp, new.mtp):
        backing = torch.randn(window, WIDTH + 16, device="npu", dtype=torch.bfloat16)
        first.attn.cache.backing, second.attn.cache.backing = backing, backing.clone()
        first.attn.kv.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
    for step in range(3):
        if step:
            q.mul_(0.75)
            for slots in mappings:
                if step == 1:
                    slots.fill_(-1)
                else:
                    slots.copy_(torch.arange(rows, device="npu").flip(0))
            for stage in old.mtp:
                stage.attn.kv.add_(0.125)
        parent(old, states, positions, mappings)
        candidate(new, states, positions, mappings)
        assert_cache_bits(old, new)
        for first, second, old_runner, new_runner in zip(old.mtp, new.mtp, old_runners, new_runners):
            old_context = first.attn.cache.field(0, 0, WIDTH)
            new_context = second.attn.cache.field(0, 0, WIDTH)
            query_kv = first.attn.kv[:block]
            kv = torch.cat((old_context, query_kv)).unsqueeze(0)
            indices = torch.arange(window + block, device="npu", dtype=torch.int32)
            indices = indices.view(1, 1, -1).expand(1, block, -1)
            expected = old_runner(q.unsqueeze(0), kv, sink, indices, WIDTH**-0.5, mode="SWA draft bidirectional")[0]
            actual = dspark.block_attention(q, new_context, query_kv, sink, WIDTH**-0.5, new_runner)
            assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
    assert all(runner.full_block_indices.cache_info().currsize == 1 for runner in new_runners)


@pytest.mark.parametrize("rows", [6, 12, 24, 48, 8192])
@pytest.mark.parametrize("shared", [True, False])
@torch.inference_mode()
def test_exact_cache_words_and_refreshed_reject_masks(monkeypatch, rows, shared):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    old, new, states, positions, mappings = setup(rows, shared)
    for step in range(3):
        if step:
            for slots in mappings:
                if step == 1:
                    slots.fill_(-1)
                else:
                    slots.copy_(torch.arange(rows, device=slots.device).flip(0))
        active, groups = dspark.prepare_context_active_rows(mappings)
        assert groups == (1 if shared else STAGES)
        if shared:
            assert active[0] is active[1] is active[2]
        saved = [s.clone() for s in mappings]
        parent(old, states, positions, mappings)
        candidate(new, states, positions, mappings)
        assert_cache_bits(old, new)
        assert all(torch.equal(s, before) for s, before in zip(mappings, saved))


@torch.inference_mode()
def test_missing_mapping_and_misaligned_rows_keep_errors(monkeypatch):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    old, new, states, positions, mappings = setup(6, True)
    with pytest.raises(RuntimeError, match="Missing cache slots"):
        candidate(new, states, positions, [None] * STAGES)
    with pytest.raises(ValueError, match="one context slot"):
        candidate(new, states, positions, mappings[:1])
    for function, model in [(parent, old), (candidate, new)]:
        with pytest.raises(ValueError, match="must align"):
            function(model, states, positions, [m[:-1] for m in mappings])


@torch.inference_mode()
def test_valid_context_write_needs_no_forward_context(monkeypatch):
    def no_context():
        raise AssertionError("Valid slot mappings must not query the forward context")

    monkeypatch.setattr(dspark, "is_draft_dummy", no_context)
    old, new, states, positions, mappings = setup(6, True)
    parent(old, states, positions, mappings)
    candidate(new, states, positions, mappings)
    assert_cache_bits(old, new)


@pytest.mark.parametrize("dummy", [True, False])
@torch.inference_mode()
def test_cpu_and_dummy_fallback_preserve_cache(monkeypatch, dummy):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: dummy)
    old, new, states, positions, mappings = setup(6, True, device="cpu")
    if dummy:
        mappings = [None] * STAGES
    parent(old, states, positions, mappings)
    candidate(new, states, positions, mappings)
    assert_cache_bits(old, new)


def elapsed(function, model, states, positions, mappings):
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin = time.perf_counter()
    start.record()
    for _ in range(CALLS):
        function(model, states, positions, mappings)
    end.record()
    submitted = time.perf_counter()
    end.synchronize()
    return 1000 * (submitted - begin), 1000 * (time.perf_counter() - begin), start.elapsed_time(end)


@pytest.mark.parametrize("rows", [6, 24, 48])
@pytest.mark.parametrize("shared", [True, False])
@torch.inference_mode()
def test_three_stage_projection_and_write_latency(monkeypatch, rows, shared):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    old, new, states, positions, mappings = setup(rows, shared, projection=True)
    parent(old, states, positions, mappings)
    candidate(new, states, positions, mappings)
    assert_cache_bits(old, new)
    measurements = {"parent": [], "candidate": []}
    for repeat in range(REPEATS):
        order = [("parent", parent, old), ("candidate", candidate, new)]
        if repeat % 2:
            order.reverse()
        for name, function, model in order:
            measurements[name].append(elapsed(function, model, states, positions, mappings))
    result = {
        name: tuple(statistics.median(values) for values in zip(*samples)) for name, samples in measurements.items()
    }
    print(
        {
            "rows": rows,
            "shared_slot_view": shared,
            "three_stage_calls": CALLS,
            "metrics": ["host_submit_ms", "wall_ready_ms", "device_ms"],
            **result,
        }
    )
    assert all(t > 0 for values in result.values() for t in values)
