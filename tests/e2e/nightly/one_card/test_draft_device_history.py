# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consume rejection-adjusted lengths without a host copy, including short rings."""

import json
import statistics
import time
from types import SimpleNamespace

import pytest
import torch

from tests.e2e.nightly.one_card.test_draft_attention_graph import inputs, parent
from vllm_ascend.models.common.ops.draft_attention_graph import DraftAttentionGraph
from vllm_ascend.models.deepseek_v41 import dspark
from vllm_ascend.models.deepseek_v41.attention import RequestSpan

pytest_plugins = ("tests.e2e.nightly.one_card.test_draft_attention_graph",)

CALLS = 12
REPEATS = 5


@pytest.fixture(autouse=True)
def real_draft_context(monkeypatch):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)


def common_metadata(metadata, lengths):
    batch = len(metadata.requests)
    return SimpleNamespace(
        num_reqs=batch,
        query_start_loc_cpu=torch.arange(batch + 1, dtype=torch.int32) * 5,
        seq_lens=lengths,
        seq_lens_cpu_upper_bound=torch.full((batch,), 153600),
        block_table_tensor=metadata.block_table_tensor,
        slot_mapping=None,
        num_actual_tokens=batch * 5,
    )


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_replay_exact_lengths_short_rings_pages_and_changing_inputs(model, batch, position_dtype, length_dtype):
    builder = object.__new__(dspark.DSparkMetadataBuilder)
    model._attention_graph = DraftAttentionGraph()
    entry = None
    for step, start in enumerate((8192, 0, 1, 123, 127, 128, 153595)):
        hidden, positions, eager = inputs(model, start=start, step=step, batch=batch, position_dtype=position_dtype)
        if step == 3:
            # Linear cache positions must never be inferred from RoPE positions.
            positions = positions + 7
        expected = parent(model, hidden, positions).clone()
        lengths = torch.tensor([span.seq_len for span in eager.requests], dtype=length_dtype, device="npu")
        metadata = builder.build(0, common_metadata(eager, lengths))
        assert not metadata.requests.materialized
        model.cache.metadata = lambda metadata=metadata: metadata
        actual = model(hidden, positions)
        assert not metadata.requests.materialized
        torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
        if entry is None:
            entry = model._attention_graph.entries[batch]
        else:
            assert model._attention_graph.entries[batch] is entry


@torch.inference_mode()
def test_unsupported_batch_materializes_exact_fallback_spans(model):
    hidden, positions, eager = inputs(model, start=8192, batch=2)
    expected = parent(model, hidden, positions).clone()
    lengths = torch.tensor([span.seq_len for span in eager.requests], dtype=torch.int32, device="npu")
    builder = object.__new__(dspark.DSparkMetadataBuilder)
    metadata = builder.build(0, common_metadata(eager, lengths))
    assert not metadata.requests.materialized
    model.cache.metadata = lambda: metadata
    model._attention_graph = DraftAttentionGraph()
    actual = model(hidden, positions)
    assert metadata.requests.materialized
    assert list(metadata.requests) == eager.requests
    torch.testing.assert_close(actual.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 4])
@torch.inference_mode()
def test_metadata_and_complete_attention_latency(model, batch):
    hidden, positions, eager = inputs(model, batch=batch)
    lengths = torch.tensor([span.seq_len for span in eager.requests], dtype=torch.int32, device="npu")
    common = common_metadata(eager, lengths)
    builder = object.__new__(dspark.DSparkMetadataBuilder)
    results = {"batch": batch, "complete_calls": CALLS}
    for name in ("parent", "candidate"):
        model._attention_graph = DraftAttentionGraph()

        def run(use_device_lengths=name == "candidate"):
            if use_device_lengths:
                metadata = builder.build(0, common)
            else:
                bounds = common.query_start_loc_cpu.tolist()
                cpu_lengths = common.seq_lens.cpu().tolist()
                metadata = SimpleNamespace(
                    requests=[RequestSpan(lo, hi, size) for lo, hi, size in zip(bounds, bounds[1:], cpu_lengths)],
                    block_table_tensor=common.block_table_tensor,
                    num_actual_tokens=common.num_actual_tokens,
                )
            model.cache.metadata = lambda: metadata
            return model(hidden, positions)

        for _ in range(3):
            run()
        torch.npu.synchronize()
        samples = []
        for _ in range(REPEATS):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            begin = time.perf_counter()
            for _ in range(CALLS):
                run()
            submitted = time.perf_counter()
            end.record()
            end.synchronize()
            samples.append(((submitted - begin) * 1000, (time.perf_counter() - begin) * 1000, start.elapsed_time(end)))
        results[name] = dict(zip(("host_ms", "ready_ms", "device_ms"), map(statistics.median, zip(*samples))))
    print("DSPARK_DEVICE_HISTORY_TIMING " + json.dumps(results), flush=True)
