# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real W8A8 and native attention graph replay with changing paged history."""

import statistics
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from torch import nn

from vllm_ascend.models.common.ops.draft_attention_graph import DraftAttentionGraph
from vllm_ascend.models.deepseek_v41 import dspark
from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Cache, DeepseekV41RMSNorm, RequestSpan
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

ROWS = 5
HIDDEN = 5120
LATENT = 1280
HEADS = 16
DIM = 512
WINDOW = 128
PHYSICAL = 1024
TABLE_PAGES = 1200
CALLS = 12
REPEATS = 5
OUTPUT_GROUPS = 2
OUTPUT_RANK = 1024


class QuantLinear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.randint(-8, 9, (outputs, inputs), dtype=torch.int8).npu(), requires_grad=False)
        self.weight_scale = nn.Parameter(
            torch.full((outputs, 1), 0.01, dtype=torch.bfloat16, device="npu"), requires_grad=False
        )
        self.weight_offset = nn.Parameter(
            torch.zeros((outputs, 1), dtype=torch.bfloat16, device="npu"), requires_grad=False
        )
        self.method = AscendW8A8DynamicLinearMethod()
        # Construct the loaded native layout without requiring a running service config.
        self.weight.data = torch_npu.npu_format_cast(self.weight.data.t().contiguous(), ACL_FORMAT_FRACTAL_NZ)
        self.weight_scale.data = self.weight_scale.data.flatten()
        self.weight_offset.data = self.weight_offset.data.flatten()
        self.gather_output = False

    def forward(self, hidden):
        return self.method.apply(self, hidden), None


@pytest.fixture(scope="module")
def model():
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(708)
    with torch.inference_mode():
        layer = dspark.DSparkAttention.__new__(dspark.DSparkAttention)
        nn.Module.__init__(layer)
        layer.tp, layer.heads, layer.dim, layer.rope_dim = 4, HEADS, DIM, 64
        layer.dsa_cp = False
        layer.config = SimpleNamespace(dspark_block_size=ROWS, sliding_window=WINDOW)
        layer.wq_a, layer.wq_b, layer.wkv = (
            QuantLinear(HIDDEN, LATENT),
            QuantLinear(LATENT, HEADS * DIM),
            QuantLinear(HIDDEN, DIM),
        )
        layer.q_norm, layer.kv_norm = DeepseekV41RMSNorm(LATENT, 1e-20), DeepseekV41RMSNorm(DIM, 1e-20)
        for norm in (layer.q_norm, layer.kv_norm):
            norm.weight = nn.Parameter(
                torch.ones_like(norm.weight, dtype=torch.bfloat16, device="npu"), requires_grad=False
            )
        layer.attn_sink = nn.Parameter(torch.linspace(-1, 1, HEADS, device="npu"), requires_grad=False)
        layer.register_buffer("frequencies", torch.linspace(0.01, 1, 32, device="npu"))
        layer.attention_kernel = DeepseekV41NativeAttention()
        layer._staging_b4_witness_pending = False
        layer._project_output = lambda output, dtype: output
        cache = DeepseekV41Cache.__new__(DeepseekV41Cache)
        nn.Module.__init__(cache)
        cache.dim, cache.index_dim, cache.block_size = DIM, 128, WINDOW
        cache.source_ratio, cache.token_history = 0, False
        cache.prefix = "component.mtp.0.attn.cache"
        # The actual cache field is a noncontiguous view with a nonzero offset.
        backing = torch.randn(PHYSICAL, DIM + 16, dtype=torch.bfloat16, device="npu")
        cache.kv_cache = (backing[:, 8 : 8 + DIM], torch.empty(PHYSICAL, 2, dtype=torch.bfloat16, device="npu"))
        layer.cache = cache
        layer.eval()
        yield layer


def inputs(model, start=8192, step=0, batch=1, position_dtype=torch.int32):
    hidden = torch.randn(batch * ROWS + 1, HIDDEN, dtype=torch.bfloat16, device="npu")[1:]
    starts = [start + row * 17 * (1 if start <= 8192 else -1) for row in range(batch)]
    positions = torch.cat([torch.arange(value, value + ROWS, dtype=position_dtype, device="npu") for value in starts])
    table = torch.stack(
        [
            (torch.arange(TABLE_PAGES, dtype=torch.int32, device="npu") * 3 + step + row) % (PHYSICAL // WINDOW)
            for row in range(batch)
        ]
    )
    if step == 2:
        for row, value in enumerate(starts):
            table[row, (value - 1) // WINDOW] = -1  # Match the parent's signed advanced indexing.
    metadata = SimpleNamespace(
        requests=[RequestSpan(row * ROWS, (row + 1) * ROWS, value + ROWS) for row, value in enumerate(starts)],
        block_table_tensor=table,
        num_actual_tokens=batch * ROWS,
    )
    model.cache.metadata = lambda: metadata
    return hidden, positions, metadata


def parent(model, hidden, positions):
    saved = getattr(model, "_attention_graph", None)
    model._attention_graph = None
    try:
        return model(hidden, positions)
    finally:
        model._attention_graph = saved


def same(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.int16).cpu(), expected.view(torch.int16).cpu())


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_complete_native_outputs_refresh_history_cache_inputs_and_parameters(model, monkeypatch, batch, position_dtype):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    replay = DraftAttentionGraph()
    model._attention_graph = replay
    address = None
    for step, start in enumerate((128, 129, 8192, 8191, 153595)):
        hidden, positions, metadata = inputs(model, start, step, batch, position_dtype)
        model.cache.field(0, 0, DIM).copy_(torch.randn_like(model.cache.field(0, 0, DIM)))
        if step == 1:
            hidden.zero_()
        if step == 3:
            model.frequencies.mul_(0.9)
            model.wq_a.weight_scale.mul_(1.01)
        saved = (
            hidden.clone(),
            positions.clone(),
            metadata.block_table_tensor.clone(),
            model.cache.field(0, 0, DIM).clone(),
        )
        expected = parent(model, hidden, positions)
        actual = model(hidden, positions)
        assert bool(replay.entries)
        same(actual, expected)
        assert address is None or actual.data_ptr() == address
        address = actual.data_ptr()
        same(hidden, saved[0])
        assert torch.equal(positions, saved[1]) and torch.equal(metadata.block_table_tensor, saved[2])
        same(model.cache.field(0, 0, DIM), saved[3])


@pytest.mark.parametrize("start,dummy", [(32, False), (8192, True)])
@torch.inference_mode()
def test_real_dispatch_falls_back_for_short_history_and_dummy(model, monkeypatch, start, dummy):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: dummy)
    hidden, positions, metadata = inputs(model, start)
    replay = DraftAttentionGraph()
    model._attention_graph = replay
    expected = parent(model, hidden, positions)
    same(model(hidden, positions), expected)
    assert not replay.entries


@torch.inference_mode()
def test_capture_stream_and_storage_changes_keep_parent(model):
    args = inputs(model)
    replay = DraftAttentionGraph()
    assert replay(model, *args) is not None
    other = torch.npu.Stream()
    other.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(other):
        assert replay(model, *args) is None
    weight = model.q_norm.weight
    model.q_norm.weight = nn.Parameter(weight.clone(), requires_grad=False)
    assert replay(model, *args) is None
    model.q_norm.weight = weight
    cache = model.cache.kv_cache
    model.cache.kv_cache = (cache[0].clone(), cache[1])
    assert replay(model, *args) is None
    model.cache.kv_cache = cache
    outer = torch.npu.NPUGraph()
    with torch.npu.graph(outer):
        assert replay(model, *args) is None
        output = args[0] + 1
    outer.replay()
    assert torch.equal(output, args[0] + 1)
    same(replay(model, *args), parent(model, *args[:2]))


@torch.inference_mode()
def test_separate_graphs_keep_outputs_private(model):
    args = inputs(model)
    first, second = DraftAttentionGraph(), DraftAttentionGraph()
    output = first(model, *args)
    snapshot = output.clone()
    args[0].add_(0.1)
    other = second(model, *args)
    assert output.data_ptr() != other.data_ptr()
    same(output, snapshot)
    same(other, parent(model, *args[:2]))


@torch.inference_mode()
def test_batch_transitions_isolate_requests_and_bound_graph_storage(model, monkeypatch):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    replay = DraftAttentionGraph()
    model._attention_graph = replay
    for batch in (1, 4, 1, 4, 2, 3, 8):
        hidden, positions, metadata = inputs(model, batch=batch)
        expected = parent(model, hidden, positions)
        same(model(hidden, positions), expected)
        assert set(replay.entries).issubset({1, 4})
        if batch == 4:
            original = expected.clone()
            hidden[:ROWS].add_(1)
            changed = model(hidden, positions)
            same(changed, parent(model, hidden, positions))
            same(changed[ROWS:], original[ROWS:])
    assert set(replay.entries) == {1, 4}


def elapsed(function):
    torch.npu.synchronize()
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin = time.perf_counter()
    start.record()
    for _ in range(CALLS):
        function()
    end.record()
    submitted = time.perf_counter()
    end.synchronize()
    return (1000 * (submitted - begin), 1000 * (time.perf_counter() - begin), start.elapsed_time(end))


@pytest.mark.parametrize("batch", [1, 4])
@torch.inference_mode()
def test_complete_attention_eager_latency(model, monkeypatch, batch):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    hidden, positions, metadata = inputs(model, batch=batch)
    model._attention_graph = DraftAttentionGraph()
    model(hidden, positions)
    parent(model, hidden, positions)
    measurements = {"parent": [], "candidate": []}
    calls = {"parent": lambda: parent(model, hidden, positions), "candidate": lambda: model(hidden, positions)}
    for repeat in range(REPEATS):
        for name in ("parent", "candidate") if repeat % 2 == 0 else ("candidate", "parent"):
            measurements[name].append(elapsed(calls[name]))
    print(
        {
            "batch": batch,
            "calls": CALLS,
            "metrics": ["host_submit_ms", "wall_ready_ms", "device_ms"],
            **{name: [statistics.median(values) for values in zip(*samples)] for name, samples in measurements.items()},
        }
    )


@pytest.fixture
def idle_model(model, monkeypatch):
    context = SimpleNamespace(in_profile_run=True, is_draft_model=True, attn_metadata=None, draft_attn_metadatas=[])
    monkeypatch.setattr(dspark, "get_forward_context", lambda: context)
    config = SimpleNamespace(**vars(model.config), o_lora_rank=OUTPUT_RANK)
    monkeypatch.setattr(model, "config", config)
    monkeypatch.setattr(model, "groups", OUTPUT_GROUPS, raising=False)
    monkeypatch.setattr(model, "local_heads", HEADS, raising=False)
    monkeypatch.setattr(model, "sequence_group", None, raising=False)
    with torch.inference_mode():
        wo_a = nn.Linear(
            DIM * HEADS // OUTPUT_GROUPS, OUTPUT_RANK * OUTPUT_GROUPS, bias=False, device="npu", dtype=torch.bfloat16
        )
        wo_b = QuantLinear(OUTPUT_RANK * OUTPUT_GROUPS, HIDDEN)
    monkeypatch.setattr(model, "wo_a", wo_a, raising=False)
    monkeypatch.setattr(model, "wo_b", wo_b, raising=False)
    monkeypatch.setattr(model, "_project_output", dspark.DSparkAttention._project_output.__get__(model))
    collective = Mock(side_effect=lambda output: output)
    monkeypatch.setattr("vllm_ascend.models.deepseek_v41.attention.tensor_model_parallel_all_reduce", collective)
    return model, context, collective


@pytest.mark.parametrize("batch", [1, 4, 8])
@torch.inference_mode()
def test_idle_native_output_keeps_projection_and_collective(idle_model, monkeypatch, batch):
    model, context, collective = idle_model
    hidden, positions, _ = inputs(model, batch=batch)
    model.cache.metadata = lambda: None
    project = Mock(wraps=model.project_kv)
    monkeypatch.setattr(model, "project_kv", project)
    expected = model(hidden, positions)
    assert project.call_count == collective.call_count == 1
    context.in_profile_run = False
    actual = model(hidden, positions)
    same(actual, expected)
    assert actual.shape == hidden.shape
    assert project.call_count == 1 and collective.call_count == 2


@torch.inference_mode()
def test_idle_context_only_skips_unwritten_nonprofile_projection(idle_model, monkeypatch):
    model, context, _ = idle_model
    hidden, positions, _ = inputs(model)
    cache = model.cache.field(0, 0, DIM)
    saved = cache.clone()
    project = Mock(wraps=model.project_kv)
    monkeypatch.setattr(model, "project_kv", project)
    model.store_context(hidden, positions, None)
    assert project.call_count == 1
    context.in_profile_run = False
    model.store_context(hidden, positions, None)
    assert project.call_count == 1
    same(cache, saved)
    expected = model.project_kv(hidden, positions)
    slots = torch.arange(ROWS, dtype=torch.int64, device="npu")
    model.store_context(hidden, positions, slots)
    assert project.call_count == 3
    same(cache[slots], expected)


@torch.inference_mode()
def test_real_missing_metadata_still_raises(idle_model):
    model, context, collective = idle_model
    hidden, positions, _ = inputs(model)
    model.cache.metadata = lambda: None
    context.in_profile_run = context.is_draft_model = False
    with pytest.raises(RuntimeError, match="requires the draft layer's paged metadata"):
        model(hidden, positions)
    collective.assert_not_called()


@torch.inference_mode()
def test_idle_outer_capture_keeps_full_projection_path(idle_model, monkeypatch):
    model, context, _ = idle_model
    hidden, positions, _ = inputs(model)
    model.cache.metadata = lambda: None
    expected = model(hidden, positions)
    context.in_profile_run = False
    project = Mock(wraps=model.project_kv)
    monkeypatch.setattr(model, "project_kv", project)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = model(hidden, positions)
    assert project.call_count == 1
    graph.replay()
    same(output, expected)


@torch.inference_mode()
def test_complete_idle_attention_latency(idle_model):
    model, context, _ = idle_model
    hidden, positions, _ = inputs(model)
    model.cache.metadata = lambda: None

    def run(profile):
        context.in_profile_run = profile
        model.store_context(hidden, positions, None)
        return model(hidden, positions)

    same(run(False), run(True))
    measurements = {"parent": [], "candidate": []}
    for repeat in range(REPEATS):
        for name in ("parent", "candidate") if repeat % 2 == 0 else ("candidate", "parent"):
            measurements[name].append(elapsed(lambda profile=name == "parent": run(profile)))
    print(
        {
            "idle_calls": CALLS,
            "metrics": ["host_submit_ms", "wall_ready_ms", "device_ms"],
            **{name: [statistics.median(values) for values in zip(*samples)] for name, samples in measurements.items()},
        }
    )
