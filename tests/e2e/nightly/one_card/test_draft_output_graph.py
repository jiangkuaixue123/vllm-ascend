# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete native draft outputs and pre-collective FP32 partials."""

import json
import statistics
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.e2e.nightly.one_card.test_draft_attention_graph import DIM, HIDDEN, QuantLinear, elapsed, inputs
from tests.e2e.nightly.one_card.test_draft_device_history import common_metadata
from vllm_ascend.models.common.ops.draft_attention_graph import DraftAttentionGraph
from vllm_ascend.models.deepseek_v41 import dspark

pytest_plugins = ("tests.e2e.nightly.one_card.test_draft_attention_graph",)

OUTPUT_RANK = 1024
OUTPUT_GROUPS = 2
REPEATS = 5


@pytest.fixture
def full_model(model, monkeypatch):
    monkeypatch.setattr(dspark, "is_draft_dummy", lambda: False)
    monkeypatch.setattr(model, "config", SimpleNamespace(**vars(model.config), o_lora_rank=OUTPUT_RANK))
    monkeypatch.setattr(model, "groups", 8 // model.tp, raising=False)
    monkeypatch.setattr(model, "local_heads", model.heads, raising=False)
    monkeypatch.setattr(model, "sequence_group", None, raising=False)
    with torch.inference_mode():
        wo_a = nn.Linear(
            DIM * model.heads // model.groups,
            OUTPUT_RANK * model.groups,
            bias=False,
            device="npu",
            dtype=torch.bfloat16,
        )
        wo_b = QuantLinear(OUTPUT_RANK * model.groups, HIDDEN)
        wo_b.reduce_results = False
    monkeypatch.setattr(model, "wo_a", wo_a, raising=False)
    monkeypatch.setattr(model, "wo_b", wo_b, raising=False)
    monkeypatch.setattr(model, "_project_output", dspark.DSparkAttention._project_output.__get__(model))
    observed = SimpleNamespace(partials=[], record=True, calls=0)

    def collective(output):
        assert not torch.npu.is_current_stream_capturing()
        assert output.dtype == torch.float32
        observed.calls += 1
        if observed.record:
            observed.partials.append(output.clone())
        # Nonidentity reduction also verifies the final cast occurs after the sum.
        return output * model.tp + 0.375

    monkeypatch.setattr(dspark, "tensor_model_parallel_all_reduce", collective)
    monkeypatch.setattr("vllm_ascend.models.deepseek_v41.attention.tensor_model_parallel_all_reduce", collective)
    return model, observed


def device_inputs(model, batch=1, start=8192, step=0, position_dtype=torch.int32, length_dtype=torch.int32):
    hidden, positions, eager = inputs(model, start=start, step=step, batch=batch, position_dtype=position_dtype)
    lengths = torch.tensor([span.seq_len for span in eager.requests], dtype=length_dtype, device="npu")
    builder = object.__new__(dspark.DSparkMetadataBuilder)
    metadata = builder.build(0, common_metadata(eager, lengths))
    model.cache.metadata = lambda: metadata
    return hidden, positions, metadata


def exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    words = torch.int32 if actual.dtype == torch.float32 else torch.int16
    torch.testing.assert_close(actual.view(words), expected.view(words), rtol=0, atol=0)


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_complete_outputs_and_collective_inputs_match_retained(full_model, batch, position_dtype, length_dtype):
    model, observed = full_model
    parent, candidate = DraftAttentionGraph(), DraftAttentionGraph(project_output=True)
    for step, start in enumerate((8192, 0, 1, 123, 127, 128, 153595)):
        hidden, positions, metadata = device_inputs(model, batch, start, step, position_dtype, length_dtype)
        if step == 3:
            positions.add_(7)
            model.wo_a.weight.mul_(0.9)
            model.wo_b.weight_scale.mul_(1.01)
        if step == 4:
            hidden.zero_()
        observed.partials.clear()
        before = observed.calls
        model._attention_graph = parent
        expected = model(hidden, positions).clone()
        model._attention_graph = candidate
        actual = model(hidden, positions)
        assert observed.calls == before + 2
        assert len(observed.partials) == 2
        exact(observed.partials[0], observed.partials[1])
        exact(actual, expected)
        assert actual.shape == hidden.shape
        assert metadata.requests.materialized == (batch == 2 and model.tp == 4)
        assert set(candidate.entries) == ({batch} if batch != 2 or model.tp == 8 else set())


@pytest.mark.parametrize("module,name", [("wo_a", "weight"), ("wo_b", "weight"), ("wo_b", "weight_scale")])
@torch.inference_mode()
def test_replaced_output_storage_falls_back_to_exact_parent(full_model, monkeypatch, module, name):
    model, _ = full_model
    args = device_inputs(model)
    candidate = DraftAttentionGraph(project_output=True)
    assert candidate(model, *args) is not None
    layer = getattr(model, module)
    monkeypatch.setattr(layer, name, nn.Parameter(getattr(layer, name).clone(), requires_grad=False))
    assert candidate(model, *args) is None
    model._attention_graph = None
    expected = model(*args[:2]).clone()
    model._attention_graph = candidate
    exact(model(*args[:2]), expected)


@torch.inference_mode()
def test_output_graph_avoids_other_streams_and_outer_capture(full_model):
    model, observed = full_model
    args = device_inputs(model)
    candidate = DraftAttentionGraph(project_output=True)
    candidate(model, *args)
    assert observed.calls == 0
    other = torch.npu.Stream()
    other.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(other):
        assert candidate(model, *args) is None
    outer = torch.npu.NPUGraph()
    with torch.npu.graph(outer):
        assert candidate(model, *args) is None
        output = args[0] + 1
    outer.replay()
    exact(output, args[0] + 1)
    assert observed.calls == 0


@torch.inference_mode()
def test_projection_configuration_change_keeps_eager_path(full_model):
    model, _ = full_model
    args = device_inputs(model)
    candidate = DraftAttentionGraph(project_output=True)
    assert candidate(model, *args) is not None
    model.wo_b.reduce_results = True
    assert candidate(model, *args) is None


@torch.inference_mode()
def test_batch_transitions_and_unsupported_batch_keep_exact_outputs(full_model):
    model, _ = full_model
    parent, candidate = DraftAttentionGraph(), DraftAttentionGraph(project_output=True)
    for batch in (1, 4, 1, 4, 2):
        hidden, positions, metadata = device_inputs(model, batch=batch)
        model._attention_graph = parent
        expected = model(hidden, positions).clone()
        model._attention_graph = candidate
        exact(model(hidden, positions), expected)
        assert metadata.requests.materialized == (batch == 2 and model.tp == 4)
    assert set(candidate.entries) == ({1, 2, 4} if model.tp == 8 else {1, 4})


@pytest.mark.parametrize("batch", [1, 2, 4])
@torch.inference_mode()
def test_complete_attention_and_projection_latency(full_model, batch):
    model, observed = full_model
    observed.record = False
    hidden, positions, _ = device_inputs(model, batch=batch)
    graphs = {"parent": DraftAttentionGraph(), "candidate": DraftAttentionGraph(project_output=True)}
    samples = {name: [] for name in graphs}
    for graph in graphs.values():
        model._attention_graph = graph
        for _ in range(3):
            model(hidden, positions)
    for repeat in range(REPEATS):
        for name in ("parent", "candidate") if repeat % 2 == 0 else ("candidate", "parent"):
            model._attention_graph = graphs[name]
            samples[name].append(elapsed(lambda: model(hidden, positions)))
    print(
        "DSPARK_OUTPUT_GRAPH_TIMING "
        + json.dumps(
            {
                "batch": batch,
                "calls": 12,
                "metrics": ["host_ms", "ready_ms", "device_ms"],
                **{name: list(map(statistics.median, zip(*values))) for name, values in samples.items()},
            }
        ),
        flush=True,
    )
