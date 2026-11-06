# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare complete HC gate words, graph updates and latency with the parent."""

import statistics

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.models.common.ops.hc_gate_fused import try_hc_gate_fused
from vllm_ascend.models.deepseek_v41.core import hc_mixes, hc_split_sinkhorn
from vllm_ascend.ops.triton.hc_sinkhorn import hc_sinkhorn4

EPS = 1e-6
NORM_EPS = 1e-20
HC_WIDTH = 4
HIDDEN = 5120
CHAIN_LENGTH = 80
REPEATS = 5


def reference(mixes, scale, base):
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + EPS
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = (mixes[:, 8:] * scale[2] + base[8:]).unflatten(-1, (4, 4))
    return pre, post, hc_sinkhorn4(comb)


def fused(mixes, scale, base):
    return try_hc_gate_fused(mixes, scale, base, HC_WIDTH, 20, EPS)


def exact(actual, expected):
    assert actual is not None
    for value, parent in zip(actual, expected):
        assert value.shape == parent.shape and value.dtype == torch.float32
        assert torch.equal(value.view(torch.int32), parent.view(torch.int32))


def make_case(rows):
    torch.npu.set_device(0)
    torch.manual_seed(711 + rows)
    mixes = torch.randn(rows, 24, device="npu", dtype=torch.float32)
    scale = torch.tensor([0.37, -1.23, 2.13], device="npu", dtype=torch.float32)
    base = torch.randn(24, device="npu", dtype=torch.float32)
    return mixes, scale, base


@pytest.mark.parametrize("rows", [1, 2, 5, 6, 12, 24, 48])
@torch.inference_mode()
def test_complete_output_words(rows):
    mixes, scale, base = make_case(rows)
    for magnitude in (0.0, 2.0**-20, 1.0, 20.0, 1000.0):
        inputs = (mixes * magnitude, scale, base)
        before = [x.clone() for x in inputs]
        exact(fused(*inputs), reference(*inputs))
        exact(hc_split_sinkhorn(*inputs, HC_WIDTH), reference(*inputs))
        for value, saved in zip(inputs, before):
            assert torch.equal(value.view(torch.int32), saved.view(torch.int32))
    mixes[:, :4] = torch.tensor([float("inf"), -float("inf"), 0.0, -0.0], device="npu", dtype=torch.float32)
    exact(fused(mixes, scale, base), reference(mixes, scale, base))


@pytest.mark.parametrize("rows", [6, 24, 48])
@torch.inference_mode()
def test_complete_hc_projection(rows):
    _, scale, base = make_case(rows)
    states = torch.randn(rows, HC_WIDTH, HIDDEN, device="npu", dtype=torch.bfloat16)
    fn = torch.randn(24, HC_WIDTH * HIDDEN, device="npu", dtype=torch.float32) * 0.01
    flat = states.flatten(-2).float()
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + NORM_EPS)
    expected = reference(F.linear(flat, fn) * rstd, scale, base)
    exact(hc_mixes(states, fn, scale, base), expected)


@pytest.mark.parametrize("rows", [6, 24, 48])
@torch.inference_mode()
def test_graph_refreshes_all_inputs(rows):
    inputs = make_case(rows)
    fused(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fused(*inputs)
    addresses = [value.data_ptr() for value in output]
    for _ in range(4):
        fresh = [torch.randn_like(value) for value in inputs]
        for value, replacement in zip(inputs, fresh):
            value.copy_(replacement)
        graph.replay()
        torch.npu.synchronize()
        exact(output, reference(*fresh))
        assert addresses == [value.data_ptr() for value in output]


@pytest.mark.parametrize("kind", ["cpu", "dtype", "strided", "large", "width", "rounds", "epsilon"])
@torch.inference_mode()
def test_unsupported_inputs_keep_parent(kind):
    mixes, scale, base = make_case(6)
    width, rounds, eps = HC_WIDTH, 20, EPS
    if kind == "cpu":
        mixes, scale, base = mixes.cpu(), scale.cpu(), base.cpu()
    elif kind == "dtype":
        mixes = mixes.bfloat16()
    elif kind == "strided":
        mixes = mixes.t().contiguous().t()
    elif kind == "large":
        mixes = mixes.repeat(10, 1)
    elif kind == "width":
        width = 2
    elif kind == "rounds":
        rounds -= 1
    elif kind == "epsilon":
        eps = 0.0
    assert try_hc_gate_fused(mixes, scale, base, width, rounds, eps) is None


def captured_chain(function, inputs):
    function(*inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(CHAIN_LENGTH):
            function(*inputs)
    graph.replay()
    torch.npu.synchronize()
    return graph


@pytest.mark.parametrize("rows", [6, 24, 48])
@torch.inference_mode()
def test_gate_graph_latency(rows):
    inputs = make_case(rows)
    graphs = [captured_chain(function, inputs) for function in (reference, fused)]
    samples = [[], []]
    for repeat in range(REPEATS):
        for choice in (0, 1) if repeat % 2 == 0 else (1, 0):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            graphs[choice].replay()
            end.record()
            end.synchronize()
            samples[choice].append(start.elapsed_time(end))
    print(
        {
            "rows": rows,
            "calls": CHAIN_LENGTH,
            "reference_ms": statistics.median(samples[0]),
            "fused_ms": statistics.median(samples[1]),
        }
    )
