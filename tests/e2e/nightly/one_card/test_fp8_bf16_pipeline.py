# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare every BF16 word with a frozen prior native oracle and replay both paths."""

from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module")
def oracle():
    path = Path(__file__).resolve().parents[4] / "fp8-oracle.pt"
    if not path.exists():
        pytest.skip("requires the task's frozen prior-native FP8 oracle")
    return torch.load(path, weights_only=True, map_location="cpu")


@pytest.mark.parametrize(
    "name",
    [
        "words-0",
        "words-128",
        "words-12672",
        "words-16256",
        "words-32624",
        "tail-1",
        "tail-3",
        "tail-65",
        "tail-129",
        "tail-6145",
    ],
)
@torch.inference_mode()
def test_all_bf16_words_and_pipeline_tails(oracle, name):
    assert enable_custom_op()
    source, expected = oracle[name]
    x = source.npu()
    output = torch.empty_like(x)
    torch.ops._C_ascend.deepseek_v41_fp8_qdq(x, output)
    assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16)), name
    assert torch.equal(x.cpu().view(torch.int16), source.view(torch.int16)), "input changed"


@torch.inference_mode()
def test_graph_replay_changes_finite_and_subnormal_paths(oracle):
    assert enable_custom_op()
    x = oracle["graph-normal"][0].npu()
    output = torch.empty_like(x)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            torch.ops._C_ascend.deepseek_v41_fp8_qdq(x, output)
    torch.npu.current_stream().wait_stream(stream)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        torch.ops._C_ascend.deepseek_v41_fp8_qdq(x, output)
    for kind in ["normal", "tiny", "zero", "nonfinite"]:
        source, expected = oracle["graph-" + kind]
        x.copy_(source)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16)), kind
