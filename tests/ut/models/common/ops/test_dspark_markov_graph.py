# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequential Markov semantics and the actual greedy/probabilistic dispatch."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.common.ops.dspark_markov_graph import DSparkMarkovGraph, greedy_markov_chain

BLOCK = 5
VOCAB = 8
ROOT = Path(__file__).resolve().parents[5]


def cycle_model():
    embedding = torch.nn.Embedding(VOCAB, VOCAB, dtype=torch.bfloat16)
    projection = torch.nn.Linear(VOCAB, VOCAB, bias=False, dtype=torch.float32)
    with torch.no_grad():
        embedding.weight.copy_(torch.eye(VOCAB))
        projection.weight.copy_(torch.roll(torch.eye(VOCAB), 1, 0))
    return embedding, projection


@torch.inference_mode()
def test_chain_uses_each_new_token_and_preserves_corrected_logits():
    embedding, projection = cycle_model()
    logits = torch.zeros(2, BLOCK, VOCAB)
    ids = torch.full((2, BLOCK + 1), -1, dtype=torch.int64)
    ids[:, 0] = torch.tensor([0, 5])
    greedy_markov_chain(logits, ids, embedding, projection)
    expected = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 0, 1, 2]])
    assert torch.equal(ids, expected)
    assert torch.equal(logits, torch.nn.functional.one_hot(expected[:, 1:], VOCAB).float())


@torch.inference_mode()
def test_bf16_logits_round_before_argmax_at_each_step():
    embedding, projection = cycle_model()
    projection.weight.zero_()
    projection.weight[1, 0] = 0.001
    logits = torch.full((1, BLOCK, VOCAB), -10.0, dtype=torch.bfloat16)
    logits[..., :2] = 1.0
    ids = torch.zeros(1, BLOCK + 1, dtype=torch.int64)
    greedy_markov_chain(logits, ids, embedding, projection)
    # The FP32 bias is rounded into BF16 logits before selecting the first tie.
    assert torch.count_nonzero(ids) == 0
    assert torch.equal(logits[..., :2], torch.ones_like(logits[..., :2]))


@torch.inference_mode()
def test_cpu_fallback_never_mutates_inputs():
    embedding, projection = cycle_model()
    logits, ids = torch.zeros(1, BLOCK, VOCAB), torch.zeros(1, BLOCK + 1, dtype=torch.int64)
    sampler = DSparkMarkovGraph()
    assert sampler(logits, ids, embedding, projection) is False
    assert not sampler.entries and torch.count_nonzero(logits) == torch.count_nonzero(ids) == 0


def dispatch_statements():
    path = ROOT / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(path.read_text())
    nodes = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in {"graph_sampler", "sampled_by_graph"}
                for target in node.targets
            )
            or (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == "sampled_by_graph"
            )
        ):
            nodes.append(node)
    assert len(nodes) == 3
    return compile(ast.Module(body=sorted(nodes, key=lambda node: node.lineno), type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("graph_result", [None, False, True])
@pytest.mark.parametrize("probabilistic", [False, True])
@torch.inference_mode()
def test_real_dispatch_keeps_probabilistic_and_unsupported_model_paths(graph_result, probabilistic):
    embedding, projection = cycle_model()
    calls = {"graph": 0, "embed": 0, "probabilistic": 0}

    def embed(ids):
        calls["embed"] += 1
        return embedding(ids.long())

    def graph(logits, ids):
        calls["graph"] += 1
        if graph_result:
            greedy_markov_chain(logits, ids, embedding, projection)
        return graph_result

    def sample(logits, metadata):
        calls["probabilistic"] += 1
        return logits.argmax(-1), logits.softmax(-1)

    model = SimpleNamespace(markov_embed=embed, markov_bias=lambda values: projection(values.float()))
    if graph_result is not None:
        model.try_sample_greedy_markov = graph
    logits, ids = torch.zeros(2, BLOCK, VOCAB), torch.zeros(2, BLOCK + 1, dtype=torch.int64)
    ids[:, 0] = torch.tensor([0, 5])
    proposer = SimpleNamespace(model=model, num_speculative_tokens=BLOCK, _sample_draft_from_logits=sample)
    probabilities = []
    scope = dict(
        self=proposer,
        logits=logits,
        draft_token_ids=ids,
        use_probabilistic=probabilistic,
        dspark_probs_list=probabilities,
        sampling_metadata=None,
    )
    exec(dispatch_statements(), scope)
    assert torch.equal(ids, torch.tensor([[0, 1, 2, 3, 4, 5], [5, 6, 7, 0, 1, 2]]))
    used_graph = not probabilistic and graph_result is True
    assert calls["graph"] == int(not probabilistic and graph_result is not None)
    assert calls["embed"] == (0 if used_graph else BLOCK)
    assert calls["probabilistic"] == len(probabilities) == (BLOCK if probabilistic else 0)


@pytest.mark.parametrize("dummy", [True, False])
def test_model_hook_excludes_profile_and_dummy_calls(dummy):
    path = ROOT / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(path.read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "try_sample_greedy_markov"
    )
    calls = []
    scope = {
        "is_draft_dummy": lambda: dummy,
        "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
        "os": os,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    head = SimpleNamespace(embed=object(), head=object())
    model = SimpleNamespace(
        mtp=[SimpleNamespace(markov_head=head)], _markov_graph=lambda *args: calls.append(args) or True
    )
    assert scope["try_sample_greedy_markov"](model, SimpleNamespace(shape=(1, 5, 8)), object()) is (not dummy)
    assert len(calls) == int(not dummy)
