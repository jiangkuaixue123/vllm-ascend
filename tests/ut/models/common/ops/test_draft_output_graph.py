# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep communication-bearing projection configurations out of local capture."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("tp", [4, 8])
@pytest.mark.parametrize("unsupported", [None, "tp", "sequence", "groups", "heads", "rank", "reduce"])
def test_output_capture_requires_the_local_tp_projection(unsupported, tp):
    source = Path(__file__).resolve().parents[5] / "vllm_ascend/models/common/ops/draft_attention_graph.py"
    tree = ast.parse(source.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name == "supports_draft_output_graph"
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    attention = SimpleNamespace(
        tp=tp,
        sequence_group=None,
        groups=8 // tp,
        local_heads=64 // tp,
        config=SimpleNamespace(o_lora_rank=1024),
        wo_b=SimpleNamespace(reduce_results=False),
    )
    if unsupported == "tp":
        attention.tp = 1
    elif unsupported == "sequence":
        attention.sequence_group = object()
    elif unsupported == "groups":
        attention.groups = 8
    elif unsupported == "heads":
        attention.local_heads = 64
    elif unsupported == "rank":
        attention.config.o_lora_rank = 512
    elif unsupported == "reduce":
        attention.wo_b.reduce_results = True
    assert namespace["supports_draft_output_graph"](attention) == (unsupported is None)
