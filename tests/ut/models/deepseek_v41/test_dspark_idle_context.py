# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Only non-profiling idle draft calls may discard local projection work."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"in_profile_run": True}, False),
        ({"is_draft_model": False}, False),
        ({"attn_metadata": {}}, False),
        ({"draft_attn_metadatas": None}, False),
        ({"draft_attn_metadatas": [{}]}, False),
    ],
)
def test_only_idle_draft_context_qualifies(changes, expected):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/dspark.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "is_idle_draft")
    context = SimpleNamespace(in_profile_run=False, is_draft_model=True, attn_metadata=None, draft_attn_metadatas=[])
    vars(context).update(changes)
    namespace = {"get_forward_context": lambda: context}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["is_idle_draft"]() is expected
