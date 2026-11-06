# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise both real head initializers and the checkpoint shard consumer."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_ascend.models.deepseek_v41.model import read_parameter_slice

ROOT = Path(__file__).resolve().parents[4]
VOCAB = 7
WIDTH = 8
TP = 2
SHARD = 4


class Reader:
    def __init__(self, value):
        self.value = value

    def get_slice(self, key):
        return self

    def get_shape(self):
        return self.value.shape

    def __getitem__(self, index):
        return self.value[index]


def head_initializers(name):
    path = ROOT / "vllm_ascend/models/deepseek_v41" / name
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in {"ParallelLMHead", "LogitsProcessor"}
    ]
    assert len(nodes) == 2
    return compile(ast.Module(body=sorted(nodes, key=lambda n: n.lineno), type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize("name", ["model.py", "dspark.py"])
@pytest.mark.parametrize("dtype", [None, torch.bfloat16, torch.float16, torch.float32, "legacy"])
@pytest.mark.parametrize("rank", [0, 1])
@torch.no_grad()
def test_configured_head_dtype_loads_exact_tp_slice_and_padding(name, dtype, rank):
    processor = SimpleNamespace() if dtype == "legacy" else SimpleNamespace(head_dtype=dtype)

    def head(vocab, width, params_dtype, **kwargs):
        assert (vocab, width) == (VOCAB, WIDTH)
        module = object.__new__(ParallelLMHead)
        nn.Module.__init__(module)
        module.weight = nn.Parameter(torch.full((SHARD, width), float("nan"), dtype=params_dtype), requires_grad=False)
        module.shard_indices = SimpleNamespace(
            org_vocab_start_index=rank * SHARD, org_vocab_end_index=min((rank + 1) * SHARD, VOCAB)
        )
        return module

    owner = SimpleNamespace(mtp=[SimpleNamespace() for _ in range(3)])
    scope = dict(
        self=owner,
        config=SimpleNamespace(vocab_size=VOCAB, hidden_size=WIDTH),
        count=3,
        torch=torch,
        ParallelLMHead=head,
        LogitsProcessor=lambda vocab: processor,
    )
    exec(head_initializers(name), scope)
    module = owner.head if name == "model.py" else owner.mtp[-1].head
    expected_dtype = torch.float32 if dtype in (None, "legacy") else dtype
    assert module.weight.dtype == expected_dtype
    # Reload the same parameter, including a final shard with vocabulary padding.
    for offset in (0.0, 0.003):
        checkpoint = torch.linspace(-1.001, 1.009, VOCAB * WIDTH, dtype=torch.float32).view(VOCAB, WIDTH) + offset
        read_parameter_slice(Reader(checkpoint), "head.weight", module.weight, module, rank, TP)
        count = min(SHARD, VOCAB - rank * SHARD)
        expected = torch.zeros(SHARD, WIDTH, dtype=expected_dtype)
        expected[:count].copy_(checkpoint[rank * SHARD : rank * SHARD + count])
        assert torch.equal(module.weight, expected)
