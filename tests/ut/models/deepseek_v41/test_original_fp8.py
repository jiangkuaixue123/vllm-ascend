# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original checkpoint contracts: raw FP4 bits, block scales and FP8 Engram rows."""

import json

import torch

from tests.ut.models.deepseek_v41.test_engram import write_shard
from vllm_ascend.models.deepseek_v41.engram import IndexedMmapStore, MmapEngramEmbedding


def test_original_fp8_engram_index_and_shards_preserve_fractional_values(tmp_path):
    prefix = "layers.1.engram.embed"
    raw = torch.tensor([0.5, -1.5, 3.0]).repeat_interleave(32).to(torch.float8_e4m3fn)
    write_shard(
        tmp_path / "table.safetensors",
        {
            prefix + ".weight": ("F8_E4M3", (3, 32), bytes(raw.view(torch.uint8).tolist())),
            prefix + ".scale": ("F8_E8M0", (3, 1), bytes([127, 128, 126])),
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {prefix + suffix: "table.safetensors" for suffix in (".weight", ".scale")}})
    )
    ids = torch.tensor([2, 0, 1, 2])
    with IndexedMmapStore(tmp_path) as store:
        parts = [MmapEngramEmbedding(store, prefix, tp_rank=i, tp_size=2)(ids, device="cpu") for i in range(2)]
    expected = torch.tensor([1.5, 0.5, -3.0, 1.5]).repeat_interleave(32).reshape(4, 32).bfloat16()
    torch.testing.assert_close(sum(parts), expected, rtol=0, atol=0)
