# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual BF16 WKV: preserve global GEMM shape and each TP shard's row offset."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from torch import nn

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

TP = 4
HIDDEN = 5120
HEAD_DIM = 512
LOCAL_ROWS = (6, 12)
INPUT_SCALES = (0.0, 0.01, 1.0, 8.0)


def fixed_m_projection_parts(weight, states, local_rows):
    assert weight.dtype == states.dtype == torch.bfloat16
    assert weight.shape == (HEAD_DIM, HIDDEN) and states.shape == (TP * local_rows, HIDDEN)
    layer = nn.Module()
    layer.prefix = "checkpoint.attn.wkv"
    layer.weight = nn.Parameter(weight, requires_grad=False)
    method = AscendUnquantizedLinearMethod()
    method.process_weights_after_loading(layer)
    expected = method.apply(layer, states)
    parts = []
    for rank in range(TP):
        offset = rank * local_rows
        padded = torch.zeros_like(states)
        padded.narrow(0, offset, local_rows).copy_(states.narrow(0, offset, local_rows))
        parts.append(method.apply(layer, padded).narrow(0, offset, local_rows))
    return torch.cat(parts), expected


@torch.inference_mode()
def check_checkpoint(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    sources = set(config["kv_source_layer_ids"])
    layers = [layer for layer in range(config["num_hidden_layers"]) if layer not in sources]
    index = json.loads((checkpoint / "quant_model_weights.safetensors.index.json").read_text())["weight_map"]
    completed = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("vllm_ascend.utils.get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=1))
        for layer in layers:
            key = f"layers.{layer}.attn.wkv.weight"
            with safe_open(checkpoint / index[key], framework="pt", device="cpu") as source:
                weight = source.get_tensor(key).npu()
            for rows in LOCAL_ROWS:
                for scale in INPUT_SCALES:
                    # Includes the exact seed/shape/scale of the rejected local-M candidate.
                    torch.manual_seed(7700 + layer)
                    states = torch.randn(rows * TP, HIDDEN, device="npu", dtype=torch.bfloat16) * scale
                    actual, expected = fixed_m_projection_parts(weight, states, rows)
                    torch.testing.assert_close(
                        actual.view(torch.int16),
                        expected.view(torch.int16),
                        rtol=0,
                        atol=0,
                        msg=lambda message, prefix=f"layer={layer} rows={rows} scale={scale}": f"{prefix}: {message}",
                    )
                    completed += 1
            print("FIXED_M_KV_PROJECTION_LAYER_PASS " + json.dumps({"layer": layer}), flush=True)
    print(
        "FIXED_M_KV_PROJECTION " + json.dumps({"status": "PASS", "layers": len(layers), "cases": completed}), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.npu.matmul.allow_hf32 = False
    check_checkpoint(args.checkpoint)
