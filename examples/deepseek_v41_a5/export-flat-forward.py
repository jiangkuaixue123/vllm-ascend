# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export the exact generated forward sources without loading model weights."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkForCausalLM
from vllm_ascend.models.deepseek_v41.flatten import generate_forward
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Block, DeepseekV41ForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = SimpleNamespace(**json.loads(args.config.read_text())["text_config"])
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "method": "source-derived backbone inlining; no tensor algebra changes",
        "reference": "jojoinfra/Agentic-Compiler@c237f86844598388d2d20d5782344bd538bf617e",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "boundaries": ["attention and KV cache", "Engram", "quantized MoE and TP/EP communication", "serving metadata"],
        "files": {},
    }
    for model, method, count, draft in (
        (DeepseekV41ForCausalLM, "forward", config.num_hidden_layers, False),
        (DeepseekV41ForCausalLM, "_forward_decode", config.num_hidden_layers, False),
        (DeepseekV41DSparkForCausalLM, "forward", len(config.compress_ratios) - config.num_hidden_layers, True),
    ):
        filename = ("draft_" if draft else "target_") + method + ".py"
        source = generate_forward(getattr(model, method), DeepseekV41Block.forward, config, count, draft=draft)
        source = "# Generated callback, executed with the original module's globals and model instance.\n" + source
        (args.output / filename).write_text(source)
        manifest["files"][filename] = {
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "lines": len(source.splitlines()),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
