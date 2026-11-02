# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Set MODEL_PATH to the indexed DeepSeek V4.1 Flash W8A8 checkpoint on a 16-die A3."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families

from tests.e2e.conftest import RemoteOpenAIServer

PARALLEL_FLAGS = {"enable_dsa_cp": True, "enable_flashcomm1": True, "enable_shared_expert_dp": True}
OUTPUT_TOKENS = 32


def get_tokens(server, dp):
    response = requests.post(
        server.url_for("v1", "completions"),
        headers={"X-data-parallel-rank": str(dp)},
        json={
            "model": "v41",
            "prompt": "1 2 3 4 5 6" if dp == 0 else "10 11 12 13 14",
            "temperature": 0.0,
            "min_tokens": OUTPUT_TOKENS,
            "max_tokens": OUTPUT_TOKENS,
            "ignore_eos": True,
            "return_token_ids": True,
        },
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    assert result["usage"]["completion_tokens"] == OUTPUT_TOKENS
    return result["choices"][0]["token_ids"]


def spec_counts(server):
    response = requests.get(server.url_for("metrics"), timeout=10)
    response.raise_for_status()
    totals = {"vllm:spec_decode_num_draft_tokens_total": 0, "vllm:spec_decode_num_accepted_tokens_total": 0}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name in totals:
                totals[sample.name] += sample.value
    return tuple(totals.values())


def test_dspark_greedy_matches_target_with_sequence_parallelism(record_property):
    model_dir = Path(os.getenv("MODEL_PATH", ""))
    if not (model_dir / "quant_model_weights.safetensors.index.json").is_file():
        pytest.skip("Requires local DeepSeek V4.1 Flash W8A8 checkpoint via MODEL_PATH")
    args = [
        "--served-model-name",
        "v41",
        "--tensor-parallel-size",
        "8",
        "--data-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "128",
        "--max-num-seqs",
        "8",
        "--additional-config",
        json.dumps(PARALLEL_FLAGS),
    ]
    outputs = []
    for label, buckets in (("target", [8]), ("dspark", [24, 48])):
        graph = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": buckets}
        argv = [*args, "--compilation-config", json.dumps(graph)]
        if label == "dspark":
            argv += ["--speculative-config", json.dumps({"method": "dspark", "num_speculative_tokens": 5})]
        with RemoteOpenAIServer(str(model_dir), argv, server_host="127.0.0.1", max_wait_seconds=2400) as server:
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda dp: get_tokens(server, dp), (0, 1)))
                start = time.perf_counter()
                ids = list(executor.map(lambda dp: get_tokens(server, dp), (0, 1)))
                record_property(f"{label}_output_tokens_per_second", 2 * OUTPUT_TOKENS / (time.perf_counter() - start))
            assert all(len(row) == OUTPUT_TOKENS for row in ids)
            outputs.append(ids)
            if label == "dspark":
                proposed, accepted = spec_counts(server)
                assert 0 < accepted <= proposed
                record_property("accepted_draft_fraction", accepted / proposed)
    assert outputs[0] == outputs[1]
