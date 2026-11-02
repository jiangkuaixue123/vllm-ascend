# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold chunked prefill on a 16-die A3 with a local DeepSeek V4.1 Flash W8A8 MODEL_PATH."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import regex as re
import requests
from prometheus_client.parser import text_string_to_metric_families
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

DP_SIZE = 4
PREFILL_TOKENS = 8208
OUTPUT_TOKENS = 3
PROMPT_PATTERN = (21750, 23333, 7767, 18392)


def test_dspark_cold_prefill_across_scheduler_steps(capfd, monkeypatch):
    model_dir = Path(os.getenv("MODEL_PATH", ""))
    if not (model_dir / "quant_model_weights.safetensors.index.json").is_file():
        pytest.skip("Requires local DeepSeek V4.1 Flash W8A8 checkpoint via MODEL_PATH")

    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", raising=False)
    args = [
        "--served-model-name",
        "v41",
        "--tensor-parallel-size",
        "4",
        "--data-parallel-size",
        str(DP_SIZE),
        "--enable-expert-parallel",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "8",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--limit-mm-per-prompt",
        '{"image":1}',
        "--additional-config",
        json.dumps({"enable_dsa_cp": True, "enable_flashcomm1": False, "enable_shared_expert_dp": True}),
        "--compilation-config",
        json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [24, 48]}),
        "--speculative-config",
        json.dumps({"method": "dspark", "num_speculative_tokens": 5}),
    ]
    prompt = [*reversed(PROMPT_PATTERN), *(PROMPT_PATTERN * (PREFILL_TOKENS // len(PROMPT_PATTERN)))][:PREFILL_TOKENS]
    barrier = Barrier(DP_SIZE)

    port = get_open_port()
    with RemoteOpenAIServer(
        str(model_dir),
        [*args, "--port", str(port)],
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        max_wait_seconds=2400,
    ) as server:
        captured = capfd.readouterr()
        logs = captured.out + captured.err
        assert "max_num_scheduled_tokens is set to 8160" in logs
        captures = re.findall(
            r"Worker_DP(\d+)_TP0(?:_EP\d+)?\b.*?Captured FULL decode ACLGraph: capture_size=(\d+)", logs
        )
        assert {(str(dp), str(size)) for dp in range(DP_SIZE) for size in (24, 48)} <= set(captures)

        def generate(dp):
            barrier.wait(timeout=30)
            response = requests.post(
                server.url_for("v1", "completions"),
                headers={"X-data-parallel-rank": str(dp)},
                json={
                    "model": "v41",
                    "prompt": prompt,
                    "seed": 0,
                    "temperature": 0,
                    "top_p": 1,
                    "add_special_tokens": False,
                    "min_tokens": OUTPUT_TOKENS,
                    "max_tokens": OUTPUT_TOKENS,
                    "ignore_eos": True,
                    "return_token_ids": True,
                },
                timeout=900,
            )
            response.raise_for_status()
            body = response.json()
            (choice,) = body["choices"]
            usage = body["usage"]
            assert choice["prompt_token_ids"] == prompt
            assert usage["prompt_tokens"] == PREFILL_TOKENS and usage["prompt_tokens_details"]["cached_tokens"] == 0
            assert usage["completion_tokens"] == OUTPUT_TOKENS and choice["finish_reason"] == "length"
            assert len(choice["token_ids"]) == OUTPUT_TOKENS and all(
                type(t) is int and t >= 0 for t in choice["token_ids"]
            )
            return body["id"], choice["token_ids"]

        with ThreadPoolExecutor(max_workers=DP_SIZE) as pool:
            rows = list(pool.map(generate, range(DP_SIZE)))
        assert len({request_id for request_id, _ in rows}) == DP_SIZE
        assert all(ids == rows[0][1] for _, ids in rows)

        # The fresh service receives only these four inference requests. Stats can publish after the HTTP replies.
        deadline = time.monotonic() + 45
        while True:
            response = requests.get(server.url_for("metrics"), timeout=10)
            response.raise_for_status()
            counts = {
                s.labels["engine"]: s.value
                for f in text_string_to_metric_families(response.text)
                for s in f.samples
                if s.name == "vllm:request_success_total" and s.labels["finished_reason"] == "length"
            }
            if counts == {str(dp): 1 for dp in range(DP_SIZE)}:
                break
            assert time.monotonic() < deadline, f"Expected one successful long prefill per engine, got {counts}"
            time.sleep(0.5)
