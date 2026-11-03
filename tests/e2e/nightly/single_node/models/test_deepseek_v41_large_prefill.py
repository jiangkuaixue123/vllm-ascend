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
PROMPT_PATTERN = (21750, 23333, 7767, 18392)


@pytest.mark.parametrize(
    ("max_model_len", "block_size", "prompt_tokens", "output_tokens", "active_dp", "timeout"),
    [
        pytest.param(32768, None, 8208, 3, DP_SIZE, 900, id="32k-four-dp"),
        # 149887 + 113 reaches the exact context limit and decode crosses the 149888 boundary.
        pytest.param(150000, 128, 149887, 113, 1, 21600, id="150k-128-token-blocks"),
    ],
)
def test_dspark_cold_prefill_across_scheduler_steps(
    capfd, monkeypatch, max_model_len, block_size, prompt_tokens, output_tokens, active_dp, timeout
):
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
        str(max_model_len),
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
    if block_size is not None:
        args.extend(["--block-size", str(block_size)])
    prompt = [*reversed(PROMPT_PATTERN), *(PROMPT_PATTERN * (prompt_tokens // len(PROMPT_PATTERN)))][:prompt_tokens]
    barrier = Barrier(active_dp)

    port = get_open_port()
    with RemoteOpenAIServer(
        str(model_dir),
        [*args, "--port", str(port)],
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        max_wait_seconds=2400 if block_size is None else 5400,
    ) as server:
        captured = capfd.readouterr()
        logs = captured.out + captured.err
        assert "max_num_scheduled_tokens is set to 8160" in logs
        captures = re.findall(
            r"Worker_DP(\d+)_TP0(?:_EP\d+)?\b.*?Captured FULL decode ACLGraph: capture_size=(\d+)", logs
        )
        assert {(str(dp), str(size)) for dp in range(DP_SIZE) for size in (24, 48)} <= set(captures)

        if block_size is not None:
            models = requests.get(server.url_for("v1", "models"), timeout=10)
            models.raise_for_status()
            assert [(m["id"], m["max_model_len"]) for m in models.json()["data"] if m["id"] == "v41"] == [
                ("v41", max_model_len)
            ]
            capacities = dict(re.findall(r"EngineCore_DP(\d+)\b.*?GPU KV cache size: ([\d,]+) tokens", logs))
            assert set(capacities) == {str(dp) for dp in range(DP_SIZE)}
            assert min(int(tokens.replace(",", "")) for tokens in capacities.values()) >= max_model_len
            assert len(prompt) + output_tokens == max_model_len
            assert (max_model_len - 1) // block_size > len(prompt) // block_size

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
                    "min_tokens": output_tokens,
                    "max_tokens": output_tokens,
                    "ignore_eos": True,
                    "return_token_ids": True,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            (choice,) = body["choices"]
            usage = body["usage"]
            assert choice["prompt_token_ids"] == prompt
            assert usage["prompt_tokens"] == prompt_tokens and usage["prompt_tokens_details"]["cached_tokens"] == 0
            assert usage["total_tokens"] == prompt_tokens + output_tokens
            assert usage["completion_tokens"] == output_tokens and choice["finish_reason"] == "length"
            assert len(choice["token_ids"]) == output_tokens and all(
                type(t) is int and t >= 0 for t in choice["token_ids"]
            )
            return body["id"], choice["token_ids"]

        with ThreadPoolExecutor(max_workers=active_dp) as pool:
            rows = list(pool.map(generate, range(active_dp)))
        assert len({request_id for request_id, _ in rows}) == active_dp
        assert all(ids == rows[0][1] for _, ids in rows)

        # Stats from the fresh service can publish after the HTTP replies.
        deadline = time.monotonic() + 45
        while True:
            response = requests.get(server.url_for("metrics"), timeout=10)
            response.raise_for_status()
            samples = [sample for family in text_string_to_metric_families(response.text) for sample in family.samples]
            if block_size is not None:
                sizes = {
                    s.labels["engine"]: s.labels["block_size"] for s in samples if s.name == "vllm:cache_config_info"
                }
                assert sizes == {str(dp): str(block_size) for dp in range(DP_SIZE)}
            counts = {
                s.labels["engine"]: s.value
                for s in samples
                if s.name == "vllm:request_success_total" and s.labels["finished_reason"] == "length" and s.value
            }
            if counts == {str(dp): 1 for dp in range(active_dp)}:
                break
            assert time.monotonic() < deadline, (
                f"Expected one successful long prefill on each requested engine, got {counts}"
            )
            time.sleep(0.5)
