# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded GSM8K API regression, separate from the full lm-eval benchmark."""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import regex as re
import yaml


def extract_number(text):
    marker = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    value = marker.group(1) if marker else numbers[-1] if numbers else None
    return str(Decimal(value.replace(",", ""))) if value is not None else None


def evaluate(item, url, model):
    index, sample = item
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": sample["question"]
                        + "\nSolve briefly. End your answer with #### followed by the number.",
                    }
                ],
                "temperature": 0,
                "max_tokens": 512,
                "chat_template_kwargs": {"thinking_mode": "chat"},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    with opener.open(request, timeout=600) as response:
        result = json.load(response)
    choice = result["choices"][0]
    answer = choice["message"]["content"] or ""
    predicted, expected = extract_number(answer), extract_number(sample["answer"])
    return {
        "index": index,
        "question": sample["question"],
        "answer": answer,
        "expected": expected,
        "predicted": predicted,
        "passed": predicted == expected and choice["finish_reason"] != "length",
        "finish_reason": choice["finish_reason"],
        "usage": result["usage"],
        "seconds": round(time.monotonic() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="deepseek-v4.1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    assert config["num_fewshot"] == 0
    assert config["tasks"][0]["name"] == "gsm8k_first8_api"
    data = args.dataset.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != config["dataset_sha256"]:
        raise ValueError("Dataset checksum differs from the evaluated snapshot")
    samples = [json.loads(line) for line in data.splitlines()][: config["limit"]]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda item: evaluate(item, args.url, args.model), enumerate(samples)))
    score = sum(item["passed"] for item in results) / len(results)
    report = {
        "task": "gsm8k_first8_api",
        "metric": "numeric_exact_match",
        "value": score,
        "num_fewshot": 0,
        "dataset_sha256": digest,
        "results": results,
        "scope": "First eight test samples; integration regression, not a full GSM8K benchmark.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if score < config["tasks"][0]["metrics"][0]["value"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
