# SPDX-License-Identifier: Apache-2.0
"""Submit independent, overlapping requests using the frozen stage-one HTTP checker."""

import argparse
import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checker", type=Path, required=True, help="Path to the frozen smoke_chat.py")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 2 or min(args.max_tokens, args.timeout) <= 0:
        parser.error("Need at least two clients and positive response limits")
    spec = importlib.util.spec_from_file_location("frozen_stage_one_chat", args.checker)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    ready = threading.Barrier(args.concurrency)
    url = args.base_url.rstrip("/") + "/chat/completions"

    def submit(index):
        question, expected = checker.CASES[index % len(checker.CASES)]
        ready.wait(timeout=args.timeout)
        start = time.monotonic_ns()
        row = checker.check_response(url, args.model, question, expected, args.max_tokens, args.timeout)
        return dict(index=index, start_ns=start, end_ns=time.monotonic_ns(), **row)

    args.output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(submit, range(args.concurrency)))
    events = sorted((t, delta) for row in rows for t, delta in ((row["start_ns"], 1), (row["end_ns"], -1)))
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    ids = {r.get("response_id") for r in rows}
    valid = all(r["passed"] for r in rows) and None not in ids and len(ids) == len(rows) and peak > 1
    with (args.output / "requests.jsonl").open("x") as file:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False), file=file)
    (args.output / "summary.json").write_text(
        json.dumps(dict(verified=valid, requests=len(rows), client_peak=peak)) + "\n"
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
