# SPDX-License-Identifier: Apache-2.0
"""Compare sequential and overlapping prefix hits and misses on one DP rank."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from context_probe import post, token_digest


def run(args, frozen):
    cases = {r["name"]: r for r in frozen["cases"]}
    seed = cases["cold-two-blocks"]
    left = cases["warm-two-blocks-new-suffix"]
    right = cases["change-first-block"]
    hit = left["expected_cached"]
    observations = {}

    def request(case, salt, expected, barrier=None):
        prompt = case["prompt"]
        if token_digest(prompt) != case["prompt_sha256"]:
            raise ValueError("Frozen prefix input has changed")
        body = dict(
            model=args.served_model,
            prompt=prompt,
            cache_salt=salt,
            temperature=0,
            min_tokens=16,
            max_tokens=16,
            return_token_ids=True,
        )
        if barrier is not None:
            barrier.wait(timeout=args.timeout)
        start = time.monotonic_ns()
        status, reply = post(args.base_url.rstrip("/") + "/completions", body, args.timeout, args.rank)
        end = time.monotonic_ns()
        choice = (reply.get("choices") or [{}])[0]
        ids = choice.get("token_ids")
        usage = reply.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return dict(
            case=case["name"],
            input_sha256=case["prompt_sha256"],
            cache_salt=salt,
            start_ns=start,
            end_ns=end,
            id=reply.get("id"),
            status=status,
            expected_cached=expected,
            cached=cached,
            output_ids=ids,
            response=reply,
            verified=(
                status == 200
                and reply.get("model") == args.served_model
                and bool(reply.get("id"))
                and choice.get("prompt_token_ids") == prompt
                and usage.get("prompt_tokens") == len(prompt)
                and isinstance(ids, list)
                and len(ids) == usage.get("completion_tokens") == 16
                and cached == expected
            ),
        )

    for mode in ("sequential", "overlapping"):
        salts = [f"{args.salt}/{mode}/{side}" for side in ("left", "right")]
        groups = {"seed": request(seed, salts[0], 0)}
        for phase, expected in (("hit_miss", (hit, 0)), ("both_hit", (hit, hit))):
            pairs = list(zip((left, right), salts, expected))
            if mode == "overlapping":
                barrier = Barrier(2)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    calls = [pool.submit(request, *p, barrier) for p in pairs]
                    rows = [call.result() for call in calls]
            else:
                rows = [request(*p) for p in pairs]
            groups[phase] = rows
        observations[mode] = groups
    comparisons = {}
    all_rows = [trial["seed"] for trial in observations.values()]
    for phase in ("hit_miss", "both_hit"):
        seq, concurrent = (observations[mode][phase] for mode in ("sequential", "overlapping"))
        overlap = min(r["end_ns"] for r in concurrent) - max(r["start_ns"] for r in concurrent)
        comparisons[phase] = dict(
            overlap_ns=overlap, same_output=[a["output_ids"] == b["output_ids"] for a, b in zip(seq, concurrent)]
        )
        all_rows.extend(seq + concurrent)
    valid = (
        all(r["verified"] for r in all_rows)
        and len({r["id"] for r in all_rows}) == len(all_rows)
        and all(c["overlap_ns"] > 0 and all(c["same_output"]) for c in comparisons.values())
    )
    return dict(rank=args.rank, comparisons=comparisons, observations=observations, valid=valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--salt", required=True, help="Fresh namespace for this trial's four isolated prefixes")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.salt or args.rank < 0 or args.timeout <= 0:
        parser.error("Use a nonempty salt, a valid rank and a positive timeout")
    report = run(args, json.loads(args.inputs.read_bytes()))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    raise SystemExit(0 if report["valid"] else 1)
