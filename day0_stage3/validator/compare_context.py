# SPDX-License-Identifier: Apache-2.0
"""Compare three salted long-context requests from the same eight-rank service."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from context_probe import token_digest


def compare(run, prompt_tokens):
    trials = {}
    identity = None
    for mode in ("fixed", "warm-fixed", "natural"):
        directory = run / f"context-{prompt_tokens}-{mode}"
        raw = (directory / "receipt.json").read_bytes()
        record = json.loads(raw)
        response = record["response"]
        usage = response["usage"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        choice = response["choices"][0]
        prompt = response["prompt_token_ids"]
        generated = choice["token_ids"]
        budget = record["requested_output"]
        count = usage["completion_tokens"]
        natural = mode == "natural"
        if not (
            record["http_status"] == 200
            and record["natural_stop"] == natural
            and record["response_token_ids_verified"]
            and len(prompt) == usage["prompt_tokens"] == record["input"]["prompt_tokens"] == prompt_tokens
            and token_digest(prompt) == record["input"]["token_ids_sha256"]
            and len(generated) == count
            and (0 < count <= budget if natural else count == budget)
            and usage["total_tokens"] == prompt_tokens + count
            and 0 <= cached <= prompt_tokens
        ):
            raise ValueError(f"{mode} request does not reproduce its input and output token counts")
        key = (record["input"], record["cache_salt_sha256"], record["rank"], response["model"], budget)
        if not key[1] or (identity is not None and key != identity):
            raise ValueError(f"{mode} has a different input, salt, rank, model or generation budget")
        identity = key
        codes = re.findall(r"\d+", choice["message"]["content"] or "")
        resources = json.loads((directory / "resources.json").read_bytes())
        ranks = {str(i) for i in range(8)}
        if (
            not 0 < resources["all_ranks_complete_snapshots"] <= resources["snapshots"]
            or set(resources["observed_peak_hbm_mb_by_rank"]) != ranks
            or set(resources["observed_peak_kv_fraction_by_rank"]) != ranks
        ):
            raise ValueError(f"{mode} has no complete eight-rank resource sample")
        trials[mode] = {
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "cached_tokens": cached,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": count,
            "output_ids": generated,
            "finish_reason": choice["finish_reason"],
            "codes": codes,
            "retrieval_verified": codes == ["73", "41", "86"],
            "resources": resources,
        }
    if trials["fixed"]["cached_tokens"] != 0:
        raise ValueError("The cold request reused a cached prefix")
    cold, warm = (trials[mode]["output_ids"] for mode in ("fixed", "warm-fixed"))
    first = next((i for i, (a, b) in enumerate(zip(cold, warm), 1) if a != b), None)
    return {
        "prompt": identity[0],
        "cache_salt_sha256": identity[1],
        "rank": identity[2],
        "model": identity[3],
        "output_budget": identity[4],
        "same_fixed_output": first is None,
        "first_different_output_token": first,
        "trials": trials,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("prompt_tokens", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run, args.prompt_tokens)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
