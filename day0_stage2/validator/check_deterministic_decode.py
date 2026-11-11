# SPDX-License-Identifier: Apache-2.0
"""Record actual greedy completion IDs and optionally compare with a serving control."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def run(args):
    case = json.loads(args.case.read_bytes())
    prompt = case["prompt_ids"]
    reference = case["generated"]
    if not prompt or not reference:
        raise ValueError("Expected nonempty diagnostic prompt and reference generation")
    body = dict(model=args.model, prompt=prompt, temperature=0, max_tokens=len(reference), return_token_ids=True)
    if args.logprobs is not None:
        if args.logprobs < 1:
            raise ValueError("Expected a positive number of ranked tokens")
        body.update(logprobs=args.logprobs, return_tokens_as_token_ids=True)
    headers = {"Content-Type": "application/json"}
    if args.data_parallel_rank is not None:
        if args.data_parallel_rank < 0:
            raise ValueError("Expected a nonnegative data-parallel rank")
        headers["X-data-parallel-rank"] = str(args.data_parallel_rank)
    req = Request(
        args.base_url.rstrip("/") + "/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    with args.output.open("x", encoding="utf-8") as output:
        with urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
            status = response.status
        choice = result["choices"][0]
        tokens = choice["token_ids"]
        valid = bool(
            status == 200
            and result["id"]
            and result["model"] == args.model
            and len(result["choices"]) == 1
            and choice["prompt_token_ids"] == prompt
            and 0 < len(tokens) <= len(reference)
            and result["usage"]["completion_tokens"] == len(tokens)
            and choice["finish_reason"] in {"stop", "length"}
        )
        if args.logprobs is not None:
            rankings = (choice.get("logprobs") or {}).get("top_logprobs")
            valid = valid and isinstance(rankings, list) and len(rankings) == len(tokens) and all(rankings)
        record = dict(
            request=body, http_status=status, response=result, valid=valid, reference_ids_match=tokens == reference
        )
        record["data_parallel_rank"] = args.data_parallel_rank
        if args.control:
            control = json.loads(args.control.read_bytes())
            if (
                not control["valid"]
                or control["request"] != body
                or control.get("data_parallel_rank") != args.data_parallel_rank
            ):
                raise ValueError("Incompatible control request or response")
            record["control_ids_match"] = tokens == control["response"]["choices"][0]["token_ids"]
        json.dump(record, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return valid and record.get("control_ids_match", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI base URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--case", type=Path, required=True, help="JSON containing prompt_ids and generated reference IDs"
    )
    parser.add_argument(
        "--control", type=Path, help="Receipt from the same revision with a different speculation setting"
    )
    parser.add_argument(
        "--logprobs", type=int, help="Request ranked token IDs and their log probabilities at each position"
    )
    parser.add_argument("--data-parallel-rank", type=int, help="Send the request to the same DP rank in both services")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
