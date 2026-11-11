# SPDX-License-Identifier: Apache-2.0
"""Freeze exact shared blocks and compare cold and warm vLLM completions on one DP rank."""

import argparse
import hashlib
import json
from pathlib import Path

from context_probe import post, token_digest


def prepare(model_dir, block):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    prose = encode(" This archive records water levels in a quiet valley. Read the final instruction carefully.")
    prefix = (prose * (2 * block // len(prose) + 1))[: 2 * block]
    variant = encode(" blue")
    if len(variant) != 1 or variant[0] in (prefix[block // 2], prefix[block + block // 2]):
        raise ValueError("Expected an unambiguous one-token edit to each selected block")
    second = prefix.copy()
    second[block + block // 2] = variant[0]
    first = prefix.copy()
    first[block // 2] = variant[0]
    suffixes = [
        " Reply with just the number four.\nAnswer:",
        " Reply with just the number seven.\nAnswer:",
        " Reply with just the number four.\nAnswer:",
        " Reply with just the number seven.\nAnswer:",
    ]
    cases = []
    names = ["cold-two-blocks", "warm-two-blocks-new-suffix", "change-second-block", "change-first-block"]
    for name, head, tail in zip(names, (prefix, prefix, second, first), suffixes, strict=True):
        ids = head + encode(tail)
        # vLLM recomputes the last token. Only complete preceding blocks can be reused.
        previous = [case["prompt"] for case in cases]
        shared = 0
        for prior in previous:
            equal = next((p for p, (a, b) in enumerate(zip(ids, prior)) if a != b), min(len(prior), len(ids)))
            shared = max(shared, min(equal, len(ids) - 1) // block * block)
        cases.append(dict(name=name, prompt=ids, prompt_sha256=token_digest(ids), expected_cached=shared))
    if [case["expected_cached"] for case in cases] != [0, block * 2, block, 0]:
        raise ValueError("Test inputs did not produce the expected shared-block geometry")
    return dict(
        block_size=block,
        tokenizer_sha256=hashlib.sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        cases=cases,
    )


def run(args, inputs):
    control = json.loads(args.control.read_bytes()) if args.control else None
    if control and (
        control["mode"] != "off"
        or not control["valid"]
        or control["rank"] != args.rank
        or control["model"] != args.served_model
    ):
        raise ValueError("Incompatible cache-off baseline")
    rows = []
    for index, case in enumerate(inputs["cases"]):
        body = dict(model=args.served_model, prompt=case["prompt"], temperature=0, max_tokens=16, return_token_ids=True)
        status, response = post(args.base_url.rstrip("/") + "/completions", body, args.timeout, args.rank)
        choice = (response.get("choices") or [{}])[0]
        usage = response.get("usage") or {}
        generated = choice.get("token_ids") or []
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        expected = 0 if args.mode == "off" else case["expected_cached"]
        valid = (
            status == 200
            and response.get("model") == args.served_model
            and choice.get("prompt_token_ids") == case["prompt"]
            and usage.get("prompt_tokens") == len(case["prompt"])
            and len(generated) == usage.get("completion_tokens")
            and bool(generated)
            and choice.get("finish_reason") in {"stop", "length"}
            and cached == expected
        )
        match = None
        if control:
            baseline = control["cases"][index]
            if baseline["name"] != case["name"] or baseline["prompt_sha256"] != case["prompt_sha256"]:
                raise ValueError("Baseline input differs from frozen probe")
            match = generated == baseline["generated"]
            valid = valid and match
        rows.append(
            dict(
                name=case["name"],
                prompt_sha256=case["prompt_sha256"],
                http_status=status,
                request=body,
                response=response,
                expected_cached=expected,
                cached_tokens=cached,
                generated=generated,
                matches_cache_off=match,
                valid=valid,
            )
        )
    return dict(
        mode=args.mode, rank=args.rank, model=args.served_model, cases=rows, valid=all(r["valid"] for r in rows)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--model-dir", type=Path, required=True)
    freeze.add_argument("--block-size", type=int, default=128)
    chase = commands.add_parser("run")
    chase.add_argument("--inputs", type=Path, required=True)
    chase.add_argument("--base-url", required=True)
    chase.add_argument("--rank", type=int, default=0)
    chase.add_argument("--served-model", required=True)
    chase.add_argument("--mode", choices=["on", "off"], required=True)
    chase.add_argument("--control", type=Path)
    chase.add_argument("--timeout", type=int, default=600)
    for command in (freeze, chase):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.block_size < 2:
            parser.error("Use at least two tokens per block")
        record = prepare(args.model_dir, args.block_size)
    else:
        if (args.mode == "on") != bool(args.control):
            parser.error("Supply a matching --control for the cache-on comparison")
        record = run(args, json.loads(args.inputs.read_bytes()))
    with args.output.open("x") as result:
        json.dump(record, result, ensure_ascii=False, indent=2)
        result.write("\n")
    raise SystemExit(0 if record.get("valid", True) else 1)
