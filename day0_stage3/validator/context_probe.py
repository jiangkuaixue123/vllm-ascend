# SPDX-License-Identifier: Apache-2.0
"""Freeze a token-exact chat transcript and verify the same request through the serving API."""

import argparse
import hashlib
import json
import re
from array import array
from pathlib import Path
from sys import byteorder
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def token_digest(tokens):
    data = array("I", tokens)
    if data.itemsize != 4:
        raise RuntimeError("Expected 32-bit token encoding for the digest")
    if byteorder != "little":
        data.byteswap()
    return hashlib.sha256(data.tobytes()).hexdigest()


def transcript(tokenizer, template, prompt_length):
    lead = "Read this archive. Remember the three color codes. The amber code is 73."
    middle = " The cobalt code is 41."
    tail = " The jade code is 86. Give only the codes for amber, cobalt and jade, in that order."
    filler = " note"
    unit = tokenizer.encode(filler, add_special_tokens=False)
    if len(unit) != 1 or tokenizer.encode(filler * 16, add_special_tokens=False) != unit * 16:
        raise ValueError("The frozen filler must contribute one token per repetition")
    left = right = 0
    for _ in range(4):
        content = lead + filler * left + middle + filler * right + tail
        messages = [{"role": "user", "content": content}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
            thinking_mode="chat",
            reasoning_effort="high",
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) == prompt_length:
            marks = {}
            for marker in ("amber code is 73", "cobalt code is 41", "jade code is 86"):
                prefix = rendered[: rendered.index(marker) + len(marker)]
                marks[marker] = len(tokenizer.encode(prefix, add_special_tokens=False)) - 1
            return messages, dict(
                prompt_tokens=len(ids),
                token_ids_sha256=token_digest(ids),
                rendered_utf8_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                marker_positions=marks,
            )
        extra = left + right + prompt_length - len(ids)
        if extra < 0:
            raise ValueError("Transcript is shorter than its fixed question and markers")
        left = extra // 2
        right = extra - left
    raise ValueError("Could not produce the exact frozen prompt length")


def post(url, body, timeout, rank):
    request = Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json", "X-data-parallel-rank": str(rank)}
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.loads(error.read())


def verify(args, messages, frozen):
    base = args.base_url.rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("Expected the OpenAI base URL ending in /v1")
    # Ask the service to apply its own default template before starting generation.
    common = dict(model=args.served_model, messages=messages)
    if not args.over_limit:
        status, encoding = post(base[:-3] + "/tokenize", common, args.timeout, args.rank)
        if (
            status != 200
            or encoding["count"] != frozen["prompt_tokens"]
            or token_digest(encoding["tokens"]) != frozen["token_ids_sha256"]
        ):
            raise ValueError("Service tokenizer/template does not reproduce the frozen input")
    body = dict(**common, temperature=0, max_tokens=args.output_tokens)
    if args.cache_salt is not None:
        body["cache_salt"] = args.cache_salt
        if not args.over_limit:
            body["return_token_ids"] = True
    if not args.natural_stop:
        body["min_tokens"] = args.output_tokens
    status, result = post(base + "/chat/completions", body, args.timeout, args.rank)
    receipt = dict(
        input=frozen, rank=args.rank, requested_output=args.output_tokens, http_status=status, response=result
    )
    receipt["natural_stop"] = args.natural_stop
    if args.cache_salt is not None:
        receipt["cache_salt_sha256"] = hashlib.sha256(args.cache_salt.encode()).hexdigest()
    if args.over_limit:
        receipt["boundary_rejected"] = 400 <= status < 500 and bool(result.get("error"))
        return receipt, receipt["boundary_rejected"]
    usage = result.get("usage", {})
    if args.cache_salt is not None:
        details = usage.get("prompt_tokens_details") or {}
        receipt["reported_cached_tokens"] = details.get("cached_tokens")
    completed = usage.get("completion_tokens")
    length_matches = isinstance(completed, int) and (
        0 < completed <= args.output_tokens if args.natural_stop else completed == args.output_tokens
    )
    receipt["usage_verified"] = (
        status == 200
        and result.get("model") == args.served_model
        and usage.get("prompt_tokens") == frozen["prompt_tokens"]
        and length_matches
        and usage.get("total_tokens") == frozen["prompt_tokens"] + completed
    )
    choice = (result.get("choices") or [{}])[0]
    if body.get("return_token_ids"):
        generated = choice.get("token_ids")
        receipt["response_token_ids_verified"] = (
            result.get("prompt_token_ids") == encoding["tokens"]
            and isinstance(generated, list)
            and all(type(token) is int and token >= 0 for token in generated)
            and len(generated) == completed
        )
    text = (choice.get("message") or {}).get("content") or ""
    receipt["reported_codes"] = re.findall(r"\d+", text)
    receipt["retrieval_verified"] = receipt["reported_codes"] == ["73", "41", "86"]
    valid = (
        receipt["usage_verified"] and receipt["retrieval_verified"] and receipt.get("response_token_ids_verified", True)
    )
    return receipt, valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, help="Original locally prepared manifest for verification")
    parser.add_argument("--served-model")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--cache-salt", help="Isolate this prefix and record exact token IDs for cold/warm comparisons."
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--over-limit", action="store_true")
    parser.add_argument(
        "--natural-stop", action="store_true", help="Additional retrieval check allowing an earlier EOS"
    )
    args = parser.parse_args()
    if args.cache_salt is not None and (not args.cache_salt or not args.base_url):
        parser.error("A cache salt requires a nonempty value and a service URL")
    if args.natural_stop and (args.over_limit or not args.base_url):
        parser.error("Use natural stopping only for an additional in-bound service request")
    if bool(args.frozen) != bool(args.base_url) or (args.base_url and not args.served_model):
        parser.error("A service run requires --frozen, --base-url and --served-model")
    if min(args.prompt_tokens, args.output_tokens, args.timeout) < 1 or args.rank < 0:
        parser.error("Lengths and timeout must be positive, and rank nonnegative")
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.model_dir / "tokenizer.json"))
    messages, record = transcript(tokenizer, args.template.read_text(), args.prompt_tokens)
    record.update(
        tokenizer_sha256=hashlib.sha256((args.model_dir / "tokenizer.json").read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256(args.template.read_bytes()).hexdigest(),
    )
    valid = True
    if args.frozen:
        if record != json.loads(args.frozen.read_bytes()):
            raise ValueError("Regenerated transcript differs from the pre-run manifest")
        record, valid = verify(args, messages, record)
    with args.output.open("x") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)
        file.write("\n")
    raise SystemExit(0 if valid else 1)
