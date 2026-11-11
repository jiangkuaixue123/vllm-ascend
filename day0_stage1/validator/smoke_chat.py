"""Check a frozen sequence of short answers against an OpenAI-compatible service."""

import argparse
import json
import sys
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


# Independent conversations, with a repeat after two prompts with other answers.
CASES = (
    ("只回复两个小写字母 ok。", "ok"),
    ("2 加 2 等于几？只回复阿拉伯数字。", "4"),
    ("10 减 3 等于几？只回复阿拉伯数字。", "7"),
    ("只回复两个小写字母 ok。", "ok"),
)


def check_response(url, model, prompt, expected, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    record = {"request": body, "expected": expected, "passed": False}
    request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            record["http_status"] = response.status
            result = json.load(response)
        choice = result["choices"][0]
        answer = choice["message"]["content"]
        record.update(
            response_id=result["id"],
            served_model=result["model"],
            answer=answer,
            finish_reason=choice["finish_reason"],
            usage=result.get("usage"),
        )
        record["passed"] = (
            record["http_status"] == 200
            and isinstance(result["id"], str)
            and bool(result["id"])
            and result["model"] == model
            and choice["finish_reason"] == "stop"
            and isinstance(answer, str)
            and answer.strip() == expected
        )
    except HTTPError as exc:
        record.update(http_status=exc.code, error=type(exc).__name__)
    except (OSError, HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
        record["error"] = type(exc).__name__
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API base ending in /v1")
    parser.add_argument("--model", required=True, help="Exact served model name")
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--timeout", required=True, type=float, help="Per-request I/O timeout in seconds")
    parser.add_argument("--output", required=True, type=Path, help="New JSONL file in an existing directory")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    url = args.base_url.rstrip("/") + "/chat/completions"
    failures = 0
    with args.output.open("x", encoding="utf-8") as stream:
        for i, (prompt, expected) in enumerate(CASES, start=1):
            record = check_response(url, args.model, prompt, expected, args.max_tokens, args.timeout)
            print(json.dumps({"case": i, **record}, ensure_ascii=False), file=stream, flush=True)
            if not record["passed"]:
                failures += 1
    if failures:
        sys.exit(f"{failures} of {len(CASES)} responses failed. See the JSONL results.")
    print("ok")


if __name__ == "__main__":
    main()
