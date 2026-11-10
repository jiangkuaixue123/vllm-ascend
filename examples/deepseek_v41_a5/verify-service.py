"""Bounded, reproducible checks against this local OpenAI-compatible service."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def chat(prompt, expected=None, *, stream=False):
    payload = {
        "model": "deepseek-v4.1",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64 if expected is None else 16,
        "temperature": 0,
        "chat_template_kwargs": {"thinking_mode": "chat"},
        "stream": stream,
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=600) as response:
        if stream:
            chunks = []
            for line in response:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    item = json.loads(line[6:])
                    chunks.extend(c["delta"].get("content") or "" for c in item.get("choices", []))
            text = "".join(chunks)
            result = {"stream_chunks": len(chunks)}
        else:
            result = json.load(response)
            text = result["choices"][0]["message"]["content"]
    passed = bool(text and text.strip())
    if expected is not None:
        passed = text.strip().rstrip("。.") == expected
    return {
        "prompt": prompt,
        "text": text,
        "expected": expected,
        "passed": passed,
        "seconds": round(time.monotonic() - start, 3),
        "response": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="logs/api-verification.json")
    args = parser.parse_args()
    cases = [("只输出 2+3 的计算结果，不要解释。", "5")]
    if not args.quick:
        cases += [
            (f"计算 {a} × {b}。只输出结果的整数，不要解释。", str(a * b))
            for a, b in [(7, 8), (9, 13), (12, 11), (17, 6), (24, 5), (31, 3), (8, 19)]
        ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1 if args.quick else 4) as executor:
        results = list(executor.map(lambda case: chat(*case), cases))
    if not args.quick:
        results.append(chat("请用一句中文介绍你自己。"))
        results.append(chat("Write a short greeting in English.", stream=True))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for item in results:
        print(json.dumps({k: v for k, v in item.items() if k != "response"}, ensure_ascii=False), flush=True)
    if not all(item["passed"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
