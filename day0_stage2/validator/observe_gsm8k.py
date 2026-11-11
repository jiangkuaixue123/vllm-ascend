# SPDX-License-Identifier: Apache-2.0
"""Record concurrent requests at the HTTP boundary while AISBench runs its own inference and evaluator."""

import asyncio
import hashlib
import json
import time
from collections import Counter

from aiohttp import ClientSession, ClientTimeout, web


def question_id(messages):
    final = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
    if not isinstance(final, str):
        return None
    prefix = "Question: "
    suffix = "\nLet's think step by step\nAnswer:"
    final = final.strip()
    if not final.startswith(prefix) or not final.endswith(suffix):
        return None
    return hashlib.sha256(final[len(prefix) : -len(suffix)].encode("utf-8")).hexdigest()


async def run_observed(argv, source, output, endpoint, expected_ids):
    active = peak = 0
    attempts = Counter()
    completed = Counter()
    run_start = time.monotonic_ns()
    with (output / "http.jsonl").open("x", buffering=1) as log:
        async with ClientSession(timeout=ClientTimeout(total=None)) as client:

            async def forward(request):
                nonlocal active, peak
                raw = await request.read()
                is_chat = request.path == "/v1/chat/completions"
                row = {}
                if is_chat:
                    payload = json.loads(raw)
                    sample = question_id(payload.get("messages", []))
                    attempts[sample] += 1
                    active += 1
                    peak = max(peak, active)
                    row = dict(
                        question_id=sample,
                        expected=sample in expected_ids,
                        attempt=attempts[sample],
                        start_ns=time.monotonic_ns() - run_start,
                        active_at_start=active,
                        model=payload.get("model"),
                        payload_sha256=hashlib.sha256(raw).hexdigest(),
                    )
                    log.write(json.dumps({"event": "start", **row}) + "\n")
                try:
                    headers = {
                        k: request.headers[k]
                        for k in ("Content-Type", "Authorization", "X-Request-Id")
                        if k in request.headers
                    }
                    async with client.request(
                        request.method, endpoint + request.path_qs, data=raw, headers=headers
                    ) as result:
                        body = await result.read()
                        if is_chat:
                            row["status"] = result.status
                            if result.status == 200:
                                reply = json.loads(body)
                                row["response_id"] = reply.get("id")
                                row["served_model"] = reply.get("model")
                                row["choices"] = reply.get("choices")
                                row["usage"] = reply.get("usage")
                                if row["choices"]:
                                    completed[sample] += 1
                        return web.Response(
                            body=body,
                            status=result.status,
                            headers={"Content-Type": result.headers.get("Content-Type", "application/json")},
                        )
                except Exception as error:
                    if is_chat:
                        row["error_type"] = type(error).__name__
                    raise
                finally:
                    if is_chat:
                        active -= 1
                        log.write(
                            json.dumps(
                                {"event": "end", **row, "end_ns": time.monotonic_ns() - run_start}, ensure_ascii=False
                            )
                            + "\n"
                        )

            app = web.Application()
            app.router.add_post("/v1/chat/completions", forward)
            app.router.add_get("/v1/models", forward)
            app.router.add_get("/health", forward)
            runner = web.AppRunner(app)
            await runner.setup()
            try:
                await web.TCPSite(runner, "127.0.0.1", 0).start()
                client_args = list(argv)
                client_args[client_args.index("--host-ip") + 1] = "127.0.0.1"
                client_args[client_args.index("--host-port") + 1] = str(runner.addresses[0][1])
                with (output / "client.log").open("x") as logfile:
                    process = await asyncio.create_subprocess_exec(
                        *client_args, cwd=source, stdout=logfile, stderr=logfile
                    )
                    try:
                        return await process.wait()
                    finally:
                        if process.returncode is None:
                            process.terminate()
                            await process.wait()
            finally:
                await runner.cleanup()
                summary = dict(
                    observed_peak=peak,
                    total_attempts=sum(attempts.values()),
                    completed=sum(completed.values()),
                    missing=sorted(expected_ids - set(completed)),
                    duplicates=[k for k, v in attempts.items() if v > 1],
                    unexpected=[k for k in attempts if k not in expected_ids],
                )
                (output / "http-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
