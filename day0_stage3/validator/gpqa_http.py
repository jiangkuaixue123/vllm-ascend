# SPDX-License-Identifier: Apache-2.0
"""Observe the actual AISBench client on loopback and write one receipt per HTTP boundary event."""

import asyncio
import json
import time
from collections import Counter
from contextlib import AsyncExitStack
from urllib.parse import urlsplit

from aiohttp import ClientSession, ClientTimeout, web
from gpqa import require, sha, text_sha


def wire_fields(settings):
    return {"stream": False, **settings["generation"], "max_tokens": settings["max_tokens"], "model": settings["model"]}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identify(payload, prompts, settings):
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or not isinstance(messages[0], dict):
        return None, None, False
    msg = messages[0]
    if set(msg) != {"role", "content"} or msg["role"] != "user" or not isinstance(msg["content"], str):
        return None, None, False
    digest = text_sha(msg["content"])
    fields = {k: v for k, v in payload.items() if k != "messages"}
    return prompts.get(digest), digest, digest in prompts and canonical(fields) == canonical(wire_fields(settings))


async def observe(argv, source, env, root, endpoint, rows, settings):
    dry = endpoint is None
    if not dry:
        target = urlsplit(endpoint)
        require(
            target.scheme in ("http", "https")
            and target.netloc
            and target.path in ("", "/")
            and not target.query
            and not target.fragment,
            "--endpoint must be the service origin",
        )
        endpoint = endpoint.rstrip("/")
    prompts = {r["prompt_sha256"]: r["question_id"] for r in rows}
    expected = set(prompts.values())
    active = peak = forwards = 0
    attempts, ended, statuses = Counter(), Counter(), Counter()
    clock = time.monotonic_ns()
    async with AsyncExitStack() as stack:
        log = stack.enter_context((root / "http.jsonl").open("x", buffering=1))
        client = None if dry else await stack.enter_async_context(ClientSession(timeout=ClientTimeout(total=None)))

        async def relay(request, raw):
            headers = {
                k: request.headers[k] for k in ("Content-Type", "Authorization", "X-Request-Id") if k in request.headers
            }
            async with client.request(
                request.method, endpoint + request.path_qs, data=raw, headers=headers, allow_redirects=False
            ) as result:
                return web.Response(
                    status=result.status,
                    body=await result.read(),
                    headers={"Content-Type": result.headers.get("Content-Type", "application/json")},
                )

        async def metadata(request):
            if dry:
                return web.json_response({"data": [{"id": settings["model"]}]} if request.path == "/v1/models" else {})
            return await relay(request, None)

        async def chat(request):
            nonlocal active, peak, forwards
            raw = await request.read()
            try:
                payload = json.loads(raw)
                qid, prompt_digest, valid = identify(payload, prompts, settings)
            except (ValueError, TypeError, AttributeError):
                payload, qid, prompt_digest, valid = {}, None, None, False
            attempts[qid] += 1
            active += 1
            peak = max(peak, active)
            token = qid if qid is not None else "<unknown>"
            row = dict(
                question_id=token,
                prompt_sha256=prompt_digest,
                expected=qid in expected,
                attempt=attempts[qid],
                active_at_start=active,
                start_ns=time.monotonic_ns() - clock,
                model=payload.get("model"),
                payload_sha256=sha(raw),
                canonical_sha256=text_sha(canonical(payload)),
                wire_valid=valid,
            )
            log.write(json.dumps({"event": "start", **row}) + "\n")
            row["forwarded"] = False
            try:
                if not valid or attempts[qid] != 1:
                    response = web.json_response({"error": "Request does not match the frozen input"}, status=422)
                elif dry:
                    # The immediate local rejection otherwise serializes the observer itself. This is not a timing run.
                    await asyncio.sleep(0.3)
                    response = web.json_response({"error": "Isolated dry-run sink"}, status=410)
                else:
                    forwards += 1
                    row["forwarded"] = True
                    response = await relay(request, raw)
                row["status"] = response.status
                row["response_sha256"] = sha(response.body)
                if response.status == 200:
                    reply = json.loads(response.body)
                    row.update(
                        response_id=reply.get("id"),
                        served_model=reply.get("model"),
                        choices=reply.get("choices"),
                        usage=reply.get("usage"),
                    )
                return response
            except Exception as error:
                row["error_type"] = type(error).__name__
                raise
            finally:
                active -= 1
                ended[token] += 1
                statuses[str(row.get("status"))] += 1
                log.write(
                    json.dumps({"event": "end", **row, "end_ns": time.monotonic_ns() - clock}, ensure_ascii=False)
                    + "\n"
                )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        app.router.add_get("/v1/models", metadata)
        app.router.add_get("/health", metadata)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, "127.0.0.1", 0).start()
            args = list(argv)
            args[args.index("--host-port") + 1] = str(runner.addresses[0][1])
            with (root / "client.log").open("x") as logfile:
                child = await asyncio.create_subprocess_exec(*args, cwd=source, env=env, stdout=logfile, stderr=logfile)
                try:
                    return await child.wait()
                finally:
                    if child.returncode is None:
                        child.terminate()
                        await child.wait()
        finally:
            await runner.cleanup()
            summary = dict(
                dry_run=dry,
                observed_peak=peak,
                total_attempts=sum(attempts.values()),
                ends=sum(ended.values()),
                upstream_chat_forwards=forwards,
                statuses=dict(statuses),
                missing=sorted(expected - set(ended)),
                duplicates=sorted(str(k) for k, n in attempts.items() if n > 1),
            )
            (root / "http-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
