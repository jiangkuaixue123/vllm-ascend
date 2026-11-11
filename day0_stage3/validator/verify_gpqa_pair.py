# SPDX-License-Identifier: Apache-2.0
"""CPU integration fixture with two genuine AISBench task/evaluator runs over six invented questions."""

import argparse
import asyncio
import copy
import json
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from gpqa import read_json, sha, text_sha
from gpqa_audit import audit, read_http
from gpqa_pair import COMPARISON, compare, join
from gpqa_template_tokens import build, check_service
from verify_gpqa_harness import fixture


def put(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


@asynccontextmanager
async def toy_server(settings, rows, answers, tokenizer, template, label):
    indexed = {r["prompt_sha256"]: i for i, r in enumerate(rows)}
    calls = {"tokenize": 0, "generation": 0}

    async def tokenize(req):
        q = await req.json()
        assert set(q) == {"messages", "model", "add_generation_prompt", "add_special_tokens", "chat_template_kwargs"}
        assert q["model"] == settings["model"] and q["add_generation_prompt"] and not q["add_special_tokens"]
        assert q["chat_template_kwargs"] == settings["generation"]["chat_template_kwargs"]
        assert text_sha(q["messages"][0]["content"]) in indexed
        calls["tokenize"] += 1
        tokens = tokenizer.apply_chat_template(
            q["messages"],
            chat_template=template,
            tokenize=True,
            add_generation_prompt=True,
            **q["chat_template_kwargs"],
        )
        if isinstance(tokens, dict) or hasattr(tokens, "keys"):
            tokens = tokens["input_ids"]
        return web.json_response(dict(count=len(tokens), tokens=tokens))

    async def chat(req):
        body = await req.json()
        i = indexed[text_sha(body["messages"][0]["content"])]
        calls["generation"] += 1
        await asyncio.sleep(0.01)
        return web.json_response(
            dict(
                id=f"{label}-{i}",
                model=settings["model"],
                object="chat.completion",
                choices=[dict(index=0, message={"role": "assistant", "content": answers[i]}, finish_reason="stop")],
                usage=dict(prompt_tokens=8, completion_tokens=3, total_tokens=11),
            )
        )

    app = web.Application()
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        yield "http://127.0.0.1:" + str(runner.addresses[0][1]), calls
    finally:
        await runner.cleanup()


async def run(tmp, args):
    from transformers import PreTrainedTokenizerFast

    data, manifest_file, manifest = fixture(tmp, args.source)
    manifest["gate_sha256"] = sha(b"invented-gate")
    put(manifest_file, manifest)
    expected = build(manifest_file, args.source, data, args.tokenizer, args.template, read_json(args.assets))
    prep = tmp / "prepared.json"
    put(prep, expected)
    gfile = tmp / "gate.json"
    md = sha(manifest_file.read_bytes())
    gate = dict(
        manifest_sha256=md,
        prior_gates_sha256=manifest["gate_sha256"],
        comparison=COMPARISON,
        reference_priority=["fixture@sha256:" + ("1" * 64)],
        candidate_image_digest="sha256:" + ("2" * 64),
    )
    put(gfile, gate)
    ext = dict(
        complete=True,
        matched=True,
        manifest_sha256=md,
        prepared_sha256=sha(prep.read_bytes()),
        gate_sha256=sha(gfile.read_bytes()),
    )
    effective = SimpleNamespace(
        manifest=manifest_file,
        gate=gfile,
        prepared=prep,
        run_pair=tmp / "external.json",
        source=args.source,
        data_root=data,
    )
    sides = {}
    counts = {}
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer))
    gpqa_script = Path(__import__("gpqa").__file__)
    for k, booleans in (
        ("reference", [True, True, False, False, None, True]),
        ("candidate", [True, False, True, False, None, False]),
    ):
        side = tmp / k
        side.mkdir()
        spec = side / "run-spec.json"
        run_dir = side / "run"
        tp = side / "tokens.json"
        image = gate["reference_priority"][0] if k == "reference" else gate["candidate_image_digest"]
        put(spec, dict(image_digest=image, model_revision=expected["hf_revision"], service_alias=expected["model"]))
        ext[k] = dict(
            run_spec=str(spec.relative_to(tmp)),
            run_spec_sha256=sha(spec.read_bytes()),
            run=str(run_dir.relative_to(tmp)),
            token_check=str(tp.relative_to(tmp)),
        )
        answers = []
        for identity, flag in zip(manifest["rows"], booleans, strict=True):
            correct = identity["reference"]
            letter = correct if flag else next(c for c in "ABCD" if c != correct)
            answers.append("Unmarked toy answer" if flag is None else "Answer: " + letter)
        async with toy_server(
            manifest["settings"], manifest["rows"], answers, tokenizer, args.template.read_text(), k
        ) as (origin, calls):
            token_report = await asyncio.to_thread(check_service, prep, manifest_file, spec, origin, 10)
            assert token_report["complete"] and calls == {"tokenize": 6, "generation": 0}
            put(tp, token_report)
            script = [
                sys.executable,
                str(gpqa_script),
                "run",
                "--source",
                str(args.source),
                "--data-root",
                str(data),
                "--manifest",
                str(manifest_file),
                "--endpoint",
                origin,
                "--output",
                str(run_dir),
            ]
            proc = await asyncio.create_subprocess_exec(*script, stdout=asyncio.subprocess.DEVNULL)
            assert await proc.wait() == 0
            assert calls == {"tokenize": 6, "generation": 6}
            counts[k] = calls
        score = audit(run_dir, manifest, manifest_file, args.source)
        assert score["complete"] and not score["dry_run"]
        put(run_dir / "audit.json", score)
        ends, _ = read_http(run_dir, manifest, False)
        sides[k] = (score, ends)
    put(effective.run_pair, ext)
    result = compare(effective)
    assert (
        result["verdict"] == "FAIL"
        and result["delta_correct"] == -1
        and result["both_correct"] == 1
        and result["candidate_only"] == 1
        and result["reference_only"] == 2
        and result["both_incorrect"] == 2
    )
    assert result["reference"]["unparseable"] == result["candidate"]["unparseable"] == 1
    assert join(manifest, expected, sides["candidate"], sides["reference"])["verdict"] == "PASS"
    assert join(manifest, expected, sides["reference"], sides["reference"])["verdict"] == "PASS"
    rejected = []

    def invalidate(path, obj, label):
        original = path.read_bytes()
        try:
            if obj is None:
                path.unlink()
            else:
                put(path, obj)
            try:
                compare(effective)
            except (OSError, ValueError, KeyError, TypeError):
                rejected.append(label)
            else:
                raise AssertionError(label)
        finally:
            path.write_bytes(original)

    audit_file = tmp / "reference/run/audit.json"
    base = read_json(audit_file)
    invalidate(audit_file, None, "missing_official_audit")
    for label, mutate in (
        ("duplicate_question", lambda j: j["rows"].__setitem__(1, j["rows"][0])),
        ("inconsistent_official_mark", lambda j: j["rows"][0].update(official_correct=False)),
        ("dry_run", lambda j: j.update(dry_run=True)),
    ):
        corrupt = copy.deepcopy(base)
        mutate(corrupt)
        invalidate(audit_file, corrupt, label)
    side_tokens = tmp / "candidate/tokens.json"
    corrupt = read_json(side_tokens)
    corrupt["rows"][0]["actual_tokens"][0] += 1
    invalidate(side_tokens, corrupt, "one_changed_token")
    corrupt = copy.deepcopy(ext)
    corrupt["candidate"]["run_spec_sha256"] = "0" * 64
    invalidate(effective.run_pair, corrupt, "foreign_runspec")
    corrupt = copy.deepcopy(ext)
    corrupt["matched"] = False
    invalidate(effective.run_pair, corrupt, "unmatched_device_or_config")
    altered = copy.deepcopy(sides["candidate"])
    altered[1][manifest["rows"][0]["question_id"]]["canonical_sha256"] = "0" * 64
    try:
        join(manifest, expected, sides["reference"], altered)
    except ValueError:
        rejected.append("different_canonical_generation")
    else:
        raise AssertionError("wire identity")
    return dict(
        complete=True,
        source="invented six-row fixture",
        calls=counts,
        paired_counts={
            k: result[k]
            for k in ("both_correct", "candidate_only", "reference_only", "both_incorrect", "delta_correct")
        },
        checked_predicates=["less", "equal", "greater"],
        rejected=rejected,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "tokenizer", "template", "assets"):
        parser.add_argument("--" + k, required=True, type=Path)
    a = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        print(json.dumps(asyncio.run(run(Path(tmp), a)), indent=2))
