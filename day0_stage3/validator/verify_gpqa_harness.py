# SPDX-License-Identifier: Apache-2.0
"""Exercise the pinned AISBench inference, extractor and scorer with invented fixture inputs and a loopback server."""

import argparse
import asyncio
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from aiohttp import web
from gpqa import CSV_NAME, DATA_DIR, DATASET, SOURCE_FILES, identity, read_json, require, sha, text_sha
from gpqa_audit import audit


def fixture(tmp, source):
    data = tmp / "toy-data"
    csv_path = data / DATA_DIR / CSV_NAME
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["metadata"] * 7
            + ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
        )
        for i in range(6):
            q = "Invented fixture " + str(i) + (" with {A}." if i == 2 else ".")
            options = [f"toy {i} option {k}" for k in range(4)]
            if i == 0:
                options[1] = options[0]
            writer.writerow([""] * 7 + [q, *options])
    rows = identity(source, csv_path)
    settings = dict(
        model="gpqa-harness-toy",
        count=len(rows),
        reader_range="[:6]",
        reader="GPQADataset",
        dataset_config=DATASET,
        max_tokens=64,
        concurrency=3,
        warmups=0,
        attempts=1,
        rate=0,
        score_floor=None,
        generation=dict(
            temperature=0, ignore_eos=False, chat_template_kwargs=dict(thinking_mode="chat", reasoning_effort="high")
        ),
    )
    m = dict(
        aisbench_revision="synthetic-fixture",
        csv_sha256=sha(csv_path.read_bytes()),
        official_sources={k: sha((source / k).read_bytes()) for k in SOURCE_FILES},
        settings=settings,
        rows=rows,
    )
    manifest_path = tmp / "toy-manifest.json"
    manifest_path.write_text(json.dumps(m))
    return data, manifest_path, m


async def run_fixture(tmp, source):
    data, manifest, m = fixture(tmp, source)
    rows = m["rows"]
    contents = [
        f"Answer: {rows[0]['reference']}",
        f"Answer: {rows[1]['reference'].lower()}",
        f"Answer: A. Rechecking. Answer: {rows[2]['reference']}",
        "No answer was marked.",
        f"Answer: {rows[4]['reference']}",
        f"Answer: {rows[5]['reference']}",
    ]
    table = {row["prompt_sha256"]: row["index"] for row in rows}
    observed = []

    async def chat(req):
        body = await req.json()
        i = table[text_sha(body["messages"][0]["content"])]
        observed.append(i)
        await asyncio.sleep(0.04)
        msg = {"role": "assistant", "content": contents[i]}
        if i == 4:
            msg["reasoning_content"] = "Answer: B"
        if i == 5:
            msg["reasoning"] = "a synthetic reasoning field"
        return web.json_response(
            dict(
                id="fixture-response-" + str(i),
                model=m["settings"]["model"],
                object="chat.completion",
                choices=[dict(index=0, message=msg, finish_reason="length" if i == 3 else "stop")],
                usage={"prompt_tokens": 7, "completion_tokens": 6, "total_tokens": 13},
            )
        )

    api = web.Application()
    api.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(api)
    await runner.setup()
    run = tmp / "run"
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        endpoint = "http://127.0.0.1:" + str(runner.addresses[0][1])
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("gpqa.py")),
            "run",
            "--source",
            str(source),
            "--data-root",
            str(data),
            "--manifest",
            str(manifest),
            "--endpoint",
            endpoint,
            "--output",
            str(run),
        ]
        process = await asyncio.create_subprocess_exec(*cmd, env=os.environ.copy())
        require(await process.wait() == 0, "Fixture AISBench process failed")
    finally:
        await runner.cleanup()
    require(sorted(observed) == list(range(len(rows))), "Fixture input did not arrive exactly once")
    result = audit(run, m, manifest, source)
    require(result["complete"], str(result["errors"]))
    require(
        [r["official_correct"] for r in result["rows"]] == [True, False, True, False, True, True],
        "Fixture official marks differ",
    )
    require(
        result["rows"][3]["official_extracted"] is None and result["rows"][1]["official_extracted"].islower(),
        "Official extractor case or unmatched behavior changed",
    )
    require(
        rows[0]["display_to_source"] == [1, 2, 3, 0] and rows[0]["reference"] == "A",
        "Repeated text first-match fixture differs",
    )
    pred = next(run.glob("work/*/predictions/*/GPQA_diamond.jsonl"))
    result_path = next(run.glob("work/*/results/*/GPQA_diamond.json"))
    outcomes = []

    def invalid(path, replacement, reason):
        original = path.read_bytes()
        try:
            path.write_bytes(replacement)
            report = audit(run, m, manifest, source)
            require(
                not report["complete"] and not any(k in report for k in ("rows", "official_accuracy")),
                "Incomplete fixture emitted a quality result",
            )
            require(any(reason in error for error in report["errors"]), f"Expected {reason}, got {report['errors']}")
            outcomes.append(reason)
        finally:
            path.write_bytes(original)

    lines = pred.read_bytes().splitlines(keepends=True)
    invalid(pred, b"".join(lines[:-1]), "Missing or extra official prediction IDs")
    invalid(pred, b"".join([*lines, lines[-1]]), "Duplicate or invalid prediction ID")
    event_path = run / "http.jsonl"
    events = event_path.read_bytes().splitlines(keepends=True)
    invalid(event_path, events[0] + b"".join(events), "Repeated HTTP attempt")
    scores = read_json(result_path)
    scores["details"]["0"]["correct"] = [False]
    invalid(result_path, json.dumps(scores).encode(), "official mark differs")
    del scores["details"]["0"]
    invalid(result_path, json.dumps(scores).encode(), "Missing or extra official per-row details")
    result_path_copy = read_json(result_path)
    result_path_copy["details"]["1"]["origin_prediction"] = "altered"
    invalid(result_path, json.dumps(result_path_copy).encode(), "raw generation differs")
    return dict(
        complete=True,
        source="invented six-row fixture",
        observed_once=len(observed),
        official_details_checked=len(rows),
        covered=[
            "official option rotation",
            "repeated display text",
            "nested brace substitution",
            "latest answer",
            "lowercase",
            "missing answer",
            "reasoning_content",
            "reasoning",
            "length stop",
        ],
        rejected=outcomes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gpqa-harness-fixture-") as directory:
        summary = asyncio.run(run_fixture(Path(directory), args.source.resolve()))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
