# SPDX-License-Identifier: Apache-2.0
"""Check every frozen GPQA chat against the service tokenizer. The only network method is POST /tokenize."""

import argparse
import json
import sys
from array import array
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from gpqa import CSV_NAME, DATA_DIR, LETTERS, check_inputs, frozen_template, read_json, require, sha, text_sha


def token_sha(ids):
    require(isinstance(ids, list) and ids and all(type(t) is int and 0 <= t < 2**32 for t in ids), "Invalid token IDs")
    raw = array("I", ids)
    require(raw.itemsize == 4, "Expected 32-bit token array")
    if sys.byteorder != "little":
        raw.byteswap()
    return sha(raw.tobytes())


def save(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def build(manifest_path, source, data_root, tokenizer_path, template_path, assets):
    from ais_bench.benchmark.datasets import gpqa as official
    from ais_bench.benchmark.utils.prompt import safe_format
    from transformers import PreTrainedTokenizerFast

    m = read_json(manifest_path)
    resource = assets["source"]
    require(sha(tokenizer_path.read_bytes()) == resource["tokenizer.json"]["sha256"], "Frozen tokenizer differs")
    require(sha(template_path.read_bytes()) == resource["chat_template.jinja"]["sha256"], "Frozen Jinja differs")
    require(
        Path(official.__file__).resolve() == (source / "ais_bench/benchmark/datasets/gpqa.py").resolve(),
        "Unexpected official reader",
    )
    frozen_rows = check_inputs(source, data_root, m)
    task_template = frozen_template(source)
    hf_template = template_path.read_text(encoding="utf-8")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    dataset = official.GPQADataset.load(str((data_root / DATA_DIR).resolve()), CSV_NAME)
    kwargs = m["settings"]["generation"]["chat_template_kwargs"]
    rows = []
    for identity, item in zip(frozen_rows, dataset, strict=True):
        require(
            text_sha(item["question"]) == identity["question_id"]
            and item["answer"] == identity["reference"]
            and [text_sha(item[k]) for k in LETTERS] == identity["option_sha256"],
            "Official input or order differs",
        )
        message = safe_format(task_template, **{**item, "answer": ""})
        require(text_sha(message) == identity["prompt_sha256"], "Official rendered user prompt differs")
        chats = [{"role": "user", "content": message}]
        rendered = tokenizer.apply_chat_template(
            chats, tokenize=False, chat_template=hf_template, add_generation_prompt=True, **kwargs
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        rows.append(
            dict(
                index=identity["index"],
                question_id=identity["question_id"],
                prompt_sha256=identity["prompt_sha256"],
                message=message,
                rendered_sha256=text_sha(rendered),
                expected_tokens=ids,
                tokens_sha256=token_sha(ids),
            )
        )
    return dict(
        schema=1,
        manifest_sha256=sha(manifest_path.read_bytes()),
        hf_revision=assets["revision"],
        tokenizer_sha256=resource["tokenizer.json"]["sha256"],
        template_sha256=resource["chat_template.jinja"]["sha256"],
        template_kwargs=kwargs,
        model=m["settings"]["model"],
        count=len(rows),
        rows=rows,
    )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def origin_tokenize(origin):
    u = urlsplit(origin)
    require(
        u.scheme in ("http", "https")
        and u.hostname
        and not u.username
        and not u.password
        and u.path.rstrip("/") in ("", "/v1")
        and not u.query
        and not u.fragment,
        "Expected a service origin",
    )
    return f"{u.scheme}://{u.netloc}/tokenize"


def send_tokenize(url, body, timeout):
    opener = build_opener(ProxyHandler({}), NoRedirect())
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener.open(req, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def check_prepared(prepared, manifest_path):
    manifest = read_json(manifest_path)
    require(
        prepared["schema"] == 1
        and prepared["manifest_sha256"] == sha(manifest_path.read_bytes())
        and prepared["model"] == manifest["settings"]["model"]
        and prepared["template_kwargs"] == manifest["settings"]["generation"]["chat_template_kwargs"]
        and prepared["count"] == manifest["settings"]["count"] == len(prepared["rows"]),
        "Foreign prepared template inputs",
    )
    for expected, frozen in zip(prepared["rows"], manifest["rows"], strict=True):
        require(
            all(expected[k] == frozen[k] for k in ("index", "question_id", "prompt_sha256"))
            and text_sha(expected["message"]) == frozen["prompt_sha256"]
            and token_sha(expected["expected_tokens"]) == expected["tokens_sha256"],
            "Altered prepared row",
        )
    return manifest


def check_service(prepared_path, manifest_path, run_spec, endpoint, timeout):
    expected = read_json(prepared_path)
    check_prepared(expected, manifest_path)
    spec = read_json(run_spec)
    require(spec["model_revision"] == expected["hf_revision"], "RunSpec/model revision mismatch")
    require(spec["service_alias"] == expected["model"], "RunSpec/model alias mismatch")
    target = origin_tokenize(endpoint)
    rows = []
    for item in expected["rows"]:
        body = dict(
            model=expected["model"],
            messages=[{"role": "user", "content": item["message"]}],
            add_generation_prompt=True,
            add_special_tokens=False,
            chat_template_kwargs=expected["template_kwargs"],
        )
        row = {k: item[k] for k in ("index", "question_id", "prompt_sha256", "rendered_sha256")}
        row.update(
            attempts=1, expected_count=len(item["expected_tokens"]), expected_tokens_sha256=item["tokens_sha256"]
        )
        try:
            status, raw = send_tokenize(target, body, timeout)
            row.update(status=status, response_sha256=sha(raw))
            require(status == 200, "HTTP " + str(status))
            reply = json.loads(raw)
            ids = reply["tokens"]
            actual = token_sha(ids)
            require(type(reply["count"]) is int and reply["count"] == len(ids), "Malformed server token count")
            row.update(actual_count=len(ids), actual_tokens_sha256=actual, actual_tokens=ids)
            if ids != item["expected_tokens"]:
                row["first_different_token"] = next(
                    (i for i, (a, b) in enumerate(zip(ids, item["expected_tokens"])) if a != b),
                    min(len(ids), len(item["expected_tokens"])),
                )
            row["equal"] = ids == item["expected_tokens"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(equal=False, error=type(exc).__name__ + ": " + str(exc))
        rows.append(row)
    return dict(
        schema=1,
        complete=all(r["equal"] for r in rows),
        method="POST",
        path="/tokenize",
        count=len(rows),
        matched=sum(r["equal"] for r in rows),
        prepared_sha256=sha(prepared_path.read_bytes()),
        manifest_sha256=expected["manifest_sha256"],
        run_spec_sha256=sha(run_spec.read_bytes()),
        hf_revision=expected["hf_revision"],
        tokenizer_sha256=expected["tokenizer_sha256"],
        template_sha256=expected["template_sha256"],
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    prepare = modes.add_parser("prepare")
    for flag in ("source", "data-root", "tokenizer", "template", "assets"):
        prepare.add_argument("--" + flag, type=Path, required=True)
    endpoint = modes.add_parser("check")
    for flag in ("prepared", "run-spec"):
        endpoint.add_argument("--" + flag, type=Path, required=True)
    endpoint.add_argument("--endpoint", required=True)
    endpoint.add_argument("--timeout", type=float, default=60)
    for cmd in (prepare, endpoint):
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    require(not a.output.exists(), "Choose a fresh output file")
    if a.command == "prepare":
        report = build(a.manifest, a.source, a.data_root, a.tokenizer, a.template, read_json(a.assets))
    else:
        require(a.timeout > 0, "Expected a positive timeout")
        report = check_service(a.prepared, a.manifest, a.run_spec, a.endpoint, a.timeout)
    save(a.output, report)
    print(json.dumps({k: v for k, v in report.items() if k in ("count", "matched", "manifest_sha256", "complete")}))
    return 0 if report.get("complete", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
