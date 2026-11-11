# SPDX-License-Identifier: Apache-2.0
"""Join a frozen GSM8K run's HTTP receipts, AISBench predictions and official evaluation."""

import argparse
import ast
import hashlib
import json
import math
from collections import Counter
from itertools import islice
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_keys(pairs):
    output = {}
    for key, value in pairs:
        require(key not in output, f"Duplicate JSON key {key}")
        output[key] = value
    return output


def read_json(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique_keys)


def read_lines(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line, object_pairs_hook=unique_keys)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def exactly_one(entries, label):
    entries = list(entries)
    require(len(entries) == 1, f"Expected exactly one {label}, found {len(entries)}")
    return entries[0]


def config_value(node):
    """Read MMEngine's literal dict(...) syntax without executing a run's config."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        args = [config_value(arg) for arg in node.args]
        return dict(*args, **{arg.arg: config_value(arg.value) for arg in node.keywords})
    if isinstance(node, ast.Dict):
        return {config_value(k): config_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.List):
        return [config_value(item) for item in node.elts]
    return ast.literal_eval(node)


def assignments(path):
    return {s.targets[0].id: s.value for s in ast.parse(path.read_bytes()).body if isinstance(s, ast.Assign)}


def check_fields(record, expected, label):
    for key, value in expected.items():
        # JSON separates booleans from numbers. Python equality alone does not.
        actual = record[key]
        if isinstance(value, (dict, list)):
            equal = json.dumps(actual, sort_keys=True) == json.dumps(value, sort_keys=True)
        else:
            equal = isinstance(actual, bool) == isinstance(value, bool) and actual == value
        require(equal, f"{label}.{key} differs")


def frozen_inputs(gates_path, inputs_path, source, data):
    gates, inputs = read_json(gates_path), read_json(inputs_path)
    cfg = gates["benchmark"]
    require(gates["sources"]["plan/gsm8k-inputs.json"] == sha(inputs_path), "Input manifest digest differs")
    for directory, hashes in ((source, inputs["scoring_sources"]), (data, inputs["inputs"])):
        for filename, expected in hashes.items():
            require(sha(directory / filename) == expected, f"Frozen source digest differs for {filename}")
    samples = list(islice(read_lines(data / "test.jsonl"), cfg["count"]))
    ids = [digest(row["question"].encode()) for row in samples]
    require(len(set(ids)) == cfg["count"] and ids == inputs["ordered_ids"], "Original question order differs")
    require(inputs["reader_test_range"] == cfg["reader_range"] == f"[:{len(ids)}]", "Frozen reader range differs")
    prompt_source = exactly_one(
        (p for p in inputs["scoring_sources"] if p.endswith("/" + cfg["dataset"] + ".py")), "frozen prompt source"
    )
    infer = assignments(source / prompt_source)["gsm8k_infer_cfg"]
    prompt = next(k.value for k in infer.keywords if k.arg == "prompt_template")
    template = config_value(next(k.value for k in prompt.keywords if k.arg == "template"))["round"]
    return gates, inputs, samples, template


def check_launch(run, gates, inputs, gates_path, inputs_path, live=True):
    launch = read_json(run / "launch.json")
    cfg = gates["benchmark"]
    check_fields(
        launch,
        dict(
            dry_run=not live,
            checked_questions=cfg["count"],
            gates_sha256=sha(gates_path),
            input_manifest_sha256=sha(inputs_path),
            aisbench_revision=inputs["aisbench_revision"],
            data_revision=inputs["dataset_revision"],
        ),
        "launch",
    )
    argv = launch["argv"]
    flags = {
        "--models": cfg["model"],
        "--datasets": cfg["dataset"],
        "--model-name": gates["served_model"],
        "--num-warmups": 0,
        "--num-prompts": cfg["count"],
        "--batch-size": cfg["client_max_inflight"],
        "--request-rate": cfg["request_rate"],
        "--retry": cfg["max_http_attempts_per_case"],
        "--max-out-len": cfg["max_output_tokens"],
    }
    expected_flags = set(flags) | {
        "--generation-kwargs",
        "--host-ip",
        "--host-port",
        "--dump-eval-details",
        "--work-dir",
    }
    if not live:
        expected_flags.add("--dry-run")
    observed_flags = [token for token in argv if token.startswith("--")]
    require(
        set(observed_flags) == expected_flags and len(observed_flags) == len(expected_flags), "Launch options differ"
    )
    for flag, expected in flags.items():
        require(argv[argv.index(flag) + 1] == str(expected), f"Launch {flag} differs")
    require(json.loads(argv[argv.index("--generation-kwargs") + 1]) == cfg["generation"], "Launch generation differs")


def resolved_config(run, gates, template, live=True):
    path = exactly_one((run / "work").glob("*/configs/*.py"), "resolved AISBench config")
    values = assignments(path)
    cli, models, datasets, evaluation = (
        config_value(values[key]) for key in ("cli_args", "models", "datasets", "eval")
    )
    model = exactly_one(models, "model")
    dataset = exactly_one(datasets, "dataset")
    cfg = gates["benchmark"]
    limits = dict(
        batch_size=cfg["client_max_inflight"],
        max_out_len=cfg["max_output_tokens"],
        request_rate=cfg["request_rate"],
        retry=cfg["max_http_attempts_per_case"],
        generation_kwargs=cfg["generation"],
    )
    check_fields(
        cli,
        dict(
            num_warmups=0,
            num_prompts=cfg["count"],
            dry_run=not live,
            dump_eval_details=True,
            mode="all",
            models=[cfg["model"]],
            datasets=[cfg["dataset"]],
            model_name=gates["served_model"],
            **limits,
        ),
        "cli_args",
    )
    check_fields(
        model,
        dict(
            model=gates["served_model"],
            stream=False,
            type="ais_bench.benchmark.models.api_models.vllm_custom_api_chat.VLLMCustomAPIChat",
            **limits,
        ),
        "model",
    )
    require(
        model["pred_postprocessor"]["type"]
        == "ais_bench.benchmark.utils.postprocess.model_postprocessors.extract_non_reasoning_content",
        "Model postprocessor differs",
    )
    require(
        dataset["reader_cfg"]
        == dict(input_columns=["question"], output_column="answer", test_range=cfg["reader_range"]),
        "Effective reader differs",
    )
    module = "ais_bench.benchmark.datasets.gsm8k."
    require(
        dataset["eval_cfg"]
        == dict(
            evaluator={"type": module + "Gsm8kEvaluator"},
            pred_postprocessor={"type": module + "gsm8k_postprocess"},
            dataset_postprocessor={"type": module + "gsm8k_dataset_postprocess"},
        ),
        "GSM8K evaluator or extractor differs",
    )
    require(dataset["infer_cfg"]["prompt_template"]["template"]["round"] == template, "Effective prompt differs")
    require(evaluation["runner"]["task"]["dump_details"] is True, "Evaluator details disabled")
    return path, model, dataset["abbr"]


def audit_http(run, ids, served_model, limit):
    active, ends, completed_ids, attempts, statuses = {}, {}, set(), Counter(), Counter()
    errors = []
    events = run / "http.jsonl"
    peak = completed = previous = 0
    for line, row in enumerate(read_lines(events), 1):
        qid, event = row["question_id"], row["event"]
        key = (qid, row["attempt"])
        stamp = row["start_ns"] if event == "start" else row["end_ns"]
        if type(stamp) is not int or stamp < previous:
            errors.append(f"HTTP line {line}: invalid event clock")
        previous = stamp
        if event == "start":
            attempts[qid] += 1
            if key in active or row["attempt"] != attempts[qid] or row["attempt"] != 1:
                errors.append(f"HTTP line {line}: duplicate attempt")
            active[key] = row
            peak = max(peak, len(active))
            if (
                row["active_at_start"] != len(active)
                or row["expected"] is not True
                or qid not in ids
                or row["model"] != served_model
            ):
                errors.append(f"HTTP line {line}: request identity or occupancy differs")
        elif event == "end":
            start = active.pop(key, None)
            if start is None or any(start[k] != row.get(k) for k in start if k != "event"):
                errors.append(f"HTTP line {line}: unmatched end or changed request")
            status = row.get("status")
            statuses[str(status)] += 1
            if status == 200 and row.get("choices"):
                completed += 1
                completed_ids.add(qid)
            if status != 200 or row.get("error_type") or row.get("served_model") != served_model:
                errors.append(f"HTTP line {line}: response status, error, or served model")
            if qid in ends:
                errors.append(f"HTTP line {line}: duplicate response")
            ends[qid] = row
        else:
            raise ValueError(f"HTTP line {line}: unknown event")
    missing = sorted(set(ids) - completed_ids)
    if active or missing or sum(attempts.values()) != len(ids) or peak > limit:
        errors.append("HTTP population, unfinished requests or concurrency limit differs")
    stats = dict(
        observed_peak=peak,
        total_attempts=sum(attempts.values()),
        completed=completed,
        missing=missing,
        duplicates=[qid for qid, n in attempts.items() if n > 1],
        unexpected=[qid for qid in attempts if qid not in ids],
    )
    summary_path = run / "http-summary.json"
    if summary_path.exists():
        observer = read_json(summary_path)
        for key in stats:
            if observer.get(key) != stats[key]:
                errors.append(f"Observer summary {key} differs from HTTP events")
    else:
        errors.append("Observer did not finish writing http-summary.json")
    return ends, dict(**stats, statuses=dict(statuses), configured_max_inflight=limit), errors


def prompt_and_wire_digest(question, template, model):
    prompt = [
        *template[:-1],
        {**template[-1], "prompt": template[-1]["prompt"].replace("{question}", question)},
    ]
    # Same key order, role conversion and default aiohttp JSON encoding used by the frozen chat adapter.
    messages = [
        {"content": msg["prompt"], "role": {"HUMAN": "user", "BOT": "assistant"}[msg["role"]]} for msg in prompt
    ]
    body = dict(
        stream=False,
        messages=messages,
        **model["generation_kwargs"],
        max_tokens=model["max_out_len"],
        model=model["model"],
    )
    return prompt, digest(json.dumps(body).encode())


def checked_row(i, sample, qid, pred, detail, http, model, template):
    label = f"row {i}"
    prompt, wire_digest = prompt_and_wire_digest(sample["question"], template, model)
    check_fields(pred, dict(id=i, success=True, origin_prompt=prompt, gold=sample["answer"]), label + ".prediction")
    require(detail["prompt"] == prompt, f"{label}: official prompt differs")
    require(wire_digest == http["payload_sha256"], f"{label}: wire payload digest differs")
    choice = exactly_one(http["choices"], f"{label} HTTP choice")
    require(choice["index"] == 0 and choice["finish_reason"] in ("stop", "length"), f"{label}: invalid choice")
    message = choice["message"]
    require(message["role"] == "assistant", f"{label}: response role differs")
    for key in ("content", "reasoning_content", "reasoning"):
        require(message.get(key) is None or isinstance(message[key], str), f"{label}: non-text {key}")
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    # Output.get_prediction in pinned AISBench uses </think> only when both fields are nonempty.
    raw = reasoning + "</think>" + content if reasoning and content else reasoning or content
    require(pred["prediction"] == detail["origin_prediction"] == raw, f"{label}: raw generation differs")
    if reasoning:
        require(pred["content"] == content, f"{label}: separated final content differs")
    fields = ("predictions", "references", "correct")
    require(all(isinstance(detail[k], list) for k in fields), f"{label}: expected official per-replica lists")
    extracted, reference, correct = (exactly_one(detail[k], f"{label} official {k}") for k in fields)
    require(
        isinstance(extracted, str) and isinstance(reference, str) and type(correct) is bool,
        f"{label}: malformed official detail",
    )
    require(reference == sample["answer"].split("#### ")[1].replace(",", ""), f"{label}: official reference differs")
    return dict(
        index=i,
        question_id=qid,
        response_id=http["response_id"],
        request_sha256=http["payload_sha256"],
        start_ns=http["start_ns"],
        end_ns=http["end_ns"],
        choice=choice,
        prediction_uuid=pred["uuid"],
        official_extracted=extracted,
        official_reference=reference,
        official_correct=correct,
    )


def audit(run, gates_path, inputs_path, source, data):
    report = {"verdict": "INCONCLUSIVE", "errors": []}
    try:
        gates, inputs, samples, template = frozen_inputs(gates_path, inputs_path, source, data)
        cfg, ids = gates["benchmark"], inputs["ordered_ids"]
        ends, report["http"], errors = audit_http(run, set(ids), gates["served_model"], cfg["client_max_inflight"])
        report["errors"].extend(errors)
        check_launch(run, gates, inputs, gates_path, inputs_path)
        check_fields(read_json(run / "exit.json"), {"exit_code": 0}, "exit")
        path, model, abbr = resolved_config(run, gates, template)
        work = path.parent.parent
        stem = work / "predictions" / model["abbr"] / abbr
        predictions, failed = stem.with_suffix(".jsonl"), stem.with_name(abbr + "_failed.jsonl")
        require(not failed.exists() or not list(read_lines(failed)), "AISBench recorded failed inference rows")
        official = work / "results" / model["abbr"] / (abbr + ".json")
        result = read_json(official)
        require(result.get("type") == "GEN" and "error" not in result, "Official GEN evaluation missing or failed")
        details = result["details"]
        indexed = {}
        for pred in read_lines(predictions):
            i = pred["id"]
            require(
                type(i) is int and i not in indexed and pred["data_abbr"] == abbr, "Duplicate or invalid prediction ID"
            )
            indexed[i] = pred
        require(set(indexed) == set(range(len(ids))), "Official prediction IDs missing or outside the frozen range")
        require(
            set(details) == {str(i) for i in indexed}, "Official per-row details missing or outside the frozen range"
        )
        if report["errors"]:
            return report
        rows = [
            checked_row(i, sample, ids[i], indexed[i], details[str(i)], ends[ids[i]], model, template)
            for i, sample in enumerate(samples)
        ]
        for key in ("response_id", "prediction_uuid"):
            require(
                all(isinstance(r[key], str) and r[key] for r in rows) and len({r[key] for r in rows}) == len(rows),
                f"Missing or duplicate {key}",
            )
        accuracy = result["accuracy"]
        correct = sum(r["official_correct"] for r in rows)
        require(
            type(accuracy) in (int, float)
            and math.isclose(accuracy, 100 * correct / len(rows), rel_tol=0, abs_tol=1e-10),
            "Official aggregate contradicts official per-question marks",
        )
        report.update(
            verdict="PASS" if correct >= cfg["required_correct"] else "FAIL",
            official_accuracy=accuracy,
            official_correct=correct,
            required_correct=cfg["required_correct"],
            total=len(rows),
            rows=rows,
            evidence_sha256={
                str(p.relative_to(run)): sha(p)
                for p in (
                    run / "launch.json",
                    run / "exit.json",
                    run / "http.jsonl",
                    run / "http-summary.json",
                    path,
                    predictions,
                    official,
                )
            },
        )
    except FileNotFoundError as exc:
        report["errors"].append(f"Missing {Path(exc.filename).name}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, StopIteration, SyntaxError) as exc:
        report["errors"].append(str(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("run", "gates", "inputs", "aisbench-source", "output"):
        parser.add_argument("--" + argument, required=True, type=Path)
    parser.add_argument("--gsm8k-data", type=Path, help="Defaults to ais_bench/datasets/gsm8k under the source")
    args = parser.parse_args()
    data = args.gsm8k_data or args.aisbench_source / "ais_bench/datasets/gsm8k"
    report = audit(args.run, args.gates, args.inputs, args.aisbench_source, data)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False)
        output.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "evidence_sha256")}, ensure_ascii=False))
    return 0 if report["verdict"] == "PASS" else 1 if report["verdict"] == "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
