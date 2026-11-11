# SPDX-License-Identifier: Apache-2.0
"""Audit one GPQA prediction and official evaluation detail against each observed request."""

import json
import math
import sys
from collections import Counter
from pathlib import Path

from gpqa import CSV_NAME, DATASET, LETTERS, MODEL, build_args, frozen_template, read_json, require, sha, text_sha
from gpqa_http import canonical

# Reuse the existing AISBench evidence reader (including duplicate-key rejection and non-executing config parsing).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from day0_stage2.validator.summarize_gsm8k import assignments, check_fields, config_value, exactly_one, read_lines


def checked_config(run, settings, template, dry):
    path = exactly_one((run / "work").glob("*/configs/*.py"), "resolved AISBench config")
    config = assignments(path)
    cli, models, tasks, evaluation = (config_value(config[k]) for k in ("cli_args", "models", "datasets", "eval"))
    model = exactly_one(models, "model")
    task = exactly_one(tasks, "dataset")
    limits = dict(
        batch_size=settings["concurrency"],
        generation_kwargs=settings["generation"],
        max_out_len=settings["max_tokens"],
        request_rate=settings["rate"],
        retry=settings["attempts"],
    )
    check_fields(
        cli,
        dict(
            num_warmups=settings["warmups"],
            num_prompts=settings["count"],
            dry_run=dry,
            dump_eval_details=True,
            mode="all",
            models=[MODEL],
            datasets=[DATASET],
            model_name=settings["model"],
            **limits,
        ),
        "CLI",
    )
    check_fields(
        model,
        dict(
            model=settings["model"],
            stream=False,
            type="ais_bench.benchmark.models.api_models.vllm_custom_api_chat.VLLMCustomAPIChat",
            pred_postprocessor={
                "type": "ais_bench.benchmark.utils.postprocess.model_postprocessors.extract_non_reasoning_content"
            },
            **limits,
        ),
        "model",
    )
    module = "ais_bench.benchmark.datasets.gpqa."
    check_fields(
        task,
        dict(
            abbr="GPQA_diamond",
            name=CSV_NAME,
            type=module + "GPQADataset",
            reader_cfg=dict(
                input_columns=["question", *LETTERS], output_column="answer", test_range=settings["reader_range"]
            ),
            eval_cfg=dict(
                evaluator={"type": module + "GPQAEvaluator"},
                pred_postprocessor={"type": module + "GPQA_Simple_Eval_postprocess"},
            ),
        ),
        "dataset",
    )
    require(
        task["infer_cfg"]["prompt_template"]["template"]["round"] == [{"role": "HUMAN", "prompt": template}],
        "Effective prompt differs",
    )
    require(evaluation["runner"]["task"]["dump_details"] is True, "Effective evaluation details disabled")
    return path, model, task["abbr"]


def read_http(run, manifest, dry):
    settings = manifest["settings"]
    ids = {r["question_id"]: r for r in manifest["rows"]}
    active, ends = {}, {}
    statuses = Counter()
    previous = peak = forwards = 0
    for line, row in enumerate(read_lines(run / "http.jsonl"), 1):
        event, qid = row["event"], row["question_id"]
        require(event in ("start", "end") and qid in ids, f"HTTP event or unknown input at line {line}")
        stamp = row["start_ns" if event == "start" else "end_ns"]
        require(type(stamp) is int and stamp >= previous, f"Nonmonotonic HTTP clock at line {line}")
        previous = stamp
        if event == "start":
            require(
                qid not in active and qid not in ends and row["attempt"] == 1, f"Repeated HTTP attempt at line {line}"
            )
            active[qid] = row
            peak = max(peak, len(active))
            check_fields(
                row,
                dict(
                    expected=True,
                    wire_valid=True,
                    model=settings["model"],
                    active_at_start=len(active),
                    prompt_sha256=ids[qid]["prompt_sha256"],
                ),
                f"HTTP line {line}",
            )
            for k in ("canonical_sha256", "payload_sha256"):
                require(
                    isinstance(row[k], str) and len(row[k]) == 64 and len(bytes.fromhex(row[k])) == 32,
                    f"Missing wire digest at line {line}",
                )
        else:
            start = active.pop(qid, None)
            require(
                start is not None
                and {k: v for k, v in start.items() if k != "event"} == {k: row.get(k) for k in start if k != "event"},
                f"HTTP response without matching start at line {line}",
            )
            require("error_type" not in row, f"Proxy failed at HTTP line {line}")
            check_fields(row, dict(status=410 if dry else 200, forwarded=not dry), f"HTTP line {line}")
            if dry:
                require(
                    not any(k in row for k in ("choices", "served_model", "response_id")),
                    "Dry-run contains a model response",
                )
            else:
                check_fields(row, dict(served_model=settings["model"]), f"HTTP line {line}")
                require(
                    isinstance(row.get("choices"), list) and len(row["choices"]) == 1,
                    f"Expected one HTTP choice at line {line}",
                )
            ends[qid] = row
            statuses[str(row["status"])] += 1
            forwards += int(row["forwarded"])
    require(not active and set(ends) == set(ids), "Missing or unfinished HTTP inputs")
    require(1 <= peak <= settings["concurrency"], "Observed concurrency exceeds the configured bound")
    stats = dict(
        dry_run=dry,
        observed_peak=peak,
        total_attempts=len(ids),
        ends=len(ends),
        upstream_chat_forwards=forwards,
        statuses=dict(statuses),
        missing=[],
        duplicates=[],
    )
    require(stats == read_json(run / "http-summary.json"), "Observer summary contradicts raw HTTP events")
    return ends, stats


def prompt_digest(prompt, frozen, origin):
    require(
        isinstance(prompt, list)
        and len(prompt) == 1
        and set(prompt[0]) == {"role", "prompt"}
        and prompt[0]["role"] == "HUMAN"
        and isinstance(prompt[0]["prompt"], str)
        and text_sha(prompt[0]["prompt"]) == frozen["prompt_sha256"],
        f"Row {frozen['index']} {origin} prompt differs",
    )
    return prompt[0]["prompt"]


def expected_wire(prompt, model):
    payload = dict(
        stream=False,
        messages=[{"content": prompt, "role": "user"}],
        **model["generation_kwargs"],
        max_tokens=model["max_out_len"],
        model=model["model"],
    )
    return sha(json.dumps(payload).encode("utf-8")), text_sha(canonical(payload))


def checked_row(frozen, prediction, detail, http, model):
    from ais_bench.benchmark.datasets.gpqa import GPQA_Simple_Eval_postprocess
    from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

    i = frozen["index"]
    label = f"Row {i}"
    check_fields(prediction, dict(id=i, data_abbr="GPQA_diamond", success=True, gold=frozen["reference"]), label)
    prompt = prompt_digest(prediction["origin_prompt"], frozen, "prediction")
    require(detail["prompt"] == prediction["origin_prompt"], label + " official detail prompt differs")
    raw_digest, normal_digest = expected_wire(prompt, model)
    require(
        raw_digest == http["payload_sha256"] and normal_digest == http["canonical_sha256"],
        label + " HTTP payload differs",
    )
    choice = exactly_one(http["choices"], label + " HTTP choice")
    require(
        type(choice["index"]) is int and choice["index"] == 0 and choice["finish_reason"] in ("stop", "length"),
        label + " invalid HTTP choice",
    )
    msg = choice["message"]
    require(
        msg["role"] == "assistant"
        and all(msg.get(k) is None or isinstance(msg[k], str) for k in ("content", "reasoning", "reasoning_content")),
        label + " invalid response message",
    )
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    # Pinned Output.get_prediction composes these same two fields when reasoning is present.
    raw = reasoning + "</think>" + content if reasoning and content else reasoning or content
    require(raw == prediction["prediction"] == detail["origin_prediction"], label + " raw generation differs")
    if reasoning:
        require(prediction["content"] == content, label + " separated final answer differs")
    for k in ("predictions", "references", "correct"):
        require(isinstance(detail[k], list) and len(detail[k]) == 1, label + " must have one official replica")
    extracted = GPQA_Simple_Eval_postprocess(extract_non_reasoning_content(raw))
    passed = detail["correct"][0]
    require(
        detail["predictions"][0] == extracted
        and detail["references"][0] == frozen["reference"]
        and type(passed) is bool,
        label + " official extraction or reference differs",
    )
    require(passed == (extracted == frozen["reference"]), label + " official mark differs")
    return dict(
        index=i,
        question_id=frozen["question_id"],
        display_to_source=frozen["display_to_source"],
        prompt_sha256=frozen["prompt_sha256"],
        request_sha256=raw_digest,
        response_sha256=http["response_sha256"],
        response_id=http["response_id"],
        prediction_uuid=prediction["uuid"],
        start_ns=http["start_ns"],
        end_ns=http["end_ns"],
        choice=choice,
        official_extracted=extracted,
        official_reference=frozen["reference"],
        official_correct=passed,
    )


def indexed_predictions(path, expected):
    indexed = {}
    for pred in read_lines(path):
        i = pred["id"]
        require(type(i) is int and i not in indexed, "Duplicate or invalid prediction ID")
        indexed[i] = pred
    require(set(indexed) == set(range(expected)), "Missing or extra official prediction IDs")
    return indexed


def audit(run, manifest, manifest_path, source):
    report = {"complete": False, "errors": []}
    try:
        settings = manifest["settings"]
        launch = read_json(run / "launch.json")
        dry = launch["dry_run"]
        require(type(dry) is bool, "Invalid dry-run marker")
        report["dry_run"] = dry
        check_fields(
            launch,
            dict(
                checked_questions=settings["count"],
                manifest_sha256=sha(manifest_path.read_bytes()),
                aisbench_revision=manifest["aisbench_revision"],
                csv_sha256=manifest["csv_sha256"],
            ),
            "launch",
        )
        argv = launch["argv"]
        expected = build_args(argv[0], settings, Path(argv[argv.index("--work-dir") + 1]).parent, dry)
        require(argv == expected, "Unfrozen launch flags or effective CLI values")
        check_fields(read_json(run / "exit.json"), dict(exit_code=0), "exit")
        path, model, abbr = checked_config(run, settings, frozen_template(source), dry)
        ends, report["http"] = read_http(run, manifest, dry)
        work = path.parent.parent
        predictions = work / "predictions" / model["abbr"] / (abbr + ".jsonl")
        failed = predictions.with_name(abbr + "_failed.jsonl")
        result_file = work / "results" / model["abbr"] / (abbr + ".json")
        files = [run / f for f in ("launch.json", "exit.json", "http.jsonl", "http-summary.json")] + [path]
        if dry:
            require(not result_file.exists(), "Unexpected official score in dry-run")
            require(
                not predictions.exists() or not list(read_lines(predictions)),
                "Unexpected successful dry-run predictions",
            )
            attempts = indexed_predictions(failed, len(manifest["rows"]))
            for f in manifest["rows"]:
                qid = f["question_id"]
                pred = attempts[f["index"]]
                require(
                    isinstance(pred.get("error_info"), str) and bool(pred["error_info"]),
                    "Missing dry-run failure receipt",
                )
                prompt = prompt_digest(pred["input"], f, "intercepted")
                digests = expected_wire(prompt, model)
                require(
                    digests == (ends[qid]["payload_sha256"], ends[qid]["canonical_sha256"]), "Intercepted wire differs"
                )
            files.append(failed)
            report["official_score_emitted"] = False
        else:
            require(not failed.exists() or not list(read_lines(failed)), "AISBench recorded failed requests")
            result = read_json(result_file)
            require(result.get("type") == "GEN" and "error" not in result, "Official GEN score missing")
            count = settings["count"]
            indexed = indexed_predictions(predictions, count)
            details = result["details"]
            require(set(details) == {str(i) for i in range(count)}, "Missing or extra official per-row details")
            rows = [
                checked_row(f, indexed[f["index"]], details[str(f["index"])], ends[f["question_id"]], model)
                for f in manifest["rows"]
            ]
            for field in ("prediction_uuid", "response_id"):
                require(
                    all(isinstance(r[field], str) and r[field] for r in rows)
                    and len({r[field] for r in rows}) == count,
                    "Missing or duplicate " + field,
                )
            correct = sum(r["official_correct"] for r in rows)
            accuracy = result["accuracy"]
            require(
                type(accuracy) in (float, int)
                and math.isclose(accuracy, 100 * correct / count, rel_tol=0, abs_tol=1e-10),
                "Official aggregate differs from per-question marks",
            )
            report.update(
                official_accuracy=accuracy,
                official_correct=correct,
                official_unparseable=sum(r["official_extracted"] is None for r in rows),
                rows=rows,
            )
            files.extend([predictions, result_file])
        report.update(
            complete=True,
            total=settings["count"],
            evidence_sha256={str(p.relative_to(run)): sha(p.read_bytes()) for p in files},
        )
    except FileNotFoundError as error:
        report["errors"].append(f"Missing {Path(error.filename).name}")
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, StopIteration, OSError, SyntaxError) as error:
        report["errors"].append(str(error))
    return report
