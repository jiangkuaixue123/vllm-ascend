# SPDX-License-Identifier: Apache-2.0
"""Run and audit the frozen AISBench GPQA-Diamond single-pass chat task."""

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

DATASET = "gpqa_gen_0_shot_cot_chat_prompt"
DATA_DIR = Path("ais_bench/datasets/gpqa")
CSV_NAME = "gpqa_diamond.csv"
BENCH = "ais_bench/benchmark/"
CONFIG = BENCH + "configs/datasets/gpqa/" + DATASET + ".py"
MODEL = "vllm_api_general_chat"
SOURCE_FILES = (
    CONFIG,
    BENCH + "configs/models/vllm_api/" + MODEL + ".py",
    BENCH + "datasets/gpqa.py",
    BENCH + "models/api_models/vllm_custom_api_chat.py",
    BENCH + "models/output.py",
    BENCH + "utils/postprocess/model_postprocessors.py",
    BENCH + "utils/prompt/prompt.py",
    BENCH + "openicl/icl_prompt_template/icl_prompt_template.py",
    BENCH + "openicl/icl_inferencer/output_handler/base_handler.py",
    BENCH + "cli/utils.py",
    BENCH + "cli/argument_parser.py",
    BENCH + "tasks/openicl_api_infer.py",
    BENCH + "tasks/openicl_eval.py",
    BENCH + "openicl/icl_evaluator/icl_base_evaluator.py",
    BENCH + "openicl/icl_inferencer/output_handler/gen_inferencer_output_handler.py",
)
LETTERS = "ABCD"


def require(test, msg):
    if not test:
        raise ValueError(msg)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(text):
    return sha(text.encode("utf-8"))


def read_json(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            require(k not in d, "Duplicate JSON key")
            d[k] = v
        return d

    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def frozen_template(source):
    tree = ast.parse((source / CONFIG).read_bytes())
    node = next(x.value for x in tree.body if isinstance(x, ast.Assign) and x.targets[0].id == "align_prompt")
    require(
        isinstance(node, ast.Call) and node.func.attr == "strip" and not node.args,
        "Unknown official prompt declaration",
    )
    return ast.literal_eval(node.func.value).strip()


def identity(source, csv_path):
    """Mirror the frozen deterministic permutation, retaining only identities in the manifest."""
    template = frozen_template(source)
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        require(
            header[7:12]
            == ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
            "Unexpected Diamond CSV schema",
        )
        for i, row in enumerate(reader):
            q, options = row[7], row[8:12]
            perm = [(j + i + 1) % 4 for j in range(4)]
            shown = [options[j] for j in perm]
            truth = LETTERS[shown.index(options[0])]
            item = {"question": q, **dict(zip(LETTERS, shown))}
            # AISBench substitutes input columns in order. Placeholders introduced by prior fields can expand.
            content = template
            for column, value in {**item, "answer": ""}.items():
                content = content.replace("{" + column + "}", value)
            rows.append(
                dict(
                    index=i,
                    question_id=text_sha(q),
                    display_to_source=perm,
                    option_sha256=list(map(text_sha, shown)),
                    reference=truth,
                    prompt_sha256=text_sha(content),
                )
            )
    require(
        len({r["question_id"] for r in rows}) == len(rows) == len({r["prompt_sha256"] for r in rows}),
        "Duplicate question or prompt",
    )
    return rows


def check_inputs(source, data_root, manifest):
    csv_path = data_root / DATA_DIR / CSV_NAME
    require(sha(csv_path.read_bytes()) == manifest["csv_sha256"], "Diamond CSV differs from the freeze")
    for filename, expected in manifest["official_sources"].items():
        require(sha((source / filename).read_bytes()) == expected, f"Different official file {filename}")
    rows = identity(source, csv_path)
    require(
        rows == manifest["rows"] and len(rows) == manifest["settings"]["count"],
        "Frozen row, option or reference differs",
    )
    return rows


def official_inputs(source, data_root, manifest):
    import ais_bench.benchmark.datasets.gpqa as gpqa
    from ais_bench.benchmark.datasets.gpqa import GPQADataset
    from ais_bench.benchmark.utils.prompt import safe_format

    require(
        Path(gpqa.__file__).resolve() == (source / (BENCH + "datasets/gpqa.py")).resolve(),
        "Unexpected imported AISBench source",
    )
    rows = check_inputs(source, data_root, manifest)
    dataset = GPQADataset.load(str((data_root / DATA_DIR).resolve()), CSV_NAME)
    require(len(dataset) == len(rows), "Official row count differs")
    template = frozen_template(source)
    for frozen, actual in zip(rows, dataset, strict=True):
        require(
            text_sha(actual["question"]) == frozen["question_id"]
            and [text_sha(actual[c]) for c in LETTERS] == frozen["option_sha256"]
            and actual["answer"] == frozen["reference"]
            and text_sha(safe_format(template, **{**actual, "answer": ""})) == frozen["prompt_sha256"],
            f"Official ordered display differs on row {frozen['index']}",
        )
    return rows


def build_args(client, settings, output, dry):
    return [
        client,
        "--models",
        MODEL,
        "--datasets",
        DATASET,
        "--model-name",
        settings["model"],
        "--host-ip",
        "127.0.0.1",
        "--host-port",
        "1",
        "--num-warmups",
        str(settings["warmups"]),
        "--retry",
        str(settings["attempts"]),
        "--num-prompts",
        str(settings["count"]),
        "--batch-size",
        str(settings["concurrency"]),
        "--request-rate",
        str(settings["rate"]),
        "--max-out-len",
        str(settings["max_tokens"]),
        "--generation-kwargs",
        json.dumps(settings["generation"]),
        "--dump-eval-details",
        "--work-dir",
        str(output / "work"),
        *(["--dry-run"] if dry else []),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run", "audit"))
    for arg in ("manifest", "source", "data-root"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    m = read_json(args.manifest)
    rows = official_inputs(args.source.resolve(), args.data_root.resolve(), m)
    if args.action == "check":
        print(
            json.dumps(
                dict(
                    official_rows=len(rows),
                    unique_question_ids=len({r["question_id"] for r in rows}),
                    displayed_options_checked=4 * len(rows),
                    references_checked=len(rows),
                    csv_sha256=m["csv_sha256"],
                )
            )
        )
        return
    if args.action == "audit":
        require(args.run and args.output, "Audit requires --run and a new --output file")
        from gpqa_audit import audit

        report = audit(args.run, m, args.manifest, args.source)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False)
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
        raise SystemExit(0 if report["complete"] else 2)
    from gpqa_http import observe

    require(args.output and (args.dry_run or args.endpoint), "Run requires --output plus --endpoint or --dry-run")
    require(not args.dry_run or args.endpoint is None, "A dry-run uses its own loopback sink")
    root = args.output.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    client = shutil.which("ais_bench")
    require(client, "The pinned ais_bench executable must be on PATH")
    argv = build_args(client, m["settings"], root, args.dry_run)
    env = {**os.environ, "AIS_BENCH_DATASETS_CACHE": str(args.data_root.resolve())}
    launch = dict(
        dry_run=args.dry_run,
        checked_questions=len(rows),
        manifest_sha256=sha(args.manifest.read_bytes()),
        aisbench_revision=m["aisbench_revision"],
        csv_sha256=m["csv_sha256"],
        argv=argv,
    )
    (root / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    status = asyncio.run(observe(argv, args.source.resolve(), env, root, args.endpoint, rows, m["settings"]))
    (root / "exit.json").write_text(json.dumps({"exit_code": status}) + "\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
