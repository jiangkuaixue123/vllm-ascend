# SPDX-License-Identifier: Apache-2.0
"""Replay a frozen AISBench GSM8K configuration against an OpenAI-compatible service."""

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs(source, manifest, benchmark):
    dataset = source / "ais_bench/datasets/gsm8k"
    for root, expected in ((source, manifest["scoring_sources"]), (dataset, manifest["inputs"])):
        for filename, digest in expected.items():
            if sha256(root / filename) != digest:
                raise ValueError(f"Identity check failed for {filename}")
    count = benchmark["count"]
    with (dataset / "test.jsonl").open(encoding="utf-8") as stream:
        ids = [
            hashlib.sha256(json.loads(line)["question"].encode()).hexdigest() for _, line in zip(range(count), stream)
        ]
    if len(ids) != count or len(set(ids)) != count or ids != manifest["ordered_ids"]:
        raise ValueError("The selected GSM8K rows do not match the frozen ordered question IDs")
    if not (manifest["reader_test_range"] == benchmark["reader_range"] == f"[:{count}]"):
        raise ValueError("The AISBench reader range does not match the input manifest")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--aisbench-source", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="HTTP service origin, including its port")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Ask AISBench to resolve the configuration and tasks")
    args = parser.parse_args()
    source = args.aisbench_source.resolve()
    output = args.output.resolve()
    gates = json.loads(args.gates.read_text())
    manifest = json.loads(args.inputs.read_text())
    expected_manifest = gates["sources"]["plan/gsm8k-inputs.json"]
    if sha256(args.inputs) != expected_manifest:
        raise ValueError("The input manifest differs from the one in the frozen gates")
    config = gates["benchmark"]
    ids = check_inputs(source, manifest, config)
    endpoint = urlsplit(args.endpoint)
    if endpoint.scheme != "http" or not endpoint.hostname or endpoint.port is None or endpoint.path not in ("", "/"):
        parser.error("--endpoint must be an HTTP origin with an explicit port")
    client = shutil.which("ais_bench")
    if client is None:
        parser.error("The AISBench executable must be on PATH")
    argv = [
        client,
        "--models",
        config["model"],
        "--datasets",
        config["dataset"],
        "--model-name",
        gates["served_model"],
        "--host-ip",
        endpoint.hostname,
        "--host-port",
        str(endpoint.port),
        # AISBench otherwise warms up on the first scored example.
        "--num-warmups",
        "0",
        "--num-prompts",
        str(len(ids)),
        "--batch-size",
        str(config["client_max_inflight"]),
        "--request-rate",
        str(config["request_rate"]),
        "--retry",
        str(config["max_http_attempts_per_case"]),
        "--max-out-len",
        str(config["max_output_tokens"]),
        "--generation-kwargs",
        json.dumps(config["generation"]),
        "--dump-eval-details",
        "--work-dir",
        str(output / "work"),
    ]
    if args.dry_run:
        argv.append("--dry-run")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        dry_run=args.dry_run,
        aisbench_revision=manifest["aisbench_revision"],
        data_revision=manifest["dataset_revision"],
        checked_questions=len(ids),
        gates_sha256=sha256(args.gates),
        input_manifest_sha256=sha256(args.inputs),
        argv=argv,
    )
    with (output / "launch.json").open("x") as file:
        json.dump(receipt, file, indent=2)
    if args.dry_run:
        with (output / "client.log").open("x") as log:
            status = subprocess.run(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    else:
        from observe_gsm8k import run_observed

        status = asyncio.run(run_observed(argv, source, output, args.endpoint.rstrip("/"), set(ids)))
    (output / "exit.json").write_text(json.dumps({"exit_code": status}) + "\n")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
