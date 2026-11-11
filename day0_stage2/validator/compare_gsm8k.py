# SPDX-License-Identifier: Apache-2.0
"""Re-audit two AISBench runs under the same frozen inputs and compare official marks by question."""

import argparse
import json
from collections import Counter
from pathlib import Path

from summarize_gsm8k import audit, sha


def compare(baseline, candidate):
    for name, run in (("baseline", baseline), ("candidate", candidate)):
        if run["errors"] or run["verdict"] == "INCONCLUSIVE":
            errors = run["errors"] or ["incomplete run"]
            return {"verdict": "INCONCLUSIVE", "errors": [f"{name}: {error}" for error in errors]}
        if not run["rows"] or len(run["rows"]) != run["total"]:
            return {"verdict": "INCONCLUSIVE", "errors": [f"{name}: incomplete question rows"]}
    if baseline["required_correct"] != candidate["required_correct"] or baseline["total"] != candidate["total"]:
        return {"verdict": "INCONCLUSIVE", "errors": ["Scoring requirements or sample counts differ"]}

    counts = Counter()
    changes = []
    for left, right in zip(baseline["rows"], candidate["rows"]):
        for key in ("index", "question_id", "request_sha256", "official_reference"):
            if left[key] != right[key]:
                return {"verdict": "INCONCLUSIVE", "errors": [f"{key} differs at index {left['index']}"]}
        flags = (left["official_correct"], right["official_correct"])
        counts[flags] += 1
        if flags[0] != flags[1]:
            changes.append(
                {
                    "index": left["index"],
                    "question_id": left["question_id"],
                    "request_sha256": left["request_sha256"],
                    "baseline_correct": flags[0],
                    "candidate_correct": flags[1],
                }
            )
    before, after = baseline["official_correct"], candidate["official_correct"]
    return {
        "verdict": "PASS" if after >= max(before, candidate["required_correct"]) else "FAIL",
        "paired": baseline["total"],
        "required_correct": candidate["required_correct"],
        "baseline_correct": before,
        "candidate_correct": after,
        "delta_correct": after - before,
        "delta_percentage_points": 100 * (after - before) / baseline["total"],
        "baseline_accuracy_pct": baseline["official_accuracy"],
        "candidate_accuracy_pct": candidate["official_accuracy"],
        "both_correct": counts[True, True],
        "both_incorrect": counts[False, False],
        "baseline_only": counts[True, False],
        "candidate_only": counts[False, True],
        "changed": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "gates", "inputs", "aisbench-source", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gsm8k-data", type=Path)
    args = parser.parse_args()
    data = args.gsm8k_data or args.aisbench_source / "ais_bench/datasets/gsm8k"
    runs = [audit(run, args.gates, args.inputs, args.aisbench_source, data) for run in (args.baseline, args.candidate)]
    result = compare(*runs)
    if result["verdict"] != "INCONCLUSIVE":
        result.update(input_manifest_sha256=sha(args.inputs), gates_sha256=sha(args.gates))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "changed"}))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
