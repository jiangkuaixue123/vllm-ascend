# SPDX-License-Identifier: Apache-2.0
"""Re-audit two official GPQA runs and pair their marks against the frozen manifest in input order."""

import argparse
import json
import math
from pathlib import Path

from gpqa import check_inputs, read_json, require, sha
from gpqa_audit import audit, read_http
from gpqa_template_tokens import check_prepared, save, token_sha

COMPARISON = "candidate.official_correct >= reference.official_correct"


def relative_file(base, value):
    require(isinstance(value, str) and value and not Path(value).is_absolute(), "Expected a relative receipt path")
    return (base / value).resolve()


def spec_identity(record, root, manifest_digest, gate, prepared):
    require(
        record["complete"] is True and record["matched"] is True, "External RunSpec comparison is incomplete or differs"
    )
    require(record["manifest_sha256"] == manifest_digest, "External RunSpec receipt has another manifest")
    sides = {}
    for label in ("reference", "candidate"):
        side = record[label]
        spec_path = relative_file(root, side["run_spec"])
        digest = sha(spec_path.read_bytes())
        require(digest == side["run_spec_sha256"], label + " RunSpec digest differs")
        spec = read_json(spec_path)
        require(
            spec["model_revision"] == prepared["hf_revision"] and spec["service_alias"] == prepared["model"],
            label + " RunSpec model differs",
        )
        image = spec["image_digest"].split("@")[-1]
        allowed = (
            [x.split("@")[-1] for x in gate["reference_priority"]]
            if label == "reference"
            else [gate["candidate_image_digest"].split("@")[-1]]
        )
        require(image in allowed, label + " image is outside the frozen identity")
        sides[label] = dict(
            spec_sha256=digest,
            image_digest=image,
            run=relative_file(root, side["run"]),
            tokens=relative_file(root, side["token_check"]),
        )
    require(
        sides["reference"]["run"] != sides["candidate"]["run"]
        and sides["reference"]["tokens"] != sides["candidate"]["tokens"]
        and sides["reference"]["spec_sha256"] != sides["candidate"]["spec_sha256"],
        "Reference and candidate must be independent observations",
    )
    return sides


def verify_tokens(side, prepared, prepared_sha):
    evidence = read_json(side["tokens"])
    require(
        evidence["schema"] == 1
        and evidence["complete"] is True
        and evidence["method"] == "POST"
        and evidence["path"] == "/tokenize",
        "Missing complete tokenizer audit",
    )
    for key, expect in (
        ("prepared_sha256", prepared_sha),
        ("manifest_sha256", prepared["manifest_sha256"]),
        ("run_spec_sha256", side["spec_sha256"]),
        ("tokenizer_sha256", prepared["tokenizer_sha256"]),
        ("template_sha256", prepared["template_sha256"]),
        ("hf_revision", prepared["hf_revision"]),
    ):
        require(evidence[key] == expect, "Foreign tokenizer receipt: " + key)
    require(
        type(evidence["count"]) is int
        and type(evidence["matched"]) is int
        and evidence["count"] == evidence["matched"] == len(prepared["rows"]),
        "Incomplete tokenizer count",
    )
    for actual, expected in zip(evidence["rows"], prepared["rows"], strict=True):
        require(
            all(actual[k] == expected[k] for k in ("question_id", "index", "prompt_sha256", "rendered_sha256")),
            "Tokenizer input order differs",
        )
        require(
            actual["status"] == 200
            and type(actual["attempts"]) is int
            and actual["attempts"] == 1
            and actual["equal"] is True
            and len(bytes.fromhex(actual["response_sha256"])) == 32,
            "Missing real single-call tokenize receipt",
        )
        ids = actual["actual_tokens"]
        require(
            token_sha(ids)
            == actual["actual_tokens_sha256"]
            == actual["expected_tokens_sha256"]
            == expected["tokens_sha256"]
            and ids == expected["expected_tokens"]
            and len(ids) == actual["actual_count"] == actual["expected_count"],
            "Template token IDs differ",
        )


def audited(side, manifest, manifest_path, source):
    run = side["run"]
    stored = read_json(run / "audit.json")
    require(
        stored["complete"] is True and stored["dry_run"] is False and stored["errors"] == [],
        "Missing complete official live audit",
    )
    fresh = audit(run, manifest, manifest_path, source)
    require(
        fresh["complete"] is True and fresh == stored,
        "Official audit no longer reproduces: " + str(fresh.get("errors", [])),
    )
    ends, _ = read_http(run, manifest, False)
    return fresh, ends, sha((run / "audit.json").read_bytes())


def join(manifest, prepared, reference, candidate):
    # A comparison delegates neither the count nor the pass/fail predicate to the recorded aggregate.
    rows = []
    counts = dict(both_correct=0, candidate_only=0, reference_only=0, both_incorrect=0)
    expected = manifest["settings"]["count"]
    require(
        len(manifest["rows"]) == expected and len({r["question_id"] for r in manifest["rows"]}) == expected,
        "Duplicate or missing frozen identity",
    )
    for i, (f, inp, rr, cc) in enumerate(
        zip(manifest["rows"], prepared["rows"], reference[0]["rows"], candidate[0]["rows"], strict=True)
    ):
        qid = f["question_id"]
        require(type(f["index"]) is int and f["index"] == i, "Manifest row order differs")
        raw, canon = rr["request_sha256"], reference[1][qid]["canonical_sha256"]
        require(
            raw == cc["request_sha256"] and canon == candidate[1][qid]["canonical_sha256"],
            "Paired HTTP generation request differs",
        )
        scoring = {}
        for label, official, http in (("reference", rr, reference[1][qid]), ("candidate", cc, candidate[1][qid])):
            require(
                official["index"] == i
                and all(official[k] == f[k] for k in ("question_id", "display_to_source", "prompt_sha256"))
                and official["official_reference"] == f["reference"]
                and raw == http["payload_sha256"]
                and official["response_sha256"] == http["response_sha256"],
                "Official identity, wire or truth differs",
            )
            extracted, mark = official["official_extracted"], official["official_correct"]
            require(
                type(mark) is bool
                and (isinstance(extracted, str) or extracted is None)
                and mark == (extracted == f["reference"]),
                "Official per-question mark contradicts its extraction",
            )
            scoring[label] = dict(
                official_extracted=extracted, official_correct=mark, response_sha256=official["response_sha256"]
            )
        r, c = scoring["reference"]["official_correct"], scoring["candidate"]["official_correct"]
        group = ("both_correct" if r else "candidate_only") if c else ("reference_only" if r else "both_incorrect")
        counts[group] += 1
        rows.append(
            dict(
                index=i,
                question_id=qid,
                display_to_source=f["display_to_source"],
                option_sha256=f["option_sha256"],
                prompt_sha256=f["prompt_sha256"],
                request_sha256=raw,
                canonical_request_sha256=canon,
                rendered_sha256=inp["rendered_sha256"],
                token_ids_sha256=inp["tokens_sha256"],
                official_reference=f["reference"],
                pair=group,
                **scoring,
            )
        )
    totals = {}
    for label, receipt in (("reference", reference[0]), ("candidate", candidate[0])):
        total = sum(row[label]["official_correct"] for row in rows)
        unparseable = sum(row[label]["official_extracted"] is None for row in rows)
        require(
            total == receipt["official_correct"]
            and unparseable == receipt["official_unparseable"]
            and receipt["total"] == expected
            and math.isclose(receipt["official_accuracy"], 100 * total / expected, rel_tol=0, abs_tol=1e-10),
            "Official total contradicts paired rows",
        )
        totals[label] = dict(
            official_correct=total, official_accuracy=receipt["official_accuracy"], unparseable=unparseable
        )
    delta = totals["candidate"]["official_correct"] - totals["reference"]["official_correct"]
    return dict(
        complete=True,
        verdict="PASS" if delta >= 0 else "FAIL",
        count=expected,
        comparison=COMPARISON,
        passes_comparison=delta >= 0,
        delta_correct=delta,
        **counts,
        **totals,
        rows=rows,
    )


def compare(args):
    manifest, gate, expected, external = map(read_json, (args.manifest, args.gate, args.prepared, args.run_pair))
    md = sha(args.manifest.read_bytes())
    gd = sha(args.gate.read_bytes())
    require(
        gate["manifest_sha256"] == md
        and gate["prior_gates_sha256"] == manifest["gate_sha256"]
        and gate["comparison"] == COMPARISON,
        "Manifest or comparison is outside the frozen gate",
    )
    require(external["gate_sha256"] == gd, "External RunSpec receipt has another gate")
    check_prepared(expected, args.manifest)
    check_inputs(args.source, args.data_root, manifest)
    sides = spec_identity(external, args.run_pair.parent, md, gate, expected)
    prepared_sha = sha(args.prepared.read_bytes())
    require(
        external["prepared_sha256"] == prepared_sha, "External RunSpec receipt has another prepared template manifest"
    )
    evidence = {}
    for key, side in sides.items():
        verify_tokens(side, expected, prepared_sha)
        evidence[key] = audited(side, manifest, args.manifest, args.source)
    output = join(manifest, expected, evidence["reference"], evidence["candidate"])
    output.update(
        manifest_sha256=md,
        gate_sha256=gd,
        prepared_sha256=prepared_sha,
        run_pair_sha256=sha(args.run_pair.read_bytes()),
        inputs={
            k: dict(
                run_spec_sha256=v["spec_sha256"],
                image_digest=v["image_digest"],
                audit_sha256=evidence[k][2],
                token_receipt_sha256=sha(v["tokens"].read_bytes()),
            )
            for k, v in sides.items()
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "gate", "prepared", "run-pair", "source", "data-root", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    args = p.parse_args()
    try:
        result = compare(args)
    except (OSError, ValueError, TypeError, KeyError, IndexError, ImportError, AttributeError) as exc:
        reason = "Missing " + Path(exc.filename).name if isinstance(exc, FileNotFoundError) else str(exc)
        result = dict(complete=False, verdict="INCONCLUSIVE", errors=[reason])
    save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
