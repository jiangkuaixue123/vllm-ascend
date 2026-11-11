"""Compare serial eager traces through their first unequal generated token, retaining numerical evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage1_trace import HERE, digest, dump_json, expected_taps, read_case


def load_trace(directory, backend):
    prompts, case = read_case()
    env = json.loads((directory / "environment.json").read_text())
    if (env["schema"], env["backend"], env["model_revision"], env["source_manifest"], env["prompts_sha256"],
        env["probe"]) != (1, backend, prompts["revision"], digest(HERE / "hf-source.json"),
                          digest(HERE / "prompts.json"), case["id"]):
        raise ValueError(f"{backend} provenance mismatch")
    if not isinstance(env["rank"], int) or env["rank"] < 0 or not env["topology"] or not env["environment"]:
        raise ValueError(f"{backend} missing rank, topology or environment")
    steps = json.loads((directory / "trace.json").read_text())["steps"]
    budget = prompts["generation"]["oracle_max_tokens"]
    eos = prompts["generation"]["stop_token_id"]
    if not 1 <= len(steps) <= budget:
        raise ValueError(f"{backend} has wrong number of steps")
    pos = 0
    ids = case["prompt_ids"]
    for i, step in enumerate(steps):
        file = f"step-{i:02d}.npz"
        if (step["step"], step["start_pos"], step["input_ids"], step["file"]) != (i, pos, ids, file):
            raise ValueError(f"{backend} broken input/position chain at {i}")
        if step["sha256"] != digest(directory / file):
            raise ValueError(f"{backend} trace digest mismatch at {i}")
        if step["token"] == eos and i != len(steps)-1:
            raise ValueError("steps after EOS")
        pos += len(ids)
        ids = [step["token"]]
    if steps[-1]["token"] != eos and len(steps) != budget:
        raise ValueError(f"{backend} missing decode steps")
    return env, steps


def check_array(name, arr, step):
    length = len(step["input_ids"])
    if name == "last_logits":
        shape = (129280,)
    elif name.endswith("pre_mix"):
        shape = (length, 4)
    else:
        shape = (length, 4, 5120)
    if arr.shape != shape or arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise ValueError(f"bad type/shape/nonfinite in step {step['step']} tensor {name}")
    if name == "last_logits":
        top = np.argsort(-arr, kind="stable")[:16].tolist()
        if step["top16"] != top or step["token"] != top[0] or step["top16_values"] != arr[top].tolist():
            raise ValueError("recorded tokens/top16 disagree with recorded logits")


def compare(reference, candidate, atol, rtol):
    ref_env, refs = load_trace(reference, "official-hf")
    cand_env, cands = load_trace(candidate, "a5-eager")
    first_numeric = first_token = None
    rows = []
    keys = expected_taps() + ["last_logits"]
    use_tolerance = atol is not None
    for a, b in zip(refs, cands):
        if a["start_pos"] != b["start_pos"] or a["input_ids"] != b["input_ids"]:
            raise ValueError("traces advanced on different inputs before first divergent token")
        i = a["step"]
        delta_rows = []
        first_in_step = None
        with np.load(reference / a["file"], allow_pickle=False) as left, np.load(candidate / b["file"], allow_pickle=False) as right:
            if set(left.files) != set(keys) or set(right.files) != set(keys):
                raise ValueError(f"missing/unknown observations on step {i}")
            for order, name in enumerate(keys):
                x, y = left[name], right[name]
                check_array(name, x, a)
                check_array(name, y, b)
                delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                bad = delta > (atol + rtol * np.abs(x.astype(np.float64))) if use_tolerance else x != y
                flat = np.flatnonzero(bad)
                delta_rows.append(dict(tensor=name, max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                                       differing=int(flat.size)))
                if flat.size:
                    idx = tuple(int(k) for k in np.unravel_index(flat[0], x.shape))
                    pos = a["start_pos"] + (len(a["input_ids"])-1 if name == "last_logits" else idx[0])
                    if first_in_step is None or (pos, order) < first_in_step[0]:
                        first_in_step = ((pos, order), dict(step=i, token_position=pos, tensor=name, index=idx,
                                              hf_rank=ref_env["rank"], candidate_rank=cand_env["rank"],
                                              reference=float(x[idx]), candidate=float(y[idx]),
                                              reference_artifact=a["file"], candidate_artifact=b["file"]))
        if first_numeric is None and first_in_step:
            first_numeric = first_in_step[1]
        rows.append(dict(step=i, top16_same=a["top16"] == b["top16"], tensors=delta_rows))
        if a["token"] != b["token"]:
            first_token = dict(step=i, token_position=a["start_pos"]+len(a["input_ids"]),
                               reference=a["token"], candidate=b["token"],
                               hf_rank=ref_env["rank"], candidate_rank=cand_env["rank"],
                               reference_top16=a["top16"], candidate_top16=b["top16"])
            break
    if first_token or (use_tolerance and first_numeric):
        verdict = "FAIL"
    elif first_numeric or len(rows) < 3:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"
    return dict(scope="independent forward numerical comparison only", verdict=verdict, tolerance=dict(atol=atol,rtol=rtol),
                numerical_policy="frozen tolerance" if use_tolerance else "exact differences reported; no approximate gate supplied",
                reference_environment=ref_env, candidate_environment=cand_env,
                first_numeric_difference=first_numeric, first_generated_token_difference=first_token, observations=rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atol", type=float)
    p.add_argument("--rtol", type=float)
    args = p.parse_args()
    if (args.atol is None) != (args.rtol is None) or any(
        not np.isfinite(x) or x < 0 for x in (args.atol,args.rtol) if x is not None
    ):
        p.error("supply both finite nonnegative tolerances from the frozen plan or omit both")
    try:
        result = compare(args.reference, args.candidate, args.atol, args.rtol)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
        result = dict(scope="trace validation", verdict="INCONCLUSIVE", error=type(exc).__name__, detail=str(exc))
    dump_json(args.output, result)
    print(result["verdict"])
    raise SystemExit(0 if result["verdict"] == "PASS" else 1)
