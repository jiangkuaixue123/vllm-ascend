# SPDX-License-Identifier: Apache-2.0
"""Sample A5 HBM and the vLLM KV occupancy of each DP rank during an actual request."""

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def hbm_rows(report):
    rows = {}
    device = None
    for line in report.splitlines():
        header = re.match(r"^\|\s*(\d+)\s*\|\s*(?:Ascend|Atlas)\w*\s*\|", line)
        if header:
            device = int(header[1])
            continue
        if device is not None:
            value = re.search(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", line)
            if value:
                rows[device] = {"used_mb": int(value[1]), "total_mb": int(value[2])}
            device = None
    return rows


def kv_rows(report, model):
    rows = {}
    for line in report.splitlines():
        match = re.fullmatch(r"vllm:kv_cache_usage_perc\{([^}]+)\}\s+([^\s]+)", line)
        if match:
            labels = dict(re.findall(r'(\w+)="([^"]*)"', match[1]))
            if labels.get("model_name") == model:
                rank = int(labels["engine"])
                if rank in rows:
                    raise ValueError(f"Duplicate KV occupancy for DP rank {rank}")
                rows[rank] = float(match[2])
    return rows


def collect(args):
    placement = [json.loads((args.run / f"rank-{rank}-placement.json").read_bytes()) for rank in range(args.ranks)]
    if [row["dp_rank"] for row in placement] != list(range(args.ranks)) or len(
        {row["device"] for row in placement}
    ) != args.ranks:
        raise ValueError("Expected one distinct physical device per DP rank")
    peak_hbm, peak_kv = {}, {}
    snapshots = complete = 0
    with args.output.open("x", buffering=1) as writer:
        while True:
            sample = {"start_ns": time.time_ns(), "ranks": {}}
            cards = hbm_rows(subprocess.check_output(["npu-smi", "info"], text=True, timeout=20))
            try:
                with urlopen(args.metrics_url, timeout=5) as response:
                    kv = kv_rows(response.read().decode(), args.model)
            except (URLError, TimeoutError) as exc:
                kv = {}
                sample["metrics_error"] = type(exc).__name__
            for row in placement:
                rank, device = row["dp_rank"], row["device"]
                memory, occupancy = cards.get(device), kv.get(rank)
                if memory is not None:
                    peak_hbm[rank] = max(memory["used_mb"], peak_hbm.get(rank, 0))
                if occupancy is not None:
                    if not 0 <= occupancy <= 1:
                        raise ValueError("KV usage is outside the expected fraction range")
                    peak_kv[rank] = max(occupancy, peak_kv.get(rank, 0.0))
                sample["ranks"][rank] = dict(device=device, hbm=memory, kv_usage_fraction=occupancy)
            complete += int(
                all(x["hbm"] is not None and x["kv_usage_fraction"] is not None for x in sample["ranks"].values())
            )
            snapshots += 1
            sample["end_ns"] = time.time_ns()
            writer.write(json.dumps(sample) + "\n")
            if args.until.exists() or (args.samples and snapshots >= args.samples):
                break
            time.sleep(args.interval)
    with args.summary.open("x") as file:
        json.dump(
            dict(
                snapshots=snapshots,
                all_ranks_complete_snapshots=complete,
                observed_peak_hbm_mb_by_rank=peak_hbm,
                observed_peak_kv_fraction_by_rank=peak_kv,
                interval_seconds=args.interval,
            ),
            file,
            indent=2,
        )
        file.write("\n")
    return complete > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--until", type=Path, required=True, help="Fresh marker created after the request ends")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--samples", type=int, help="Optional collection limit for a bounded preflight")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks < 1 or args.interval <= 0 or (args.samples is not None and args.samples < 1) or args.until.exists():
        parser.error("Use positive observation bounds and a fresh stop marker")
    raise SystemExit(0 if collect(args) else 1)
