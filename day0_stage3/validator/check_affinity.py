# SPDX-License-Identifier: Apache-2.0
"""Read the live Linux masks for the identified A5 workers and their runtime threads."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def cpu_set(mask):
    result = set()
    for interval in mask.split(","):
        first, _, last = interval.strip().partition("-")
        start = int(first)
        end = int(last or first)
        if start < 0 or end < start:
            raise ValueError(f"Invalid CPU interval {interval}")
        result.update(range(start, end + 1))
    return result


def collect(directory, ranks, mode):
    topo = subprocess.check_output(["npu-smi", "info", "-t", "topo"], text=True)
    local = {}
    for line in topo.splitlines():
        match = re.match(r"^\s*NPU(\d+)\s+.*?\s+([\d,\-]+)\s*$", line)
        if match:
            local[int(match[1])] = match[2]
    nodes = {p.parent.name: p.read_text().strip() for p in Path("/sys/devices/system/node").glob("node*/cpulist")}
    init_mask = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/1/status").read_text().splitlines()
        if line.startswith("Cpus_allowed_list:")
    )
    rows = []
    for rank in range(ranks):
        placement = json.loads((directory / f"rank-{rank}-placement.json").read_bytes())
        dev, pid = placement["device"], placement["worker_pid"]
        if placement["dp_rank"] != rank or dev not in local:
            raise ValueError(f"Incomplete A5 placement for DP rank {rank}")
        proc = Path("/proc", str(pid))
        threads = []
        for task in sorted((proc / "task").iterdir()):
            try:
                status = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines() if ":" in line)
            except FileNotFoundError:
                continue
            name = status["Name"].strip()
            main = int(task.name) == pid
            if not main and name not in {"acl_thread", "release_thread"}:
                continue
            if main and not name.startswith("VLLM"):
                raise ValueError(f"Worker PID no longer identifies the expected process on rank {rank}")
            mask = status["Cpus_allowed_list"].strip()
            threads.append(
                dict(
                    role="main" if main else name,
                    effective_mask=mask,
                    within_device_cpus=cpu_set(mask) <= cpu_set(local[dev]),
                )
            )
        roles = {t["role"] for t in threads}
        if "main" not in roles:
            raise ValueError(f"No live main thread for rank {rank}")
        rows.append(
            dict(
                rank=rank,
                device=dev,
                device_local_cpus=local[dev],
                device_numa_nodes=sorted(n for n, cpus in nodes.items() if cpu_set(cpus) & cpu_set(local[dev])),
                threads=threads,
                absent=sorted({"acl_thread", "release_thread"} - roles),
            )
        )
    report = dict(mode=mode, container_init_mask=init_mask, numa=nodes, ranks=rows)
    if mode == "on":
        report["all_observed_threads_local"] = all(t["within_device_cpus"] for r in rows for t in r["threads"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory with per-rank load placement receipts")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--mode", choices=["on", "off"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks <= 0:
        parser.error("At least one rank is required")
    result = collect(args.run, args.ranks, args.mode)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    raise SystemExit(0 if result.get("all_observed_threads_local", True) else 1)
