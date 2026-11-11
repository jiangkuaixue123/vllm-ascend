# SPDX-License-Identifier: Apache-2.0
"""Summarize live decode replay and actual sampler verification from a service run."""

import argparse
import json
from collections import Counter
from pathlib import Path


def records(path):
    if path.is_file():
        with path.open() as entries:
            for line in entries:
                yield json.loads(line)


def summarize_rank(directory, rank):
    graph_path = directory / f"graph-rank-{rank}.jsonl"
    draft_path = directory / f"dspark-worker-rank-{rank}.jsonl"
    captures = []
    live_widths = Counter()
    live_rows = Counter()
    for event in records(graph_path):
        if event["rank"] != rank or event["api"] != "torch.npu.NPUGraph":
            raise ValueError(f"Unexpected graph receipt on rank {rank}")
        if event["event"] == "capture":
            captures.append(event["width"])
        elif event["event"] == "replay" and event["live"] > 0:
            if event["width"] not in captures or event["live"] != sum(row["tokens"] for row in event["requests"]):
                raise ValueError(f"Replay does not match its capture or request geometry on rank {rank}")
            live_widths[event["width"]] += 1
            live_rows[len(event["requests"])] += 1

    roles = Counter()
    submitted = accepted = rejected_blocks = 0
    requests = set()
    for record in records(draft_path):
        if isinstance(record, dict):
            roles[record["role"]] += 1
            continue
        for row in record:
            proposed = row["proposed"]
            count = row["accepted"]
            if not 0 <= count <= len(proposed) or proposed[:count] != row["emitted"][:count]:
                raise ValueError(f"Draft prefix differs from emitted tokens on rank {rank}")
            if row["rejected"] != len(proposed) - count:
                raise ValueError(f"Invalid draft rejection count on rank {rank}")
            if row["committed_input_count"] != row["scheduled"] - len(proposed) + count:
                raise ValueError(f"Committed inputs disagree with target verification on rank {rank}")
            requests.add(row["req_id"])
            submitted += len(proposed)
            accepted += count
            rejected_blocks += int(row["rejected"] > 0)

    return {
        "rank": rank,
        "graph_receipt_present": graph_path.is_file(),
        "draft_receipt_present": draft_path.is_file(),
        "capture_widths": captures,
        "live_replay_widths": dict(live_widths),
        "live_replay_request_counts": dict(live_rows),
        "draft_worker_roles": dict(roles),
        "requests_seen_by_verifier": len(requests),
        "draft_tokens_scheduled": submitted,
        "draft_tokens_accepted": accepted,
        "draft_tokens_discarded": submitted - accepted,
        "blocks_with_a_rejection": rejected_blocks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing per-rank worker receipts")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error("Expected at least one rank")
    rows = [summarize_rank(args.directory, rank) for rank in range(args.ranks)]
    with args.output.open("x") as output:
        json.dump(rows, output, indent=2)
        output.write("\n")
