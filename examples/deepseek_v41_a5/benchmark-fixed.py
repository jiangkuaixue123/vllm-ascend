# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the frozen eight-request throughput workload against the existing service."""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def fetch(base_url, path, body=None):
    request = urllib.request.Request(
        base_url + path,
        data=None if body is None else json_bytes(body),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=600 if body is not None else 15)


def metrics(base_url):
    with fetch(base_url, "/metrics") as response:
        raw = response.read().decode()
    result = {}
    for line in raw.splitlines():
        if line.startswith("vllm:"):
            name = line.split("{", 1)[0].split(" ", 1)[0]
            result[name] = result.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return result


def require_idle(values):
    if values["vllm:num_requests_running"] != 0 or values["vllm:num_requests_waiting"] != 0:
        raise RuntimeError("Service is busy; run this benchmark when no other requests are active")


def runtime_metadata(container, scenario):
    info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
    devices = {item["PathOnHost"] for item in info["HostConfig"]["Devices"]}
    expected = {f"/dev/davinci{i}" for i in scenario["hardware"]["physical_devices"]}
    actual = {path for path in devices if path.removeprefix("/dev/davinci").isdigit()}
    if actual != expected:
        raise RuntimeError(f"Hardware differs from the frozen four-card scenario: {actual}")
    mounts = {item["Destination"]: item["Source"] for item in info["Mounts"]}
    if mounts.get("/models/deepseek-v41") != scenario["hardware"]["checkpoint"]:
        raise RuntimeError("Checkpoint mount differs from the frozen scenario")
    command = subprocess.check_output(["docker", "exec", container, "cat", "/proc/1/cmdline"]).decode().split("\0")
    flags = (
        "--tensor-parallel-size",
        "--quantization",
        "--dtype",
        "--max-model-len",
        "--block-size",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--gpu-memory-utilization",
        "--speculative-config",
        "--compilation-config",
        "--additional-config",
        "--chat-template",
        "--chat-template-content-format",
    )
    settings = {flag: command[command.index(flag) + 1] for flag in flags if flag in command}
    settings["--enable-expert-parallel"] = "--enable-expert-parallel" in command
    settings["--enforce-eager"] = "--enforce-eager" in command
    if settings.get("--tensor-parallel-size") != "4" or not settings["--enable-expert-parallel"]:
        raise RuntimeError("The frozen scenario requires TP4/EP4")
    return {
        "container": container,
        "image_id": info["Image"],
        "settings": settings,
        "labels": {k: v for k, v in info["Config"]["Labels"].items() if k.startswith("source.")},
    }


def check_tokenization(base_url, scenario):
    for sample in [scenario["warmup"], *scenario["samples"]]:
        with fetch(
            base_url,
            "/tokenize",
            {
                "model": scenario["model"],
                "messages": sample["messages"],
                "chat_template_kwargs": scenario["chat_template_kwargs"],
            },
        ) as response:
            tokens = json.load(response)["tokens"]
        digest = hashlib.sha256(json_bytes(tokens)).hexdigest()
        if len(tokens) != sample["prompt_tokens"] or digest != sample["prompt_token_ids_sha256"]:
            raise RuntimeError(f"Tokenizer/template changed the frozen input: {sample['id']}")


def request(base_url, scenario, sample, output_tokens, barrier=None):
    body = {
        key: scenario[key]
        for key in (
            "model",
            "temperature",
            "seed",
            "ignore_eos",
            "stream",
            "chat_template_kwargs",
        )
    }
    body.update(
        messages=sample["messages"],
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        stream_options={"include_usage": True},
        cache_salt=uuid.uuid4().hex,
    )
    if barrier is not None:
        barrier.wait(timeout=30)
    started = time.perf_counter()
    first, usage, finish = None, None, None
    chunks = []
    with fetch(base_url, "/v1/chat/completions", body) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content") or ""
                if content:
                    first = first or time.perf_counter()
                    chunks.append(content)
                finish = choice.get("finish_reason") or finish
    ended = time.perf_counter()
    if not usage or first is None or finish != "length":
        raise RuntimeError(f"Incomplete response for {sample['id']}: {usage}, {finish}")
    if usage["completion_tokens"] != output_tokens or usage["prompt_tokens"] != sample["prompt_tokens"]:
        raise RuntimeError(f"Token counts changed for {sample['id']}: {usage}")
    if usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0:
        raise RuntimeError(f"Cached or missing usage for {sample['id']}; result is not comparable")
    return {
        "id": sample["id"],
        "usage": usage,
        "finish_reason": finish,
        "response_text": "".join(chunks),
        "ttft_seconds": first - started,
        "latency_seconds": ended - started,
        "decode_tokens_per_second": (output_tokens - 1) / (ended - first),
    }


def measure(base_url, scenario):
    require_idle(metrics(base_url))
    warmup = request(base_url, scenario, scenario["warmup"], scenario["warmup"]["output_tokens"])
    before = metrics(base_url)
    require_idle(before)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=scenario["concurrency"]) as pool:
        barrier = threading.Barrier(scenario["concurrency"])
        jobs = [
            pool.submit(request, base_url, scenario, sample, scenario["output_tokens"], barrier)
            for sample in scenario["samples"]
        ]
        samples = [job.result() for job in jobs]
    duration = time.perf_counter() - started
    after = metrics(base_url)
    require_idle(after)
    delta = {k: after[k] - before.get(k, 0) for k in after if k.endswith("_total")}
    total = sum(s["usage"]["completion_tokens"] for s in samples)
    if delta["vllm:request_success_total"] != scenario["requests"] or delta["vllm:generation_tokens_total"] != total:
        raise RuntimeError("Server counts differ from client counts; possible overlapping traffic")
    drafted = delta.get("vllm:spec_decode_num_draft_tokens_total", 0)
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    return {
        "warmup": warmup,
        "samples": samples,
        "concurrency": scenario["concurrency"],
        "requests": len(samples),
        "total_output_tokens": total,
        "wall_seconds": duration,
        "aggregate_output_tokens_per_second": total / duration,
        "mean_latency_seconds": statistics.mean(s["latency_seconds"] for s in samples),
        "mean_ttft_seconds": statistics.mean(s["ttft_seconds"] for s in samples),
        "mean_decode_tokens_per_second": statistics.mean(s["decode_tokens_per_second"] for s in samples),
        "draft_acceptance_ratio": accepted / drafted if drafted else None,
        "metrics_delta": delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--container", default="jojo-v41-d4ac431")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "fixed-throughput"
    raw = (directory / "scenario.json").read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != (directory / "scenario.sha256").read_text().split()[0]:
        raise RuntimeError("Frozen scenario checksum changed")
    scenario = json.loads(raw)
    if args.describe:
        print(json.dumps(scenario, ensure_ascii=False, indent=2))
        return
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    output = (
        args.output
        or Path(__file__).resolve().parents[3] / "logs" / "fixed-throughput" / f"{now:%Y%m%d-%H%M%S-%f}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario_id": scenario["scenario_id"],
        "scenario_sha256": checksum,
        "started_at": now.isoformat(),
        "status": "running",
        "output": str(output),
    }
    # Reserve a unique artifact before generating requests; never replace baseline files.
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    try:
        baseline_path = directory / "baseline.json"
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        if baseline and baseline["scenario_sha256"] != checksum:
            raise RuntimeError("Baseline belongs to a different scenario")
        report["runtime"] = runtime_metadata(args.container, scenario)
        check_tokenization(args.base_url, scenario)
        report.update(measure(args.base_url.rstrip("/"), scenario))
        if baseline:
            previous = baseline["aggregate_output_tokens_per_second"]
            report["baseline_output_tokens_per_second"] = previous
            report["throughput_change_percent"] = (report["aggregate_output_tokens_per_second"] / previous - 1) * 100
            report["runtime_settings_changed"] = report["runtime"]["settings"] != baseline["runtime"]["settings"]
        report["status"] = "valid"
    except Exception as error:
        report.update(status="invalid", error=str(error))
        raise
    finally:
        report["finished_at"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("warmup", "samples", "metrics_delta", "runtime")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
