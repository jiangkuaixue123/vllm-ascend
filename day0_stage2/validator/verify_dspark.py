# SPDX-License-Identifier: Apache-2.0
"""Load the compact original draft and compare its predictions against frozen HF CPU outputs."""

import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn

from vllm_ascend.models.deepseek_v41 import weights
from vllm_ascend.models.deepseek_v41.backend import ParallelHead
from vllm_ascend.models.deepseek_v41.config import DeepseekV41Config
from vllm_ascend.models.deepseek_v41.dspark import DSparkDraft


@torch.inference_mode()
def run(checkpoint, oracle, output, device, ticks, threads, native):
    device = torch.device(device)
    torch.set_num_threads(threads)
    if device.type == "npu":

        torch.npu.set_device(device)
    output.mkdir(parents=True, exist_ok=False)
    settings = DeepseekV41Config.from_pretrained(checkpoint, local_files_only=True).text_config
    target = nn.Module()
    with torch.device(device):
        target.embed = nn.Embedding(
            settings.vocab_size,
            settings.hidden_size,
            _weight=torch.empty(settings.vocab_size, settings.hidden_size, dtype=torch.bfloat16),
        )
        target.head = ParallelHead(settings.vocab_size, settings.hidden_size)
    refs, _ = weights.read_checkpoint(checkpoint)
    with ExitStack() as stack:
        files = {
            ref.file: stack.enter_context(ref.file.open("rb", buffering=0))
            for ref in (refs["embed.weight"], refs["head.weight"])
        }
        for name, p in target.named_parameters():
            weights._copy_tensor(refs[name], p, files)
    draft = DSparkDraft(settings, target, device=device, max_requests=2, max_model_len=128)
    result = draft.load_weights(checkpoint, receipt=output / "weights.json")
    print("loaded", result["tensors"], result["parameter_bytes"], flush=True)
    if native:
        from vllm_ascend.models.deepseek_v41.native_ops import replace_native_leaves

        sites = replace_native_leaves(draft)
        (output / "native.json").write_text(json.dumps(sites, indent=2) + "\n")
    events = [json.loads(s) for s in (oracle / "steps.jsonl").read_text().splitlines()]
    replay = torch.load(oracle / "target-aux.pt", map_location="cpu", weights_only=True)
    rows = []
    for tick, (event, h) in enumerate(zip(events, replay["auxiliaries"])):
        if tick >= ticks:
            break
        pos = event["position"]
        n = h.shape[0]
        positions = torch.arange(pos, pos + n, device=device)
        slots = torch.zeros(n, dtype=torch.long, device=device)
        draft.commit_target(h.to(device), slots, positions)
        if not tick:
            continue
        t0 = time.monotonic()
        answer = draft.propose(torch.tensor([event["target"]], device=device), slots[-1:], positions[-1:])
        predictions = answer.token_ids[0].tolist()
        conf = answer.confidence[0].float().cpu()
        if not torch.isfinite(answer.logits).all() or not torch.isfinite(conf).all():
            raise ValueError("non-finite draft")
        if answer.logits[0].argmax(-1).tolist() != predictions:
            raise ValueError("draft IDs differ from Markov-adjusted logits")
        row = dict(
            tick=tick,
            target_position=pos,
            seed=event["target"],
            predictions=predictions,
            reference=event["proposals"],
            matches=sum(a == b for a, b in zip(predictions, event["proposals"])),
            confidence=conf.tolist(),
            ref_confidence=event["confidence"],
            max_abs_confidence=float((conf - torch.tensor(event["confidence"])).abs().max()),
            seconds=time.monotonic() - t0,
        )
        rows.append(row)
        with (output / "steps.jsonl").open("a") as f:
            print(json.dumps(row), file=f)
        print("candidate", device, tick, row["matches"], row["max_abs_confidence"], flush=True)
    check = dict(
        device_type=device.type,
        native_leaves=native,
        proposed=sum(len(r["predictions"]) for r in rows),
        matched=sum(r["matches"] for r in rows),
        max_abs_confidence=max(r["max_abs_confidence"] for r in rows),
        real_shared_target_head=True,
        raw_mtp_tensors=result["tensors"],
        reference_sha256=hashlib.sha256((oracle / "target-aux.pt").read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "summary.json").write_text(json.dumps(check, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "oracle", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ticks", type=int, default=12)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--native", action="store_true")
    run(**vars(p.parse_args()))
