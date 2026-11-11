"""Replay an independent HF Engram vector against the loaded eager module on one rank."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tokenizers import Tokenizer

from vllm_ascend.models.deepseek_v41 import ngram
from vllm_ascend.models.deepseek_v41.backend import Engram
from vllm_ascend.models.deepseek_v41.engram import HostEngramEmbedding
from vllm_ascend.models.deepseek_v41.weights import _copy_tensor, read_checkpoint


class TokenizerView:
    def __init__(self, path):
        self.backend_tokenizer = Tokenizer.from_file(str(path))

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def compare(actual, expected):
    value = actual.cpu().float()
    ref = torch.from_numpy(expected).float()
    if value.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {value.shape}, {ref.shape}")
    delta = (value - ref).abs()
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite Engram projection or gate")
    earliest = (delta != 0).nonzero()
    first = None
    if len(earliest):
        index = tuple(earliest[0].tolist())
        first = {"index": list(index), "candidate": value[index].item(), "reference": ref[index].item()}
    return {"exact": not len(earliest), "max_abs": float(delta.max()), "different_elements": len(earliest),
            "elements": delta.numel(), "first_difference": first}


@torch.inference_mode()
def probe(args):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "npu":
        import torch_npu
        torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}" if args.device == "npu" else "cpu")
    tensors, _ = read_checkpoint(args.checkpoint)
    settings = SimpleNamespace(**json.loads((args.reference / "config.json").read_text()),
                               max_seq_len=256, max_batch_size=1)
    layout = ngram.EngramLayout.from_args(settings)
    state = ngram.NgramHashState(settings, layout, TokenizerView(args.checkpoint / "tokenizer.json"))
    digest = hashlib.sha256(args.oracle.read_bytes()).hexdigest()
    desc = json.loads(args.oracle.with_suffix(".json").read_text())
    if digest != desc["vector_sha256"]:
        raise ValueError("Independent HF operator vector integrity check failed")
    result = {"scope": "isolated Engram against CPU HF oracle with fixed diagnostic residual",
              "device": str(device), "rank": rank, "torch": str(torch.__version__),
              "npu_package": torch_npu.__version__ if device.type == "npu" else None,
              "oracle_sha256": digest, "layers": {}}
    with np.load(args.oracle, allow_pickle=False) as obs:
        ids = torch.from_numpy(obs["prompt_ids"]).unsqueeze(0)
        expected_hashes = torch.from_numpy(obs["all_hashes"])
        full = state(ids, 0).clone()
        split = torch.cat([state(ids[:, :-2], 0), state(ids[:, -2:-1], ids.size(1)-2),
                           state(ids[:, -1:], ids.size(1)-1)], 1)
        state(ids[:, -3:].flip(1), 0)
        reset = state(ids, 0)
        if not all(torch.equal(actual, expected_hashes) for actual in (full, split, reset)):
            raise ValueError("Engram row IDs differ on prefill, continuation or request reset")
        result["hashes_prefill_decode_reset_match"] = True
        for layer_index, layer in enumerate(layout.layer_ids):
            prefix = f"layers.{layer}.engram."
            embedding = HostEngramEmbedding(layout.num_embeddings[layer_index], layout.head_dim, transfer_rows=65)
            embedding.bind(tensors[prefix + "embed.weight"], tensors[prefix + "embed.scale"], device)
            row_ids = full[:, desc["sample_positions"], layer_index].contiguous()
            key = f"layer{layer}."
            if not np.array_equal(row_ids[0].numpy(), obs[key + "rows"]):
                raise ValueError(f"Wrong layer or row order at layer {layer}")
            if not np.array_equal(embedding._weight[row_ids[0].numpy()], obs[key + "weight_bytes"]):
                raise ValueError(f"FP8 row byte mismatch at layer {layer}")
            if not np.array_equal(embedding._scale[row_ids[0].numpy()], obs[key + "scale_bytes"]):
                raise ValueError(f"E8M0 row byte mismatch at layer {layer}")
            rows = embedding(row_ids)
            if not torch.equal(rows[0].cpu().float(), torch.from_numpy(obs[key + "lookup"])):
                raise ValueError(f"Dequantized embedding differs at layer {layer}")
            # Exercise duplicate IDs, out-of-order restoration and a batch larger
            # than the staging bound on each worker against the same reference.
            repeated = row_ids.flip(-1).repeat_interleave(2, dim=-1)
            before = embedding.transferred_rows
            wanted = torch.from_numpy(obs[key + "lookup"]).flip(1).repeat_interleave(2, dim=1)
            if not torch.equal(embedding(repeated)[0].cpu().float(), wanted):
                raise ValueError(f"Engram row restoration mismatch at layer {layer}")
            if embedding.transferred_rows - before >= repeated.numel():
                raise ValueError("Repeated lookups did not reduce the number of transferred rows")
            report = {"raw_rows_scale_and_bf16_match": True, "repeats_match": True,
                      "sampled_rows": row_ids.numel(), "transfer_rows_bound": embedding.transfer_rows}
            if not args.rows_only:
                previous = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                try:
                    with torch.device(device):
                        block = Engram(settings, layer, layout, lambda *_: embedding)
                finally:
                    torch.set_default_dtype(previous)
                for name, param in block.named_parameters():
                    ref = tensors[prefix + name]
                    with ref.file.open("rb") as f:
                        _copy_tensor(ref, param, {ref.file: f})
                h = torch.from_numpy(obs["residual"]).to(device=device, dtype=torch.bfloat16)
                projections = []
                hook = block.wkv.register_forward_hook(lambda _m, _a, x: projections.append(x))
                output = block(h, row_ids)
                hook.remove()
                projected, = projections
                report["wkv"] = compare(projected[0], obs[key + "wkv"])
                report["gated_output"] = compare(output, obs[key + "output"])
                # Inspect the gate for this actual wkv result, including the
                # per-HC normalization, q/k product and signed sqrt.
                k = projected[..., :settings.hc_mult * settings.dim].float().reshape_as(h)
                x = h.float()
                rstd = (x.square().mean(-1) + settings.norm_eps).rsqrt() * (k.square().mean(-1) + settings.norm_eps).rsqrt()
                dot = (x * (block.q_weight.float() * block.k_weight.float()) * k).sum(-1) * rstd * settings.dim**-0.5
                magnitude = dot.abs().clamp_min(block.clamp_value).sqrt()
                gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
                report["gate_coefficients"] = compare(gate, obs[key + "gate"])
            report.update(lookup_calls=embedding.lookup_calls, requested_rows=embedding.requested_rows,
                          transferred_rows=embedding.transferred_rows,
                          selected_fp8_scale_payload_bytes=embedding.row_payload_bytes,
                          largest_payload_bytes=embedding.largest_payload_bytes)
            result["layers"][layer] = report
    if device.type == "npu":
        torch.npu.synchronize(device)
        result["peak_allocated_device_bytes"] = torch.npu.max_memory_allocated(device)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="Frozen HF inference/ directory")
    p.add_argument("--oracle", type=Path, required=True, help="HF Engram NPZ with its sibling JSON sidecar")
    p.add_argument("--device", required=True, choices=["cpu", "npu"])
    p.add_argument("--rows-only", action="store_true")
    p.add_argument("--output", type=Path, required=True, help="New JSON file; {rank} is the local rank")
    options = p.parse_args()
    torch.set_num_threads(2)
    record = probe(options)
    with Path(str(options.output).replace("{rank}", str(record["rank"]))).open("x") as f:
        json.dump(record, f, indent=2)
