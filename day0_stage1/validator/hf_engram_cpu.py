"""Independent CPU Engram vectors from the pinned HF hash and gate, plus real checkpoint rows."""

import argparse
import ast
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from freeze_prompts import verify_source
from stage1_trace import digest, dump_json, read_case


def official_gate(hf):
    # Load only the upstream Engram.forward expression. This standalone CPU probe does
    # not import unrelated model backends or allocate a 384M-row parameter.
    path = hf / "inference/model.py"
    module = ast.parse(path.read_text())
    cls, = [c for c in module.body if isinstance(c, ast.ClassDef) and c.name == "Engram"]
    forward, = [f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "forward"]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), path.name, "exec"), scope)
    return scope["forward"]


def fp8_projection(values, weight, scales):
    # kernel.py act_quant and fp8_gemm_kernel use FP32 products for a 32x32
    # tile and apply both scales before adding to the running accumulator.
    block = 32
    x = values.flatten(-2).float().reshape(len(values), -1, block)
    amax = x.abs().amax(-1).clamp_min(1e-4)
    sx = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (x / sx.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    weights = weight.reshape(weight.shape[0], -1, block)
    sw = scales.float().repeat_interleave(block, dim=0)
    acc = torch.zeros(len(x), weight.shape[0])
    for k in range(x.shape[1]):
        product = q[:, k] @ weights[:, k].float().T
        acc.add_(product * sx[:, k, None] * sw[None, :, k])
    return acc.bfloat16()


def lookup(file, name, ids):
    sl = file.get_slice(name)
    return torch.stack([sl[int(i) : int(i) + 1][0] for i in ids.flatten()]).view(*ids.shape, -1)


@torch.inference_mode()
def run(hf, weights, out):
    from engram import EngramLayout, NgramHashState

    prompts, case = read_case()
    settings = json.loads((hf / "inference/config.json").read_bytes())
    settings.update(max_batch_size=1, max_seq_len=prompts["max_model_len"])
    args = SimpleNamespace(**settings)
    layout = EngramLayout.from_args(args)
    tk = AutoTokenizer.from_pretrained(hf, local_files_only=True)
    state = NgramHashState(args, layout, tk)
    ids = torch.tensor([case["prompt_ids"]])
    all_hashes = state(ids, 0).clone()
    n = len(case["prompt_ids"])
    # Supply the last two known prompt tokens one at a time after an independent
    # prefill. Compare the official cache to its complete-sequence forward.
    split = torch.cat([state(ids[:, :n-2], 0), state(ids[:, n-2:n-1], n-2), state(ids[:, -1:], n-1)], dim=1)
    if not torch.equal(all_hashes, split):
        raise ValueError("official full-sequence and incremental hashes differ")

    index_path = weights / "model.safetensors.index.json"
    index_raw = index_path.read_bytes()
    index_info = json.loads((Path(__file__).parent / "hf-source.json").read_bytes())["source"][index_path.name]
    git_oid = hashlib.sha1(b"blob " + str(len(index_raw)).encode() + b"\0" + index_raw).hexdigest()
    if git_oid != index_info["git_oid"]:
        raise ValueError("checkpoint weight map differs from the frozen source")
    index = json.loads(index_raw)["weight_map"]
    gate_fn = official_gate(hf)
    # Deterministic independent residual input for isolating the gate; it is not a
    # hidden state from an actual model prefill.
    residual = ((torch.arange(3 * args.hc_mult * args.dim) % 257 - 128).float() / 64)
    residual = residual.reshape(1, 3, args.hc_mult, args.dim).bfloat16()
    arrays = {
        "prompt_ids": np.array(case["prompt_ids"]),
        "all_hashes": all_hashes.numpy(),
        "residual": residual.float().numpy(),
    }
    layers = []
    for j, layer in enumerate(args.engram_layer_ids):
        prefix = f"layers.{layer}.engram."
        names = [prefix + suffix for suffix in ("embed.weight", "embed.scale", "wkv.weight", "wkv.scale",
                                                 "q_weight", "k_weight")]
        shard, = {index[name] for name in names}
        rows = all_hashes[0, -3:, j, :]
        with safe_open(weights / shard, framework="pt", device="cpu") as f:
            raw = lookup(f, names[0], rows)
            scale = lookup(f, names[1], rows)
            values = (raw.float().unflatten(-1, (-1, 32)) * scale.float().unsqueeze(-1)).flatten(-2).bfloat16()
            kv = fp8_projection(values, f.get_tensor(names[2]), f.get_tensor(names[3]))
            q, k = f.get_tensor(names[4]), f.get_tensor(names[5])
        # Run the unmodified official CPU-capable gate with the real projection,
        # then separately record the gate coefficients for first-divergence checks.
        obj = SimpleNamespace(embed=lambda _: values.unsqueeze(0), wkv=lambda _: kv.unsqueeze(0),
                              hc_mult=args.hc_mult, dim=args.dim, eps=args.norm_eps,
                              q_weight=q, k_weight=k, clamp_value=1e-6)
        out_tensor = gate_fn(obj, residual, rows.unsqueeze(0))
        key = kv[:, :args.hc_mult * args.dim].float().reshape_as(residual)
        h = residual.float()
        rstd = torch.rsqrt(h.square().mean(-1) + args.norm_eps) * torch.rsqrt(key.square().mean(-1) + args.norm_eps)
        dot = (h * (q.float() * k.float()) * key).sum(-1) * rstd * args.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(obj.clamp_value).sqrt(), dot))
        expected = (h + gate.unsqueeze(-1) * kv[:, -args.dim:].float()[None, :, None]).bfloat16()
        if not torch.equal(expected, out_tensor) or not torch.isfinite(out_tensor).all():
            raise ValueError("recovered gate differs from the upstream forward")
        tensors = dict(rows=rows, weight_bytes=raw.view(torch.uint8), scale_bytes=scale.view(torch.uint8),
                       lookup=values.float(), wkv=kv.float(), gate=gate, output=out_tensor.float())
        arrays.update({f"layer{layer}.{name}": t.numpy() for name, t in tensors.items()})
        layers.append(dict(layer=layer, shard=shard, shape=list(rows.shape), unique_rows=rows.unique().numel(),
                           exact_full_vs_incremental_hash=True, exact_reconstructed_gate_output=True))
    with out.open("xb") as f:
        np.savez_compressed(f, **arrays)
    dump_json(out.with_suffix(".json"), dict(
        scope="CPU operator reference; fixed diagnostic residual and teacher-forced tokens",
        model_revision=prompts["revision"], checkpoint_index_git_oid=git_oid, checkpoint_data="sampled rows and projections",
        backend="CPU PyTorch, HF hash and gate with explicit FP8 grouped projection", input_case=case["id"],
        sample_positions=list(range(n-3,n)), layers=layers, vector_sha256=digest(out),
        versions={n:version(n) for n in ("torch","safetensors","transformers","tokenizers","numpy")},
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf", type=Path)
    p.add_argument("weights", type=Path)
    p.add_argument("output", type=Path)
    opt = p.parse_args()
    for name in ("inference/model.py","inference/engram.py","inference/kernel.py","inference/config.json",
                 "tokenizer.json","tokenizer_config.json"):
        verify_source(opt.hf, name)
    sys.path.insert(0, str((opt.hf / "inference").resolve()))
    torch.set_num_threads(4)
    run(opt.hf, opt.weights, opt.output)
