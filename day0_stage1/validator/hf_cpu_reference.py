"""Run the frozen HF Transformer on a CPU, directly binding the official compact text weights."""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

import cpu_math
from freeze_prompts import verify_source
from stage1_trace import HERE, TraceWriter, digest, read_case


def load_raw(model, weights):
    raw = (weights / "model.safetensors.index.json").read_bytes()
    manifest = json.loads((HERE / "hf-source.json").read_bytes())
    git_oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if git_oid != manifest["source"]["model.safetensors.index.json"]["git_oid"]:
        raise ValueError("original weight index does not match pinned HF")
    index = json.loads(raw)["weight_map"]
    used = set()
    with ExitStack() as stack:
        files = {}
        for shard in sorted(set(index.values())):
            path = weights / shard
            if path.stat().st_size != manifest["original_weights"][shard]["bytes"]:
                raise ValueError(f"original shard size differs: {shard}")
            f = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            if any(index[name] != shard for name in f.keys()):
                raise ValueError(f"index/header mismatch: {shard}")
            files[shard] = f
        def read(name):
            value = files[index[name]].get_tensor(name)
            used.add(name)
            return value

        # Keep the original Parameter objects: HF Linear stores a scale-object
        # reference on weight.scale; replacing tensors via assign would break it.
        for name, param in model.named_parameters():
            value = read(name)
            if name.endswith(".attn.wo_a.weight"):
                scale = read(name.removesuffix(".weight") + ".scale")
                value = cpu_math.dequant_weight(value, scale)
            if param.dtype == torch.float4_e2m1fn_x2:
                if value.dtype != torch.int8:
                    raise ValueError(f"expected packed original I8: {name}")
                value = value.view(param.dtype)
            if value.shape != param.shape:
                raise ValueError(f"{name} {value.shape} vs {param.shape}")
            param.data = value.to(param.dtype)
    for name in set(index) - used:
        if not (name.startswith(("vision.", "aligner.", "mtp.", "image_")) or name.endswith(".gate.bias_vl")):
            raise ValueError(f"unconsumed backbone weight {name}")
    return {"used_source_keys":len(used),"original_index_git_oid":git_oid,
            "original_shards":len(files), "transformation":"direct mmap; I8 expert reinterpretation, wo_a QDQ, HF parameter dtypes"}


@torch.inference_mode()
def run(hf, weights, out, threads):
    for name in ("inference/kernel.py", "inference/model.py", "inference/config.json", "inference/engram.py",
                 "inference/vision.py","inference/image_processor.py","tokenizer.json","tokenizer_config.json"):
        verify_source(hf, name)
    sys.path.insert(0, str((hf / "inference").resolve()))
    sys.modules["kernel"] = cpu_math
    from model import ModelArgs, Transformer

    prompts, case = read_case()
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.bfloat16)
    args = ModelArgs(**json.loads((hf / "inference/config.json").read_bytes()))
    args.max_batch_size = 1
    args.max_seq_len = prompts["max_model_len"]
    args.temperature = 0
    args.vision_n_layers = 0
    args.dspark_block_size = 0
    tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=True)
    model = Transformer(args, tokenizer).eval()
    provenance = load_raw(model, weights)
    versions = {p:version(p) for p in ("torch","transformers","numpy","safetensors","tokenizers")}
    env = dict(versions=versions, kernels="CPU PyTorch mathematical reconstruction; BF16 operands / FP32 accumulation",
               cpu_kernel_sha256=digest(HERE/"cpu_math.py"), driver_sha256=digest(HERE/"hf_cpu_reference.py"),
               threads=threads, device="cpu", **provenance)
    writer = TraceWriter(out, backend="official-hf", rank=0, topology={"TP":1,"DP":1,"EP":1}, environment=env)
    writer.attach(model.layers)
    print("CPU reference loaded", provenance, flush=True)
    ids = case["prompt_ids"]
    pos = 0
    for step in range(prompts["generation"]["oracle_max_tokens"]):
        writer.begin(ids, pos)
        chosen, logits, _ = model(torch.tensor([ids], dtype=torch.long), pos)
        token = writer.end(logits, chosen.item())
        print("step",step,"input_position",pos,"next_token",token,flush=True)
        if token == prompts["generation"]["stop_token_id"]:
            break
        pos += len(ids)
        ids = [token]
    writer.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf", type=Path)
    p.add_argument("weights",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--threads",type=int,default=16)
    opt = p.parse_args()
    run(opt.hf, opt.weights, opt.output, opt.threads)
