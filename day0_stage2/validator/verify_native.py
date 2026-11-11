# SPDX-License-Identifier: Apache-2.0
"""Compare the actual A5 operators with HF text/draft gating, weighted RMSNorm and block-MX rules."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F


def mx_reference(x, w, s, fp4):
    width = x.shape[-1]
    g = x.float().reshape(-1, width // 32, 32)
    power = torch.pow(2.0, torch.ceil(torch.log2(g.abs().amax(-1).clamp_min(1e-4) / 448.0)))
    a = (g / power[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    if fp4:
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
        byte = w.view(torch.uint8).long()
        w = lut[torch.stack((byte & 15, byte >> 4), -1).flatten(-2)]
        s = s.view(torch.float8_e8m0fnu).float()
    else:
        w = w.view(torch.float8_e4m3fn).float()
        s = s.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 0)[: w.shape[0]]
    w = w.reshape(w.shape[0], width // 32, 32)
    result = torch.zeros(a.shape[0], w.shape[0])
    for j in range(a.shape[1]):
        result += F.linear(a[:, j], w[:, j]) * power[:, j, None] * s[:, j]
    return result.bfloat16()


def diff(a, b):
    delta = (a.cpu().float() - b.cpu().float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "unequal": (delta != 0).sum().item(),
        "n": delta.numel(),
    }


def reference_route(value, bias, temperature, top_k):
    scores = F.softplus(value / temperature).sqrt()
    ids = (scores + bias).topk(top_k, -1).indices
    chosen = scores.gather(1, ids)
    return chosen / (chosen.sum(-1, keepdim=True) + 1e-20) * 1.5, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    from vllm_ascend.models.deepseek_v41 import native_ops as ops

    ops.initialize_native_ops()

    torch.npu.set_device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(14305)
    root = args.model
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def read(k):
        with safe_open(root / idx[k], framework="pt", device="cpu") as file:
            return file.get_tensor(k)

    report = {
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_ops_sha256": hashlib.sha256(Path(ops.__file__).read_bytes()).hexdigest(),
        "device": torch.npu.get_device_name(args.device),
        "seed": 14305,
        "gate": [],
        "norm": {},
        "mx": {},
    }
    for pref, k in [("layers.0.ffn.gate", 6), ("mtp.0.ffn.gate", 3)]:
        w, b = read(pref + ".weight"), read(pref + ".bias")
        for temp in (1.0, 0.8):
            logits = F.linear(torch.randn(13, w.shape[1]).bfloat16().float(), w.float())
            update = logits.roll(3, dims=1).contiguous()
            x = logits.npu()
            bias = b.npu()

            actual = ops.hash_route(x, bias, top_k=k, temperature=temp)
            expected = reference_route(logits, b, temp, k)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                capture = ops.hash_route(x, bias, top_k=k, temperature=temp)
            x.copy_(update.npu())
            graph.replay()
            updated = reference_route(update, b, temp, k)
            comparison = {
                "key": pref,
                "temperature": temp,
                "experts": w.shape[0],
                "k": k,
                "eager_ids": diff(actual[1], expected[1]),
                "eager_weights": diff(actual[0], expected[0]),
                "replay_ids": diff(capture[1], updated[1]),
                "replay_weights": diff(capture[0], updated[0]),
                "updated_selections": (expected[1] != updated[1]).sum().item(),
            }
            report["gate"].append(comparison)
    for key in sorted(idx):
        if "norm" not in key or not key.endswith(".weight") or not key.startswith(("norm.", "layers.", "mtp.")):
            continue
        w = read(key)
        x = torch.randn(3, w.numel()).bfloat16()
        x[0].zero_()
        x[1] *= 1.0e-10
        expect = (w.float() * (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1.0e-20))).bfloat16()
        report["norm"][key] = diff(ops.rms_norm(x.npu(), w.npu(), 1.0e-20), expect)
    for pref, fp4 in [("layers.0.attn.wq_a", False), ("layers.0.ffn.experts.0.w1", True)]:
        w, s = read(pref + ".weight"), read(pref + ".scale")
        holder = torch.nn.Module()
        holder.packed_fp4 = fp4
        if fp4:
            w = w.view(torch.uint8)
        holder.weight = torch.nn.Parameter(w.npu(), requires_grad=False)
        holder.scale = torch.nn.Parameter(s.view(torch.uint8).npu(), requires_grad=False)
        native = ops.NativeMXLinear(holder)
        x = torch.randn(3, w.shape[1] * (2 if fp4 else 1)).bfloat16()
        ref = mx_reference(x, w, s, fp4)
        value = native(x.npu())
        report["mx"][pref] = dict(
            logical_shape=[w.shape[0], x.shape[-1]],
            weight_dtype=str(w.dtype),
            scale_dtype=str(s.dtype),
            comparison=diff(value, ref),
        )
    report["operators"] = [
        str(torch.ops._C_ascend.moe_gating_top_k_hash.default._schema),
        str(torch.ops.npu.npu_rms_norm.default._schema),
        str(torch.ops.npu.npu_quant_matmul.default._schema),
    ]
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
