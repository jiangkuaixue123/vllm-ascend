# SPDX-License-Identifier: Apache-2.0
"""Strict loading of original V4.1 compact safetensors and host Engram tables."""

import gzip
import hashlib
import json
import math
import re
import struct
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_WIDTH = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}
EXPERT = re.compile(r"(?:layers|mtp)\.\d+\.ffn\.experts\.(\d+)\.")
INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class TensorRef:
    file: Path
    offset: int
    shape: tuple
    dtype: str
    nbytes: int


def _unique_keys(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key {name!r}")
        result[name] = value
    return result


def read_checkpoint(checkpoint_dir):
    root = Path(checkpoint_dir)
    index_path = root / INDEX_NAME
    weight_map = None
    indexed_bytes = None
    if index_path.is_file():
        index = json.loads(index_path.read_bytes(), object_pairs_hook=_unique_keys)
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Empty or invalid checkpoint weight_map")
        indexed_bytes = index.get("metadata", {}).get("total_size")
        files = sorted(set(weight_map.values()))
    else:
        files = ["model.safetensors"]
    tensors, header_hashes = {}, {}
    for basename in files:
        if not isinstance(basename, str) or Path(basename).name != basename or not basename.endswith(".safetensors"):
            raise ValueError(f"Invalid shard basename {basename!r}")
        shard = root / basename
        with shard.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"Truncated safetensors length in {basename}")
            size, = struct.unpack("<Q", prefix)
            if not 0 < size <= 100_000_000:
                raise ValueError(f"Invalid safetensors header size in {basename}")
            header_bytes = stream.read(size)
            if len(header_bytes) != size:
                raise ValueError(f"Truncated safetensors header in {basename}")
        header_hashes[basename] = hashlib.sha256(header_bytes).hexdigest()
        header = json.loads(header_bytes, object_pairs_hook=_unique_keys)
        header.pop("__metadata__", None)
        payload = 8 + size
        cursor = 0
        for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"][0]):
            if (weight_map is not None and weight_map.get(name) != basename) or name in tensors:
                raise ValueError(f"Misindexed or duplicate key {name}")
            start, end = entry["data_offsets"]
            shape = tuple(entry["shape"])
            dtype = entry["dtype"]
            if (type(start) is not int or type(end) is not int or start != cursor
                    or not shape or any(type(d) is not int or d < 1 for d in shape)
                    or dtype not in CHECKPOINT_WIDTH or end - start != math.prod(shape) * CHECKPOINT_WIDTH[dtype]):
                raise ValueError(f"Invalid tensor descriptor for {name}")
            tensors[name] = TensorRef(shard, payload + start, shape, dtype, end - start)
            cursor = end
        if shard.stat().st_size != payload + cursor:
            raise ValueError(f"Truncated shard or trailing payload in {basename}")
    if weight_map is not None and weight_map.keys() != tensors.keys():
        raise ValueError("Checkpoint is missing indexed tensors")
    if indexed_bytes is not None and sum(t.nbytes for t in tensors.values()) != indexed_bytes:
        raise ValueError("Indexed payload size mismatch")
    return tensors, header_hashes


def classify(name):
    if name.startswith("mtp."):
        return "dspark"
    if (name.startswith(("vision.", "aligner.", "image_"))
            or name.endswith(".ffn.gate.bias_vl")):
        return "vision"
    if ".engram.embed." in name:
        return "engram_host"
    if EXPERT.match(name):
        return "expert"
    return "replicated"


def expert_owner(name, num_experts, ep_size):
    if num_experts % ep_size:
        raise ValueError("Expert count must divide evenly across EP ranks")
    expert = int(EXPERT.match(name)[1])
    if expert >= num_experts:
        raise ValueError(f"Invalid expert index in {name}")
    return expert // (num_experts // ep_size)


def _validate_expert_headers(model, tensors, count):
    # Check off-rank experts against one local expert per layer. This touches
    # only header metadata: discarded experts never enter CPU or device RAM.
    actual = {k for k in tensors if classify(k) == "expert"}
    for layer_id, block in enumerate(model.layers):
        ffn = block.ffn
        if len(ffn.experts) != count:
            raise ValueError("Inconsistent expert count across text blocks")
        first = ffn.experts[ffn.experts_start_idx]
        params = list(first.named_parameters())
        for expert_id in range(count):
            prefix = f"layers.{layer_id}.ffn.experts.{expert_id}."
            for suffix, param in params:
                name = prefix + suffix
                if name not in actual:
                    raise ValueError(f"Missing routed expert tensor {name}")
                actual.remove(name)
                ref = tensors[name]
                dtype = "F8_E8M0" if suffix.endswith(".scale") else "I8"
                if ref.dtype != dtype or tuple(param.shape) != ref.shape:
                    raise ValueError(f"Invalid {ref.dtype} {ref.shape} for {name}")
    if actual:
        raise ValueError(f"Unexpected routed expert tensors {sorted(actual)[:6]}")


def _field(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _validate_unused_headers(hf_config, tensors):
    """Check the unallocated vision/draft schemas against the checkpoint configuration."""
    unseen = {k for k in tensors if classify(k) in ("vision", "dspark")}

    def expect(name, dtype, *shape):
        ref = tensors.get(name)
        if ref is None or ref.dtype != dtype or ref.shape != shape:
            raise ValueError(f"Expected {name} with {dtype} {shape}")
        unseen.remove(name)

    text = _field(hf_config, "text_config")
    dim = _field(text, "hidden_size")
    vision = _field(hf_config, "checkpoint_vision_config") or _field(hf_config, "vision_config")
    if vision:
        vdim = _field(vision, "hidden_size")
        inter = _field(vision, "intermediate_size")
        patch = _field(vision, "patch_size")
        ratio = _field(vision, "downsample_ratio")
        for name in ("image_start", "image_end", "image_newline", "aligner.w1.bias", "aligner.w2.bias"):
            expect(name, "BF16", dim)
        expect("aligner.w1.weight", "BF16", dim, ratio * ratio * vdim)
        expect("aligner.w2.weight", "BF16", dim, dim)
        expect("vision.patch_embed.proj.weight", "BF16", vdim, 3 * patch * patch)
        expect("vision.patch_embed.proj.bias", "BF16", vdim)
        expect("vision.norm.weight", "BF16", vdim)
        for i in range(_field(vision, "num_hidden_layers")):
            for suffix, shape in {
                "attn.wo.bias": (vdim,), "attn.wo.weight": (vdim, vdim),
                "attn.wqkv.bias": (3 * vdim,), "attn.wqkv.weight": (3 * vdim, vdim),
                "mlp.w1.weight": (2 * inter, vdim), "mlp.w2.weight": (vdim, inter),
                "norm1.weight": (vdim,), "norm2.weight": (vdim,),
            }.items():
                expect(f"vision.blocks.{i}.{suffix}", "BF16", *shape)
        for i in range(_field(text, "num_hidden_layers")):
            expect(f"layers.{i}.ffn.gate.bias_vl", "F32", _field(text, "n_routed_experts"))

    depth = _field(text, "num_nextn_predict_layers", 0)
    if depth:
        experts = _field(text, "dspark_n_routed_experts")
        rank = _field(text, "dspark_markov_rank")
        targets = _field(text, "dspark_target_layer_ids")
        # A draft block shares the target block's checkpoint geometry for these
        # defined leaves. Its expert count and its first/last-layer heads differ.
        common = [
            "attn.attn_sink", "attn.kv_norm.weight", "attn.q_norm.weight", "attn_norm.weight", "ffn_norm.weight",
            *(f"hc_{which}_{part}" for which in ("attn", "ffn") for part in ("base", "fn", "scale")),
            *(f"attn.{proj}.{part}" for proj in ("wkv", "wo_a", "wo_b", "wq_a", "wq_b")
              for part in ("weight", "scale")),
            *(f"ffn.shared_experts.{proj}.{part}" for proj in ("w1", "w2", "w3")
              for part in ("weight", "scale")),
        ]
        for i in range(depth):
            prefix = f"mtp.{i}."
            for suffix in common:
                ref = tensors["layers.0." + suffix]
                expect(prefix + suffix, ref.dtype, *ref.shape)
            expect(prefix + "ffn.gate.weight", "BF16", experts, dim)
            expect(prefix + "ffn.gate.bias", "F32", experts)
            if vision:
                expect(prefix + "ffn.gate.bias_vl", "F32", experts)
            for expert in range(experts):
                for proj in ("w1", "w2", "w3"):
                    for part in ("weight", "scale"):
                        ref = tensors[f"layers.0.ffn.experts.0.{proj}.{part}"]
                        expect(prefix + f"ffn.experts.{expert}.{proj}.{part}", ref.dtype, *ref.shape)
        expect("mtp.0.main_norm.weight", "BF16", dim)
        expect("mtp.0.main_proj.weight", "F8_E4M3", dim, dim * len(targets))
        expect("mtp.0.main_proj.scale", "F8_E8M0", (dim + 31) // 32, (dim * len(targets) + 31) // 32)
        prefix = f"mtp.{depth - 1}."
        expect(prefix + "norm.weight", "BF16", dim)
        expect(prefix + "confidence_head.proj.weight", "BF16", 1, dim + rank)
        for projection in ("embed", "head"):
            expect(prefix + f"markov_head.{projection}.weight", "BF16", _field(text, "vocab_size"), rank)
    if unseen:
        raise ValueError(f"Unexpected vision or draft tensors {sorted(unseen)[:6]}")


def _parameter_contract(name, ref, param):
    import torch

    expected = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
                "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8}[ref.dtype]
    if name.endswith(".scale") and ref.dtype != "F8_E8M0":
        raise ValueError(f"Expected UE8M0 scale at {name}, found {ref.dtype}")
    valid = {expected}
    if ref.dtype in ("F8_E4M3", "F8_E8M0", "I8"):
        valid.add(torch.uint8)
    if ref.dtype == "I8":
        valid.add(torch.float4_e2m1fn_x2)
    if ref.dtype == "BF16" and (
        name == "head.weight" or re.fullmatch(r"layers\.\d+\.attn\.compressor\.(wgate|wkv)\.weight", name)
    ):
        valid.add(torch.float32)
    if param.dtype not in valid or tuple(param.shape) != ref.shape:
        raise ValueError(f"{name}: checkpoint {ref.dtype} {ref.shape}, target {param.dtype} {tuple(param.shape)}")


def _read_rows(ref, stream, start, end):
    import torch

    stride = ref.nbytes // ref.shape[0]
    stream.seek(ref.offset + start * stride)
    buf = bytearray((end - start) * stride)
    if stream.readinto(buf) != len(buf):
        raise EOFError(f"Truncated tensor in {ref.file.name}")
    # Keep exotic formats as raw bytes from their original file until conversion/transfer.
    dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(ref.dtype, torch.uint8)
    return torch.frombuffer(buf, dtype=dtype).view(end - start, *ref.shape[1:])


def _copy_tensor(ref, param, files, wo_scale=None):
    import torch

    row_bytes = ref.nbytes // ref.shape[0]
    rows = max(1, (8 << 20) // row_bytes)
    if wo_scale is not None:
        rows = max(32, (rows // 32) * 32)
    for start in range(0, ref.shape[0], rows):
        end = min(start + rows, ref.shape[0])
        data = _read_rows(ref, files[ref.file], start, end)
        target = param[start:end]
        if wo_scale is not None:
            # Frozen convert.py stores the block-diagonal output projection in BF16.
            scale = _read_rows(wo_scale, files[wo_scale.file], start // 32, end // 32)
            expanded = scale.view(torch.float8_e8m0fnu).float()[:, None, :, None]
            value = data.view(torch.float8_e4m3fn).float().reshape((end - start) // 32, 32, -1, 32)
            target.copy_((value * expanded).reshape_as(target).bfloat16())
        elif ref.dtype in ("F8_E4M3", "F8_E8M0", "I8"):
            target.view(torch.uint8).copy_(data)
        else:
            target.copy_(data)


def load_checkpoint(model, checkpoint_dir, *, ep_rank, device, ep_size=8, manifest=None, hf_config=None):
    """Populate one rank's text backbone. Returns its loaded named-parameter set."""
    import torch
    from .engram import HostEngramEmbedding

    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"Invalid EP rank {ep_rank} of {ep_size}")
    device = torch.device(device)
    before = None
    if manifest is not None and device.type == "npu":
        before = torch.npu.memory_allocated(device)
    tensors, header_hashes = read_checkpoint(checkpoint_dir)
    params = dict(model.named_parameters())
    experts = len(model.layers[0].ffn.experts) if len(model.layers) else 0
    _validate_expert_headers(model, tensors, experts)
    if hf_config is None:
        hf_config = json.loads((Path(checkpoint_dir) / "config.json").read_bytes())
    _validate_unused_headers(hf_config, tensors)
    local, host, converted, actions = {}, {}, {}, {}
    for name, ref in tensors.items():
        family = classify(name)
        action = family
        if family == "expert" and expert_owner(name, experts, ep_size) != ep_rank:
            action = "other_ep_rank"
        elif family == "engram_host":
            prefix, _, leaf = name.rpartition(".")
            host.setdefault(prefix, {})[leaf] = ref
        elif family in ("expert", "replicated"):
            prefix = name.removesuffix(".scale").removesuffix(".weight")
            wo_param = params.get(prefix + ".weight") if prefix.endswith(".attn.wo_a") else None
            if wo_param is not None and wo_param.dtype == torch.bfloat16:
                converted[prefix] = tensors[prefix + ".scale"]
                action = "wo_a_to_bfloat16"
                if name.endswith(".scale"):
                    actions[name] = action
                    continue
            local[name] = ref
        actions[name] = action

    if local.keys() != params.keys():
        missing = sorted(params.keys() - local.keys())[:6]
        unexpected = sorted(local.keys() - params.keys())[:6]
        raise ValueError(f"Checkpoint/target mismatch: missing={missing}, unexpected={unexpected}")
    embeddings = {name: module for name, module in model.named_modules() if isinstance(module, HostEngramEmbedding)}
    if host.keys() != embeddings.keys():
        raise ValueError(f"Engram module paths differ: {sorted(embeddings)}, expected {sorted(host)}")
    for name, pair in host.items():
        if set(pair) != {"weight", "scale"}:
            raise ValueError(f"Incomplete or unexpected Engram table at {name}")
        embeddings[name].validate_layout(pair["weight"], pair["scale"])
    for name, ref in local.items():
        param = params[name]
        if param.device.type != device.type or (device.index is not None and device.index != param.device.index):
            raise ValueError(f"{name}: expected allocated target on {device}, got {param.device}")
        prefix = name.removesuffix(".weight")
        if prefix in converted:
            scale = converted[prefix]
            if (ref.dtype != "F8_E4M3" or scale.dtype != "F8_E8M0" or len(ref.shape) != 2
                    or ref.shape != tuple(param.shape) or ref.shape != tuple(32 * x for x in scale.shape)):
                raise ValueError(f"Invalid wo_a block geometry at {name}")
        else:
            _parameter_contract(name, ref, param)
    # All rank-local shape/dtype/key checks precede the first data copy.
    with torch.no_grad(), ExitStack() as stack:
        files = {file: stack.enter_context(file.open("rb", buffering=0))
                 for file in {r.file for r in (*local.values(), *converted.values())}}
        for name, ref in local.items():
            _copy_tensor(ref, params[name], files, converted.get(name.removesuffix(".weight")))
        for name, pair in host.items():
            embeddings[name].bind(pair["weight"], pair["scale"], device)
    if manifest is not None:
        path = Path(manifest)
        key_path = path.with_name(path.name + ".keys.jsonl.gz")
        with gzip.open(key_path, "wt", encoding="utf-8") as log:
            for name in sorted(actions):
                print(json.dumps([name, actions[name]], separators=(",", ":")), file=log)
        memory = {}
        if before is not None:
            torch.npu.synchronize(device)
            free, total = torch.npu.mem_get_info(device)
            memory = {"before_copy_allocated": before, "after_bind_allocated": torch.npu.memory_allocated(device),
                      "process_peak_allocated": torch.npu.max_memory_allocated(device),
                      "device_free_after_bind": free, "device_total": total}
        index = Path(checkpoint_dir) / INDEX_NAME
        index_digest = hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None
        with path.open("w") as log:
            json.dump({"device_type": device.type, "device_memory_bytes": memory, "ep_rank": ep_rank, "ep_size": ep_size, "index_sha256": index_digest,
                       "shard_header_sha256": header_hashes, "key_actions": dict(Counter(actions.values())),
                       "all_keys": key_path.name, "named_device_parameters": len(params),
                       "parameter_bytes": sum(p.numel() * p.element_size() for p in params.values()),
                       "engram_host_file_bytes": sum(ref.nbytes for pair in host.values() for ref in pair.values())}, log)
    return set(local)
