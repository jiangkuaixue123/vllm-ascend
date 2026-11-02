# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load indexed V4.1 tensors into regular layers and Ascend routed experts."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


@dataclass(frozen=True)
class IndexedPart:
    source: str
    destination: str
    projection: str = ""
    expert_id: int = -1


def _owned_experts(owner):
    mapping = owner.expert_map_manager.expert_map
    if mapping is None:
        ids = list(range(owner.global_num_experts))
    else:
        local = mapping[: owner.global_num_experts].cpu().tolist()
        ids = [i for i, offset in enumerate(local) if offset >= 0]
        if sorted(local[i] for i in ids) != list(range(owner.local_num_experts)):
            raise ValueError(f"Invalid V4.1 expert placement for {owner.layer_name}")
    if len(ids) != owner.local_num_experts:
        raise ValueError(f"Unexpected fused or redundant V4.1 experts in {owner.layer_name}")
    return ids


def _list_parts(parameters, modules):
    owners = {key: module for key, module in modules.items() if isinstance(module, RoutedExperts)}
    expert_ids = {key: _owned_experts(owner) for key, owner in owners.items()}
    for destination in parameters:
        path, _, field = destination.rpartition(".")
        owner = owners.get(path)
        packed, _, suffix = field.partition("_")
        if owner is None or packed not in ("w13", "w2") or suffix not in ("weight", "weight_scale", "weight_offset"):
            yield IndexedPart(destination, destination)
            continue
        # The registered owner may itself live in a nested "routed_experts"
        # module. Its layer_name is the checkpoint / ModelSlim prefix.
        if not owner.layer_name.endswith(".ffn.experts"):
            raise ValueError(f"Unexpected V4.1 routed-expert prefix: {owner.layer_name}")
        projections = ("w1", "w3") if packed == "w13" else ("w2",)
        for expert_id in expert_ids[path]:
            for proj in projections:
                source = f"{owner.layer_name}.{expert_id}.{proj}.{suffix}"
                yield IndexedPart(source, destination, proj, expert_id)


def _load_expert_tensor(reader, part, parameter, config):
    # Materialize at most one CPU expert projection or channel vector. The
    # registered RoutedExperts loader owns global-to-local and TP slicing.
    source = reader.get_tensor(part.source)
    is_weight = part.source.endswith(".weight")
    h, i = config.hidden_size, config.moe_intermediate_size
    out_dim, in_dim = (h, i) if part.projection == "w2" else (i, h)
    expected = (out_dim, in_dim if is_weight else 1)
    if tuple(source.shape) != expected:
        raise ValueError(f"Invalid V4.1 expert tensor shape for {part.source}: {source.shape} != {expected}")
    if (is_weight and source.dtype != parameter.dtype) or (not is_weight and not source.is_floating_point()):
        raise ValueError(f"Invalid V4.1 expert tensor dtype for {part.source}: {source.dtype}")
    loaded = parameter.weight_loader(
        param=parameter,
        loaded_weight=source,
        weight_name=part.destination,
        shard_id=part.projection,
        expert_id=part.expert_id,
        return_success=True,
    )
    if loaded is not True:
        raise ValueError(f"Local V4.1 expert refused checkpoint tensor {part.source}")


@torch.no_grad()
def load_indexed_v41_weights(model, weight_map, tp_rank, tp_size, read_parameter_slice):
    """Return loaded parameter names only after every required source part was read."""
    parameters = dict(model.named_parameters())
    modules = dict(model.named_modules())
    by_shard = defaultdict(list)
    for part in _list_parts(parameters, modules):
        if part.source not in weight_map:
            raise KeyError(f"Required V4.1 parameter absent from index: {part.source}")
        by_shard[weight_map[part.source]].append(part)

    root = Path(model.model_path).resolve()
    loaded = set()
    for shard, parts in sorted(by_shard.items()):
        path = (root / shard).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Checkpoint shard escapes model directory")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for part in parts:
                param = parameters[part.destination]
                if part.expert_id >= 0:
                    _load_expert_tensor(reader, part, param, model.config)
                else:
                    module = modules[part.destination.rpartition(".")[0]]
                    read_parameter_slice(reader, part.source, param, module, tp_rank, tp_size)
                loaded.add(part.destination)
        logger.info("V4.1 loaded indexed shard %s (%d local tensors)", shard, len(parts))
    if loaded != set(parameters):
        raise RuntimeError(f"Unloaded V4.1 parameters: {set(parameters) - loaded}")
    return loaded
