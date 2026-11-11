# SPDX-License-Identifier: Apache-2.0
"""Map local checkpoint E2M1/E8M0 expert matrices into the native MoE scheme."""

import torch


@torch.no_grad()
def install_expert_weights(layer, source, *, process=True):
    """Fill the native scheme's compact [expert,N,K/2] and [expert,N,K/32] parameters."""
    start, end = source.experts_start_idx, source.experts_end_idx
    if not 0 <= start < end <= len(source.experts):
        raise ValueError("Invalid checkpoint EP range")
    inter = source.experts[start].w1.weight.shape[0]
    n, dim = end - start, source.dim
    shape = {
        "w13_weight": (n, 2 * inter, dim // 2),
        "w2_weight": (n, dim, inter // 2),
        "w13_weight_scale": (n, 2 * inter, dim // 32),
        "w2_weight_scale": (n, dim, inter // 32),
    }
    for name, required in shape.items():
        value = getattr(layer, name)
        allowed = (torch.uint8, torch.float8_e8m0fnu) if name.endswith("scale") else (torch.uint8,)
        if tuple(value.shape) != required or value.dtype not in allowed:
            raise ValueError(f"Unexpected {name} layout {tuple(value.shape)} {value.dtype}")
    if hasattr(layer, "expert_map") and layer.expert_map is not None:
        # Weight loading happens once outside graphs. Refuse non-contiguous or
        # redundant placements until there is a matching official checkpoint mapping.
        actual = layer.expert_map.to(device="cpu", dtype=torch.int64)
        expected = torch.full((len(source.experts),), -1, dtype=torch.int64, device="cpu")
        expected[start:end] = torch.arange(n, device="cpu")
        if not torch.equal(actual, expected):
            raise ValueError("vLLM EP placement differs from the loaded expert shard")
    for local, global_id in enumerate(range(start, end)):
        expert = source.experts[global_id]
        if expert is None:
            raise ValueError(f"Missing local expert {global_id}")
        for part, name in (("weight", "weight"), ("scale", "weight_scale")):
            w13 = getattr(layer, "w13_" + name)[local].view(torch.uint8)
            w2 = getattr(layer, "w2_" + name)[local].view(torch.uint8)
            # Packed E2M1 nibbles and biased UE8M0 exponents are bytes, not
            # integers to cast numerically (particularly UE8M0 zero and 255).
            w13[:inter].copy_(getattr(expert.w1, part).view(torch.uint8))
            w13[inter:].copy_(getattr(expert.w3, part).view(torch.uint8))
            w2.copy_(getattr(expert.w2, part).view(torch.uint8))
    if process:
        layer.quant_method.process_weights_after_loading(layer)
    return {
        "expert_range": [start, end],
        "input_layouts": {k: list(v) for k, v in shape.items()},
        "native_layouts": {
            k: {"shape": list(getattr(layer, k).shape), "dtype": str(getattr(layer, k).dtype)} for k in shape
        },
    }


@torch.no_grad()
def make_shared_expert(source, quant_config, prefix):
    """Use the installed shared-MLP API so AscendMoERunner owns its stream/reduction."""
    from vllm_ascend.models.deepseek_v4.model import DeepseekV2MLP

    inter, dim = source.w1.weight.shape
    shared = DeepseekV2MLP(
        dim,
        inter,
        "silu",
        swiglu_limit=source.swiglu_limit,
        quant_config=quant_config,
        reduce_results=False,
        prefix=prefix,
    )
    for layer, original in ((shared.gate_up_proj, (source.w1, source.w3)), (shared.down_proj, (source.w2,))):
        for name, dest in layer.named_parameters(recurse=False):
            if name == "weight":
                leaves = [p.weight for p in original]
            elif name == "weight_scale":
                # Each original FP8 scale covers 32 output rows and 32 K values.
                # The generic MXFP8 quant scheme accepts one UE8M0 byte per row/K group.
                leaves = [p.scale.view(torch.uint8).repeat_interleave(32, dim=0)[: p.weight.shape[0]] for p in original]
            else:
                raise ValueError(f"Unexpected shared expert destination: {name}")
            shape = (sum(t.shape[0] for t in leaves), leaves[0].shape[1])
            if dest.shape != shape or dest.element_size() != 1:
                raise ValueError(f"Unsupported shared {name} destination {dest.dtype} {tuple(dest.shape)}")
            for dest_leaf, src in zip(dest.split([t.shape[0] for t in leaves], dim=0), leaves):
                dest_leaf.view(torch.uint8).copy_(src.view(torch.uint8))
        layer.quant_method.process_weights_after_loading(layer)
    return shared
