# SPDX-License-Identifier: Apache-2.0
"""Check every FP32 gate coefficient and neutral combine on real uneven eight-rank HCCL."""

import argparse
import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm_ascend.ascend_forward_context import set_mc2_tokens_capacity
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.quantization.quant_type import QuantType

from vllm_ascend.models.deepseek_v41.weighted_experts import _route_weights
from vllm_ascend.ops.fused_moe import token_dispatcher as td


class AllToAll(td.TokenDispatcherWithAll2AllV):
    # Use torchrun's real HCCL group as the isolated EP group.
    ep_group = property(lambda self: dist.group.WORLD)
    ep_rank = property(lambda self: dist.get_rank())
    ep_size = property(lambda self: dist.get_world_size())


def assignments(experts, topk):
    book = {}
    for rank in range(8):
        for token in range(7 + rank):
            tag = 1 + 16 * rank + token
            for lane in range(topk):
                eid = (37 * tag + 17 * lane) % experts
                weight = torch.tensor(0.07142857 + (tag * 11 + lane * 73) / 5003).item()
                book[tag, eid] = (lane, weight)
    return book


def replay(rank, experts, topk, kind, config):
    book = assignments(experts, topk)
    m = 7 + rank
    # The real ALL2ALL has varying physical token counts. MC2 also tests the
    # standard service envelope: uniform storage, rank-dependent x_active_mask.
    slots = 16 if kind == "mc2" else m
    x = torch.ones((slots, 5120), dtype=torch.bfloat16)
    ids = torch.zeros((slots, topk), dtype=torch.int32)
    ws = torch.zeros((slots, topk), dtype=torch.float32)
    for token in range(m):
        tag = 1 + 16 * rank + token
        x[token, :8] = torch.tensor([1 if tag & (1 << bit) else -1 for bit in range(8)])
        for eid in range(experts):
            if (tag, eid) in book:
                lane, weight = book[tag, eid]
                ids[token, lane] = eid
                ws[token, lane] = weight
    # Padding has valid but ignored IDs, with a token marker that cannot appear in the book.
    x[m:, :8] = -1
    local = experts // 8
    mapping = torch.full((experts,), -1, dtype=torch.int32)
    mapping[rank * local : (rank + 1) * local] = torch.arange(local)
    if kind == "mc2":
        group = SimpleNamespace(device_group=dist.group.WORLD, world_size=8, rank_in_group=rank)
        with patch.object(td, "get_mc2_group", return_value=group):
            dispatcher = td.TokenDispatcherWithMC2(top_k=topk, num_experts=experts)
        dispatcher.global_bs = 0
    else:
        dispatcher = AllToAll(top_k=topk, num_experts=experts, num_local_experts=local)
    intent = MoETokenDispatchInput(
        x.npu(),
        ws.npu(),
        ids.npu(),
        MoeRouterInput(mapping.npu(), 0, (torch.arange(slots) < m).npu() if kind == "mc2" else None, False),
        MoEQuantParams(quant_type=QuantType.W4A8MXFP, mxfp=MoEMxfpParams(act_quant_type=torch.float8_e4m3fn)),
    )
    dispatched = dispatcher.token_dispatch(intent)
    aligned, neutral = _route_weights(dispatched, dispatcher)
    counts = dispatched.group_list.cpu().long()
    if dispatched.group_list_type == 0:
        counts = torch.diff(counts, prepend=torch.zeros(1, dtype=counts.dtype))
    expert_ids = torch.arange(rank * local, (rank + 1) * local).repeat_interleave(counts)
    active = expert_ids.numel()
    encoded = dispatched.hidden_states[:active, :8].contiguous().cpu()
    if encoded.dtype == torch.uint8:
        encoded = encoded.view(torch.float8_e4m3fn)
    tags = ((encoded.float() > 0).int() * (2 ** torch.arange(8))).sum(-1).tolist()
    row_pairs = list(zip(tags, expert_ids.tolist()))
    expected_pairs = [(tag, eid) for tag, eid in book if rank * local <= eid < (rank + 1) * local]
    record = {
        "kind": kind,
        "experts": experts,
        "top_k": topk,
        "active_input_tokens": m,
        "physical_input_tokens": slots,
        "active_recv_rows": active,
        "dispatch_shape": list(dispatched.hidden_states.shape),
        "weight_shape": list(aligned.shape),
        "weight_dtype": str(aligned.dtype),
        "all_expected_rows_present": sorted(row_pairs) == sorted(expected_pairs),
    }
    if not record["all_expected_rows_present"]:
        record["unexpected_pairs"] = [p for p in row_pairs if p not in expected_pairs][:8]
    aligned_cpu = aligned[:active].cpu().float()
    expect = torch.tensor([book.get(key, (-1, float("nan")))[1] for key in row_pairs], dtype=torch.float32)
    record.update(
        weight_neq=int((aligned_cpu != expect).sum()),
        fp32_distinct_from_bf16=int((expect.bfloat16().float() != expect).sum()),
    )
    # A unique channel per routing lane verifies both the reverse transport and
    # the absence of a second application of the router coefficient.
    payload = torch.zeros(dispatched.hidden_states.shape, dtype=torch.bfloat16)
    for row, key in enumerate(row_pairs):
        if key in book:
            payload[row, book[key][0]] = aligned_cpu[row].bfloat16()
    result = dispatcher.token_combine(payload.npu(), neutral).cpu().float()
    expected_output = ws.bfloat16().float()
    record["combine_neq"] = int((result[:m, :topk] != expected_output[:m]).sum())
    record["combine_n"] = m * topk
    if kind == "mc2":
        record["native_expand_scales_shape"] = list(dispatched.combine_metadata.expand_scales.shape)
        record["dispatch_comm_alg"] = dispatcher.mc2_comm_alg
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=["all2all", "mc2", "both"], default="both")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=65))
    assert dist.get_world_size() == 8
    result = {"rank": rank, "hccl_expansion": os.getenv("HCCL_OP_EXPANSION_MODE"), "cases": []}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"rank-{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    try:
        config = EngineArgs(
            model=str(args.model),
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=128,
            max_num_seqs=16,
            enable_prefix_caching=False,
        ).create_engine_config()
        with set_current_vllm_config(config):
            set_mc2_tokens_capacity(config, 16, 1)
            for kind in ["all2all", "mc2"] if args.kind == "both" else [args.kind]:
                for n, k in ((384, 6), (128, 3)):
                    result["cases"].append(replay(rank, n, k, kind, config))
                    path.write_text(json.dumps(result, indent=2) + "\n")
                    dist.barrier()
        for case in result["cases"]:
            if (
                not case["all_expected_rows_present"]
                or case["weight_neq"]
                or case["combine_neq"]
                or case["weight_dtype"] != "torch.float32"
            ):
                raise ValueError("Routed coefficients or reverse combine differ from the FP32 source")
    except Exception as err:
        result["error"] = f"{type(err).__name__}: {str(err).splitlines()[0]}"
        raise
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
