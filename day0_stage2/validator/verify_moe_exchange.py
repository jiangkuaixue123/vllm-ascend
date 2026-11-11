# SPDX-License-Identifier: Apache-2.0
"""AIV eight-rank replay with the installed dispatcher and real dynamic MX scales."""

import argparse
import datetime
import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from vllm_ascend.ops.fused_moe.moe_utils import async_all_to_all
from vllm_ascend.ops.fused_moe.token_dispatcher import DeviceOperator, TokenDispatcherWithAll2AllV


class Dispatch(TokenDispatcherWithAll2AllV):
    """Use the real eight-rank HCCL default group as the isolated EP group."""

    ep_group = property(lambda self: dist.group.WORLD)
    ep_rank = property(lambda self: dist.get_rank())
    ep_size = property(lambda self: dist.get_world_size())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=70))
    size = dist.get_world_size()
    assert size == 8
    out = {"rank": rank, "size": size, "aiv": os.getenv("HCCL_OP_EXPANSION_MODE"), "cases": []}
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        from vllm.config import VllmConfig, set_current_vllm_config

        config = VllmConfig()
        for m, e, k, handler_e, handler_k in (
            (7 + rank, 384, 6, 384, 6),
            (31, 128, 3, 128, 3),
            (7 + rank, 384, 6, 128, 3),
            (128, 384, 6, 384, 6),
        ):
            dispatcher = Dispatch(top_k=handler_k, num_experts=handler_e, num_local_experts=handler_e // size)
            g = torch.Generator().manual_seed(1800 + rank)
            cpu_ids = torch.rand(m, e, generator=g).topk(k, -1).indices.int()
            x = torch.randn(m, 5120, generator=g).bfloat16().npu()
            ids = cpu_ids.npu()
            pre = dispatcher._dispatch_preprocess(x, ids)
            sorted_x, rev, counts, ins, outs, local_ids, shape0, shape1 = pre
            raw_ins, raw_outs = ins.copy(), outs.copy()
            with set_current_vllm_config(config):
                q, scale = DeviceOperator.npu_dynamic_quant(
                    sorted_x, act_quant_type=torch.float8_e4m3fn, use_mxfp_quant=True
                )
            expected = torch.bincount(cpu_ids.flatten() // (e // size), minlength=size).tolist()
            rec = {
                "tokens": m,
                "k": k,
                "experts": e,
                "dispatcher_experts": handler_e,
                "dispatcher_k": handler_k,
                "hidden": list(q.shape),
                "scale": list(scale.shape),
                "scale_stride": list(scale.stride()),
                "raw_input_splits": raw_ins.tolist(),
                "raw_output_splits": raw_outs.tolist(),
                "input_splits": ins.tolist(),
                "output_splits": outs.tolist(),
                "expected_input_splits": expected,
            }
            out["cases"].append(rec)
            (args.out / f"rank-{rank}.json").write_text(json.dumps(out, indent=2))
            bad = torch.tensor([int(sum(ins) != scale.shape[0])], dtype=torch.int32, device="npu")
            dist.all_reduce(bad)
            if bad.item():
                rec["input_count_bad_on_n_ranks"] = bad.item()
                if bad.item() == size:
                    try:
                        _, recv, h = async_all_to_all(scale, outs, ins, dist.group.WORLD)
                        h.wait()
                    except Exception as err:
                        rec["original_error"] = str(err).splitlines()[0]
                continue
            for layout in ("original", "rows"):
                y = scale if layout == "original" else scale.reshape(scale.shape[0], -1)
                try:
                    _, gather, w = async_all_to_all(y, outs, ins, dist.group.WORLD)
                    w.wait()
                    rec[layout + "_shape"] = list(gather.shape)
                    rec[layout + "_status"] = "ok"
                except Exception as err:
                    rec[layout + "_error"] = str(err).splitlines()[0]
            try:
                _, grouped, h = async_all_to_all(q, outs, ins, dist.group.WORLD)
                h.wait()
                recv_scale = gather.reshape(grouped.shape[0], *scale.shape[1:])
                post = dispatcher._dispatch_postprocess(grouped, recv_scale, local_ids, True, torch.float8_e4m3fn)
                rec["post_shapes"] = [list(v.shape) if v is not None else None for v in post]
            except Exception as err:
                rec["post_error"] = str(err).splitlines()[0]
            dist.barrier()
    except Exception as err:
        out["error"] = type(err).__name__ + ": " + str(err).splitlines()[0]
        out["functions"] = [frame.name for frame in traceback.extract_tb(err.__traceback__)]
    finally:
        (args.out / f"rank-{rank}.json").write_text(json.dumps(out, indent=2))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
