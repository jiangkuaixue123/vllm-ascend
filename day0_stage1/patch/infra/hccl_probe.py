"""Gather per-rank device, memory and collective results inside the target container."""
import datetime
import importlib.metadata as md
import json
import os
from pathlib import Path
import torch
import torch_npu
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(rank)
dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=90))
value = torch.tensor([rank + 1], dtype=torch.int32, device=f"npu:{rank}")
dist.all_reduce(value)
free, total = torch.npu.mem_get_info(rank)
row = dict(rank=rank, local_device=rank, soc=torch.npu.get_device_name(rank),
           available=torch.npu.is_available(), affinity=sorted(os.sched_getaffinity(0)),
           device_count=torch.npu.device_count(), free_bytes=free, total_bytes=total,
           backend=dist.get_backend(), group_size=dist.get_world_size(), sum=value.item(),
           packages={p: md.version(p) for p in ["torch", "torch-npu", "vllm", "vllm-ascend", "transformers"]})
p = Path(os.environ["HCCL_PROBE_RESULTS"])
with (p / f"rank-{rank}.json").open("x") as f:
    json.dump(row, f)
dist.barrier()
dist.destroy_process_group()
assert row["sum"] == 36 and row["group_size"] == row["device_count"] == 8
