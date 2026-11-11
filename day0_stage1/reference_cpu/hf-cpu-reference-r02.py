"""Supplement the frozen CPU mathematical oracle with the reference kernel's ordered 32-wide dot products."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validator"))
import cpu_math
import hf_cpu_reference
from stage1_trace import TraceWriter, digest


def grouped(a, sa, w, sw, block):
    # kernel.py does one FP32 dot per K block, then accumulates scaled dot products
    # in forward K order. CPU bmm forms only that small dot, leaving the order
    # of inter-block accumulation explicit.
    assert block == 32
    features = a.shape[-1]
    width = w.shape[0]
    groups = features // block
    raw = cpu_math._LUT[w.view(torch.uint8).int()] if w.dtype == torch.float4_e2m1fn_x2 else w
    w_blocks = raw.reshape(width, groups, block).permute(1,2,0).float()
    a_blocks = a.reshape(-1, groups, block).transpose(0,1).float()
    partials = torch.bmm(a_blocks,w_blocks)
    s_w = sw.float()
    if w.dtype != torch.float4_e2m1fn_x2:
        s_w = s_w.repeat_interleave(block,dim=0)
    partials.mul_(sa.float().reshape(-1,groups).T.unsqueeze(-1))
    partials.mul_(s_w.T.unsqueeze(1))
    total = torch.zeros_like(partials[0])
    for part in partials.unbind(0):
        total += part
    return total.reshape(*a.shape[:-1],width).bfloat16()


def fp8(a, a_s, b, b_s, scale_dtype, block_size):
    return grouped(a, a_s, b, b_s, block_size)


def fp4(a, a_s, b, b_s, scale_dtype, act_block_size):
    return grouped(a, a_s, b, b_s, act_block_size)


class OrderedCPUTrace(TraceWriter):
    def __init__(self, directory, *, backend, rank, topology, environment):
        environment = dict(environment, kernels="CPU PyTorch; FP8 operands/FP4 packed experts, FP32 32-column dot products, ordered FP32 block accumulation",
                           ordered_dot_sha256=digest(Path(__file__)))
        super().__init__(directory,backend=backend,rank=rank,topology=topology,environment=environment)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf", type=Path)
    p.add_argument("weights",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--threads",type=int,default=16)
    args=p.parse_args()
    cpu_math.fp8_gemm=fp8
    cpu_math.fp4_gemm=fp4
    hf_cpu_reference.TraceWriter=OrderedCPUTrace
    hf_cpu_reference.run(args.hf,args.weights,args.output,args.threads)
