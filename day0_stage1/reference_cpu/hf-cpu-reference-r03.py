"""Use meta weight placeholders and file-backed raw shard views for the ordered CPU reference."""

import argparse
import hashlib
import importlib.util
import json
import mmap
import struct
import sys
from pathlib import Path

import torch
from torch import nn

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/"validator"))
import cpu_math
import hf_cpu_reference
from stage1_trace import HERE, digest

module_spec=importlib.util.spec_from_file_location("cpu_ordered", Path(__file__).with_name("hf-cpu-reference-r02.py"))
ordered=importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(ordered)
TYPES={"F32":torch.float32, "BF16":torch.bfloat16,"F8_E8M0":torch.float8_e8m0fnu,"F8_E4M3":torch.float8_e4m3fn,"I8":torch.int8}


def mapped_weights(model, weights):
    manifest=json.loads((HERE/"hf-source.json").read_bytes())
    raw=(weights/"model.safetensors.index.json").read_bytes()
    oid=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if oid!=manifest["source"]["model.safetensors.index.json"]["git_oid"]:
        raise ValueError("source map identity mismatch")
    index=json.loads(raw)["weight_map"]
    mapped={}
    meta={}
    for shard, identity in manifest["original_weights"].items():
        path=weights/shard
        if path.stat().st_size!=identity["bytes"]:
            raise ValueError(shard)
        with path.open("rb") as f:
            size,=struct.unpack("<Q",f.read(8))
            header=json.loads(f.read(size))
            m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY)
            m.madvise(mmap.MADV_RANDOM)
            mapped[shard]=m
            for name,entry in header.items():
                if name=="__metadata__": continue
                if index[name]!=shard or name in meta:raise ValueError(name)
                lo,hi=entry["data_offsets"]
                if not 0<=lo<hi<=len(m)-size-8:raise ValueError(name)
                meta[name]=(8+size+lo,hi-lo,entry["shape"],entry["dtype"])
    if set(meta)!=set(index):raise ValueError("header inventory mismatch")
    used=set()
    def read(name):
        start,length,shape,dtype=meta[name]
        used.add(name)
        return torch.frombuffer(mapped[index[name]],dtype=torch.uint8,count=length,offset=start).view(TYPES[dtype]).reshape(shape)
    count=0
    for name,param in model.named_parameters():
        value=read(name)
        if name.endswith(".attn.wo_a.weight"):
            value=cpu_math.dequant_weight(value,read(name.removesuffix(".weight")+".scale"))
        if param.dtype==torch.float4_e2m1fn_x2:
            if value.dtype!=torch.int8:raise ValueError(name)
            value=value.view(param.dtype)
        if tuple(param.shape)!=tuple(value.shape):raise ValueError(name)
        mod,field=name.rsplit(".",1)
        setattr(model.get_submodule(mod),field,nn.Parameter(value.to(param.dtype),requires_grad=False))
        count+=1
    for sub in model.modules():
        if hasattr(sub,"scale") and isinstance(getattr(sub,"weight",None),nn.Parameter):
            sub.weight.scale=sub.scale
    if any(not n.startswith(("mtp.","vision.","aligner.","image_")) and not n.endswith(".gate.bias_vl") for n in set(index)-used):
        raise ValueError("unconsumed text input")
    print("Bound",count,"parameters using raw mmap views",flush=True)
    return dict(used_source_keys=len(used),original_index_git_oid=oid,original_shards=len(mapped),
                transformation="verified original shard headers, lazy mmap/metadata construction, official parameter dtypes and wo_a QDQ",
                mmap_loader_sha256=digest(Path(__file__)))


def bind(hf):
    sys.path.insert(0,str((hf/"inference").resolve()))
    cpu_math.fp8_gemm=ordered.fp8
    cpu_math.fp4_gemm=ordered.fp4
    sys.modules["kernel"]=cpu_math
    import model
    # These classes contain weights and shape metadata only; genuine KV, RoPE and
    # n-gram state still use the CPU constructors in the original model.py.
    for cls in (model.Linear, model.ParallelEmbedding, model.ParallelEngramEmbedding, model.ParallelHead):
        original=cls.__init__
        def make_meta(self,*args,_init=original,**kwargs):
            with torch.device("meta"):
                _init(self,*args,**kwargs)
        cls.__init__=make_meta
    hf_cpu_reference.load_raw=mapped_weights
    hf_cpu_reference.TraceWriter=ordered.OrderedCPUTrace


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf",type=Path);p.add_argument("weights",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--threads",type=int,default=16)
    args=p.parse_args()
    bind(args.hf)
    hf_cpu_reference.run(args.hf,args.weights,args.output,args.threads)
