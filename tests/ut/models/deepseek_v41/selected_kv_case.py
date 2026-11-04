# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real decode entry with independent cache fields and controlled top-k IDs."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.models.deepseek_v41.native_attention import DeepseekV41NativeAttention


def selected_kv_case(device, ratio, *, strided_columns=False):
    device = torch.device(device)
    rows, heads, dim, window, extra = 6, 16, 512, 128, 32
    generator = torch.Generator().manual_seed(718)

    def metadata(max_len, block):
        columns = (max_len + block - 1) // block
        return attention.DeepseekV41DecodeMetadata(
            torch.zeros(rows, dtype=torch.long, device=device),
            torch.arange(rows, device=device),
            torch.zeros(rows, columns * 2 + 3, dtype=torch.int32, device=device)[:, 3::2],
            torch.zeros(rows, dtype=torch.bool, device=device),
            max_len,
            6,
        )

    def cache(block, expected_offset, dtype, step):
        stride = dim * step + 11
        storage = torch.randn((352, stride), generator=generator).to(dtype).to(device)
        logical = storage[8:, 5 : 5 + dim * step : step]
        obj = attention.DeepseekV41Cache.__new__(attention.DeepseekV41Cache)
        nn.Module.__init__(obj)
        obj.block_size = block

        def field(bank, offset, width):
            assert (bank, offset, width) == (0, expected_offset, dim)
            return logical

        obj.field = field
        return obj

    raw = cache(16, 0, torch.bfloat16, 2 if strided_columns else 1)
    source = cache(32, dim, torch.float32 if ratio == 2 else torch.bfloat16, 1)
    dst_md, src_md = metadata(256, raw.block_size), metadata(224, source.block_size)
    topk = torch.zeros(rows, 2 * extra + 1, dtype=torch.int64, device=device)[:, 1::2]
    runtime = SimpleNamespace(source_cache=source, source_decode=src_md, decode_topk=topk, decode=dst_md)
    layer = attention.DeepseekV41Attention.__new__(attention.DeepseekV41Attention)
    nn.Module.__init__(layer)
    layer.dim, layer.heads, layer.ratio, layer.owns_kv = dim, heads, ratio, False
    layer.config = SimpleNamespace(sliding_window=window)
    layer.cache, layer.sequence_group, layer.dsa_cp = raw, None, False
    raw.metadata = lambda: SimpleNamespace(decode=dst_md)
    layer._sequence_inputs = lambda x, mode: (x, 0)
    layer.wq_a = layer.wkv = lambda x: (x,)
    layer.q_norm = layer.kv_norm = nn.Identity()
    layer.wq_b = lambda qr: (qr.repeat(1, heads),)
    layer.rotate = lambda *args, **kwargs: None
    layer._decode_compressed = lambda *args: None
    layer._decode_index = lambda *args: None
    layer._project_output = lambda attended, dtype: attended[:, 0]
    layer.attn_sink = torch.linspace(-1, 1, heads, device=device)
    x = torch.randn((rows, dim), generator=generator).bfloat16().to(device)
    native = DeepseekV41NativeAttention()
    if device.type == "npu":
        native.pin_decode_plan(x.device, heads, rows, window + (extra if ratio else 0))
    received = []

    def attend(q, kv, sink, ids, scale, **kwargs):
        assert kwargs["uniform_decode"]
        received.append((kv, ids))
        return native(q, kv, sink, ids, scale, **kwargs)

    layer.attention_kernel = attend

    def inputs(step):
        lengths = torch.tensor([1, 5, 127, 128, 129, 221]).roll(step)
        for i, (md, block) in enumerate(((dst_md, raw.block_size), (src_md, source.block_size))):
            md.seq_lens.copy_(lengths.roll(i).clamp_max(md.max_model_len))
            md.active.fill_(step > 0)
            if step:
                md.active[(i + step) % rows] = False
            cols = torch.arange(md.block_table_tensor.shape[1])
            table = (torch.arange(rows)[:, None] + cols * 3 + step + i) % (320 // block)
            table[0, 0], table[2, 0] = -1, 9999
            md.block_table_tensor.copy_(table)
        indices = torch.arange(extra).repeat(rows, 1).roll(step, 1)
        indices[:, ::5] = -1
        indices[:, 1:3] = 0  # Deliberate duplicates.
        indices[::2, -1] = torch.iinfo(torch.int64).max
        topk.copy_(indices)

    def run():
        received.clear()
        result = layer(x, runtime)
        return (*received[0], result)

    return inputs, run
