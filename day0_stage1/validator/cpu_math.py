"""Independent CPU expressions for the six math entry points used by the pinned HF text forward."""

import torch
import torch.nn.functional as F


def act_quant(x, block_size, scale_fmt, scale_dtype, inplace=False):
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    if scale_fmt != "ue8m0":
        raise ValueError("expected the frozen UE8M0 activation format")
    s = torch.exp2(torch.ceil(torch.log2(amax / 448))).to(scale_dtype)
    q = (blocks / s.float().unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn)
    if inplace:
        return x.copy_((q.float() * s.float().unsqueeze(-1)).flatten(-2))
    return q.reshape(x.shape), s


# E2M1 nibble to signed value, packed low then high. Both signed-zero encodings have value zero.
_E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.)
_LUT = torch.tensor([[_E2M1[i & 15], _E2M1[i >> 4]] for i in range(256)], dtype=torch.bfloat16)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if not inplace:
        raise ValueError("the pinned indexer and compressor request only in-place FP4 QDQ")
    groups = x.float().unflatten(-1, (-1, block_size))
    amax = groups.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2**-9) / 6).to(scale_dtype).float()
    else:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(6 * 2**-126) / 6)))
    scaled = (groups / s.unsqueeze(-1)).clamp(-6, 6)
    # T.Cast for E2M1 uses round-to-nearest-even. At a midpoint, prefer the
    # code with an even low bit. Between 4 and 6 these codes are 6 and 7.
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32)
    mag = scaled.abs()
    codes = torch.bucketize(mag, boundaries)
    ties_to_upper_even = (codes % 2 != 0) & (mag == boundaries[codes.clamp_max(6)])
    vals = torch.tensor(_E2M1[:8], dtype=torch.float32)[codes + ties_to_upper_even]
    return x.copy_((torch.copysign(vals, scaled) * s.unsqueeze(-1)).reshape_as(x))


def dequant_weight(w, sw, block=32):
    n = w.size(0)
    if w.dtype == torch.float4_e2m1fn_x2:
        packed = w.view(torch.uint8)
        values = _LUT[packed.int()].reshape(n, -1, block)
        return (values * sw.bfloat16().unsqueeze(-1)).flatten(1)
    q = w.bfloat16().unflatten(0, (-1, block)).unflatten(-1, (-1, block))
    return (q * sw.bfloat16()[:,None,:,None]).reshape_as(w)


def fp8_gemm(a, a_s, b, b_s, scale_dtype, block_size):
    grouped = a.bfloat16().unflatten(-1, (-1, block_size)) * a_s.bfloat16().unsqueeze(-1)
    return F.linear(grouped.flatten(-2), dequant_weight(b, b_s, block_size))


def fp4_gemm(a, a_s, b, b_s, scale_dtype, act_block_size):
    grouped = a.bfloat16().unflatten(-1, (-1, act_block_size)) * a_s.bfloat16().unsqueeze(-1)
    return F.linear(grouped.flatten(-2), dequant_weight(b, b_s))


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    a, z, c = torch.split(mixes, [hc_mult, hc_mult, hc_mult**2], dim=-1)
    ba, bz, bc = torch.split(hc_base, [hc_mult, hc_mult, hc_mult**2])
    pre = (a * hc_scale[0] + ba).sigmoid() + eps
    post = 2 * (z * hc_scale[1] + bz).sigmoid()
    c = (c * hc_scale[2] + bc).unflatten(-1, (hc_mult,hc_mult)).softmax(-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters-1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    return pre, post, c


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch, seq, heads, dim = q.shape
    rows = torch.arange(batch)[:, None, None]
    maximum = torch.full((batch,seq,heads,1), -1e30, dtype=torch.float32)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((*maximum.shape[:-1], dim), dtype=torch.float32)
    for ix in topk_idxs.split(64, dim=-1):
        keys = kv[rows, ix.clamp_min(0).long()].float()
        scores = (q.float() @ keys.transpose(-1,-2)) * softmax_scale
        scores.masked_fill_(ix.unsqueeze(-2) < 0, -torch.inf)
        new_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        rescale = (maximum - new_maximum).exp()
        prob = (scores - new_maximum).exp()
        numerator = numerator * rescale + prob.bfloat16().float() @ keys
        denominator = denominator * rescale + prob.sum(-1, keepdim=True)
        maximum = new_maximum
    denominator += (attn_sink.float().reshape(1,1,heads,1) - maximum).exp()
    return (numerator / denominator).to(q.dtype)
