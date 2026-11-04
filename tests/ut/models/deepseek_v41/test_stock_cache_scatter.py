# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import attention


class Descriptor:
    """Host-only metadata for testing a large physical cache without allocating it."""

    def __init__(self, shape, dtype, stride, device):
        self.shape = torch.Size(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device
        self.requires_grad = False
        self._stride = stride
        self.offset = 1024
        self.format = 2

    def is_contiguous(self):
        return self._stride == ((1,) if self.ndim == 1 else (self.shape[1], 1))

    def stride(self, dim=None):
        return self._stride if dim is None else self._stride[dim]

    def storage_offset(self):
        return self.offset

    def element_size(self):
        return self.dtype.itemsize

    def numel(self):
        return self.shape.numel()


@pytest.mark.parametrize(("dtype", "width"), [(torch.float32, 512), (torch.bfloat16, 512), (torch.bfloat16, 128)])
@torch.inference_mode()
def test_stock_scatter_dispatch(dtype, width):
    dev = SimpleNamespace(type="npu")
    limit = attention._MAX_DECODE_SCATTER_BYTES // (dtype.itemsize * width)
    page = Descriptor((limit, width), dtype, (1024, 1), dev)
    slots = Descriptor((24,), torch.int64, (1,), dev)
    updates = Descriptor((24, width), dtype, (width, 1), dev)
    md = attention.DeepseekV41DecodeMetadata(None, slots, None, None, 16384, 6, 0)
    ctx = SimpleNamespace(cudagraph_runtime_mode=attention.CUDAGraphMode.FULL, is_draft_model=False)
    npu = SimpleNamespace(Format=SimpleNamespace(ND=2), get_npu_format=lambda tensor: tensor.format)

    def supports():
        return attention._supports_decode_scatter(page, slots, updates, md, 2)

    with (
        patch.dict(sys.modules, {"torch_npu": npu}),
        patch.object(attention.envs, "VLLM_BATCH_INVARIANT", False),
        patch.object(attention, "is_forward_context_available", return_value=True),
        patch.object(attention, "get_forward_context", return_value=ctx),
        patch.object(attention, "get_ascend_device_type", return_value=attention.AscendDeviceType.A3) as device,
        patch.object(attention, "enable_custom_op", return_value=True) as custom,
        patch.object(torch.ops._C_ascend, "npu_scatter_nd_update_v2", create=True),
    ):
        assert supports()
        for obj, key, bad in (
            (page, "shape", torch.Size([limit + 1, width])),
            (page, "offset", torch.iinfo(torch.int32).max),
            (page, "_stride", (width, 1)),
            (page, "dtype", torch.float16),
            (updates, "requires_grad", True),
            (slots, "dtype", torch.int32),
            (slots, "device", torch.device("cpu")),
            (slots, "shape", torch.Size([12])),
            (updates, "shape", torch.Size([24, width - 1])),
            (page, "format", 29),
            (slots, "format", 29),
            (updates, "format", 29),
            (md, "grouped_row_start", None),
            (md, "query_width", 1),
            (ctx, "cudagraph_runtime_mode", attention.CUDAGraphMode.NONE),
            (ctx, "is_draft_model", True),
            (attention.envs, "VLLM_BATCH_INVARIANT", True),
        ):
            with patch.object(obj, key, bad):
                assert not supports(), key
        with torch.inference_mode(False):
            assert not supports()
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            assert not supports()
        assert not attention._supports_decode_scatter(page, slots, updates, md, 0)
        device.return_value = None
        assert not supports()
        device.return_value = attention.AscendDeviceType.A3
        custom.return_value = False
        assert not supports()


@pytest.mark.parametrize("query_width", [1, 6])
def test_stock_scatter_cpu_keeps_parent(query_width):
    page = torch.zeros(4, 1024)[:, :512]
    cache = SimpleNamespace(field=lambda *args: page, source_ratio=2)
    slots = torch.tensor([0, 0, -1, 2, 4, 1])
    active = torch.tensor([True, True, True, True, True, False])
    md = attention.DeepseekV41DecodeMetadata(None, slots, None, active, 16384, query_width, 0)
    values = torch.arange(6 * 512).reshape(6, 512).float()
    with patch.object(attention, "_scatter_decode_field", side_effect=AssertionError):
        attention.DeepseekV41Cache.decode_write(cache, md, values, 0, 0, active)
    assert torch.equal(page[0], values[1 if query_width == 1 else 0])
    assert torch.equal(page[2], values[3])
    assert torch.count_nonzero(page[[1, 3]]) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_stock_scatter_transfers_written_words(dtype):
    word = torch.int16 if dtype == torch.bfloat16 else torch.int32
    backing = torch.arange(7 * 1024, dtype=word).reshape(7, 1024)
    expected = backing.clone()
    page = backing[1:-1, 512:640].view(dtype)
    slots = torch.tensor([0, 4, 0])
    bits = torch.tensor([0, -1, 1, -32768, 0x7F81, 0x7FC2, -127, -62], dtype=word)
    if dtype == torch.float32:
        bits = torch.tensor([0, -1, 1, -(2**31), 0x7F800001, 0x7FC12345, -8388607, -4123456], dtype=word)
    updates = bits.repeat(3, 16).view(dtype)
    updates[1].view(word).copy_(bits.flip(0).repeat(16))
    expected[1:-1, 512:640].index_copy_(0, slots, updates.view(word))

    def dma(dst, idx, src):
        assert dst is page and idx.dtype == torch.int32 and torch.equal(idx[:, 0].long(), slots)
        dst.view(word).index_copy_(0, idx[:, 0].long(), src.view(word))

    with patch.object(torch.ops._C_ascend, "npu_scatter_nd_update_v2", side_effect=dma, create=True) as native:
        attention._scatter_decode_field(page, slots, updates)
    assert native.call_count == 1
    assert torch.equal(backing, expected)
