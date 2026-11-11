"""Read only the hits from shared FP8 checkpoint pages for the eager text backbone."""

import numpy as np
import torch
from torch import nn


class HostEngramEmbedding(nn.Module):
    """Drop-in embedding for HF Engram. The projection, norm and gate stay on device."""

    def __init__(self, num_embeddings, dim, *, transfer_rows=512):
        super().__init__()
        if num_embeddings <= 0 or dim <= 0 or dim % 32 or transfer_rows <= 0:
            raise ValueError("Invalid FP8 Engram layout or row batch")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.transfer_rows = transfer_rows
        self._weight = self._scale = None
        self.register_buffer("_value_lut", None, persistent=False)
        self.register_buffer("_scale_lut", None, persistent=False)
        self.lookup_calls = self.requested_rows = self.transferred_rows = 0
        self.row_payload_bytes = self.largest_payload_bytes = 0

    def validate_layout(self, weight, scale):
        if (weight.dtype, weight.shape) != ("F8_E4M3", (self.num_embeddings, self.dim)):
            raise ValueError(f"Wrong Engram weight layout: {weight.dtype} {weight.shape}")
        if (scale.dtype, scale.shape) != ("F8_E8M0", (self.num_embeddings, self.dim // 32)):
            raise ValueError(f"Wrong Engram scale layout: {scale.dtype} {scale.shape}")

    def bind(self, weight, scale, device):
        self.validate_layout(weight, scale)
        # MAP_SHARED/read-only file-backed pages are common to every worker on the host.
        # Keep storage outside Module buffers, which vLLM/PyTorch can move to the NPU.
        self._weight = np.memmap(weight.file, mode="r", offset=weight.offset,
                                 shape=weight.shape, dtype=np.uint8)
        self._scale = np.memmap(scale.file, mode="r", offset=scale.offset,
                                shape=scale.shape, dtype=np.uint8)
        # Decode codes on device using the exact torch format, including subnormals.
        codes = torch.arange(256, dtype=torch.int32, device="cpu").byte()
        self._value_lut = codes.view(torch.float8_e4m3fn).float().to(device)
        self._scale_lut = codes.view(torch.float8_e8m0fnu).float().to(device)

    @torch.inference_mode()
    def forward(self, indices):
        if self._weight is None or self._scale is None:
            raise RuntimeError("Engram checkpoint tables have not been bound")
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram rows require integer indices")
        flat = indices.reshape(-1)
        device = self._value_lut.device
        result = torch.empty((flat.numel(), self.dim), device=device, dtype=torch.bfloat16)
        self.lookup_calls += 1
        for start in range(0, flat.numel(), self.transfer_rows):
            ids = flat[start:start + self.transfer_rows].to(device="cpu", dtype=torch.int64).numpy()
            unique, inverse = np.unique(ids, return_inverse=True)
            if unique[0] < 0 or unique[-1] >= self.num_embeddings:
                raise IndexError("Engram row outside its layer's table")
            # Fancy indexing copies only the selected rows into bounded writable host arrays.
            raw = torch.from_numpy(self._weight[unique]).to(device)
            scales = torch.from_numpy(self._scale[unique]).to(device)
            values = self._value_lut[raw.long()].reshape(-1, self.dim // 32, 32)
            values = (values * self._scale_lut[scales.long()].unsqueeze(-1)).flatten(1).bfloat16()
            positions = torch.from_numpy(inverse).to(device=device, dtype=torch.long)
            result[start:start + ids.size].copy_(values.index_select(0, positions))
            self.requested_rows += ids.size
            self.transferred_rows += unique.size
            payload = unique.size * (self.dim + self.dim // 32)
            self.row_payload_bytes += payload
            self.largest_payload_bytes = max(self.largest_payload_bytes, payload)
        return result.view(*indices.shape, self.dim)
