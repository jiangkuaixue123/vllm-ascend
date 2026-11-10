# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Generated, instance-bound callbacks: globals come from the original methods.
# Preserve the measured source's imports and SSA temporaries for AST comparison.
# ruff: noqa: F401, F821, F841, I001

from vllm_ascend.models.deepseek_v41.attention import DeepseekV41Cache
from vllm_ascend.ops.triton.deepseek_v41_cache import decode_read as fused_decode_read, decode_write


def decode_read(self, md, positions, bank, offset, width):
    # Microbenchmarks favor native large raw/index reads. Keep only proven
    # short gathers and large FP32 compressor-field gathers on this kernel.
    if positions.shape[1] > 8 and not (bank == 0 and offset == self.dim and self.source_ratio > 1):
        return DeepseekV41Cache.decode_read(self, md, positions, bank, offset, width)
    return fused_decode_read(self, md, positions, bank, offset, width)
