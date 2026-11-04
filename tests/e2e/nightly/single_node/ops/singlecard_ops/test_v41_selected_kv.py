# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch

from tests.ut.models.deepseek_v41.selected_kv_case import selected_kv_case
from vllm_ascend.models.deepseek_v41 import attention
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@pytest.mark.parametrize("ratio,strided_columns", [(0, False), (1, False), (2, False), (2, True)])
@torch.inference_mode()
def test_decode_selected_kv_native_graph(ratio, strided_columns):
    if get_ascend_device_type() != AscendDeviceType.A3 or attention.try_prepare_selected_kv is None:
        pytest.skip("Requires A3 with Triton and standard batch mode")
    init_device_properties_triton()
    assert attention.enable_custom_op()
    update, run = selected_kv_case("npu:0", ratio, strided_columns=strided_columns)
    update(0)
    accepted = []
    original = attention.try_prepare_selected_kv

    def record(*args):
        output = original(*args)
        accepted.append(output is not None)
        return output

    with patch.object(attention, "try_prepare_selected_kv", record):
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replay = run()
    assert accepted == ([not strided_columns] * 2 if ratio else [])
    for step in (1, 2, 0, 3):
        update(step)
        with patch.object(attention, "try_prepare_selected_kv", None):
            baseline = run()
        candidate = run()
        graph.replay()
        torch.npu.synchronize()
        for a, b, c in zip(replay, candidate, baseline):
            # Inspect the selected latent, logical IDs and actual native-attention result.
            dtype = torch.int16 if c.dtype == torch.bfloat16 else c.dtype
            assert torch.equal(a.view(dtype), c.view(dtype))
            assert torch.equal(b.view(dtype), c.view(dtype))
