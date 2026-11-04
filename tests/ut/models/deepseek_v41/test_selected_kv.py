# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.ut.models.deepseek_v41.selected_kv_case import selected_kv_case
from vllm_ascend.models.deepseek_v41 import attention


@pytest.mark.parametrize("ratio", [0, 1, 2])
@torch.inference_mode()
def test_selected_kv_cpu_decode_fallback(monkeypatch, ratio):
    def forbidden(*args):
        pytest.fail("CPU decode must use the existing paged read")

    monkeypatch.setattr(attention, "try_prepare_selected_kv", forbidden)
    update, run = selected_kv_case("cpu", ratio)
    update(0)
    kv, ids, output = run()
    assert kv.dtype == output.dtype == torch.bfloat16 and ids.dtype == torch.int32
    assert not kv.count_nonzero() and not output.count_nonzero() and (ids == -1).all()
    update(1)
    kv, ids, output = run()
    assert kv.shape == (6, 128 + (32 if ratio else 0), 512)
    assert (ids[1] == -1).all() and not output[1].count_nonzero()
    if ratio:
        # The inactive consumer row can retain data from the independent live source.
        assert kv[1].count_nonzero()
    assert output[0].count_nonzero()
