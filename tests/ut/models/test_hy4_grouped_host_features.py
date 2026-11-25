# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.hy_v4_flash.grouped_host_features import supports_grouped_feature_rows


@pytest.mark.parametrize("tokens", [6, 42, 96])
def test_grouped_rows_accept_strided_hash_buffers_and_separate_shards(tokens):
    hashes = torch.empty(4, 192, 16, dtype=torch.int32)[:, :tokens]
    tables = [SimpleNamespace(head_start=h, head_end=h + 2, dim=96, host_weight=object()) for h in (0, 4, 10, 14)]
    rows = [torch.empty(tokens, 2, 96) for _ in tables]
    assert supports_grouped_feature_rows(tables, hashes, rows)
    assert not supports_grouped_feature_rows(tables[:-1], hashes, rows)
    assert not supports_grouped_feature_rows(tables, hashes, [rows[0].bfloat16(), *rows[1:]])
    tables[-1].head_end += 1
    assert not supports_grouped_feature_rows(tables, hashes, rows)
