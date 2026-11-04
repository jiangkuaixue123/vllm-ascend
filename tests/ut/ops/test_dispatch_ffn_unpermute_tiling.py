# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest


def test_unpermute_tiling_covers_small_decode_buckets_without_zero_division(tmp_path):
    """Compile the actual host/device tiler, removing only vendor annotations."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the host tiling regression")
    root = Path(__file__).resolve().parents[3]
    header = root / "csrc/mc2/dispatch_ffn_combine/op_kernel/unpermute/moe_token_unpermute_tiling.h"
    source = header.read_text().replace("__forceinline__ [host, aicore]", "inline")
    program = tmp_path / "tiling.cpp"
    program.write_text(
        "#include <algorithm>\n#include <cstdint>\n#include <cassert>\nusing std::min;\n"
        + source
        + """
int main() {
    for (int rows : {0, 1, 6, 12, 39, 40, 41, 80, 24001}) {
        for (int topk : {3, 6}) {
            MoeTokenUnpermuteTilingData t{};
            MoeTokenUnpermuteTiling(rows * topk, 5120, topk, t, 40);
            assert(t.tokens_splited_length > 0);
            int64_t count = 0;
            for (int core = 0; core < 40; ++core) {
                int64_t remainder = t.tokens_splited_remain + (core < t.tokens_core_remain);
                int64_t assigned = t.tokens_splited_num * t.tokens_splited_length + remainder;
                assert(assigned == t.tokens_core_length + (core < t.tokens_core_remain));
                count += assigned;
            }
            assert(count == rows);
        }
    }
}
"""
    )
    binary = tmp_path / "tiling"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(program), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
