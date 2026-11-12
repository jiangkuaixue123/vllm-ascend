# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile the actual output-queue budget against the A3 unified buffer limit."""

import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

ROOT = Path(__file__).resolve().parents[5]


def test_complete_tile_fits_only_with_single_tile_buffers(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "csrc/moe/hc_pre_delayed/op_host/hc_pre_delayed_tiling.cpp").read_text()
    constants = "\n".join(re.findall(r"constexpr (?:int64_t|float) \w+ = [^;]+;", source))
    function = "int64_t OutputUbBytes(" + source.split("int64_t OutputUbBytes(", 1)[1].split("\n}", 1)[0] + "\n}"
    unit = tmp_path / "budget.cpp"
    unit.write_text(
        "#include <cassert>\n#include <cstdint>\n" + constants + "\n" + function + "\n"
        "int main() {\n"
        "constexpr int64_t UB_BYTES = 192 * 1024;\n"
        "for (int64_t cores : {20, 24}) {\n"
        "assert(OutputUbBytes(cores, HIDDEN_SIZE, SINGLE_TILE_BUFFERS) + UB_RESERVE_BYTES <= UB_BYTES);\n"
        "assert(OutputUbBytes(cores, HIDDEN_SIZE, DOUBLE_BUFFER) + UB_RESERVE_BYTES > UB_BYTES);\n"
        "assert(OutputUbBytes(cores, D_TILE_SIZE, DOUBLE_BUFFER) + UB_RESERVE_BYTES <= UB_BYTES);\n"
        "}\n}\n"
    )
    # Initializer-list support is explicit for both supported host compilers.
    unit.write_text("#include <initializer_list>\n" + unit.read_text())
    binary = tmp_path / "budget"
    subprocess.run([compiler, "-std=c++17", str(unit), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
