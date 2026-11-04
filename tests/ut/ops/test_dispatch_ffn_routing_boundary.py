# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest


def test_route_before_down_rejects_gather_from_the_actual_routing_tiler(tmp_path):
    """Compile the production tiler and the same guard used by the host entry."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the routing boundary regression")
    root = Path(__file__).resolve().parents[3]
    operator = root / "csrc/mc2/dispatch_ffn_combine"
    tiler = operator / "op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
    guard = operator / "op_host/tiling_args.h"
    program = tmp_path / "routing.cpp"
    program.write_text(
        "#include <algorithm>\n#include <array>\n#include <cassert>\n#include <cmath>\n#include <cstdint>\n"
        f'#include "{tiler}"\n#include "{guard}"\n'
        + """
uint64_t route(int64_t rows, int64_t hidden, int64_t topk, int64_t localExperts,
               int64_t ep, bool expectedSupport) {
    optiling::MoeInitRoutingQuantV2TilingBase tiler;
    // Same arguments as the dispatch host: no smooth/drop, AIV40, UB196352.
    bool ok = tiler.DoTiling(rows, hidden, topk, 0, localExperts * ep + 1,
                            0, 0, 2, false, 2, 1, 0, 40, 196352);
    assert(ok);
    assert(Moe::IsRouteBeforeDownRoutingSupported(true, ok, tiler.tilingKey_) == expectedSupport);
    assert(Moe::IsRouteBeforeDownRoutingSupported(false, ok, tiler.tilingKey_));
    return tiler.tilingKey_;
}
int main() {
    for (int rows : {6, 12}) {
        assert(route(rows, 5120, 6, 24, 16, true) == 21000);  // target
        assert(route(rows, 5120, 3, 8, 16, true) == 21000);   // draft
    }
    for (int hidden : {64, 5120}) {
        assert(route(64, hidden, 8, 24, 16, true) == 21000);
        assert(route(64, hidden, 8, 8, 16, true) == 21000);
    }
    // M1/K64/N64/topk1/localE4096/EP16/capacity16 passes the raw shape
    // limits, but its expert-count workspace prevents full-load routing.
    assert(route(1, 64, 1, 4096, 16, false) == 11000);
    assert(!Moe::IsRouteBeforeDownRoutingSupported(true, false, 21000));
    assert(!Moe::IsRouteBeforeDownRoutingSupported(true, true, 11010));
    assert(Moe::IsRouteBeforeDownRoutingSupported(false, false, 11000));
}
"""
    )
    binary = tmp_path / "routing"
    subprocess.run([compiler, "-std=c++17", str(program), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
