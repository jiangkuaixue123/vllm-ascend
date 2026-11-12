# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_triton_lock.py"
spec = importlib.util.spec_from_file_location("patch_triton_lock", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LEGACY = """
std::vector<int64_t> lockInitData(8, 0);
ret = rtMemcpy(
    syncBlockLock_ptr, syncBlockLockSize,
    reinterpret_cast<void *>(lockInitData.data()), syncBlockLockSize,
    RT_MEMCPY_HOST_TO_DEVICE
);
"""


@pytest.mark.parametrize("initial", [0, 7])
def test_initialization_preserves_both_launcher_paths_and_nonzero_values(initial):
    source = LEGACY + "kernel_launch();" + LEGACY
    generate = module.stream_ordered_lock_init(lambda *args: source)
    actual = generate({}, {}, SimpleNamespace(lock_num=8, lock_init_val=initial))
    if initial == 0:
        assert actual.count("rtMemsetAsync(") == 2
        assert "rtMemcpy(" not in actual
        assert "kernel_launch();" in actual
        assert module.stream_ordered_lock_init(generate)({}, {}, SimpleNamespace(lock_num=8)) == actual
    else:
        assert actual == source


def test_unknown_legacy_layout_fails_before_launch():
    generate = module.stream_ordered_lock_init(lambda *args: LEGACY.replace("rtMemcpy(", "otherCopy("))
    with pytest.raises(RuntimeError, match="Unsupported legacy"):
        generate({}, {}, SimpleNamespace(lock_num=8, lock_init_val=0))
