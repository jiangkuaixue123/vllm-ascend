# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep legacy Triton-Ascend zero-initialized locks in graph replay order."""

import re
from functools import wraps


def stream_ordered_lock_init(generate):
    @wraps(generate)
    def wrapped(constants, signature, metadata):
        source = generate(constants, signature, metadata)
        initial = getattr(metadata, "lock_init_value", getattr(metadata, "lock_init_val", 0))
        if getattr(metadata, "lock_num", 0) > 0 and initial == 0:
            # Host copies run at capture time only; reset on the kernel's stream.
            source = re.sub(
                r"std::vector<int64_t> lockInitData\([^;]+\);\s*"
                r"ret = rtMemcpy\(\s*syncBlockLock_ptr, syncBlockLockSize,\s*"
                r"reinterpret_cast<void \*>\(lockInitData.data\(\)\), syncBlockLockSize,\s*"
                r"RT_MEMCPY_HOST_TO_DEVICE\s*\);",
                "ret = rtMemsetAsync(syncBlockLock_ptr, syncBlockLockSize, 0, syncBlockLockSize, stream);",
                source,
            )
            if "std::vector<int64_t> lockInitData" in source:
                raise RuntimeError("Unsupported legacy Triton lock initialization")
        return source

    return wrapped


def patch_triton_lock():
    from triton.backends.ascend import driver

    if hasattr(driver, "generate_npu_wrapper_src"):
        driver.generate_npu_wrapper_src = stream_ordered_lock_init(driver.generate_npu_wrapper_src)
