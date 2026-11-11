# SPDX-License-Identifier: CANN-Open-Software-License-Agreement-Version-2.0
add_op_to_compiled_list()
set(kv_compress_epilog_v2_track_kernel_inputs TRUE CACHE INTERNAL "Rebuild binary variants when staged source changes")
set(kv_compress_epilog_v2_depends attention/kv_compress_epilog CACHE INTERNAL "Shared row writer and DMA helpers")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE kv_compress_epilog_v2_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME KvCompressEpilogV2
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE kv_compress_epilog_v2 ACLNNTYPE aclnn)
endif()
