# SPDX-License-Identifier: Apache-2.0
# Headers only: do not register/rebuild HcPre in the independent vendor package.
set(hc_pre_delayed_depends moe/hc_pre/op_kernel PARENT_SCOPE)

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE hc_pre_delayed_def.cpp)
    add_ops_compile_options(
        OP_NAME HcPreDelayed
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE hc_pre_delayed ACLNNTYPE aclnn)
endif()
