/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
// Prefer the package's freshly copied header-only dependency when present.
#if __has_include("../../op_kernel/hc_pre_m_k_split_core.h")
#include "../../op_kernel/hc_pre_m_k_split_core.h"
#else
#include "../../hc_pre/op_kernel/hc_pre_m_k_split_core.h"
#endif

using namespace AscendC;
extern "C" __global__ __aicore__ void hc_pre_delayed(
    GM_ADDR x, GM_ADDR hc_fn, GM_ADDR hc_scale, GM_ADDR hc_base, GM_ADDR previous_pre,
    GM_ADDR y, GM_ADDR next_pre, GM_ADDR post, GM_ADDR comb_frag, GM_ADDR projection,
    GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(HcPreTilingData, data, tiling);
        TPipe pipe;
        HcPre::HcPreMembaseKSplitCorePart1<DTYPE_X, true> projectionOp;
        projectionOp.Init(x, hc_fn, userWs, &data, &pipe);
        projectionOp.Process();
        pipe.Destroy();
        TPipe outputPipe;
        HcPre::HcPreMembaseKSplitCorePart2<DTYPE_X, true> outputOp;
        outputOp.Init(x, hc_scale, hc_base, y, post, comb_frag, userWs, &data, &outputPipe,
            previous_pre, next_pre, projection, rstd);
        outputOp.Process();
        outputPipe.Destroy();
    }
}
