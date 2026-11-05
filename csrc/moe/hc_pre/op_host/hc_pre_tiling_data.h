/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HC_PRE_SHARED_TILING_DATA_H
#define HC_PRE_SHARED_TILING_DATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HcPreTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, hcMix);
TILING_DATA_FIELD_DEF(int64_t, hcMult);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, hcMultAlign);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(int64_t, iterTimes);
TILING_DATA_FIELD_DEF(float, hcEps);
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(int64_t, kBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, kFactor);
TILING_DATA_FIELD_DEF(int64_t, tailKFactor);
TILING_DATA_FIELD_DEF(int64_t, kLoop);
TILING_DATA_FIELD_DEF(int64_t, stage2RowFactor);

TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, kLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, kLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, stage1KFactor);
TILING_DATA_FIELD_DEF(int64_t, kFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, kL1Size);
TILING_DATA_FIELD_DEF(int64_t, mL1Size);
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDimK);
TILING_DATA_FIELD_DEF(int64_t, kUbSize); // ub一次处理k的size
TILING_DATA_FIELD_DEF(int64_t, cvLoopKSize); // cv同步使用的k轴长度
TILING_DATA_FIELD_DEF(int64_t, multCoreSplitKSize); // 多核切K后的单核k处理长度
TILING_DATA_FIELD_DEF(int64_t, multCoreSplitMSize); // 多核切K后的单核k处理长度
TILING_DATA_FIELD_DEF(int64_t, tailKSizeOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailKSizeOfTailBlock);

TILING_DATA_FIELD_DEF(int64_t, mLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, mLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, formerMSize);
TILING_DATA_FIELD_DEF(int64_t, tailMSizeOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailMSizeOfTailBlock);

TILING_DATA_FIELD_DEF(int64_t, firstUsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, secondUsedCoreNum);

TILING_DATA_FIELD_DEF(int64_t, rowInnerFactor);

TILING_DATA_FIELD_DEF(int64_t, cubeCoreNum);
TILING_DATA_FIELD_DEF(int64_t, stage1MFactor);

TILING_DATA_FIELD_DEF(int64_t, bufferPool0Size);
TILING_DATA_FIELD_DEF(int64_t, bufferPool1Size);
TILING_DATA_FIELD_DEF(int64_t, mUbSize);

END_TILING_DATA_DEF;
}  // namespace optiling
#endif
