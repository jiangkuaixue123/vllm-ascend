/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include "hc_pre_delayed_tiling.h"

namespace optiling {
namespace {
constexpr int64_t HC_WIDTH = 4;
constexpr int64_t ALIGNED_HC_WIDTH = 8;
constexpr int64_t VECTOR_BLOCK_BYTES = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t SINKHORN_ITERS = 20;
constexpr float HC_EPS = 1e-6f;
constexpr int64_t HIDDEN_SIZE = 5120;
constexpr int64_t MIX_WIDTH = 24;
constexpr int64_t CUBE_BLOCK = 16;
constexpr int64_t K_SPLIT_ALIGNMENT = 256;
constexpr int64_t K_LOOP_SIZE = 1024;
constexpr int64_t D_TILE_SIZE = 2560;
constexpr int64_t FLOAT_CACHE_LINE = 128;
constexpr int64_t WORKSPACE_ALIGNMENT = 512;
constexpr int64_t UB_RESERVE_BYTES = 8192;
constexpr int64_t RUNTIME_WORKSPACE_BYTES = 16 * 1024 * 1024;

int64_t RoundUp(int64_t value, int64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

int64_t OutputUbBytes(int64_t splitCores)
{
    // Queues for two projections, comb projection, and both square-sum
    // allocations in the shared kernel. Each queue has two buffers.
    int64_t partials = DOUBLE_BUFFER * splitCores * sizeof(float) *
        (2 * ALIGNED_HC_WIDTH + HC_WIDTH * ALIGNED_HC_WIDTH + 2 * CUBE_BLOCK);
    int64_t inputOutputTiles = (DOUBLE_BUFFER * sizeof(uint16_t) + sizeof(float)) *
        (HC_WIDTH + 1) * D_TILE_SIZE;
    int64_t gateQueues = DOUBLE_BUFFER * (1 + HC_WIDTH) * ALIGNED_HC_WIDTH * sizeof(float);
    int64_t bases = (2 + HC_WIDTH) * ALIGNED_HC_WIDTH * sizeof(float);
    int64_t broadcasts = 2 * ALIGNED_HC_WIDTH * VECTOR_BLOCK_BYTES;
    int64_t reductions = (1 + 2 + HC_WIDTH + 1 + 1) * ALIGNED_HC_WIDTH * sizeof(float);
    int64_t diagonalMasks = CUBE_BLOCK * ALIGNED_HC_WIDTH * sizeof(float);
    int64_t delayedQueues = (1 + HC_WIDTH) * ALIGNED_HC_WIDTH * sizeof(float);
    return partials + inputOutputTiles + gateQueues + bases + broadcasts +
        reductions + diagonalMasks + delayedQueues;
}

bool ShapeMatches(const gert::StorageShape *storage, std::initializer_list<int64_t> expected)
{
    if (storage == nullptr || storage->GetStorageShape().GetDimNum() != expected.size()) {
        return false;
    }
    size_t axis = 0;
    for (auto size : expected) {
        if (storage->GetStorageShape().GetDim(axis++) != size) {
            return false;
        }
    }
    return true;
}
}  // namespace

ge::graphStatus TilingForHcPreDelayed(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *x = context->GetInputShape(0);
    const auto *attrs = context->GetAttrs();
    if (x == nullptr || x->GetStorageShape().GetDimNum() != 3 || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t rows = x->GetStorageShape().GetDim(0);
    if ((rows != 3 && rows != 6 && rows != 12) || !ShapeMatches(x, {rows, HC_WIDTH, HIDDEN_SIZE}) ||
        !ShapeMatches(context->GetInputShape(1), {MIX_WIDTH, HC_WIDTH * HIDDEN_SIZE}) ||
        !ShapeMatches(context->GetInputShape(2), {3}) ||
        !ShapeMatches(context->GetInputShape(3), {MIX_WIDTH}) ||
        !ShapeMatches(context->GetInputShape(4), {rows, HC_WIDTH})) {
        return ge::GRAPH_FAILED;
    }
    const auto *hc = attrs->GetAttrPointer<int64_t>(0);
    const auto *iters = attrs->GetAttrPointer<int64_t>(1);
    const auto *eps = attrs->GetAttrPointer<float>(2);
    const auto *normEps = attrs->GetAttrPointer<float>(3);
    if ((hc != nullptr && *hc != HC_WIDTH) || (iters != nullptr && *iters != SINKHORN_ITERS) ||
        (eps != nullptr && *eps != HC_EPS) ||
        (normEps != nullptr && (!std::isfinite(*normEps) || *normEps <= 0.0f))) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    int64_t cubeCores = platform.GetCoreNumAic();
    int64_t vectorCores = platform.GetCoreNumAiv();
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (cubeCores <= 0 || vectorCores < rows || ubBytes <= UB_RESERVE_BYTES) {
        return ge::GRAPH_FAILED;
    }
    int64_t k = HC_WIDTH * HIDDEN_SIZE;
    int64_t splitK = RoundUp((k + cubeCores - 1) / cubeCores, K_SPLIT_ALIGNMENT);
    int64_t usedKCores = (k + splitK - 1) / splitK;
    // Include both legacy squareSumQue InitBuffer calls, both x/y buffers,
    // intermediate FP32 buffers, and the new previous-pre/diagnostic queues.
    int64_t stage2UbBytes = OutputUbBytes(usedKCores);
    int64_t stage1Rows = (ubBytes - UB_RESERVE_BYTES) / (12 * K_LOOP_SIZE);
    if (stage2UbBytes + UB_RESERVE_BYTES > ubBytes || stage1Rows < 1) {
        return ge::GRAPH_FAILED;
    }
    HcPreTilingData data;
    data.set_bs(rows);
    data.set_hcMix(MIX_WIDTH);
    data.set_hcMult(HC_WIDTH);
    data.set_d(HIDDEN_SIZE);
    data.set_hcMultAlign(ALIGNED_HC_WIDTH);
    data.set_k(k);
    data.set_cubeBlockDimM(1);
    data.set_cubeBlockDimK(usedKCores);
    data.set_multCoreSplitMSize(CUBE_BLOCK);
    data.set_mL1Size(CUBE_BLOCK);
    data.set_multCoreSplitKSize(splitK);
    data.set_kL1Size(K_LOOP_SIZE);
    data.set_cvLoopKSize(K_LOOP_SIZE);
    data.set_cubeCoreNum(cubeCores);
    data.set_stage1MFactor(stage1Rows);
    // T <= 12: one vector core owns one token; each HC copy is read in two D tiles.
    data.set_rowOfFormerBlock(1);
    data.set_rowOfTailBlock(1);
    data.set_rowLoopOfFormerBlock(1);
    data.set_rowLoopOfTailBlock(1);
    data.set_stage2RowFactor(1);
    data.set_secondUsedCoreNum(rows);
    data.set_tailRowFactorOfFormerBlock(1);
    data.set_tailRowFactorOfTailBlock(1);
    data.set_dLoop(HIDDEN_SIZE / D_TILE_SIZE);
    data.set_dFactor(D_TILE_SIZE);
    data.set_tailDFactor(D_TILE_SIZE);
    data.set_iterTimes(SINKHORN_ITERS);
    data.set_hcEps(HC_EPS);
    data.set_normEps(normEps == nullptr ? 1e-20f : *normEps);
    int64_t castBytes = cubeCores * 2 * CUBE_BLOCK * K_LOOP_SIZE * sizeof(float);
    int64_t projectionBytes = RoundUp(usedKCores * rows * FLOAT_CACHE_LINE * sizeof(float), WORKSPACE_ALIGNMENT);
    int64_t squareBytes = RoundUp(usedKCores * CUBE_BLOCK * CUBE_BLOCK * sizeof(float), WORKSPACE_ALIGNMENT);
    context->GetWorkspaceSizes(1)[0] = castBytes + projectionBytes + squareBytes + RUNTIME_WORKSPACE_BYTES;
    context->SetTilingKey(0);
    context->SetBlockDim(cubeCores);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(HcPreDelayed).Tiling(TilingForHcPreDelayed);
}  // namespace optiling
