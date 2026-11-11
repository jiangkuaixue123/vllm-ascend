// SPDX-License-Identifier: CANN-Open-Software-License-Agreement-Version-2.0
#include <algorithm>
#include <cstdint>
#include <limits>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "kv_compress_epilog_v2_tiling.h"
#include "../../kv_compress_epilog/op_host/kv_compress_row_partition.h"

namespace optiling {
struct KvCompressEpilogV2CompileInfo {};

static ge::graphStatus TileKvCompressEpilogV2(gert::TilingContext *ctx)
{
    auto *cache = ctx->GetInputShape(0);
    auto *input = ctx->GetInputShape(1);
    auto *slots = ctx->GetInputShape(2);
    auto *attrs = ctx->GetAttrs();
    if (cache == nullptr || input == nullptr || slots == nullptr || attrs == nullptr ||
        ctx->GetPlatformInfo() == nullptr || attrs->GetInt(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &x = input->GetStorageShape();
    const auto &c = cache->GetStorageShape();
    const auto &s = slots->GetStorageShape();
    const int64_t mode = *attrs->GetInt(0);
    if (mode < 0 || mode > 2 || x.GetDimNum() != 2 || c.GetDimNum() != 2 || s.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t width = x.GetDim(1), rows = x.GetDim(0), stride = c.GetDim(1);
    const int64_t group = mode == 1 ? 16 : 32;
    if (rows <= 0 || rows != s.GetDim(0) || rows > std::numeric_limits<int32_t>::max() ||
        width <= 0 || width > 65536 || width % group != 0 || c.GetDim(0) <= 0 ||
        stride != (mode == 0 ? width : width / 2) + width / group) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const int64_t cores = platform.GetCoreNumAiv();
    if (cores <= 0 || ub <= 1024) {
        return ge::GRAPH_FAILED;
    }
    const auto partition = PartitionKvCompressRows(rows, cores);
    const int64_t rowBytes = 2 * (2 * width + ((stride + 31) / 32) * 32) + sizeof(int64_t);
    const int64_t blockRows = std::min<int64_t>(partition.perCore, (ub - 1024) / rowBytes);
    if (blockRows == 0) {
        return ge::GRAPH_FAILED;
    }
    KvCompressEpilogV2TilingData data;
    data.set_d(width);
    data.set_kvCacheCol(stride);
    data.set_cacheRows(c.GetDim(0));
    data.set_quantMode(mode);
    data.set_rowFactor(blockRows);
    data.set_rowOfFormerBlock(partition.perCore);
    data.set_rowLoopOfFormerBlock((partition.perCore + blockRows - 1) / blockRows);
    data.set_rowLoopOfTailBlock((partition.lastCore + blockRows - 1) / blockRows);
    data.set_tailRowFactorOfFormerBlock((partition.perCore - 1) % blockRows + 1);
    data.set_tailRowFactorOfTailBlock((partition.lastCore - 1) % blockRows + 1);
    ctx->SetBlockDim(partition.usedCores);
    ctx->SetTilingKey(0);
    auto *buffer = ctx->GetRawTilingData();
    if (buffer == nullptr || buffer->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    data.SaveToBuffer(buffer->GetData(), buffer->GetCapacity());
    buffer->SetDataSize(data.GetDataSize());
    size_t *ws = ctx->GetWorkspaceSizes(1);
    if (ws == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseKvCompressEpilogV2(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCompressEpilogV2).Tiling(TileKvCompressEpilogV2)
    .TilingParse<KvCompressEpilogV2CompileInfo>(ParseKvCompressEpilogV2);
}  // namespace optiling
