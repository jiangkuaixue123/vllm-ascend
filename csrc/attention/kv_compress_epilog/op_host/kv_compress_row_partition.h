// SPDX-License-Identifier: CANN-Open-Software-License-Agreement-Version-2.0
#pragma once
#include <algorithm>
#include <cstdint>

namespace optiling {
struct KvCompressRowPartition {
    int64_t perCore;
    int64_t usedCores;
    int64_t lastCore;
};

inline KvCompressRowPartition PartitionKvCompressRows(int64_t rows, int64_t cores)
{
    const int64_t perCore = (rows + cores - 1) / cores;
    const int64_t used = std::min(cores, (rows + perCore - 1) / perCore);
    return {perCore, used, rows - (used - 1) * perCore};
}
}  // namespace optiling
