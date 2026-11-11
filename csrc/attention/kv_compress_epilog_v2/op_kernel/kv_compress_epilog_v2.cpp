// SPDX-License-Identifier: CANN-Open-Software-License-Agreement-Version-2.0
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "kv_compact_codec.h"
// The binary builder stages each op_kernel directory under its operator name.
#if __has_include("../kv_compress_epilog/kv_compress_epilog.h")
#include "../kv_compress_epilog/kv_compress_epilog.h"
#else
#include "../../kv_compress_epilog/op_kernel/kv_compress_epilog.h"
#endif

namespace {
constexpr uint32_t kThreads = 128;
constexpr uint32_t kMixedMode = 3;
constexpr uint32_t kMixedPage = 128;
constexpr uint32_t kNope = 448;
constexpr uint32_t kMixedValues = kNope + 64 * sizeof(uint16_t);
constexpr uint32_t kMixedScales = 8;
using namespace AscendC;
using namespace KvCompactCodec;

__simt_vf__ __aicore__ LAUNCH_BOUND(kThreads) inline void EncodeRows(
    __ubuf__ uint8_t *output, __ubuf__ uint16_t *input, uint32_t rows, uint32_t dim,
    uint32_t rowBytes, uint32_t mode)
{
    const bool mixed = mode == kMixedMode;
    const bool fp8 = mode == 0 || mixed;
    const bool e4Scale = mode == 1;
    const uint32_t block = mixed ? 64U : e4Scale ? 16U : 32U;
    const uint32_t groups = dim / block;
    const uint32_t stride = (rowBytes + 31U) / 32U * 32U;
    const uint32_t payload = mixed ? kMixedValues : fp8 ? dim : dim / 2;
    for (uint32_t work = Simt::GetThreadIdx(); work < rows * groups; work += kThreads) {
        const uint32_t row = work / groups;
        const uint32_t group = work % groups;
        __ubuf__ uint16_t *src = input + row * dim + group * block;
        __ubuf__ uint8_t *dst = output + row * stride;
        if (mixed && group == groups - 1) {
            // Mixed MLA consumes the last 64 RoPE components as original BF16
            // and ignores the eighth scale byte. Initialize every stored byte.
            __ubuf__ uint16_t *rope = (__ubuf__ uint16_t *)(dst + kNope);
            for (uint32_t i = 0; i < block; ++i) {
                rope[i] = src[i];
            }
            dst[payload + group] = 0;
            continue;
        }
        uint16_t maximum = 0;
        for (uint32_t i = 0; i < block; ++i) {
            const uint16_t v = src[i] & 0x7fffU;
            maximum = v > maximum ? v : maximum;
        }
        const uint8_t scaleByte = e4Scale ? E4Scale(maximum) : PowerScale(maximum, fp8);
        dst[payload + group] = scaleByte;
        if (fp8) {
            for (uint32_t i = 0; i < block; ++i) {
                dst[group * block + i] = PayloadFP8(src[i], scaleByte);
            }
        } else {
            const float scale = e4Scale ? E4M3Value(scaleByte) : 0.0f;
            for (uint32_t i = 0; i < block; i += 2) {
                const uint8_t even = e4Scale ? PayloadFP4(src[i], scale) : PayloadMXFP4(src[i], scaleByte);
                const uint8_t odd = e4Scale ? PayloadFP4(src[i + 1], scale) : PayloadMXFP4(src[i + 1], scaleByte);
                dst[(group * block + i) / 2] = even | (odd << 4);
            }
        }
    }
}

struct CompactRowWriter {
    static __aicore__ inline void Store(LocalTensor<uint8_t>& local, GlobalTensor<uint8_t>& cache,
        int64_t row, int64_t slot, const KvCompressEpilogV2TilingData *tiling)
    {
        using namespace KvCompressEpilogOps;
        const int64_t offset = row * RoundUp<uint8_t>(tiling->kvCacheCol);
        if (tiling->quantMode == kMixedMode) {
            CopyBlockSplitKv(local[offset], local[offset + kMixedValues], cache, slot, kMixedPage,
                             kMixedValues, kMixedScales, kMixedPage * (kMixedValues + kMixedScales));
        } else {
            CopyOut(local[offset], cache[slot * tiling->kvCacheCol], 1, tiling->kvCacheCol);
        }
    }

    static __aicore__ inline void Run(LocalTensor<uint8_t> &out, LocalTensor<bfloat16_t> &in,
                                      int64_t rows, const KvCompressEpilogV2TilingData *tiling)
    {
        Simt::VF_CALL<EncodeRows>(Simt::Dim3(kThreads), (__ubuf__ uint8_t *)out.GetPhyAddr(),
            (__ubuf__ uint16_t *)in.GetPhyAddr(), static_cast<uint32_t>(rows), static_cast<uint32_t>(tiling->d),
            static_cast<uint32_t>(tiling->kvCacheCol), static_cast<uint32_t>(tiling->quantMode));
        PipeBarrier<PIPE_ALL>();
    }
};
}  // namespace

extern "C" __global__ __aicore__ void kv_compress_epilog_v2(
    GM_ADDR cache, GM_ADDR x, GM_ADDR slot_mapping, GM_ADDR cache_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(KvCompressEpilogV2TilingData, data, tiling);
    AscendC::TPipe pipe;
    KvCompressEpilogOps::KvCompressEpilogRegBase<DTYPE_X, DTYPE_SLOT_MAPPING, uint8_t,
                                                 KvCompressEpilogV2TilingData, CompactRowWriter> op(&pipe);
    op.Init(x, slot_mapping, cache, &data);
    op.Process();
}
