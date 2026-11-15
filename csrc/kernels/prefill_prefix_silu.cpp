// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

namespace {
constexpr uint32_t TILE = 4096;

class PrefixSilu {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR groups, GM_ADDR out,
                              uint32_t rows, uint32_t width, uint32_t count) {
        AscendC::GlobalTensor<int64_t> counts;
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groups));
        int64_t live = count == 0 ? rows : 0;
        for (uint32_t i = 0; i < count; ++i) {
            live += counts.GetValue(i);
        }
        live = live < 0 ? 0 : (live > rows ? rows : live);
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        pipe_.InitBuffer(in_, 2, TILE * sizeof(float));
        pipe_.InitBuffer(out_, 2, TILE * sizeof(float));
        pipe_.InitBuffer(tmp_, TILE * sizeof(float));
        const uint64_t elements = live * width;
        for (uint64_t offset = AscendC::GetBlockIdx() * TILE; offset < elements;
             offset += AscendC::GetBlockNum() * TILE) {
            const uint32_t n = elements - offset < TILE ? elements - offset : TILE;
            auto xLocal = in_.AllocTensor<float>();
            AscendC::DataCopy(xLocal, input_[offset], n);
            in_.EnQue(xLocal);
            xLocal = in_.DeQue<float>();
            auto y = out_.AllocTensor<float>();
            auto tmp = tmp_.Get<float>();
            AscendC::Muls(tmp, xLocal, -1.0f, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tmp, tmp, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmp, tmp, 1.0f, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(y, xLocal, tmp, n);
            out_.EnQue(y);
            in_.FreeTensor(xLocal);
            y = out_.DeQue<float>();
            AscendC::DataCopy(output_[offset], y, n);
            out_.FreeTensor(y);
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_;
    AscendC::GlobalTensor<float> input_, output_;
};
}

extern "C" __global__ __aicore__ void prefill_prefix_silu_kernel(
    GM_ADDR x, GM_ADDR groups, GM_ADDR out, uint32_t rows, uint32_t width, uint32_t count) {
    PrefixSilu op;
    op.Run(x, groups, out, rows, width, count);
}

void prefill_prefix_silu_launch(void* stream, void* x, void* groups, void* out,
                               uint32_t rows, uint32_t width, uint32_t count) {
    constexpr uint32_t VECTOR_BLOCKS = 48;
    prefill_prefix_silu_kernel<<<VECTOR_BLOCKS, nullptr, stream>>>(
        static_cast<uint8_t*>(x), static_cast<uint8_t*>(groups), static_cast<uint8_t*>(out),
        rows, width, count);
}
