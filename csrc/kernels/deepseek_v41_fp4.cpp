// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t TILE = 2048;
constexpr uint32_t CORES = 48;

class Fp4 {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t n, uint32_t tile, bool e4) {
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        n_ = n; tile_ = tile; e4_ = e4; group_ = e4 ? 16 : 32;
        pipe_.InitBuffer(in_, 1, TILE * 2);
        pipe_.InitBuffer(out_, 1, TILE * 2);
        pipe_.InitBuffer(value_, TILE * 4);
        pipe_.InitBuffer(scale_, TILE * 4);
        pipe_.InitBuffer(result_, TILE * 4);
        pipe_.InitBuffer(temp_, TILE * 4);
        pipe_.InitBuffer(level_, TILE * 4);
        pipe_.InitBuffer(words_, TILE * 2);
        pipe_.InitBuffer(spare_, TILE * 2);
        pipe_.InitBuffer(maximum_, TILE / 16 * 4);
        pipe_.InitBuffer(coeff_, TILE / 16 * 4);
        pipe_.InitBuffer(compact_, TILE / 16 * 2);
        pipe_.InitBuffer(broadcast_, TILE / 16 * 8 * 4);
        pipe_.InitBuffer(mask_, TILE / 8);
    }

    __aicore__ inline void Process() {
        for (uint32_t base = GetBlockIdx() * tile_; base < n_; base += GetBlockNum() * tile_) {
            const uint32_t count = n_ - base < tile_ ? n_ - base : tile_;
            auto input = in_.AllocTensor<bfloat16_t>();
            DataCopy(input, x_[base], count);
            in_.EnQue(input); input = in_.DeQue<bfloat16_t>();
            auto output = out_.AllocTensor<bfloat16_t>();
            Quantize(input, output, count);
            out_.EnQue(output); in_.FreeTensor(input);
            output = out_.DeQue<bfloat16_t>();
            DataCopy(y_[base], output, count); out_.FreeTensor(output);
        }
    }

private:
    __aicore__ inline void Broadcast(LocalTensor<float> dst, LocalTensor<float> src, uint32_t groups) {
        auto tmp = broadcast_.Get<float>();
        Brcb(tmp, src, (groups + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        Copy(dst, tmp, uint64_t(group_), uint8_t(groups), {1, 0, uint8_t(group_ / 8), 1});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Quantize(LocalTensor<bfloat16_t> input,
                                    LocalTensor<bfloat16_t> output, uint32_t count) {
        auto raw = input.ReinterpretCast<uint16_t>();
        auto words = words_.Get<uint16_t>(); auto spare = spare_.Get<uint16_t>();
        auto value = value_.Get<float>(); auto scale = scale_.Get<float>();
        auto result = result_.Get<float>(); auto temp = temp_.Get<float>(); auto level = level_.Get<float>();
        auto maximum = maximum_.Get<float>(); auto coeff = coeff_.Get<float>();
        auto compact = compact_.Get<int16_t>(); auto mask = mask_.Get<uint8_t>();
        const uint32_t groups = count / group_, compareCount = (count + 63) / 64 * 64;
        Duplicate(value, 0.0f, compareCount); Duplicate(scale, 0.0f, compareCount);
        Duplicate(temp, 0.0f, compareCount); Duplicate(result, 0.0f, compareCount);
        PipeBarrier<PIPE_V>();
        ShiftLeft(words, raw, uint16_t(1), count); PipeBarrier<PIPE_V>();
        ShiftRight(words, words, uint16_t(1), count); PipeBarrier<PIPE_V>();
        Cast(value, words.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, count); PipeBarrier<PIPE_V>();
        WholeReduceMax(maximum, value, group_, groups, 1, 1, group_ / 8, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        Cast(compact, maximum, RoundMode::CAST_RINT, groups); PipeBarrier<PIPE_V>();
        if (e4_) {
            Cast(coeff, compact.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, groups); PipeBarrier<PIPE_V>();
            Maxs(coeff, coeff, 6.0f / 512, groups); PipeBarrier<PIPE_V>();
            Mins(coeff, coeff, 2784.0f, groups); PipeBarrier<PIPE_V>();
            Muls(coeff, coeff, 1.0f / 6, groups); PipeBarrier<PIPE_V>();
            // BF16 exponent approximation cannot cross an E4M3 midpoint here.
            Cast(compact.ReinterpretCast<bfloat16_t>(), coeff, RoundMode::CAST_RINT, groups); PipeBarrier<PIPE_V>();
            ShiftRight(compact.ReinterpretCast<uint16_t>(), compact.ReinterpretCast<uint16_t>(), uint16_t(7), groups); PipeBarrier<PIPE_V>();
            ShiftLeft(compact.ReinterpretCast<uint16_t>(), compact.ReinterpretCast<uint16_t>(), uint16_t(7), groups); PipeBarrier<PIPE_V>();
            Maxs(compact, compact, int16_t(121 << 7), groups); PipeBarrier<PIPE_V>();
            Adds(compact, compact, int16_t(20 << 7), groups); PipeBarrier<PIPE_V>();
            Cast(level, compact.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, groups); PipeBarrier<PIPE_V>();
            Add(temp, coeff, level, groups); PipeBarrier<PIPE_V>();
            Sub(coeff, temp, level, groups); PipeBarrier<PIPE_V>();
        } else {
            // ceil(log2(amax/6)): the BF16 mantissa carries only above 1.5.
            Adds(compact, compact, int16_t(63), groups); PipeBarrier<PIPE_V>();
            ShiftRight(compact.ReinterpretCast<uint16_t>(), compact.ReinterpretCast<uint16_t>(), uint16_t(7), groups); PipeBarrier<PIPE_V>();
            Maxs(compact, compact, int16_t(3), groups); PipeBarrier<PIPE_V>();
            Adds(compact, compact, int16_t(-2), groups); PipeBarrier<PIPE_V>();
            ShiftLeft(compact.ReinterpretCast<uint16_t>(), compact.ReinterpretCast<uint16_t>(), uint16_t(7), groups); PipeBarrier<PIPE_V>();
            Cast(coeff, compact.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, groups); PipeBarrier<PIPE_V>();
        }
        Broadcast(scale, coeff, groups);
        Cast(value, input, RoundMode::CAST_NONE, count); PipeBarrier<PIPE_V>();
        Abs(value, value, count); PipeBarrier<PIPE_V>();
        constexpr float boundaries[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        constexpr float levels[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        for (uint32_t i = 0; i < 7; ++i) {
            // Both operands are exact dyadic products; even midpoint ties stay below.
            Muls(temp, scale, boundaries[i], count); PipeBarrier<PIPE_V>();
            Compare(mask, value, temp, (i == 1 || i == 3 || i == 5) ? CMPMODE::GE : CMPMODE::GT, compareCount);
            PipeBarrier<PIPE_V>();
            Duplicate(level, levels[i], count); PipeBarrier<PIPE_V>();
            Select(result, mask, level, result, SELMODE::VSEL_TENSOR_TENSOR_MODE, count); PipeBarrier<PIPE_V>();
        }
        Mul(result, result, scale, count); PipeBarrier<PIPE_V>();
        Cast(output, result, RoundMode::CAST_RINT, count); PipeBarrier<PIPE_V>();
        ShiftRight(spare, raw, uint16_t(15), count); PipeBarrier<PIPE_V>();
        ShiftLeft(spare, spare, uint16_t(15), count); PipeBarrier<PIPE_V>();
        Or(output.ReinterpretCast<uint16_t>(), output.ReinterpretCast<uint16_t>(), spare, count); PipeBarrier<PIPE_V>();
        Duplicate(words, uint16_t(0x7fff), count); PipeBarrier<PIPE_V>();
        if (!e4_) {
            // NaN-scale groups preserve signed finite zeros; zero/NaN lanes stay NaN.
            ShiftLeft(spare, raw, uint16_t(1), count); PipeBarrier<PIPE_V>();
            ShiftRight(spare, spare, uint16_t(1), count); PipeBarrier<PIPE_V>();
            Cast(value, spare.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, count); PipeBarrier<PIPE_V>();
            ShiftRight(spare, raw, uint16_t(15), count); PipeBarrier<PIPE_V>();
            ShiftLeft(spare, spare, uint16_t(15), count); PipeBarrier<PIPE_V>();
            CompareScalar(mask, value, 0.0f, CMPMODE::EQ, compareCount); PipeBarrier<PIPE_V>();
            Select(spare.ReinterpretCast<half>(), mask, words.ReinterpretCast<half>(), spare.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count); PipeBarrier<PIPE_V>();
            CompareScalar(mask, value, 32640.0f, CMPMODE::GT, compareCount); PipeBarrier<PIPE_V>();
            Select(spare.ReinterpretCast<half>(), mask, words.ReinterpretCast<half>(), spare.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count); PipeBarrier<PIPE_V>();
        }
        Broadcast(temp, maximum, groups);
        if (!e4_) {
            CompareScalar(mask, temp, 32640.0f, CMPMODE::GT, compareCount); PipeBarrier<PIPE_V>();
            Select(output.ReinterpretCast<half>(), mask, spare.ReinterpretCast<half>(), output.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count); PipeBarrier<PIPE_V>();
        }
        // 2784/6 is the last E4M3 scale midpoint (464 ->448); larger scales are NaN.
        CompareScalar(mask, temp, e4_ ? 17710.0f : 32640.0f, e4_ ? CMPMODE::GT : CMPMODE::EQ, compareCount); PipeBarrier<PIPE_V>();
        Select(output.ReinterpretCast<half>(), mask, words.ReinterpretCast<half>(), output.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, count); PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> in_;
    TQue<TPosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> value_, scale_, result_, temp_, level_, words_, spare_;
    TBuf<TPosition::VECCALC> maximum_, coeff_, compact_, broadcast_, mask_;
    GlobalTensor<bfloat16_t> x_, y_;
    uint32_t n_, tile_, group_;
    bool e4_;
};
}

extern "C" __global__ __aicore__ void deepseek_v41_fp4_qdq_kernel(
    GM_ADDR x, GM_ADDR out, uint32_t n, uint32_t tile, bool e4) {
    Fp4 op; op.Init(x, out, n, tile, e4); op.Process();
}

void deepseek_v41_fp4_qdq_launch(void* stream, void* x, void* out, uint32_t n, bool e4) {
    uint32_t tile = 256;
    while (tile < TILE && tile * CORES < n) tile *= 2;
    uint32_t blocks = (n - 1) / tile + 1;
    blocks = blocks < CORES ? blocks : CORES;
    deepseek_v41_fp4_qdq_kernel<<<blocks, nullptr, stream>>>(
        static_cast<uint8_t*>(x), static_cast<uint8_t*>(out), n, tile, e4);
}
