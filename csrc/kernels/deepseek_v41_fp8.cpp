// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "../attention/inplace_partial_rotary_mul/op_kernel/rotate_interleaved_common.h"

using namespace AscendC;

namespace {
constexpr uint32_t ELEMENTS = 4096;
constexpr uint32_t GROUPS = ELEMENTS / 32;
constexpr uint32_t VECTOR_BLOCKS = 48;

class Qdq {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR tail, GM_ADDR out, uint32_t n,
                              uint32_t totalRows, uint32_t start, uint32_t tp, uint32_t tileElements) {
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        tail_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tail));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        hasTail_ = tail != nullptr;
        n_ = n; totalRows_ = totalRows; start_ = start; tp_ = tp; tileElements_ = tileElements;
        const uint32_t buffers = (n - 1) / tileElements + 1 > VECTOR_BLOCKS ? 2 : 1;
        pipe_.InitBuffer(in_, buffers, ELEMENTS * 2);
        pipe_.InitBuffer(output_, buffers, ELEMENTS * 2);
        pipe_.InitBuffer(value_, ELEMENTS * 4);
        pipe_.InitBuffer(scale_, ELEMENTS * 4);
        pipe_.InitBuffer(bias_, ELEMENTS * 4);
        pipe_.InitBuffer(work_, ELEMENTS * 4);
        pipe_.InitBuffer(words_, ELEMENTS * 2);
        pipe_.InitBuffer(spare_, ELEMENTS * 2);
        pipe_.InitBuffer(maxWord_, GROUPS * 4);
        pipe_.InitBuffer(coeff_, GROUPS * 4);
        pipe_.InitBuffer(compact_, GROUPS * 2);
        pipe_.InitBuffer(compact2_, GROUPS * 2);
        pipe_.InitBuffer(broadcast_, GROUPS * 8 * 4);
        pipe_.InitBuffer(mask_, ELEMENTS / 8);
    }

    __aicore__ inline void InitRotary(GM_ADDR cosine, GM_ADDR sine) {
        cosine_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cosine));
        sine_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sine));
        pipe_.InitBuffer(phase_, 1, 128 * sizeof(float));
        pipe_.InitBuffer(ropeOffsets_, 512 * sizeof(uint32_t));
        auto indices = ropeOffsets_.Get<int32_t>();
        for (uint32_t i = 0; i < 8; ++i) indices.SetValue(i, i ^ 1);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        for (uint32_t span = 8; span < 512; span *= 2) {
            Adds(indices[span], indices, static_cast<int32_t>(span), span);
            PipeBarrier<PIPE_V>();
        }
        Muls(indices, indices, int32_t(sizeof(float)), 512);
        PipeBarrier<PIPE_V>();
    }

    template<bool PACK, bool ROTARY = false>
    __aicore__ inline void Process() {
        const uint32_t tiles = (n_ - 1) / tileElements_ + 1;
        if (GetBlockIdx() >= tiles) return;
        CopyIn<PACK>(GetBlockIdx());
        for (uint32_t tile = GetBlockIdx(); tile < tiles;
             tile += GetBlockNum()) {
            uint32_t source = tile * tileElements_;
            uint32_t destination = source;
            uint32_t count = n_ - source < tileElements_ ? n_ - source : tileElements_;
            auto input = in_.DeQue<bfloat16_t>();
            // DeQue first keeps depth one while the second buffer is prefetched.
            if (tile + GetBlockNum() < tiles) CopyIn<PACK>(tile + GetBlockNum());
            if constexpr (PACK) {
                const uint32_t row = tile / 8, head = tile % 8 * 8;
                const uint32_t headsPerRank = 64 / tp_;
                destination = ((head / headsPerRank * totalRows_ + start_ + row) * headsPerRank
                               + head % headsPerRank) * 512;
            }
            auto result = output_.AllocTensor<bfloat16_t>();
            if constexpr (ROTARY) RotateTail(input, tile / 8);
            Quantize(input, result, count);
            output_.EnQue(result);
            in_.FreeTensor(input);
            result = output_.DeQue<bfloat16_t>();
            DataCopy(out_[destination], result, count);
            output_.FreeTensor(result);
        }
    }

private:
    __aicore__ inline void RotateTail(LocalTensor<bfloat16_t> input, uint32_t row) {
        auto phase = phase_.AllocTensor<float>();
        DataCopy(phase, cosine_[row * 64], 64);
        DataCopy(phase[64], sine_[row * 64], 64);
        phase_.EnQue(phase);
        phase = phase_.DeQue<float>();
        auto dense = words_.Get<uint16_t>();
        auto value = value_.Get<float>();
        auto rotated = work_.Get<float>();
        // Eight heads share the same 64 FP32 phases; only their BF16 tail is gathered.
        Copy(dense, input[448].ReinterpretCast<uint16_t>(), uint64_t(64), uint8_t(8), {1, 1, 4, 32});
        PipeBarrier<PIPE_V>();
        Cast(value, dense.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, 512);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams broadcast{1, 1, 1, 8, 8, 0};
        Mul(rotated, value, phase, uint64_t(64), uint8_t(8), broadcast);
        PipeBarrier<PIPE_V>();
        Gather(value, value, ropeOffsets_.Get<uint32_t>(), 0, 512);
        PipeBarrier<PIPE_V>();
        Mul(value, value, phase[64], uint64_t(64), uint8_t(8), broadcast);
        PipeBarrier<PIPE_V>();
        // Inverse interleave changes the odd partner sign, after the FP32 product.
        RotateInterleavedN::InterleavedInversion(value, 512, true);
        PipeBarrier<PIPE_V>();
        Add(rotated, rotated, value, 512);
        PipeBarrier<PIPE_V>();
        Cast(dense.ReinterpretCast<bfloat16_t>(), rotated, RoundMode::CAST_RINT, 512);
        PipeBarrier<PIPE_V>();
        Copy(input[448].ReinterpretCast<uint16_t>(), dense, uint64_t(64), uint8_t(8), {1, 1, 32, 4});
        PipeBarrier<PIPE_V>();
        phase_.FreeTensor(phase);
    }

    template<bool PACK>
    __aicore__ inline void CopyIn(uint32_t tile) {
        const uint32_t source = tile * tileElements_;
        const uint32_t count = n_ - source < tileElements_ ? n_ - source : tileElements_;
        auto input = in_.AllocTensor<bfloat16_t>();
        if constexpr (PACK) {
            if (hasTail_) {
                const uint32_t row = tile / 8, head = tile % 8 * 8;
                // Body and replacement tail have disjoint DMA destinations.
                DataCopy(input, x_[source], DataCopyParams{8, 28, 4, 4});
                DataCopy(input[448], tail_[(row * 64 + head) * 64], DataCopyParams{8, 4, 0, 28});
            } else {
                DataCopy(input, x_[source], count);
            }
        } else {
            DataCopy(input, x_[source], count);
        }
        in_.EnQue(input);
    }

    __aicore__ inline void Broadcast32(LocalTensor<float> dst, LocalTensor<float> src, uint32_t groups) {
        auto tmp = broadcast_.Get<float>();
        Brcb(tmp, src, (groups + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        Copy(dst, tmp, uint64_t(32), uint8_t(groups), {1, 0, 4, 1});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Quantize(LocalTensor<bfloat16_t> input,
                                    LocalTensor<bfloat16_t> output, uint32_t n) {
        auto raw = input.ReinterpretCast<uint16_t>();
        auto word = words_.Get<uint16_t>();
        auto spare = spare_.Get<uint16_t>();
        auto value = value_.Get<float>();
        auto scale = scale_.Get<float>();
        auto bias = bias_.Get<float>();
        auto work = work_.Get<float>();
        auto maximum = maxWord_.Get<float>();
        auto coeff = coeff_.Get<float>();
        auto compact = compact_.Get<int16_t>();
        auto compact2 = compact2_.Get<int16_t>();
        auto predicate = mask_.Get<uint8_t>();
        const uint32_t groups = n / 32;
        const uint32_t compareCount = (n + 63) / 64 * 64;
        // A3's count-based CompareScalar only processes complete FP32 repeats.
        if (compareCount != n) {
            Duplicate(work, 0.0f, compareCount);
            Duplicate(scale, 0.0f, compareCount);
            PipeBarrier<PIPE_V>();
        }

        // Numeric positive BF16 words preserve NaN ordering without a floating max of NaNs.
        ShiftLeft(word, raw, uint16_t(1), n);
        PipeBarrier<PIPE_V>();
        ShiftRight(word, word, uint16_t(1), n);
        PipeBarrier<PIPE_V>();
        Cast(work, word.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(maximum, work, 32, groups, 1, 1, 4, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        Cast(compact, maximum, RoundMode::CAST_RINT, groups);
        PipeBarrier<PIPE_V>();
        Mins(compact, compact, int16_t(0x7f80), groups);
        PipeBarrier<PIPE_V>();
        Adds(compact, compact, int16_t(31), groups);
        PipeBarrier<PIPE_V>();
        ShiftRight(compact.ReinterpretCast<uint16_t>(), compact.ReinterpretCast<uint16_t>(), uint16_t(7), groups);
        PipeBarrier<PIPE_V>();
        Maxs(compact, compact, int16_t(113), groups);
        PipeBarrier<PIPE_V>();
        // The tile-wide guard pays off only when many bulk tiles share each core.
        if ((n_ - 1) / tileElements_ + 1 > VECTOR_BLOCKS) {
            // In the finite normal E4M3 domain, power-of-two scaling only
            // changes the exponent; rounding BF16's low four bits is exact.
            Adds(compact2, compact, int16_t(-14), groups);
            PipeBarrier<PIPE_V>();
            ShiftLeft(compact2.ReinterpretCast<uint16_t>(), compact2.ReinterpretCast<uint16_t>(), uint16_t(7), groups);
            PipeBarrier<PIPE_V>();
            Cast(coeff, compact2, RoundMode::CAST_NONE, groups);
            PipeBarrier<PIPE_V>();
            Broadcast32(scale, coeff, groups);
            CompareScalar(predicate, work, 0.0f, CMPMODE::EQ, compareCount);
            PipeBarrier<PIPE_V>();
            Select(work, predicate, scale, work, SELMODE::VSEL_TENSOR_TENSOR_MODE, compareCount);
            PipeBarrier<PIPE_V>();
            Sub(value, work, scale, compareCount);
            PipeBarrier<PIPE_V>();
            Duplicate(bias, 32639.0f, compareCount);
            PipeBarrier<PIPE_V>();
            Sub(bias, bias, work, compareCount);
            PipeBarrier<PIPE_V>();
            Min(value, value, bias, compareCount);
            PipeBarrier<PIPE_V>();
            const uint32_t repeats = compareCount / 64;
            WholeReduceMin(coeff, value, 64, repeats, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
            WholeReduceMin(coeff, coeff, repeats, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            if (coeff.GetValue(0) >= 0.0f) {
                ShiftRight(word, raw, uint16_t(4), n);
                PipeBarrier<PIPE_V>();
                ShiftLeft(word, word, uint16_t(15), n);
                PipeBarrier<PIPE_V>();
                ShiftRight(word, word, uint16_t(15), n);
                PipeBarrier<PIPE_V>();
                Adds(word.ReinterpretCast<int16_t>(), word.ReinterpretCast<int16_t>(), int16_t(7), n);
                PipeBarrier<PIPE_V>();
                Add(word.ReinterpretCast<int16_t>(), raw.ReinterpretCast<int16_t>(), word.ReinterpretCast<int16_t>(), n);
                PipeBarrier<PIPE_V>();
                ShiftRight(word, word, uint16_t(4), n);
                PipeBarrier<PIPE_V>();
                ShiftLeft(output.ReinterpretCast<uint16_t>(), word, uint16_t(4), n);
                PipeBarrier<PIPE_V>();
                return;
            }

        }

        Adds(compact2, compact, int16_t(-8), groups);
        PipeBarrier<PIPE_V>();
        ShiftLeft(compact2.ReinterpretCast<uint16_t>(), compact2.ReinterpretCast<uint16_t>(), uint16_t(7), groups);
        PipeBarrier<PIPE_V>();
        Cast(coeff, compact2.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, groups);
        PipeBarrier<PIPE_V>();
        Broadcast32(scale, coeff, groups);

        Duplicate(compact2, int16_t(262), groups);
        PipeBarrier<PIPE_V>();
        Sub(compact2, compact2, compact, groups);
        PipeBarrier<PIPE_V>();
        ShiftLeft(compact2.ReinterpretCast<uint16_t>(), compact2.ReinterpretCast<uint16_t>(), uint16_t(7), groups);
        PipeBarrier<PIPE_V>();
        Cast(coeff, compact2.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, groups);
        PipeBarrier<PIPE_V>();
        Broadcast32(bias, coeff, groups);
        Cast(value, input, RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Mul(value, value, bias, n);
        PipeBarrier<PIPE_V>();
        Abs(value, value, n);
        PipeBarrier<PIPE_V>();

        // Power-of-two scaling preserves BF16 mantissas at every nonzero E4M3 boundary.
        Cast(word.ReinterpretCast<bfloat16_t>(), value, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        ShiftRight(word, word, uint16_t(7), n);
        PipeBarrier<PIPE_V>();
        ShiftLeft(word, word, uint16_t(7), n);
        PipeBarrier<PIPE_V>();
        Maxs(word.ReinterpretCast<int16_t>(), word.ReinterpretCast<int16_t>(), int16_t(121 << 7), n);
        PipeBarrier<PIPE_V>();
        Adds(word.ReinterpretCast<int16_t>(), word.ReinterpretCast<int16_t>(), int16_t(20 << 7), n);
        PipeBarrier<PIPE_V>();
        Cast(bias, word.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        Add(work, bias, value, n);
        PipeBarrier<PIPE_V>();
        Sub(work, work, bias, n);
        PipeBarrier<PIPE_V>();
        Mul(work, work, scale, n);
        PipeBarrier<PIPE_V>();
        Cast(output, work, RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        ShiftRight(word, raw, uint16_t(15), n);
        PipeBarrier<PIPE_V>();
        ShiftLeft(word, word, uint16_t(15), n);
        PipeBarrier<PIPE_V>();
        Or(output.ReinterpretCast<uint16_t>(), output.ReinterpretCast<uint16_t>(), word, n);
        PipeBarrier<PIPE_V>();

        // Match native Inf/NaN-group canonicalization, including signed finite zeros.
        ShiftLeft(spare, raw, uint16_t(1), n);
        PipeBarrier<PIPE_V>();
        ShiftRight(spare, spare, uint16_t(1), n);
        PipeBarrier<PIPE_V>();
        Cast(work, spare.ReinterpretCast<int16_t>(), RoundMode::CAST_NONE, n);
        PipeBarrier<PIPE_V>();
        ShiftRight(spare, raw, uint16_t(15), n);
        PipeBarrier<PIPE_V>();
        ShiftLeft(spare, spare, uint16_t(15), n);
        PipeBarrier<PIPE_V>();
        Duplicate(word, uint16_t(0x7fff), n);
        PipeBarrier<PIPE_V>();
        CompareScalar(predicate, work, 0.0f, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        Select(spare.ReinterpretCast<half>(), predicate, word.ReinterpretCast<half>(),
               spare.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(predicate, work, 32640.0f, CMPMODE::GT, compareCount);
        PipeBarrier<PIPE_V>();
        Select(spare.ReinterpretCast<half>(), predicate, word.ReinterpretCast<half>(),
               spare.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        Broadcast32(scale, maximum, groups);
        CompareScalar(predicate, scale, 32640.0f, CMPMODE::GT, compareCount);
        PipeBarrier<PIPE_V>();
        Select(output.ReinterpretCast<half>(), predicate, spare.ReinterpretCast<half>(),
               output.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(predicate, scale, 32640.0f, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        Select(output.ReinterpretCast<half>(), predicate, word.ReinterpretCast<half>(),
               output.ReinterpretCast<half>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> in_;
    TQue<TPosition::VECOUT, 1> output_;
    TQue<TPosition::VECIN, 1> phase_;
    TBuf<TPosition::VECCALC> ropeOffsets_;
    GlobalTensor<float> cosine_, sine_;
    TBuf<TPosition::VECCALC> value_, scale_, bias_, work_, words_, spare_;
    TBuf<TPosition::VECCALC> maxWord_, coeff_, compact_, compact2_, broadcast_, mask_;
    GlobalTensor<bfloat16_t> x_, tail_, out_;
    uint32_t n_, totalRows_, start_, tp_, tileElements_;
    bool hasTail_;
};
}

extern "C" __global__ __aicore__ void deepseek_v41_fp8_qdq_kernel(
    GM_ADDR x, GM_ADDR out, uint32_t n, uint32_t tileElements) {
    Qdq op;
    op.Init(x, nullptr, out, n, 0, 0, 0, tileElements);
    op.Process<false>();
}

extern "C" __global__ __aicore__ void deepseek_v41_fp8_pack_kernel(
    GM_ADDR x, GM_ADDR tail, GM_ADDR out, uint32_t n, uint32_t totalRows, uint32_t start, uint32_t tp) {
    Qdq op;
    op.Init(x, tail, out, n, totalRows, start, tp, ELEMENTS);
    op.Process<true>();
}

void deepseek_v41_fp8_qdq_launch(void* stream, void* x, void* out, uint32_t n) {
    uint32_t tileElements = 256;
    while (tileElements < ELEMENTS && tileElements * VECTOR_BLOCKS < n) tileElements *= 2;
    const uint32_t tiles = (n - 1) / tileElements + 1;
    const uint32_t blocks = tiles < VECTOR_BLOCKS ? tiles : VECTOR_BLOCKS;
    deepseek_v41_fp8_qdq_kernel<<<blocks, nullptr, stream>>>(
        static_cast<uint8_t*>(x), static_cast<uint8_t*>(out), n, tileElements);
}

void deepseek_v41_fp8_pack_launch(void* stream, void* x, void* tail, void* out,
                                uint32_t n, uint32_t totalRows, uint32_t start, uint32_t tp) {
    const uint32_t tiles = (n - 1) / ELEMENTS + 1;
    const uint32_t blocks = tiles < VECTOR_BLOCKS ? tiles : VECTOR_BLOCKS;
    deepseek_v41_fp8_pack_kernel<<<blocks, nullptr, stream>>>(static_cast<uint8_t*>(x), static_cast<uint8_t*>(tail),
        static_cast<uint8_t*>(out), n, totalRows, start, tp);
}

extern "C" __global__ __aicore__ void deepseek_v41_fp8_rotary_pack_kernel(
    GM_ADDR x, GM_ADDR cosine, GM_ADDR sine, GM_ADDR out, uint32_t n, uint32_t rows, uint32_t tp) {
    Qdq op;
    op.Init(x, nullptr, out, n, rows, 0, tp, ELEMENTS);
    op.InitRotary(cosine, sine);
    op.Process<true, true>();
}

void deepseek_v41_fp8_rotary_pack_launch(void* stream, void* x, void* cosine, void* sine, void* out,
                                        uint32_t n, uint32_t rows, uint32_t tp) {
    const uint32_t tiles = (n - 1) / ELEMENTS + 1;
    const uint32_t blocks = tiles < VECTOR_BLOCKS ? tiles : VECTOR_BLOCKS;
    deepseek_v41_fp8_rotary_pack_kernel<<<blocks, nullptr, stream>>>(static_cast<uint8_t*>(x),
        static_cast<uint8_t*>(cosine), static_cast<uint8_t*>(sine), static_cast<uint8_t*>(out), n, rows, tp);
}
