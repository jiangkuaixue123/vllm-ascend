// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "../mc2/dispatch_ffn_combine/op_kernel/utils/route_before_down_quant.hpp"

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

class PrefixRouteQuant {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR factor, GM_ADDR groups, GM_ADDR out,
                              GM_ADDR scales, uint32_t rows, uint32_t width,
                              uint32_t count, float limit) {
        AscendC::GlobalTensor<int64_t> counts;
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groups));
        AscendC::GlobalTensor<float> factors, outputScales;
        factors.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(factor));
        outputScales.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
        AscendC::GlobalTensor<bfloat16_t> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        AscendC::GlobalTensor<int8_t> output;
        output.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
        pipe_.InitBuffer(meta_, 224);
        auto metadata = meta_.Get<int64_t>();
        int64_t live = count == 0 ? rows : 0;
        if (count) {
            AscendC::DataCopyPad(metadata, counts, {1, static_cast<uint32_t>(count * sizeof(int64_t)), 0, 0, 0}, {});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t i = 0; i < count; ++i) live += metadata.GetValue(i);
        }
        live = live < 0 ? 0 : (live > rows ? rows : live);
        pipe_.InitBuffer(in_, 2, 2 * width * sizeof(bfloat16_t));
        pipe_.InitBuffer(out_, 2, width + 32);
        pipe_.InitBuffer(gate_, width * sizeof(float));
        pipe_.InitBuffer(up_, width * sizeof(float));
        pipe_.InitBuffer(tmp_, width * sizeof(float));
        pipe_.InitBuffer(abs_, width * sizeof(float));
        pipe_.InitBuffer(reduce_, width * sizeof(float));
        pipe_.InitBuffer(max_, 32);
        pipe_.InitBuffer(rounded_, width * sizeof(bfloat16_t));
        for (uint32_t row = AscendC::GetBlockIdx(); row < live; row += AscendC::GetBlockNum()) {
            auto values = in_.AllocTensor<bfloat16_t>();
            AscendC::DataCopy(values, input[row * 2 * width], 2 * width);
            in_.EnQue(values);
            values = in_.DeQue<bfloat16_t>();
            auto gate = gate_.Get<float>();
            auto up = up_.Get<float>();
            auto tmp = tmp_.Get<float>();
            auto absolute = abs_.Get<float>();
            auto maximum = max_.Get<float>();
            auto rounded = rounded_.Get<bfloat16_t>();
            auto quantized = out_.AllocTensor<int8_t>();
            AscendC::Cast(gate, values, AscendC::RoundMode::CAST_NONE, width);
            AscendC::Cast(up, values[width], AscendC::RoundMode::CAST_NONE, width);
            AscendC::PipeBarrier<PIPE_V>();
            in_.FreeTensor(values);
            if (limit > 0) {
                AscendC::Mins(gate, gate, limit, width);
                AscendC::Mins(up, up, limit, width);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Maxs(up, up, -limit, width);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Muls(tmp, gate, -1.0f, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tmp, tmp, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmp, tmp, 1.0f, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(gate, gate, tmp, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(gate, gate, up, width);
            auto scalar = meta_.Get<float>()[48];
            AscendC::DataCopyPad(scalar, factors[row], {1, sizeof(float), 0, 0, 0}, {});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            const float route = scalar.GetValue(0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(gate, gate, route, width);
            AscendC::PipeBarrier<PIPE_V>();
            // W2 quantization consumes the original BF16 activation, not its FP32 producer.
            AscendC::Cast(rounded, gate, AscendC::RoundMode::CAST_RINT, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(gate, rounded, AscendC::RoundMode::CAST_NONE, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Abs(absolute, gate, width);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceMax<float>(maximum, absolute, reduce_.Get<float>(), width, false);
            AscendC::PipeBarrier<PIPE_V>();
            const float scale = VllmAscend::QuantizeRouteBeforeDown(gate, absolute, maximum, quantized, width);
            auto scaleBlock = quantized.ReinterpretCast<float>()[width / sizeof(float)];
            scaleBlock.SetValue(0, scale);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            out_.EnQue(quantized);
            quantized = out_.DeQue<int8_t>();
            AscendC::DataCopy(output[row * width], quantized, width);
            AscendC::DataCopyPad(outputScales[row], quantized.ReinterpretCast<float>()[width / sizeof(float)],
                                {1, sizeof(float), 0, 0, 0});
            out_.FreeTensor(quantized);
        }
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> meta_, gate_, up_, tmp_, abs_, reduce_, max_, rounded_;
};
class OrderedCombine {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR factor, GM_ADDR ids, GM_ADDR owned,
                              GM_ADDR out, GM_ADDR destinations, uint32_t rows,
                              uint32_t expanded, uint32_t outputRows) {
        using namespace AscendC;
        GlobalTensor<bfloat16_t> input;
        GlobalTensor<float> factors, weights, output;
        GlobalTensor<int32_t> indices, mapping;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        factors.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(factor));
        weights.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(owned));
        indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ids));
        mapping.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(destinations));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        pipe_.InitBuffer(in_, 2, H * sizeof(bfloat16_t));
        pipe_.InitBuffer(out_, 2, H * sizeof(float));
        pipe_.InitBuffer(tmp_, H * sizeof(float));
        pipe_.InitBuffer(meta_, 896);
        auto index = meta_.Get<int32_t>();
        auto weight = meta_.Get<float>()[96];
        auto destination = meta_.Get<int32_t>()[192];
        auto scale = meta_.Get<float>()[208];
        for (uint32_t begin = GetBlockIdx() * B; begin < rows; begin += GetBlockNum() * B) {
            const uint32_t count = rows - begin < B ? rows - begin : B;
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopyPad(index, indices[begin * K], {1, static_cast<uint32_t>(count * K * sizeof(int32_t)), 0, 0, 0}, {});
            DataCopyPad(weight, weights[begin * K], {1, static_cast<uint32_t>(count * K * sizeof(float)), 0, 0, 0}, {});
            if (destinations) DataCopyPad(destination, mapping[begin], {1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0}, {});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t row = 0; row < count; ++row) {
                const int32_t target = destinations ? destination.GetValue(row) : begin + row;
                if (target < 0 || target >= outputRows) continue;
                auto total = out_.AllocTensor<float>();
                Duplicate(total, 0.0f, H);
                PipeBarrier<PIPE_V>();
                for (uint32_t slot = 0; slot < K; ++slot) {
                    int64_t source = index.GetValue(row * K + slot);
                    source = source < 0 ? -source : source;
                    const float coefficient = weight.GetValue(row * K + slot);
                    if (source >= expanded || coefficient == 0.0f) continue;
                    SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
                    DataCopyPad(scale, factors[source], {1, sizeof(float), 0, 0, 0}, {});
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                    auto value = tmp_.Get<float>();
                    if (scale.GetValue(0) != 0.0f) {
                        auto local = in_.AllocTensor<bfloat16_t>();
                        DataCopy(local, input[source * H], H);
                        in_.EnQue(local);
                        local = in_.DeQue<bfloat16_t>();
                        Cast(value, local, RoundMode::CAST_NONE, H);
                        PipeBarrier<PIPE_V>();
                        in_.FreeTensor(local);
                    } else {
                        Duplicate(value, 0.0f, H);
                        PipeBarrier<PIPE_V>();
                    }
                    Muls(value, value, coefficient, H);
                    PipeBarrier<PIPE_V>();
                    // Preserve slot zero assignment and every later FP32 addition.
                    if (slot == 0) Muls(total, value, 1.0f, H);
                    else Add(total, total, value, H);
                    PipeBarrier<PIPE_V>();
                }
                out_.EnQue(total);
                total = out_.DeQue<float>();
                DataCopy(output[static_cast<uint64_t>(target) * H], total, H);
                out_.FreeTensor(total);
            }
        }
    }
private:
    static constexpr uint32_t H = 5120, K = 6, B = 16;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_, meta_;
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

extern "C" __global__ __aicore__ void prefill_prefix_route_quant_kernel(
    GM_ADDR x, GM_ADDR factor, GM_ADDR groups, GM_ADDR out, GM_ADDR scales,
    uint32_t rows, uint32_t width, uint32_t count, float limit) {
    PrefixRouteQuant op;
    op.Run(x, factor, groups, out, scales, rows, width, count, limit);
}

void prefill_prefix_route_quant_launch(void* stream, void* x, void* factor, void* groups,
                                     void* out, void* scales, uint32_t rows, uint32_t width,
                                     uint32_t count, float limit) {
    prefill_prefix_route_quant_kernel<<<48, nullptr, stream>>>(
        static_cast<uint8_t*>(x), static_cast<uint8_t*>(factor), static_cast<uint8_t*>(groups),
        static_cast<uint8_t*>(out), static_cast<uint8_t*>(scales), rows, width, count, limit);
}

extern "C" __global__ __aicore__ void prefill_ordered_combine_kernel(
    GM_ADDR x, GM_ADDR factor, GM_ADDR ids, GM_ADDR owned, GM_ADDR out,
    GM_ADDR destinations, uint32_t rows, uint32_t expanded, uint32_t outputRows) {
    OrderedCombine op;
    op.Run(x, factor, ids, owned, out, destinations, rows, expanded, outputRows);
}

void prefill_ordered_combine_launch(void* stream, void* x, void* factor, void* ids,
                                    void* owned, void* out, void* destinations,
                                    uint32_t rows, uint32_t expanded, uint32_t outputRows) {
    prefill_ordered_combine_kernel<<<48, nullptr, stream>>>(
        static_cast<uint8_t*>(x), static_cast<uint8_t*>(factor), static_cast<uint8_t*>(ids),
        static_cast<uint8_t*>(owned), static_cast<uint8_t*>(out), static_cast<uint8_t*>(destinations),
        rows, expanded, outputRows);
}
