// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef VLLM_ASCEND_ROUTE_BEFORE_DOWN_QUANT_HPP
#define VLLM_ASCEND_ROUTE_BEFORE_DOWN_QUANT_HPP

#include "kernel_operator.h"

namespace VllmAscend {

// maxAndScale contains ReduceMax's result and has room for one FP32 block.
// scratch has count FP32 elements and may be reinterpreted while narrowing.
// Match CANN's generic BF16 DynamicQuant: vector 127/max, multiplication,
// RINT to an integer, and max*(1/127) for the separately returned scale.
// Div(input, max/127) and a scalar reciprocal differ at rounding boundaries.
__aicore__ inline float QuantizeRouteBeforeDown(
    const AscendC::LocalTensor<float>& input,
    const AscendC::LocalTensor<float>& scratch,
    const AscendC::LocalTensor<float>& maxAndScale,
    const AscendC::LocalTensor<int8_t>& output,
    uint32_t count)
{
    constexpr uint32_t SCALE_BLOCK_ELEMENTS = 8;
    constexpr float QUANT_MAX = 127.0f;
    constexpr float DEQUANT_FACTOR = 1.0f / QUANT_MAX;

    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    const float maxAbs = maxAndScale.GetValue(0);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    if (maxAbs == 0.0f) {
        // Do not form 127/0 or 0*inf. Preserve the vendor's zero scale.
        AscendC::Duplicate(scratch, 0.0f, count);
    } else {
        AscendC::Duplicate(scratch, maxAbs, count);
        AscendC::Duplicate(maxAndScale, QUANT_MAX, SCALE_BLOCK_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(maxAndScale, maxAndScale, scratch, SCALE_BLOCK_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float reciprocal = maxAndScale.GetValue(0);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::Duplicate(scratch, reciprocal, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(scratch, input, scratch, count);
    }
    AscendC::PipeBarrier<PIPE_V>();

    auto integers = scratch.template ReinterpretCast<int32_t>();
    auto halves = scratch.template ReinterpretCast<half>();
    AscendC::Cast(integers, scratch, AscendC::RoundMode::CAST_RINT, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetDeqScale(static_cast<half>(1.0));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(halves, integers, AscendC::RoundMode::CAST_ROUND, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(output, halves, AscendC::RoundMode::CAST_TRUNC, count);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Duplicate(maxAndScale, maxAbs, SCALE_BLOCK_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(maxAndScale, maxAndScale, DEQUANT_FACTOR, SCALE_BLOCK_ELEMENTS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return maxAndScale.GetValue(0);
}

}  // namespace VllmAscend
#endif
