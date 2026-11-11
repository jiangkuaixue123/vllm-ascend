#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Replay the eight-rank text service using the installed vllm-ascend distribution.
set -eu
: "${MODEL_DIR:?checkpoint directory}" "${SERVED_MODEL:?service alias}" "${PORT:?reserved port}"
: "${MAX_MODEL_LEN:?per-request token limit}" "${PREFIX_CACHING:?true or false}" "${CPU_BINDING:?true or false}"
case "$PREFIX_CACHING" in
    true) prefix_flag=--enable-prefix-caching ;;
    false) prefix_flag=--no-enable-prefix-caching ;;
    *) exit 2 ;;
esac
case "$CPU_BINDING" in true|false) ;; *) exit 2 ;; esac
set --
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    set -- --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
fi
chat_kwargs=${CHAT_TEMPLATE_KWARGS:-}
if [[ -z "$chat_kwargs" ]]; then
    chat_kwargs='{"thinking_mode":"chat","reasoning_effort":"high","enable_thinking":false}'
fi
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
additional=$(printf '{"enable_cpu_binding":%s,"enable_shared_expert_dp":true,"enable_flashcomm1":true,"enable_dsa_cp":false}' "$CPU_BINDING")
exec vllm serve "$MODEL_DIR" --served-model-name "$SERVED_MODEL" \
    --api-server-count 1 --host 127.0.0.1 --port "$PORT" \
    --dtype bfloat16 --data-parallel-size 8 --data-parallel-size-local 8 \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --enable-expert-parallel --block-size 128 \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[6,12,18,24]}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
    --additional-config "$additional" "$prefix_flag" --enable-chunked-prefill --no-async-scheduling \
    --default-chat-template-kwargs "$chat_kwargs" \
    --enable-prompt-tokens-details --generation-config vllm "$@"
