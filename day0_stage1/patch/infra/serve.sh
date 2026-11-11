#!/usr/bin/env bash
set -euo pipefail
: "${PORT:?reserved loopback port}"
exec vllm serve /models/dsv41 --served-model-name dsv41-stage1 \
    --api-server-count 1 --host 127.0.0.1 --port "$PORT" \
    --default-chat-template-kwargs '{"thinking_mode":"chat","reasoning_effort":"high"}' \
    --dtype bfloat16 --max-model-len 256 \
    --data-parallel-size 8 --data-parallel-size-local 8 --enable-expert-parallel \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --max-num-seqs 1 --max-num-batched-tokens 256 \
    --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling \
    --kv-cache-memory-bytes 16777216 --generation-config vllm
