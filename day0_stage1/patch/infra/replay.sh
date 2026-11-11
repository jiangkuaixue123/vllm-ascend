#!/usr/bin/env bash
# Run within the target container after copying the patch, validator and pinned
# HF assets to /stage1. The API is reachable from this container's loopback.
set -euo pipefail
: "${PORT:?reserved port}" "${RUN_NAME:?unique attempt name}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ && "$RUN_NAME" != .* ]]
export VLLM_ASCEND_DSV41_RESULTS_DIR="/stage1/results/$RUN_NAME"
mkdir "$VLLM_ASCEND_DSV41_RESULTS_DIR"
export VLLM_ASCEND_DSV41_TRACE_DIR="$VLLM_ASCEND_DSV41_RESULTS_DIR/numerical"
export VLLM_ASCEND_DSV41_TRACE_INPUTS=/stage1/validator
envrun=(bash /stage1/patch/infra/in-env.sh)
"${envrun[@]}" bash /stage1/patch/infra/serve.sh >"$VLLM_ASCEND_DSV41_RESULTS_DIR/server.log" 2>&1 &
server=$!
trap 'kill -TERM "$server" 2>/dev/null || true; wait "$server" || true' EXIT
api="http://127.0.0.1:$PORT/v1"
deadline=$((SECONDS + 2400))
until curl -fsS --max-time 3 "$api/models" > "$VLLM_ASCEND_DSV41_RESULTS_DIR/models.json.tmp" 2>/dev/null; do
    kill -0 "$server" 2>/dev/null || { wait "$server"; exit 1; }
    (( SECONDS < deadline )) || { echo 'Service startup deadline expired' >&2; exit 1; }
    sleep 3
done
mv "$VLLM_ASCEND_DSV41_RESULTS_DIR"/models.json{.tmp,}
"${envrun[@]}" python /stage1/validator/check_service.py --base-url "$api" --output "$VLLM_ASCEND_DSV41_RESULTS_DIR/template-check.json"
"${envrun[@]}" python /stage1/validator/smoke_chat.py --base-url "$api" --model dsv41-stage1 \
    --max-tokens 32 --timeout 600 --output "$VLLM_ASCEND_DSV41_RESULTS_DIR/smoke.jsonl"
"${envrun[@]}" python /stage1/patch/infra/request-numerical.py --base-url "$api" --output "$VLLM_ASCEND_DSV41_RESULTS_DIR/numerical-response.json"
