#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/d4ac431/.venv/bin:$PATH
export OMP_NUM_THREADS=8
export HCCL_CONNECT_TIMEOUT=120
export HCCL_EXEC_TIMEOUT=600
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export GLOO_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
mkdir -p /workspace
cd /workspace
exec vllm serve /models/deepseek-v41 \
  --served-model-name deepseek-v4.1 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --quantization deepseek_v4_fp8 --dtype bfloat16 \
  --max-model-len 65536 --block-size 128 \
  --max-num-batched-tokens 512 --max-num-seqs 16 \
  --gpu-memory-utilization 0.93 \
  --chat-template /deployment/reference/chat_template.jinja \
  --chat-template-content-format openai --reasoning-parser deepseek_v3 \
  --limit-mm-per-prompt '{"image":1}' \
  --enable-prefix-caching --enable-prompt-tokens-details \
  --additional-config '{"enable_dsa_cp":true,"enable_flashcomm1":false,"enable_shared_expert_dp":true,"deepseek_v41_flatten":true,"deepseek_v41_flatten_source_dir":"/deployment/experiments/fusion-20261109/selected"}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[24,48]}' "$@"
