# Hy4-Flash-FP8 long-context decode on eight Ascend 950DT devices

Run the commands below from the root of the installed vLLM Ascend checkout inside the `quay.io/ascend/vllm-ascend:hy4-flash-a5-openeuler` image. Use the matching vLLM core at `478dc119f61259aefd945bd7973c164c4841093c`. Mount the Hy4-Flash-FP8 checkpoint at `/models/Hy4-Flash-FP8`.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AICPU_TS
export HCCL_SOCKET_IFNAME=enp34s0f1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

recipe=examples/hy4_flash_a5/serve-overrides.json
context_length=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["--max-model-len"])' "$recipe")
additional_config=$(python3 -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1]))["--additional-config"]))' "$recipe")

vllm serve /models/Hy4-Flash-FP8 \
  --host 127.0.0.1 --port 8024 --served-model-name hy4-flash --trust-remote-code \
  --tensor-parallel-size 8 --data-parallel-size 1 --enable-expert-parallel \
  --max-model-len "$context_length" --max-num-seqs 32 --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.85 --kv-cache-dtype fp8 \
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}' \
  --speculative-config '{"method":"dfly","num_speculative_tokens":5}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2,3,4,5,6,7,8,16,32],"max_decode_cudagraph_capture_tokens":512}' \
  --additional-config "$additional_config" \
  --no-enable-prefix-caching --async-scheduling --seed 1024
```

The tested host uses the network interface `enp34s0f1`. Set `HCCL_SOCKET_IFNAME` to the appropriate interface on another host.

The recipe reserves 131,584 positions for requests with 130,048 input and 1,024 output tokens so DFly can still verify five proposals at the end. Its `--additional-config` enables the following decode optimizations.

| Option | Effect |
| --- | --- |
| `hy4flash_native_sparse_c8_verify` | Runs C8 sparse attention on the six target verification rows of a single request. |
| `hy4flash_fuse_ihc_pre` | Fuses iHC residual-channel gating for small decode batches. |
| `hy4flash_share_verify_layout` | Reuses the current request's positions and window rows across target layers in a captured forward. |
| `hy4flash_shard_indexpool_scores` | Divides long-context IndexPool scoring across eight tensor ranks and preserves the global top-k order. |
| `hy4flash_fuse_indexpool_rotate_encode` | Encodes short rotated IndexPool queries and keys in one Hadamard and E4M3 rounding kernel, using scale boundaries obtained on the device. |

The fixed-length test measures one client request at a time. Each reported TPOT divides the elapsed time between its first and last generated-token events by 1,023. The reported P50 is the median across twenty distinct requests. The accuracy run uses all thirty AIME2026 questions at concurrency sixteen. [Per-request timings and hardware results](https://github.com/InferMatrixLab/InferRSI/issues/613)
