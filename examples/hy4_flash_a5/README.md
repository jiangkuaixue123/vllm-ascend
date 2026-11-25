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
| `hy4flash_native_sparse_c8_verify` | Uses native C8 sparse target verification. With native pages, each request supplies its own page table and valid candidates are packed ahead of padding. |
| `hy4flash_fuse_ihc_pre` | Fuses iHC residual-channel gating for small decode batches. |
| `hy4flash_share_verify_layout` | Reuses positions, window candidates and sparse group boundaries across target layers for every request in a captured decode batch. Each request keeps its own valid rows. |
| `hy4flash_shard_indexpool_scores` | Divides long-context IndexPool work across eight tensor ranks. Multiple native-page decode requests distribute complete row lists using the CCU copy group, preserving each request’s global top-k order. |
| `hy4flash_fuse_indexpool_rotate_encode` | Encodes rotated IndexPool queries for batched decode and short pooled keys with calibrated FP8 rounding. Target layers can reuse the same query positions’ sine and cosine values. |
| `hy4flash_pack_window_operands` | Reads overlapping C8 windows once per request and packs FP32 key and value operands for six verification queries per request. |
| `hy4flash_pack_mla_query` | Prepares MLA projection weights during loading and packs the absorbed queries and interleaved rotary components for all token rows in a decode batch. Uses about 134.8 MiB per rank for this TP8 checkpoint. |
| `hy4flash_batch_host_lookup` | Transfers the hash indices for four CPU-resident Engram tables together during single-request, six-token verification. Each table supplies its trained BF16 rows to the captured graph. |
| `hy4flash_pack_feature_channels` | After the Engram CPU lookup, projects four feature keys and a value together and reduces them across the eight TP ranks in one operation for captured decode batches with prepared lookup rows. |
| `hy4flash_verify_host_features` | Registers each CPU-resident BF16 Engram shard for direct reads during captured A5 decode. The index kernel uses each request’s own token range and lookback, including when a batch contains padding or requests of different lengths. Each NPU reads its selected rows. Four matching registered tables share one read launch. |
| `hy4flash_local_vocabulary` | Replicates the ordinary token embedding on each of eight NPUs for token lookups up to the configured decode capacity. For each DFly request, the eight vocabulary shards exchange one winning score and token ID per rank. Each rank needs about 0.86 GiB of additional memory. |
| `hy4flash_share_neighbor_recall` | Lets five target sparse layers reuse the preceding sparse layer’s logical retrieval positions during decode. Each layer retains its own attention and KV contents. This approximation can change token selection. |
| `hy4flash_sharded_greedy_verify` | Verifies plain greedy drafts from rank-local vocabulary winners. When a CCU copy group is available it handles winner packets through the configured decode token capacity. |
| `hy4flash_native_index_pages` | Requests 64-token KV pages for native C8 and contiguous pooled IndexPool keys. This increases each cached token from 1,000 to 1,025 bytes. Other explicitly requested page sizes keep the standard layout. |
| `hy4flash_ccu_copy_gathers` | Uses a separate CCU group to gather BF16 token shards up to 288 KiB per rank and exchange vocabulary winners on eight A5 devices. |
| `hy4flash_ordered_verify_scatter` | On eight A5 devices, exchanges each destination’s BF16 rows and reproduces the native reduction order. Each configured decode size uses this path after its padded geometry matches the original reduction at startup. |
| `hy4flash_resident_index_keys` | Keeps a per-layer BF16 view of active pooled keys, refreshing changed pages and invalidating it on other cache writes. Requires native pages and disabled prefix caching and KV transfer. |
| `hy4flash_draft_projection_sums` | Checks the native BF16 sum before switching supported single-request DFly projection widths to the measured transport. |
| `hy4flash_overlap_decode_kv` | Computes KV normalization and C8 quantization alongside the independent query projection during native-page decode. |
| `hy4flash_overlap_verify_projections` | Overlaps independent target output-gate and index-query projections with singleton verification attention and cache updates. |
| `hy4flash_shared_fp32_draft_history` | Shares each request’s dequantized DFly history across its six queries, in bounded batches, while retaining the original per-query FP32 attention reductions and trained five-token proposals. |

The fixed-length test measures one client request at a time. Each reported TPOT divides the elapsed time between its first and last generated-token events by 1,023. The reported P50 is the median across twenty distinct requests. The accuracy run uses all thirty AIME2026 questions at concurrency sixteen. [Per-request timings and hardware results](https://github.com/InferMatrixLab/InferRSI/issues/613)
