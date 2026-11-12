# DeepSeek V4.1 retained parallel layouts

Both layouts use the same W8A8 checkpoint, EP16, DSpark with five speculative
tokens, DSA-CP, FlashComm1, shared-expert DP, and fused MC2. Rebuild both the native delayed HC PyTorch binding and its host tiling with
three-row support, then point ASCEND_CUSTOM_OPP_PATH at the rebuilt vendor
package while preserving the other installed operator packages. Set the model path and existing runtime library paths for the host.

| Setting | DP4TP4 | DP2TP8 |
| --- | --- | --- |
| tensor-parallel-size | 4 | 8 |
| data-parallel-size | 4 | 2 |
| compilation_config.mode | 0 | 0 |
| cudagraph_mode | FULL_DECODE_ONLY | FULL_DECODE_ONLY |
| cudagraph_capture_sizes | [24,48] | [24,48] |
| max-model-len | 153600 | 153600 |
| block-size | 128 | 128 |
| max-num-seqs per DP | 8 | 8 |
| max-num-batched-tokens per DP | 8192 | 8192 |

Use enable_expert_parallel=true, enable_dsa_cp=true, enable_flashcomm1=true,
enable_shared_expert_dp=true, enable_fused_mc2=1 (integer), prefix caching,
and VLLM_ASCEND_ENABLE_FLASHCOMM1=1 in both deployments.

Both layouts use capture sizes [24,48], as requested. TP8 uses three or six
local rows. Three-row shards cover either half of a six-token proposal group;
the slot arena preserves the shard's exact positions and causal window. The
native HC binding and tiling must be rebuilt with three-row support. TP4
retains its original six/twelve-row execution.

## Optimization coverage

### Optional six-row gather policy

For single-node A3 DP4TP4, `"dsv41_small_gather_aiv": true` in
`--additional-config` creates one auxiliary TP AIV communicator, owned and
released by the existing group coordinator. The default is false; the option
requires a boolean and rejects other model, hardware or parallel layouts.
DBO, EPLB and pipeline/context parallelism are outside its scope.

Only target full-graph decode all-gather of BF16 hidden states shaped
`[6,5120]` uses that communicator. Twelve-row inputs, draft, prefill, other
shapes/dtypes and all other collectives use their original groups. Capture
sizes remain `[24,48]`. The main TP/EP/MC2/DP/world options do not change.
The startup/run marker is `DSV41_SMALL_GATHER_AIV_RUNTIME`.

The whole-TP AIV experiment is not retained: twelve-row BF16 all-gather
replay returned nonzero values after zeroing its input. The six-row path and
the original twelve-row path must both pass changing-input replay and full
model consumer checks before measuring this narrower candidate.

See [TorchNPU communication-domain settings](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/Frameworkfeatures/docs/en/framework_feature_guide_pytorch/setting_HCCL_communicator_parameter.md)
and [HCCL AIV constraints](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0096.html).
Only one AIV communicator is used per worker; no global communication
environment variable is set.

### Retained model paths

- HC delayed mixing, HC post, RMS and Sinkhorn: local three/six/twelve-row kernels.
- Target query/rotary overlap, paged index scoring and slot arena: TP4/TP8
  dispatch with actual TP size; original arithmetic and collective order.
- Target selected FIA and native BF16 sink: unchanged 64-head DSA-CP queries.
- Index sorting, cutoff ties, compact candidate masks, paged reads, slot buffers
  and middle-field scatter: qualified three/six/twelve-local-row and 24/48-global-row layouts.
- FP8 QDQ: TP8 output projection has one group of width4096 rather than two.
- Draft staging, batched attention and device history: request-owned metadata
  with TP8's eight local heads.
- Draft context, attention and local output graphs: TP8 supports per-DP batches
  1, 2 and 4; communication stays outside local graph capture. TP4 keeps its
  original supported batches 1 and 4.
- Grouped W8A8 and route-before-down fused MC2 retain EP16 and use actual TP
  metadata. No weight conversion or re-quantization is introduced.

## Validation results

Measured on 2026-11-09 with 8192 input tokens and 1024 output tokens.
Each layout reused one service for R1 and R2. Only R1 warmed the tiers;
R2 did not restart or repeat warmup. Request slots were routed modulo DP size.

| Layout | C1 ms/token | C4 ms/token | C16 ms/token | GSM8K fixed300 |
| --- | ---: | ---: | ---: | --- |
| DP2TP8 | 37.057937 | 28.698287 | Not measured | 294/300; zero failed requests |
| DP4TP4 | 29.401772 | 24.017719 | 35.548399 | Not rerun for this candidate |

Values are the means of two rounds. DP2TP8 C4 was 31.083016 in R1 and
26.313557 in R2; the generated sequences and speculative acceptance rates
differed, so both rounds remain part of the result.

DP4TP4 C1 was 1.71% slower than its historical retained mean of 28.906816.
A fresh run of the unchanged retained version measured 28.855630, putting
the candidate 1.89% above that reference. The original performance gate
therefore failed. The user explicitly accepted the measured DP4TP4 result
and authorized merging and pushing this implementation. This is a user
acceptance of the measured difference, not evidence of zero regression.
C4 and C16 remained within their previously declared measurement resolution,
and C4 remained below 25 ms/token.

Component coverage passed: 165 host checks; 307 device integration checks
with two skips; 32 native HC/slot-arena checks; 106 draft-graph checks;
real TP4 and TP8 attention, cache, collectives and output consumers; and
the single-group FP8 path. The slot-arena bitwise test uses the same native
backend on both sides; real FIA consumers are covered separately.

The deployment settings and acceptance record are also tracked in
[InferRSI issue 353](https://github.com/InferMatrixLab/InferRSI/issues/353#issuecomment-6730962302).
