# Replay the stage-one A5 text service

The vllm-ascend distribution registers the V4.1 text configuration, model and weight loader through its existing Ascend registration hooks. With the default load format, the platform selects the V4.1 loader for this architecture. The server uses the checkpoint tokenizer and a packaged template fallback. Each DP worker keeps its own compressed KV, indexes and n-gram history. The eight workers form one EP group. Each worker loads its share of routed experts and participates in collectives even during another rank's request. The two Engram tables and their scales are file-backed on the host. Workers gather bounded groups of hit rows, then run dequantization and gating on the device. The original checkpoint also contains vision and speculative weights. The text loader records a disposition for every key in each rank's ledger.

The launch config covers one 256-token sequence per DP rank, a single API frontend and sequential requests. Prefix caching, speculative decoding and graph capture are disabled. Use the A5 image identified by digest in `infra/container.sh`.

1. Copy `patch/`, `validator/`, `reference_cpu/` and `plan/` from `day0_stage1/` into a private directory. Set `WORK` to its absolute path, `MODEL_PATH` to the original 48-shard checkpoint and `CONTAINER` to a new name. Run these commands from `$WORK`:

   ```sh
   mkdir -p results
   python3 patch/infra/fetch-reference.py plan/input-freeze.json reference/hf
   ```

2. For the optional independent numerical reference, in a CPU-only instance of the same image, mount `$WORK` at `/stage1` and `$MODEL_PATH` read-only at `/models/dsv41`. Generate the independent reference from the frozen official model and its tokenizer. The ordered CPU implementation accumulates 32-column FP32 dots and binds the original weight files through copy-on-write mmap. Select fresh output paths for every attempt.

   ```sh
   cd /stage1
   export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
   python validator/hf_engram_cpu.py reference/hf /models/dsv41 results/hf-engram-cpu.npz
   python reference_cpu/hf-cpu-reference-r03.py reference/hf /models/dsv41 results/hf-cpu-trace --threads 16
   ```

3. Acquire an exclusive eight-card reservation and its service port. On the reserved host, export `WORK`, `MODEL_PATH`, `CONTAINER` and `LEASED_DEVICES` using the allocated IDs in physical order 0 through 7. Run `bash patch/infra/container.sh`. Build the vllm-ascend distribution from this checkout using its normal installation procedure for the target chip. Place the resulting wheel under `$WORK/wheels/`. Inside the new container run the following:

   ```sh
   cd /stage1
   run_env() { bash /stage1/patch/infra/in-env.sh "$@"; }
   run_env python -m pip install --no-deps --force-reinstall /stage1/wheels/vllm_ascend-*.whl
   export HCCL_PROBE_RESULTS=/stage1/results/hccl
   mkdir "$HCCL_PROBE_RESULTS" results/engram-npu
   run_env torchrun --standalone --nproc-per-node=8 patch/infra/hccl_probe.py
   run_env torchrun --standalone --nproc-per-node=8 validator/verify_engram.py \
     --device npu --checkpoint /models/dsv41 --reference reference/hf/inference \
     --oracle results/hf-engram-cpu.npz --output 'results/engram-npu/rank-{rank}.json'
   ```

On the tested A5 host, the default HCCL communication engine failed to resolve `RunAicpuIndOpCommInit` on two devices. The eight-rank probe and the following service runs succeeded with `export HCCL_OP_EXPANSION_MODE=AIV` in the launch environment. This is a host-specific replay setting.

For an interactive eager service, use the ordinary vLLM entry point, as shown in `patch/infra/serve.sh`. The example sets standard topology, scheduler and evaluation options. The independent reference directory and diagnostic switches are used only by the evidence checks.

4. In the container export `PORT` with the reserved value and `RUN_NAME` with a new attempt name. Run the launcher, then compare the numerical request on the serving rank against the independent trace:

   ```sh
   bash patch/infra/replay.sh
   python validator/compare_traces.py results/hf-cpu-trace \
     "results/$RUN_NAME"/numerical/rank-* --output "results/$RUN_NAME/comparison.json"
   ```

   The launcher checks the model alias and tokenization, runs all four chat requests, captures the numerical request, then terminates its server. The comparator records the first differing token or tensor. Without a predeclared floating-point tolerance, a nonidentical trace yields `INCONCLUSIVE` and exit code 1. Retrieve the rank load ledgers, per-rank checks, HTTP records and traces. Stop the task container and release the reservation.
