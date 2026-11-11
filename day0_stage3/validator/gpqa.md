# GPQA-Diamond with AISBench

Run these commands in the pinned AISBench client's Python environment. Put the authorized Diamond CSV at `$DATA_ROOT/ais_bench/datasets/gpqa/gpqa_diamond.csv`. The private JSON manifest supplies the CSV and AISBench source digests, ordered question and option identities, model alias and generation settings. Treat the data, predictions, HTTP receipts and per-question audit as private.

```bash
GPQA=day0_stage3/validator/gpqa.py
python "$GPQA" check --source "$AISBENCH_SOURCE" --data-root "$DATA_ROOT" --manifest "$MANIFEST"
python "$GPQA" run --dry-run --source "$AISBENCH_SOURCE" --data-root "$DATA_ROOT" --manifest "$MANIFEST" --output "$NEW_DRY_DIR"
python "$GPQA" audit --source "$AISBENCH_SOURCE" --data-root "$DATA_ROOT" --manifest "$MANIFEST" --run "$NEW_DRY_DIR" --output "$NEW_DRY_DIR/audit.json"
```

The pinned AISBench CLI sends chat requests during its dry run. The harness intercepts them on loopback, waits 300 ms per request so the client's concurrency is visible, and returns HTTP 410. The audit validates the input bytes for each intercepted request and the official client's failure receipt. All timing from this exercise describes the local sink.

```bash
python "$GPQA" run --endpoint "$SERVICE_ORIGIN" --source "$AISBENCH_SOURCE" --data-root "$DATA_ROOT" --manifest "$MANIFEST" --output "$NEW_RUN_DIR"
python "$GPQA" audit --source "$AISBENCH_SOURCE" --data-root "$DATA_ROOT" --manifest "$MANIFEST" --run "$NEW_RUN_DIR" --output "$NEW_RUN_DIR/audit.json"
```

Use a fresh private output directory for every run. `SERVICE_ORIGIN` is the HTTP(S) scheme, host and optional port. For live runs the observer forwards only the first valid POST for each expected prompt. The audit joins those raw model messages to AISBench predictions and the official evaluator's per-input details in original CSV order. It emits the recorded official score only when the evidence is complete. A quality threshold can be applied separately by a campaign that defines one.

To exercise the scoring and receipt validation on a CPU, run the following command with the same installed AISBench client. The fixture creates six invented questions, launches a temporary loopback service, checks the official marks, and verifies that altered, duplicated or missing receipts are detected.

```bash
python day0_stage3/validator/verify_gpqa_harness.py --source "$AISBENCH_SOURCE"
```

The stage-three launcher takes its chat defaults from `CHAT_TEMPLATE_KWARGS`. For an author-settings run, export the exact JSON object at `settings.generation.chat_template_kwargs` in the frozen manifest before launching. Record the installed vLLM and vllm-ascend versions, the source and wheel digests, and the resolved model class and loader. Check each rendered prompt through the service's `/tokenize` route against the pinned HF encoding before sending generation requests. The long-context retrieval fixture uses the chat defaults under which its token IDs were frozen.
