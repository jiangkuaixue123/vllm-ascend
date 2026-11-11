# Paired GPQA-Diamond evaluation

Prepare the private expected chat IDs once in the official AISBench CPU client. The preparer verifies the CSV and official sources, consumes one GPQADataset pass, checks every displayed answer option against the frozen input manifest, then applies the frozen Hugging Face tokenizer and chat template. The private asset manifest supplies their revision and hashes. The prepared file contains licensed prompts and token IDs.

```bash
python gpqa_template_tokens.py prepare --source "$AISBENCH_SOURCE" --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" --assets "$HF_ASSETS" --tokenizer "$TOKENIZER_JSON" \
  --template "$HF_CHAT_TEMPLATE" --output "$EXPECTED"
```

For each server, use its own RunSpec and fresh output file. The check sends one POST to `/tokenize` per original question, retains the response token IDs, and checks the complete sequence against the local HF result. The server applies its own chat template with the frozen chat arguments. The command rejects HTTP redirects. It can run before starting the official generations.

```bash
python gpqa_template_tokens.py check --prepared "$EXPECTED" --manifest "$MANIFEST" \
  --run-spec "$SIDE_RUNSPEC" --endpoint "$SIDE_SERVICE_ORIGIN" --output "$SIDE_TOKEN_RECEIPT"
```

Run and audit each side using the existing `gpqa.py run` and `gpqa.py audit` instructions in `gpqa.md`. Save the returned live audit as `audit.json` in the run directory. The independent RunSpec reviewer verifies the original checkpoint, stock reference image, devices and equivalent requested and effective serving options required by the frozen paired gate. Its result references both RunSpecs, raw official run directories and tokenizer receipts. Paths below are relative to that review file. Use stable RunSpecs when running the preflight so their content digests remain available.

```json
{
  "complete": true,
  "matched": true,
  "gate_sha256": "<paired gate digest>",
  "manifest_sha256": "<ordered AISBench input manifest digest>",
  "prepared_sha256": "<prepared HF template file digest>",
  "reference": {
    "run_spec": "reference/run-spec.json",
    "run_spec_sha256": "<reference RunSpec digest>",
    "run": "reference/gpqa",
    "token_check": "reference/template-tokens.json"
  },
  "candidate": {
    "run_spec": "candidate/run-spec.json",
    "run_spec_sha256": "<candidate RunSpec digest>",
    "run": "candidate/gpqa",
    "token_check": "candidate/template-tokens.json"
  }
}
```

```bash
python gpqa_pair.py --manifest "$MANIFEST" --gate "$PAIRED_GATE" --prepared "$EXPECTED" \
  --run-pair "$RUNSPEC_REVIEW" --source "$AISBENCH_SOURCE" --data-root "$DATA_ROOT" \
  --output "$PRIVATE_PAIRED_REPORT"
```

The pair checker reloads both RunSpecs and verifies their digests and frozen image identities. It validates each actual token list and reproduces both complete audits from the raw single-call HTTP events, the resolved AISBench configuration, the original generations and the official scorer details. For each original index it checks the question, ordered options, user prompt, request bytes, canonical generation parameters, gold reference and individual official marks. The report retains both extracted letters and marks with their response digests. It prints both-correct, reference-only, candidate-only, both-incorrect, unparseable counts, each official total, and the difference. Exit status is 0 when the candidate has at least as many official correct answers as the reference, 1 when it has fewer, and 2 for absent, changed or incomplete evidence. Only the final paired report contains a score comparison.

The CPU fixture uses six invented questions, two independent loopback services and the frozen official AISBench reader and evaluator. No real question is used to select or change a model configuration.

```bash
python verify_gpqa_pair.py --source "$AISBENCH_SOURCE" --tokenizer "$TOKENIZER_JSON" \
  --template "$HF_CHAT_TEMPLATE" --assets "$HF_ASSETS"
```
