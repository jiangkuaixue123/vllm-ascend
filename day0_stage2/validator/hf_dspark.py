"""Teacher-force the frozen HF target and score its actual DSpark proposals on CPU."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


@torch.inference_mode()
def run(stage1, hf, weights, directory, steps, threads, chat_prompt):
    directory.mkdir(parents=True, exist_ok=False)
    driver = next(
        stage1 / path / "hf-cpu-reference-r03.py"
        for path in ("results", "reference_cpu")
        if (stage1 / path / "hf-cpu-reference-r03.py").is_file()
    )
    spec = importlib.util.spec_from_file_location("hf_ordered_cpu", driver)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.bind(hf)
    from freeze_prompts import verify_source
    from model import ModelArgs, Transformer
    from stage1_trace import digest, read_case
    from transformers import AutoTokenizer

    for name in ("inference/model.py", "inference/config.json", "inference/kernel.py", "inference/engram.py"):
        verify_source(hf, name)
    torch.set_num_threads(threads)
    torch.set_default_dtype(torch.bfloat16)
    args = ModelArgs(**json.loads((hf / "inference/config.json").read_bytes()))
    args.max_batch_size, args.max_seq_len, args.temperature, args.vision_n_layers = 1, 128, 0, 0
    tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=True)
    model = Transformer(args, tokenizer).eval()
    provenance = reference.mapped_weights(model, weights)
    prompts, case = read_case()
    if chat_prompt:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": chat_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            thinking_mode="chat",
            reasoning_effort="high",
            return_dict=False,
        )
        case = {"id": "independent-diagnostic", "prompt_ids": list(ids)}
    ids = case["prompt_ids"]
    generated, suggestions, fed_inputs, auxiliaries = [], [], [], []
    position = 0
    with (directory / "steps.jsonl").open("x") as log:
        for tick in range(steps):
            fed_inputs.append(list(ids))
            seed, target_logits, aux = model(torch.tensor([ids]), position)
            if not torch.isfinite(target_logits).all() or not torch.isfinite(aux).all():
                raise RuntimeError("nonfinite target output")
            auxiliaries.append(aux[0].cpu())
            result = model.forward_spec(seed, aux, position)
            token = int(seed[0])
            generated.append(token)
            row = {"tick": tick, "position": position, "target": token}
            if result is not None:
                draft, logits, conf = result
                if not torch.isfinite(logits).all() or not torch.isfinite(conf).all():
                    raise RuntimeError("nonfinite draft output")
                row.update(
                    proposals=draft[0, 1:].tolist(),
                    confidence=conf[0].tolist(),
                    markov_seed=int(draft[0, 0]),
                    block_top1=logits[0].argmax(-1).tolist(),
                )
                suggestions.append((tick, row["proposals"]))
            print(json.dumps(row), file=log, flush=True)
            print("official CPU", tick, token, row.get("proposals"), flush=True)
            position += len(ids)
            ids = [token]
    adjudication = []
    for tick, draft in suggestions:
        truth = generated[tick + 1 : tick + 1 + len(draft)]
        if not truth:
            continue
        accepted = next((i for i, (a, b) in enumerate(zip(draft, truth)) if a != b), len(truth))
        adjudication.append(
            dict(
                tick=tick,
                draft=draft,
                target=truth,
                accepted=accepted,
                rejected=accepted < len(truth),
                first_replacement=truth[accepted] if accepted < len(truth) else None,
            )
        )
    torch.save(
        dict(auxiliaries=auxiliaries, target_generated=generated, target_input_ids=fed_inputs),
        directory / "target-aux.pt",
    )
    summary = dict(
        kind="frozen official full target + official full draft, CPU greedy teacher-forced",
        driver_sha256=digest(Path(__file__)),
        diagnostic_user_text=chat_prompt,
        input_case=case["id"],
        prompt_ids=case["prompt_ids"],
        source=prompts["revision"],
        threads=threads,
        loader=provenance,
        generated=generated,
        adjudication=adjudication,
        actual_accepted=sum(x["accepted"] for x in adjudication),
        actual_rejected=sum(x["rejected"] for x in adjudication),
        aux_sha256=digest(directory / "target-aux.pt"),
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("stage1", "hf", "weights", "directory"):
        p.add_argument(arg, type=Path)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--chat-prompt")
    a = p.parse_args()
    run(**vars(a))
