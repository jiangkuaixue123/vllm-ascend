"""Record an opt-in eager diagnostic forward from either model. Each trace has one request."""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_case():
    prompts = json.loads((HERE / "prompts.json").read_text())
    case, = [row for row in prompts["cases"] if row["id"] == "numerical"]
    return prompts, case


def expected_taps():
    taps = []
    for layer in range(40):
        if layer in (1, 14):
            taps.append(f"layers.{layer}.engram")
        taps.extend((f"layers.{layer}.hidden", f"layers.{layer}.pre_mix"))
    return taps


def dump_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def as_array(tensor):
    return tensor.detach().to("cpu").float().numpy().copy()


class TraceWriter:
    """Call begin -> actual forward (with taps attached) -> end once per step, then finish."""

    def __init__(self, directory, *, backend, rank, topology, environment):
        if backend not in ("official-hf", "a5-eager"):
            raise ValueError(f"unsupported trace backend {backend}")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.prompts, self.case = read_case()
        self.rank = rank
        self.inputs = []
        self.steps = []
        self.data = None
        self.handles = []
        dump_json(self.root / "environment.json", dict(
            schema=1, backend=backend, rank=rank, topology=topology, environment=environment,
            model_revision=self.prompts["revision"], source_manifest=digest(HERE / "hf-source.json"),
            prompts_sha256=digest(HERE / "prompts.json"), probe=self.case["id"],
        ))

    def attach(self, layers):
        if len(layers) != 40 or self.handles:
            raise ValueError("expected forty backbone layers and one hook registration")
        for i, layer in enumerate(layers):
            def block_hook(module, args, result, i=i):
                if self.data is not None:
                    h, pre_mix = result
                    self.tap(f"layers.{i}.hidden", h)
                    self.tap(f"layers.{i}.pre_mix", pre_mix)

            self.handles.append(layer.register_forward_hook(block_hook))
            if i in (1, 14):
                def engram_hook(module, args, result, i=i):
                    if self.data is not None:
                        self.tap(f"layers.{i}.engram", result)

                self.handles.append(layer.engram.register_forward_hook(engram_hook))

    def begin(self, input_ids, start_pos):
        if self.data is not None:
            raise ValueError("previous forward has not ended")
        ids = list(input_ids)
        if not self.steps:
            if ids != self.case["prompt_ids"] or start_pos != 0:
                raise ValueError("first forward does not have the frozen prompt IDs/position")
        elif ids != [self.steps[-1]["token"]] or start_pos != len(self.inputs):
            raise ValueError("decode did not continue from the preceding sampled token")
        self.data = {}
        self.input_ids = ids
        self.start_pos = start_pos
        self.inputs.extend(ids)

    def tap(self, name, tensor):
        if name in self.data:
            raise ValueError(f"duplicate hook {name}")
        # Remove the batch dimension only. Full prefill positions allow first-token diagnosis.
        if tensor.shape[0] != 1 or tensor.shape[1] != len(self.input_ids):
            raise ValueError(f"shape {tuple(tensor.shape)} at {name} disagrees with this forward")
        self.data[name] = as_array(tensor[0])

    def end(self, logits, sampled_token=None):
        if self.data is None or list(self.data) != expected_taps():
            raise ValueError("missing or reordered layer observations")
        raw = as_array(logits).reshape(-1, logits.shape[-1])[-1]
        if raw.shape != (129280,) or not np.isfinite(raw).all():
            raise ValueError("expected 129280 finite final-position logits")
        top = np.argsort(-raw, kind="stable")[:16].tolist()
        token = top[0]
        if sampled_token is not None and int(sampled_token) != token:
            raise ValueError("model's sampled token differs from greedy argmax")
        self.data["last_logits"] = raw
        filename = f"step-{len(self.steps):02d}.npz"
        with (self.root / filename).open("xb") as f:
            np.savez(f, **self.data)
        self.steps.append(dict(step=len(self.steps), start_pos=self.start_pos, input_ids=self.input_ids,
                               token=token, top16=top, top16_values=raw[top].tolist(),
                               file=filename, sha256=digest(self.root / filename)))
        self.data = None
        return token

    def finish(self):
        for handle in self.handles:
            handle.remove()
        generation = self.prompts["generation"]
        if self.data is not None or not self.steps:
            raise ValueError("incomplete forward")
        tokens = [step["token"] for step in self.steps]
        if generation["stop_token_id"] in tokens[:-1] or len(tokens) > generation["oracle_max_tokens"]:
            raise ValueError("sequence continued after stop")
        if tokens[-1] != generation["stop_token_id"] and len(tokens) != generation["oracle_max_tokens"]:
            raise ValueError("sequence ended without EOS or generation budget")
        dump_json(self.root / "trace.json", {"steps": self.steps})
