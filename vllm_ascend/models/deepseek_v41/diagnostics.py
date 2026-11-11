"""Optional single-request eager tensor recording for DeepSeek V4.1."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _array(tensor):
    return tensor.detach().cpu().float().numpy().copy()


class EagerTrace:
    """Only the requested prompt on this rank installs hooks or materializes tensors."""
    def __init__(self, directory, inputs, dp_rank, ep_rank):
        self.root = Path(directory) / f"rank-{dp_rank}"
        self.inputs = Path(inputs)
        self.prompts = json.loads((self.inputs / "prompts.json").read_text(encoding="utf-8"))
        case, = (row for row in self.prompts["cases"] if row["id"] == "numerical")
        self.case = case
        self.dp_rank = dp_rank
        self.ep_rank = ep_rank
        self.steps = []
        self.handles = []
        self.expected = []
        self.data = None
        self.done = False

    def _capture(self, name, tensor):
        if self.data is None:
            return
        if name in self.data or tensor.shape[:2] != (1, len(self.ids)):
            raise ValueError(f"Unexpected eager activation {name} with shape {tuple(tensor.shape)}")
        self.data[name] = _array(tensor[0])

    def _attach(self, layers):
        self.root.mkdir(parents=True, exist_ok=False)
        packages = ("torch", "torch-npu", "vllm", "vllm-ascend")
        _write_json(self.root / "environment.json", dict(
            schema=1, backend="a5-eager", rank=self.dp_rank,
            topology=dict(dp=8, ep=8, tp=1, pp=1, ep_rank=self.ep_rank),
            environment={name: version(name) for name in packages},
            model_revision=self.prompts["revision"], probe=self.case["id"],
            source_manifest=_digest(self.inputs / "hf-source.json"),
            prompts_sha256=_digest(self.inputs / "prompts.json"),
        ))
        for index, layer in enumerate(layers):
            prefix = f"layers.{index}"
            if layer.engram is not None:
                self.expected.append(prefix + ".engram")
                def engram_hook(_module, _args, result, prefix=prefix):
                    self._capture(prefix + ".engram", result)
                self.handles.append(layer.engram.register_forward_hook(engram_hook))
            self.expected.extend((prefix + ".hidden", prefix + ".pre_mix"))
            def block_hook(_module, _args, result, prefix=prefix):
                hidden, pre_mix = result
                self._capture(prefix + ".hidden", hidden)
                self._capture(prefix + ".pre_mix", pre_mix)
            self.handles.append(layer.register_forward_hook(block_hook))

    def _detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.done = True

    def begin(self, token_ids, start_pos, layers):
        if self.done or (not self.steps and start_pos != 0):
            return
        ids = token_ids.tolist()
        if not self.steps:
            if ids != self.case["prompt_ids"]:
                return
            self._attach(layers)
        else:
            previous = self.steps[-1]
            if (ids != [previous["token"]]
                    or start_pos != previous["start_pos"] + len(previous["input_ids"])):
                _write_json(self.root / "interrupted.json", dict(
                    completed_steps=len(self.steps), reason="request did not continue the recorded greedy prefix"))
                self._detach()
                return
        self.ids, self.start, self.data = ids, start_pos, {}

    def end(self, logits):
        if self.data is None:
            return
        if list(self.data) != self.expected:
            raise ValueError("Missing or reordered eager tensor observations")
        raw = _array(logits.reshape(-1, logits.shape[-1])[-1])
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite final-position logits")
        top = np.argsort(-raw, kind="stable")[:16].tolist()
        token = top[0]
        self.data["last_logits"] = raw
        step = len(self.steps)
        name = f"step-{step:02d}.npz"
        with (self.root / name).open("xb") as stream:
            np.savez(stream, **self.data)
        self.steps.append(dict(
            step=step, start_pos=self.start, input_ids=self.ids, token=token, top16=top,
            top16_values=raw[top].tolist(), file=name, sha256=_digest(self.root / name),
        ))
        self.data = None
        generation = self.prompts["generation"]
        if token == generation["stop_token_id"] or len(self.steps) == generation["oracle_max_tokens"]:
            _write_json(self.root / "trace.json", {"steps": self.steps})
            self._detach()
