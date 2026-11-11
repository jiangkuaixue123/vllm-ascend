"""Reproduce the frozen text/ID parity check using the pinned HF tokenizer and template."""

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

from smoke_chat import CASES


PROBE = "请按顺序列出从二十到三十的整数，用逗号分隔。"
TEMPLATE_KWARGS = {"thinking_mode": "chat", "reasoning_effort": "high"}
HERE = Path(__file__).resolve().parent


def verify_source(root, name):
    data = (root / name).read_bytes()
    manifest = json.loads((HERE / "hf-source.json").read_bytes())
    if hashlib.sha256(data).hexdigest() != manifest["source"][name]["sha256"]:
        raise ValueError(f"source identity mismatch: {name}")


def frozen_inputs(hf):
    from transformers import AutoTokenizer

    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "encoding/encoding.py"):
        verify_source(hf, name)
    sys.path.insert(0, str(hf / "encoding"))
    from encoding import encode_messages

    tokenizer = AutoTokenizer.from_pretrained(hf, local_files_only=True)
    template = (hf / "chat_template.jinja").read_text(encoding="utf-8")
    cases = [(f"smoke-{i}", text, expected) for i, (text, expected) in enumerate(CASES, 1)]
    cases.append(("numerical", PROBE, None))
    rows = []
    for name, text, expected in cases:
        messages = [{"role": "user", "content": text}]
        rendered = encode_messages(messages, **TEMPLATE_KWARGS)
        kwargs = dict(chat_template=template, add_generation_prompt=True, **TEMPLATE_KWARGS)
        rendered_template = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.encode(rendered)
        template_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        if rendered != rendered_template or ids != template_ids:
            raise ValueError(f"official encoder and template disagree on {name}")
        if ids[0] != tokenizer.bos_token_id or rendered.count(tokenizer.bos_token) != 1:
            raise ValueError(f"unexpected BOS handling on {name}")
        row = dict(id=name, messages=messages, rendered=rendered, prompt_ids=ids)
        if expected is not None:
            row["expected"] = expected
        rows.append(row)
    return {
        "revision": json.loads((HERE / "hf-source.json").read_bytes())["revision"],
        "template_kwargs": TEMPLATE_KWARGS,
        "generation": dict(temperature=0, smoke_max_tokens=32, oracle_max_tokens=8,
                           stop_token_id=tokenizer.eos_token_id, add_generation_prompt=True),
        "served_model": "dsv41-stage1",
        "max_model_len": 256,
        "smoke_concurrency": 1,
        "smoke_timeout_s": 600,
        "request_retries": 0,
        "packages": {name: version(name) for name in ("transformers", "tokenizers", "jinja2")},
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hf_snapshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(frozen_inputs(args.hf_snapshot.resolve()), indent=2, ensure_ascii=False))
