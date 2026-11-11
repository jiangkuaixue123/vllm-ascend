"""Fetch and verify the small official HF reference files in plan/input-freeze.json."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("manifest", type=Path)
p.add_argument("output_dir", type=Path)
args = p.parse_args()
frozen = json.loads(args.manifest.read_text())
base = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{frozen['revision']}"
for name, want in frozen["reference_sha256"].items():
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unexpected reference path")
    target = args.output_dir / relative
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != want:
            raise ValueError(f"Existing file differs: {relative}")
        continue
    with urlopen(f"{base}/{name}", timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != want:
        raise ValueError(f"Reference digest mismatch: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as out:
        out.write(data)
