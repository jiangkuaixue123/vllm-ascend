# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy a verified historical fusion profile into a new editable directory."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare_bundle(output, profile="selected", *, root=None, config=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    if profile not in ("selected", "optional"):
        raise ValueError(f"Unknown fusion profile: {profile}")
    manifest = json.loads((root / profile / "manifest.json").read_text())
    if config is not None and hashlib.sha256(Path(config).read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Checkpoint config differs from the measured V4.1 configuration")
    sources = {}
    for name, entry in manifest["files"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError(f"Invalid callback filename: {name}")
        source = root / profile / name
        if not source.is_file():
            source = root / "selected" / name
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Callback checksum mismatch: {name}")
        compile(raw, name, "exec")
        sources[name] = raw
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in sources.items():
        (output / name).write_bytes(raw)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--profile", choices=("selected", "optional"), default="selected")
    parser.add_argument("--config", type=Path, help="Optionally verify the checkpoint config.json SHA256")
    args = parser.parse_args()
    manifest = prepare_bundle(args.output, args.profile, config=args.config)
    print(f"Prepared {args.profile}: {len(manifest['files'])} callbacks in {args.output.resolve()}")


if __name__ == "__main__":
    main()
