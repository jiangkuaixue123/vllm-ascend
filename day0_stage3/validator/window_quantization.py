# SPDX-License-Identifier: Apache-2.0
"""Recompute current FP8 window bytes from BF16 values captured by the model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def expected_bytes(bits):
    """Select the smallest power-of-two scale and round to nearest-even E4M3FN."""
    values = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite captured KV input")
    groups = values.reshape(*values.shape[:-1], -1, 32)
    maximum = np.maximum(np.abs(groups).max(axis=-1), np.float32(1e-4))
    fraction, exponent = np.frexp(maximum / 448)
    exponent -= fraction == 0.5
    normalized = np.ldexp(groups, -exponent[..., None])
    # Positive E4M3FN codes 0 through 126 are monotone, including subnormals.
    code = np.arange(127)
    magnitude = np.where(code < 8, code / 512, np.ldexp(1 + (code % 8) / 8, code // 8 - 7))
    midpoints = (magnitude[:-1] + magnitude[1:]) / 2
    absolute = np.abs(normalized)
    lower = np.searchsorted(midpoints, absolute)
    ties = absolute == midpoints[np.minimum(lower, len(midpoints) - 1)]
    rounded = lower + (ties & (lower % 2 == 1))
    signed = rounded | (np.signbit(normalized).astype(np.uint8) << 7)
    return signed.reshape(values.shape).astype(np.uint8), (exponent + 127).astype(np.uint8)


def check_capture(receipt):
    header = json.loads(receipt.read_bytes())
    archive = receipt.with_suffix(".npz")
    if (
        archive.name != header["archive"]
        or hashlib.sha256(archive.read_bytes()).hexdigest() != header["archive_sha256"]
    ):
        raise ValueError(f"Capture digest mismatch in {receipt.name}")
    active = np.asarray(header["host"]["active"])
    if active.ndim != 1 or not np.isin(active, (0, 1)).all() or not active.any():
        raise ValueError(f"Invalid active lanes in {receipt.name}")
    with np.load(archive, allow_pickle=False) as data:
        kv, packed = data["kv"], data["packed"]
        width = int(data["head_dim"].item())
        if (
            not header["layers"]
            or kv.dtype != np.uint16
            or packed.dtype != np.uint8
            or width <= 0
            or width % 32
            or kv.shape != (len(header["layers"]), active.size, width)
            or packed.shape != (*kv.shape[:-1], width + width // 32)
        ):
            raise ValueError(f"Invalid KV geometry in {receipt.name}")
        kv, packed = kv[:, active.astype(bool)], packed[:, active.astype(bool)]
        codes, scales = expected_bytes(kv)
        layers = []
        for i in range(len(codes)):
            layers.append(
                dict(
                    layer=i,
                    code_bytes=codes[i].size,
                    wrong_codes=int(np.count_nonzero(packed[i, :, :width] != codes[i])),
                    scale_bytes=scales[i].size,
                    wrong_scales=int(np.count_nonzero(packed[i, :, width:] != scales[i])),
                )
            )
    return dict(receipt=receipt.name, archive_sha256=header["archive_sha256"], layers=layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.run.glob("window-rank-*-replay-*.json"))
    if not paths:
        parser.error("No window captures found")
    captures = [check_capture(path) for path in paths]
    layers = [layer for capture in captures for layer in capture["layers"]]
    totals = {
        key: sum(layer[key] for layer in layers) for key in ("code_bytes", "wrong_codes", "scale_bytes", "wrong_scales")
    }
    with args.output.open("x") as f:
        json.dump(dict(captures=captures, totals=totals), f, indent=2)
        f.write("\n")
    print(json.dumps(totals))
    return int(bool(totals["wrong_codes"] or totals["wrong_scales"]))


if __name__ == "__main__":
    raise SystemExit(main())
