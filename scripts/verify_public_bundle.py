#!/usr/bin/env python3
"""Verify public images, trace coefficients, and evidence-table completeness."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "public/demo"
SCHEMES = ("cfg", "fitted", "interval")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_path(path: str) -> Path:
    prefix = "/demo/"
    if not path.startswith(prefix):
        raise AssertionError(f"unexpected public path: {path}")
    return DEMO / path.removeprefix(prefix)


def expected_coefficient(scheme: str, w: float, r: float, h: float | None) -> float:
    cfg = w * (r - 1.0)
    if scheme == "cfg":
        return cfg
    if scheme == "fitted":
        return r ** (1.0 + w) - r
    h_value = math.inf if h is None else h
    return 0.0 if h_value > math.log1p(1.0 / w) else cfg


def main() -> int:
    final = json.loads((DEMO / "final/manifest.json").read_text())
    trace = json.loads((DEMO / "trace/manifest.json").read_text())
    evidence = json.loads((DEMO / "evidence.json").read_text())

    if len(final["samples"]) != 8:
        raise AssertionError("expected eight fixed prompt samples")
    for sample in final["samples"]:
        for scheme in SCHEMES:
            entry = sample["schemes"][scheme]
            image = public_path(entry["image"])
            mask = public_path(entry["mask"])
            if not image.is_file() or not mask.is_file():
                raise AssertionError(f"missing final asset for prompt {sample['promptIndex']} / {scheme}")
            if sha256(image) != entry["sha256"]:
                raise AssertionError(f"SHA256 mismatch: {image}")

    primary = final["samples"][0]
    if trace["prompt"] != primary["prompt"] or trace["seed"] != primary["seed"]:
        raise AssertionError("trace prompt/seed does not match the primary fixed sample")
    if trace["numSteps"] != 12:
        raise AssertionError("portfolio trace must contain twelve DDIM steps")

    w = float(trace["w"])
    for scheme in SCHEMES:
        steps = trace["schemes"][scheme]
        if len(steps) != trace["numSteps"]:
            raise AssertionError(f"{scheme}: incomplete trace")
        for expected_step, step in enumerate(steps, start=1):
            if step["step"] != expected_step:
                raise AssertionError(f"{scheme}: nonconsecutive trace at step {expected_step}")
            expected = expected_coefficient(scheme, w, float(step["r"]), step["h"])
            if abs(float(step["coefficient"]) - expected) > 1e-10:
                raise AssertionError(f"{scheme}: coefficient mismatch at step {expected_step}")
            if not public_path(step["image"]).is_file() or not public_path(step["mask"]).is_file():
                raise AssertionError(f"{scheme}: missing frame at step {expected_step}")

        fixed = np.asarray(Image.open(public_path(primary["schemes"][scheme]["image"])).convert("RGB"), dtype=np.int16)
        traced = np.asarray(Image.open(public_path(steps[-1]["image"])).convert("RGB"), dtype=np.int16)
        difference = np.abs(fixed - traced)
        mean_error = float(difference.mean())
        p99_error = float(np.quantile(difference, 0.99))
        if mean_error > 3.0 or p99_error > 30.0:
            raise AssertionError(
                f"{scheme}: final-frame parity failed (mean={mean_error:.3f}, p99={p99_error:.1f})"
            )
        print(f"[parity] {scheme}: mean={mean_error:.3f}/255 p99={p99_error:.1f}/255")

    cells = evidence["cifar"]["cells"]
    if len(cells) != 9:
        raise AssertionError("expected the complete 3x3 CIFAR atlas")
    for cell in cells:
        if set(cell["schemes"]) != set(SCHEMES):
            raise AssertionError(f"incomplete scheme table at w={cell['w']}, N={cell['n']}")
    if len(evidence["latentDiffusion"]["cells"]) != 2:
        raise AssertionError("expected two Stable Diffusion transfer cells")

    print("[bundle] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
