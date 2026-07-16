#!/usr/bin/env python3
"""Verify public images, display derivatives, traces, and evidence tables."""

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
    retina = json.loads((DEMO / "retina/manifest.json").read_text())
    sdxl = json.loads((DEMO / "sdxl/manifest.json").read_text())
    evidence = json.loads((DEMO / "evidence.json").read_text())

    expected_retina_sources: set[str] = set()
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
            expected_retina_sources.add(entry["image"])

    primary = final["samples"][0]
    if trace["prompt"] != primary["prompt"] or trace["seed"] != primary["seed"]:
        raise AssertionError("trace prompt/seed does not match the primary fixed sample")
    if trace["numSteps"] != 12:
        raise AssertionError("portfolio trace must contain twelve DDIM steps")
    expected_retina_sources.add(trace["sharedInitial"]["image"])

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
            expected_retina_sources.add(step["image"])

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

    retina_assets = retina["assets"]
    if set(retina_assets) != expected_retina_sources:
        missing = sorted(expected_retina_sources - set(retina_assets))
        extra = sorted(set(retina_assets) - expected_retina_sources)
        raise AssertionError(f"retina manifest coverage mismatch: missing={missing}, extra={extra}")
    for source_name, entry in retina_assets.items():
        source = public_path(source_name)
        display = public_path(entry["display"])
        if sha256(source) != entry["sourceSha256"] or sha256(display) != entry["displaySha256"]:
            raise AssertionError(f"retina SHA256 mismatch: {source_name}")
        with Image.open(source) as source_image, Image.open(display) as display_image:
            if display_image.size != (source_image.width * 2, source_image.height * 2):
                raise AssertionError(f"retina dimensions mismatch: {source_name}")
    print(f"[retina] {len(retina_assets)} display derivatives verified")

    if (sdxl["width"], sdxl["height"]) != (1024, 1024):
        raise AssertionError("SDXL hero must remain native 1024")
    if set(sdxl["schemes"]) != set(SCHEMES):
        raise AssertionError("SDXL hero is missing a matched scheme")
    sdxl_w = float(sdxl["w"])
    cfg_final = float(sdxl["schemes"]["cfg"]["coefficientFinal"])
    final_r = 1.0 + cfg_final / sdxl_w
    if not 0.0 <= final_r <= 1.0:
        raise AssertionError("SDXL final schedule ratio is invalid")
    final_h = math.inf if final_r == 0.0 else -math.log(final_r)
    expected_final_coefficients = {
        "cfg": sdxl_w * (final_r - 1.0),
        "fitted": final_r ** (1.0 + sdxl_w) - final_r,
        "interval": 0.0 if final_h > math.log1p(1.0 / sdxl_w) else sdxl_w * (final_r - 1.0),
    }
    for scheme in SCHEMES:
        entry = sdxl["schemes"][scheme]
        image = public_path(entry["image"])
        mask = public_path(entry["mask"])
        if not image.is_file() or not mask.is_file():
            raise AssertionError(f"SDXL hero asset missing: {scheme}")
        if sha256(image) != entry["sha256"]:
            raise AssertionError(f"SDXL SHA256 mismatch: {scheme}")
        with Image.open(image) as opened:
            if opened.size != (1024, 1024):
                raise AssertionError(f"SDXL dimensions mismatch: {scheme}")
        if abs(float(entry["coefficientFinal"]) - expected_final_coefficients[scheme]) > 1e-10:
            raise AssertionError(f"SDXL terminal coefficient mismatch: {scheme}")
    print("[sdxl] native-1024 matched triplet verified")

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
