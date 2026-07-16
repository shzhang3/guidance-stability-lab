#!/usr/bin/env python3
"""Export a compact, provenance-rich portfolio bundle from the SD1.5 shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_INDICES = [36, 37, 3, 12, 15, 20, 23, 40]
SCHEMES = ("cfg", "fitted", "interval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saturation_mask(image: np.ndarray) -> np.ndarray:
    clipped = ((image <= 2) | (image >= 253)).any(axis=2)
    mask = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    mask[clipped] = np.asarray([255, 69, 58, 190], dtype=np.uint8)
    return mask


def load_rows(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            (int(row["prompt_index"]), row["scheme"]): row
            for row in csv.DictReader(handle)
        }


def load_prompts(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["prompt_index"]): row for row in csv.DictReader(handle)}


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.metrics)
    prompts = load_prompts(args.prompts)
    shards = {scheme: np.load(args.shards / f"{scheme}.npz") for scheme in SCHEMES}

    samples = []
    for index in args.indices:
        sample = {
            "promptIndex": index,
            "prompt": prompts[index]["prompt"],
            "seed": int(prompts[index]["seed"]),
            "schemes": {},
        }
        for scheme in SCHEMES:
            archive = shards[scheme]
            positions = np.flatnonzero(archive["prompt_indices"] == index)
            if len(positions) != 1:
                raise ValueError(f"{scheme}: expected one image for prompt {index}, got {len(positions)}")
            image = archive["images"][int(positions[0])]
            image_name = f"p{index:04d}-{scheme}.webp"
            mask_name = f"p{index:04d}-{scheme}-mask.webp"
            image_path = output / image_name
            mask_path = output / mask_name
            Image.fromarray(image).save(image_path, "WEBP", quality=91, method=6)
            Image.fromarray(saturation_mask(image), "RGBA").save(mask_path, "WEBP", lossless=True, method=6)
            metric = rows[(index, scheme)]
            sample["schemes"][scheme] = {
                "image": f"/demo/final/{image_name}",
                "mask": f"/demo/final/{mask_name}",
                "sha256": sha256(image_path),
                "saturation": float(metric["pixel_saturation"]),
                "contrast": float(metric["pixel_contrast"]),
                "laplacianEnergy": float(metric["laplacian_energy"]),
                "latentNormMax": float(metric["latent_norm_max"]),
                "clipScore": float(metric["clip_score"]),
            }
        samples.append(sample)

    manifest = {
        "formatVersion": 1,
        "title": "Guidance Stability Lab matched-seed bundle",
        "source": {
            "repository": "guided-cfg-ap",
            "run": "results/fitted_cfg_ldm_20260710/sd15_g12_N12_n1000",
            "model": "runwayml/stable-diffusion-v1-5",
            "scheduler": "deterministic DDIM",
            "guidanceScale": 12.0,
            "w": 11.0,
            "numSteps": 12,
            "sourceTreeSha256": "d954e5686324ac71c6867afdf68b94d0db44c0cb1fc0642f92f1a3c284fead4a",
            "selection": (
                "Primary sample p0036 was chosen for visual legibility after ranking the first "
                "64 matched prompts by CFG-minus-fitted saturation. Additional examples are fixed."
            ),
        },
        "samples": samples,
    }
    with (output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=Path("/tmp/guidance-stability-assets"))
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "/Users/zhangxiaoning/Documents/guided-cfg-ap/results/"
            "fitted_cfg_ldm_20260710/sd15_g12_N12_n1000/ldm_benchmark_raw.csv"
        ),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(
            "/Users/zhangxiaoning/Documents/guided-cfg-ap/results/"
            "fitted_cfg_ldm_20260710/sd15_g12_N12_n1000/prompts.csv"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("public/demo/final"))
    parser.add_argument("--indices", nargs="+", type=int, default=DEFAULT_INDICES)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
