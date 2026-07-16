#!/usr/bin/env python3
"""Export the compact experiment table consumed by the portfolio atlas."""

from __future__ import annotations

import csv
import json
from pathlib import Path


SOURCE = Path("/Users/zhangxiaoning/Documents/guided-cfg-ap")
GRID = SOURCE / "results/fitted_cfg_cifar_corrected_grid_20260710/aggregate/cifar_coefficient_combined.csv"
LDM_ROOT = SOURCE / "results/fitted_cfg_ldm_20260710"
OUTPUT = Path("public/demo/evidence.json")


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    cells: dict[tuple[float, int], dict] = {}
    with GRID.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scheme"] not in {"cfg", "fitted", "interval"}:
                continue
            key = (number(row, "w"), int(row["N"]))
            cell = cells.setdefault(key, {"w": key[0], "n": key[1], "schemes": {}})
            cell["schemes"][row["scheme"]] = {
                "fid": number(row, "fid"),
                "kid": number(row, "kid_mean"),
                "ampP95": number(row, "guided_E1_p95"),
                "clipP95": number(row, "clip_frac_p95"),
                "targetAccuracy": number(row, "target_accuracy"),
                "targetConfidence": number(row, "target_confidence"),
                "terminalCoefficient": number(row, "terminal_coefficient"),
            }

    latent_cells = []
    for name, label in (
        ("sd15_g12_N12_n5000", "g=12, N=12"),
        ("sd15_g7p5_N20_n5000", "g=7.5, N=20"),
    ):
        path = LDM_ROOT / name / "ldm_benchmark_summary.json"
        payload = json.loads(path.read_text())
        latent_cells.append({
            "id": name,
            "label": label,
            "summary": {row["scheme"]: row for row in payload["summary"]},
            "bootstrap": {row["scheme"]: row for row in payload["bootstrap"]},
            "shutdownContrast": payload["shutdown_contrast"][0],
        })

    output = {
        "formatVersion": 1,
        "cifar": {
            "dataset": "CIFAR-10 EDM",
            "samplesPerCell": 5000,
            "source": "results/fitted_cfg_cifar_corrected_grid_20260710/aggregate/cifar_coefficient_combined.csv",
            "cells": [cells[key] for key in sorted(cells)],
        },
        "latentDiffusion": {
            "dataset": "COCO 2017 validation captions",
            "model": "Stable Diffusion 1.5",
            "samplesPerCell": 5000,
            "cells": latent_cells,
        },
        "scope": [
            "Fitted CFG is presented as a high-guidance stabilizer, not a universal quality improvement.",
            "KID can favor vanilla CFG on CIFAR-10 even when FID and clipping favor fitted CFG.",
            "Stable Diffusion evidence covers two deterministic DDIM cells, not every model or scheduler.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
