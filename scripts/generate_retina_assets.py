#!/usr/bin/env python3
"""Build deterministic 2x display derivatives without changing evidence assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "public/demo"
RETINA = DEMO / "retina"
COLLECTIONS = ("final", "trace")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_path(path: Path) -> str:
    return f"/demo/{path.relative_to(DEMO).as_posix()}"


def build(quality: int, clean: bool) -> None:
    if clean and RETINA.exists():
        shutil.rmtree(RETINA)

    assets: dict[str, dict[str, object]] = {}
    for collection in COLLECTIONS:
        source_root = DEMO / collection
        for source in sorted(source_root.rglob("*.webp")):
            if source.name.endswith("-mask.webp"):
                continue

            relative = source.relative_to(DEMO)
            destination = RETINA / relative
            destination.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(source) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                display = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
                display = display.filter(ImageFilter.UnsharpMask(radius=1.0, percent=40, threshold=3))
                display.save(destination, "WEBP", quality=quality, method=6, exact=True)

            assets[public_path(source)] = {
                "display": public_path(destination),
                "sourceWidth": width,
                "sourceHeight": height,
                "displayWidth": width * 2,
                "displayHeight": height * 2,
                "sourceSha256": sha256(source),
                "displaySha256": sha256(destination),
            }

    manifest = {
        "formatVersion": 1,
        "scope": "Display-only derivatives. Raw 512 experiment assets remain the evidence source.",
        "processor": {
            "scale": 2,
            "resample": "Pillow LANCZOS",
            "sharpen": "UnsharpMask(radius=1.0, percent=40, threshold=3)",
            "format": "WebP",
            "quality": quality,
            "method": 6,
        },
        "assets": assets,
    }
    RETINA.mkdir(parents=True, exist_ok=True)
    (RETINA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[retina] wrote {len(assets)} display derivatives to {RETINA}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    build(args.quality, args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
