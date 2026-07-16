#!/usr/bin/env python3
"""Generate exact matched-seed SD1.5 DDIM trajectories for the portfolio lab."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCHEMES = ("cfg", "fitted", "interval")


def alpha_sigma(scheduler, timestep, device, dtype):
    index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    alpha = scheduler.alphas_cumprod[index].to(device=device, dtype=dtype)
    sigma = ((1.0 - alpha) / alpha).clamp_min(0).sqrt()
    return alpha, sigma


def previous_alpha_sigma(scheduler, step, timesteps, device, dtype):
    if step + 1 < len(timesteps):
        return alpha_sigma(scheduler, timesteps[step + 1], device, dtype)
    alpha = scheduler.final_alpha_cumprod.to(device=device, dtype=dtype)
    sigma = ((1.0 - alpha) / alpha).clamp_min(0).sqrt()
    return alpha, sigma


def coefficient(scheme: str, w: float, r: float, h: float) -> float:
    if scheme == "cfg":
        return w * (r - 1.0)
    if scheme == "fitted":
        return r ** (1.0 + w) - r
    if scheme == "interval":
        h_flat = math.log1p(1.0 / w) if w > 0 else math.inf
        return 0.0 if h > h_flat else w * (r - 1.0)
    raise ValueError(f"unknown scheme: {scheme}")


def encode_prompt(pipe, prompt: str, device, dtype):
    encoded = pipe.encode_prompt(
        prompt=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=[""],
    )
    prompt_embeds, negative_embeds = encoded[0], encoded[1]
    return torch.cat([negative_embeds, prompt_embeds], dim=0).to(device=device, dtype=dtype)


@torch.no_grad()
def decode(pipe, latents: torch.Tensor) -> np.ndarray:
    decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return (
        (decoded.float() / 2.0 + 0.5)
        .clamp(0, 1)
        .detach()
        .cpu()[0]
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def save_image_bundle(image: np.ndarray, destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(destination, "WEBP", quality=91, method=6)
    clipped_channels = (image <= 2) | (image >= 253)
    clipped = clipped_channels.any(axis=2)
    mask = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    mask[clipped] = np.asarray([255, 69, 58, 190], dtype=np.uint8)
    Image.fromarray(mask, "RGBA").save(
        destination.with_name(f"{destination.stem}-mask.webp"),
        "WEBP",
        lossless=True,
        method=6,
    )
    return float(clipped_channels.mean())


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=args.cache_dir,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    text_embeddings = encode_prompt(pipe, args.prompt, device, dtype)

    shape = (1, pipe.unet.config.in_channels, args.height // pipe.vae_scale_factor, args.width // pipe.vae_scale_factor)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = torch.randn(shape, generator=generator, device=device, dtype=dtype) * pipe.scheduler.init_noise_sigma
    w = args.guidance_scale - 1.0

    initial_image = decode(pipe, initial)
    initial_saturation = save_image_bundle(initial_image, output / "shared" / "step-00.webp")
    manifest = {
        "formatVersion": 1,
        "model": args.model_id,
        "scheduler": "DDIMScheduler",
        "prompt": args.prompt,
        "seed": args.seed,
        "guidanceScale": args.guidance_scale,
        "w": w,
        "numSteps": args.num_steps,
        "height": args.height,
        "width": args.width,
        "sharedInitial": {
            "image": "/demo/trace/shared/step-00.webp",
            "mask": "/demo/trace/shared/step-00-mask.webp",
            "saturation": initial_saturation,
        },
        "schemes": {},
    }

    for scheme in SCHEMES:
        latents = initial.clone()
        records = []
        for step, timestep in enumerate(timesteps):
            alpha, sigma = alpha_sigma(pipe.scheduler, timestep, device, dtype)
            alpha_next, sigma_next = previous_alpha_sigma(pipe.scheduler, step, timesteps, device, dtype)
            sigma_value = float(sigma.detach().cpu())
            sigma_next_value = float(sigma_next.detach().cpu())
            r = 0.0 if sigma_value <= 0 else sigma_next_value / sigma_value
            h = math.inf if sigma_next_value <= 0 else math.log(sigma_value / sigma_next_value)

            model_input = torch.cat([latents, latents], dim=0)
            model_input = pipe.scheduler.scale_model_input(model_input, timestep)
            eps_u, eps_c = pipe.unet(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                return_dict=False,
            )[0].chunk(2)
            y = latents / alpha.sqrt()
            d_conditional = y - sigma * eps_c
            d_unconditional = y - sigma * eps_u
            c = coefficient(scheme, w, r, h)
            y_next = d_conditional + r * (y - d_conditional) + c * (d_unconditional - d_conditional)
            latents = alpha_next.sqrt() * y_next

            image = decode(pipe, latents)
            image_path = output / scheme / f"step-{step + 1:02d}.webp"
            saturation = save_image_bundle(image, image_path)
            records.append({
                "step": step + 1,
                "timestep": int(timestep.item()),
                "sigma": sigma_value,
                "sigmaNext": sigma_next_value,
                "r": r,
                "h": None if not math.isfinite(h) else h,
                "coefficient": c,
                "latentNorm": float(latents.float().norm().detach().cpu()),
                "saturation": saturation,
                "image": f"/demo/trace/{scheme}/step-{step + 1:02d}.webp",
                "mask": f"/demo/trace/{scheme}/step-{step + 1:02d}-mask.webp",
            })
        manifest["schemes"][scheme] = records

    with (output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def self_test() -> None:
    for w in (0.0, 4.0, 11.0):
        for h in (1e-6, 0.1, 1.0):
            r = math.exp(-h)
            cfg = coefficient("cfg", w, r, h)
            fitted = coefficient("fitted", w, r, h)
            if w == 0.0 and (cfg != 0.0 or fitted != 0.0):
                raise AssertionError("w=0 must collapse to DDIM")
            if h < 1e-4 and abs(cfg / h + w) > 2e-3:
                raise AssertionError("CFG first-order coefficient mismatch")
            if h < 1e-4 and abs(fitted / h + w) > 2e-3:
                raise AssertionError("fitted first-order coefficient mismatch")
    if coefficient("fitted", 11.0, 0.0, math.inf) != 0.0:
        raise AssertionError("fitted one-jump coefficient must vanish")
    print("[self-test] PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--cache-dir", default="/gscratch/amath/shzhang3/hf_cache")
    parser.add_argument("--prompt", default="a group of people riding skis on a snowy surface")
    parser.add_argument("--seed", type=int, default=20300036)
    parser.add_argument("--guidance-scale", type=float, default=12.0)
    parser.add_argument("--num-steps", type=int, default=12)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/demo-trace"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
