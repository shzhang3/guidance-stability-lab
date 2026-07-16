#!/usr/bin/env python3
"""Generate a native-1024 matched SDXL triplet for the portfolio hero."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMES = ("cfg", "fitted", "interval")


def coefficient(scheme: str, w: float, r: float, h: float) -> float:
    if scheme == "cfg":
        return w * (r - 1.0)
    if scheme == "fitted":
        return r ** (1.0 + w) - r
    if scheme == "interval":
        h_flat = math.log1p(1.0 / w) if w > 0 else math.inf
        return 0.0 if h > h_flat else w * (r - 1.0)
    raise ValueError(f"unknown scheme: {scheme}")


def alpha_sigma(scheduler: Any, timestep: Any, device: Any, dtype: Any) -> tuple[Any, Any]:
    import torch

    index = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
    alpha = scheduler.alphas_cumprod[index].to(device=device, dtype=dtype)
    sigma = ((1.0 - alpha) / alpha).clamp_min(0).sqrt()
    return alpha, sigma


def previous_alpha_sigma(
    scheduler: Any,
    step: int,
    timesteps: Any,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    if step + 1 < len(timesteps):
        return alpha_sigma(scheduler, timesteps[step + 1], device, dtype)
    alpha = scheduler.final_alpha_cumprod.to(device=device, dtype=dtype)
    sigma = ((1.0 - alpha) / alpha).clamp_min(0).sqrt()
    return alpha, sigma


def encode_sdxl(pipe: Any, prompt: str, height: int, width: int, device: Any) -> tuple[Any, Any, Any]:
    import torch

    prompt_embeds, negative_embeds, pooled, negative_pooled = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )
    text_embeddings = torch.cat([negative_embeds, prompt_embeds], dim=0).to(device)
    pooled_embeddings = torch.cat([negative_pooled, pooled], dim=0).to(device)
    projection_dim = pipe.text_encoder_2.config.projection_dim
    time_ids = pipe._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=text_embeddings.dtype,
        text_encoder_projection_dim=projection_dim,
    )
    time_ids = torch.cat([time_ids, time_ids], dim=0).to(device)
    return text_embeddings, pooled_embeddings, time_ids


def decode_sdxl(pipe: Any, latents: Any) -> np.ndarray:
    import torch

    original_dtype = pipe.vae.dtype
    needs_upcast = original_dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcast:
        pipe.vae.to(dtype=torch.float32)
    latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

    has_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
    has_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None
    if has_mean and has_std:
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        std = torch.tensor(pipe.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        decoded_latents = latents * std / pipe.vae.config.scaling_factor + mean
    else:
        decoded_latents = latents / pipe.vae.config.scaling_factor

    decoded = pipe.vae.decode(decoded_latents, return_dict=False)[0]
    if pipe.watermark is not None:
        decoded = pipe.watermark.apply_watermark(decoded)
    image = pipe.image_processor.postprocess(decoded, output_type="np")[0]
    if needs_upcast:
        pipe.vae.to(dtype=original_dtype)
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def save_image_bundle(image: np.ndarray, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(destination, "WEBP", quality=95, method=6, exact=True)
    clipped_channels = (image <= 2) | (image >= 253)
    clipped = clipped_channels.any(axis=2)
    mask = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    mask[clipped] = np.asarray([255, 69, 58, 190], dtype=np.uint8)
    mask_path = destination.with_name(f"{destination.stem}-mask.webp")
    Image.fromarray(mask, "RGBA").save(mask_path, "WEBP", lossless=True, method=6)
    return {
        "image": f"/demo/sdxl/{destination.name}",
        "mask": f"/demo/sdxl/{mask_path.name}",
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "saturation": float(clipped_channels.mean()),
    }


def generate(args: argparse.Namespace) -> None:
    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    load_options: dict[str, Any] = {
        "torch_dtype": dtype,
        "cache_dir": args.cache_dir,
        "use_safetensors": True,
    }
    if dtype == torch.float16:
        load_options["variant"] = "fp16"
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id, **load_options)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler.set_timesteps(args.num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    text_embeddings, pooled_embeddings, time_ids = encode_sdxl(
        pipe, args.prompt, args.height, args.width, device
    )
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    shape = (
        1,
        pipe.unet.config.in_channels,
        args.height // pipe.vae_scale_factor,
        args.width // pipe.vae_scale_factor,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = torch.randn(shape, generator=generator, device=device, dtype=dtype) * pipe.scheduler.init_noise_sigma
    w = args.guidance_scale - 1.0
    manifest: dict[str, Any] = {
        "formatVersion": 1,
        "scope": "Native-1024 visual transfer demo; not a benchmark or paper-quality endpoint.",
        "selection": "Prompt, seed, guidance, and NFE fixed in the launch script before generation.",
        "model": args.model_id,
        "scheduler": "DDIMScheduler",
        "prompt": args.prompt,
        "seed": args.seed,
        "guidanceScale": args.guidance_scale,
        "w": w,
        "numSteps": args.num_steps,
        "height": args.height,
        "width": args.width,
        "schemes": {},
    }

    for scheme in SCHEMES:
        latents = initial.clone()
        coefficients: list[float] = []
        latent_norm_max = 0.0
        for step, timestep in enumerate(timesteps):
            alpha, sigma = alpha_sigma(pipe.scheduler, timestep, device, dtype)
            alpha_next, sigma_next = previous_alpha_sigma(pipe.scheduler, step, timesteps, device, dtype)
            sigma_value = float(sigma.detach().cpu())
            sigma_next_value = float(sigma_next.detach().cpu())
            r = 0.0 if sigma_value <= 0 else sigma_next_value / sigma_value
            h = math.inf if sigma_next_value <= 0 else math.log(sigma_value / sigma_next_value)

            model_input = pipe.scheduler.scale_model_input(latents, timestep)
            noise_u = pipe.unet(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings[:1],
                added_cond_kwargs={"text_embeds": pooled_embeddings[:1], "time_ids": time_ids[:1]},
                return_dict=False,
            )[0]
            noise_c = pipe.unet(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings[1:],
                added_cond_kwargs={"text_embeds": pooled_embeddings[1:], "time_ids": time_ids[1:]},
                return_dict=False,
            )[0]
            y = latents / alpha.sqrt()
            denoiser_c = y - sigma * noise_c
            denoiser_u = y - sigma * noise_u
            step_coefficient = coefficient(scheme, w, r, h)
            y_next = denoiser_c + r * (y - denoiser_c) + step_coefficient * (denoiser_u - denoiser_c)
            latents = alpha_next.sqrt() * y_next
            coefficients.append(step_coefficient)
            latent_norm_max = max(latent_norm_max, float(latents.float().norm().detach().cpu()))

        image = decode_sdxl(pipe, latents)
        entry = save_image_bundle(image, output / f"{scheme}.webp")
        entry.update({
            "coefficientMin": min(coefficients),
            "coefficientFinal": coefficients[-1],
            "latentNormMax": latent_norm_max,
        })
        manifest["schemes"][scheme] = entry
        print(f"[sdxl] {scheme} complete; saturation={entry['saturation']:.6f}", flush=True)

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[sdxl] wrote matched native-{args.width} triplet to {output}")


def self_test() -> None:
    w = 11.0
    for h in (1e-6, 0.1, 1.0):
        r = math.exp(-h)
        if abs(coefficient("cfg", w, r, h) - w * (r - 1.0)) > 1e-14:
            raise AssertionError("CFG coefficient mismatch")
        if h < 1e-4 and abs(coefficient("fitted", w, r, h) / h + w) > 2e-3:
            raise AssertionError("fitted first-order mismatch")

    y, sigma, r = 0.7, 2.3, 0.61
    eps_u, eps_c = -0.4, 0.9
    denoiser_u = y - sigma * eps_u
    denoiser_c = y - sigma * eps_c
    fitted_form = denoiser_c + r * (y - denoiser_c) + w * (r - 1.0) * (denoiser_u - denoiser_c)
    epsilon_form = y - (1.0 - r) * sigma * ((1.0 + w) * eps_c - w * eps_u)
    if abs(fitted_form - epsilon_form) > 1e-12:
        raise AssertionError("CFG DDIM oracle mismatch")
    print("[self-test] PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--cache-dir", default="/gscratch/amath/shzhang3/hf_cache")
    parser.add_argument(
        "--prompt",
        default=(
            "a futuristic glass research station beneath the ocean, surrounded by bioluminescent coral "
            "and divers, cinematic architectural photography, natural colors, intricate detail"
        ),
    )
    parser.add_argument("--seed", type=int, default=260715)
    parser.add_argument("--guidance-scale", type=float, default=12.0)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/sdxl-hero"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
