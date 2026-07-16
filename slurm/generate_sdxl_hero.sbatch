#!/bin/bash
#SBATCH -A amath
#SBATCH -p gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=guidance_lab_sdxl
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR="${WORKDIR:-/gscratch/amath/shzhang3/guidance-stability-lab}"
LDM_ENV="${LDM_ENV:-/gscratch/amath/shzhang3/venvs/ldm-smoke}"

cd "$WORKDIR"
mkdir -p slurm/logs results/sdxl-hero
source "$LDM_ENV/bin/activate"

python scripts/generate_sdxl_hero.py --self-test
python scripts/generate_sdxl_hero.py \
  --model-id stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a futuristic glass research station beneath the ocean, surrounded by bioluminescent coral and divers, cinematic architectural photography, natural colors, intricate detail" \
  --seed 260715 \
  --guidance-scale 12 \
  --num-steps 20 \
  --height 1024 \
  --width 1024 \
  --output results/sdxl-hero
