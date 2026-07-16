#!/bin/bash
#SBATCH -A amath
#SBATCH -p gpu-rtx6k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=guidance_lab_trace
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

WORKDIR="${WORKDIR:-/gscratch/amath/shzhang3/guidance-stability-lab}"
LDM_ENV="${LDM_ENV:-/gscratch/amath/shzhang3/venvs/ldm-smoke}"

cd "$WORKDIR"
mkdir -p slurm/logs results/demo-trace
source "$LDM_ENV/bin/activate"

python scripts/generate_sd_trace.py --self-test
python scripts/generate_sd_trace.py --output results/demo-trace
