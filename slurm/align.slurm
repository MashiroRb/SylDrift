#!/bin/bash
#SBATCH --job-name=syldrift_align
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/align-%j.out
#SBATCH --error=logs/align-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
$HOME/.conda/envs/syldrift/bin/python align_whisper.py "${N_UTT:-800}"
echo "[align] exit=$?"
