#!/bin/bash
#SBATCH --job-name=syldrift_d
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/f5d-%j.out
#SBATCH --error=logs/f5d-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[f5d] start $(date)"
$HOME/.conda/envs/syldrift/bin/python -m syldrift.f5_drift --steps "${STEPS:-800}" --batch "${BATCH:-4}" --lr "${LR:-1e-5}" --lambda_drift "${LAM:-0.1}" --out "${OUT:-runs/f5_drift_l0.1}"
echo "[f5d] exit=$? $(date)"
