#!/bin/bash
#SBATCH --job-name=syldrift_f5ft
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/f5ft-%j.out
#SBATCH --error=logs/f5ft-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DISABLE_CUDNN=0
PY=$HOME/.conda/envs/syldrift/bin/python
echo "[f5ft] start $(date) host=$(hostname)"
$PY -m syldrift.f5_finetune --steps "${STEPS:-2000}" --batch "${BATCH:-4}" \
    --lr "${LR:-1e-5}" --drift_w "${DRIFT_W:-0.05}" --out "${OUT:-runs/f5_ft}"
echo "[f5ft] exit=$? $(date)"
