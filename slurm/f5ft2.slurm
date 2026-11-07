#!/bin/bash
#SBATCH --job-name=syldrift_f5ft2
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/f5ft2-%j.out
#SBATCH --error=logs/f5ft2-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=$HOME/.conda/envs/syldrift/bin/python
echo "[f5ft2] start $(date) host=$(hostname)"
$PY -m syldrift.f5_finetune_cached --steps "${STEPS:-3000}" --batch "${BATCH:-8}" --lr "${LR:-1e-5}" --out "${OUT:-runs/f5_ft}"
echo "[f5ft2] exit=$? $(date)"
