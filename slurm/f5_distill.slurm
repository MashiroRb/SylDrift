#!/bin/bash
#SBATCH --job-name=syldrift_distill
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/distill-%j.out
#SBATCH --error=logs/distill-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[distill] start $(date) $(hostname)"
$HOME/.conda/envs/syldrift/bin/python -m syldrift.f5_distill --steps "${STEPS:-600}" --batch "${BATCH:-2}" --lr "${LR:-2e-6}" --cons_w "${CONS_W:-0.0}" --out "${OUT:-runs/f5_distill}"
echo "[distill] exit=$? $(date)"
