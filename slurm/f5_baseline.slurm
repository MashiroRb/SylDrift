#!/bin/bash
#SBATCH --job-name=syldrift_f5base
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/f5-%j.out
#SBATCH --error=logs/f5-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=$HOME/.conda/envs/syldrift/bin/python
echo "[f5] start $(date) host=$(hostname)"
$PY -m syldrift.f5_baseline
echo "[f5] exit=$? $(date)"
