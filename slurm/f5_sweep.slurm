#!/bin/bash
#SBATCH --job-name=syldrift_f5sweep
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/f5sweep-%j.out
#SBATCH --error=logs/f5sweep-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=$HOME/.conda/envs/syldrift/bin/python
echo "[sweep] start $(date)"
$PY -m syldrift.f5_sweep --n "${N:-10}" --nfe 1 2 4 8 16 32
echo "[sweep] exit=$? $(date)"
