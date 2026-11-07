#!/bin/bash
#SBATCH --job-name=syldrift_premel
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/premel-%j.out
#SBATCH --error=logs/premel-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
$HOME/.conda/envs/syldrift/bin/python precompute_f5_mel.py "${LIMIT:-0}"
echo "[premel] exit=$?"
