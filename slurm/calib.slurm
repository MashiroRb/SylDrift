#!/bin/bash
#SBATCH --job-name=syldrift_calib
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/calib-%j.out
#SBATCH --error=logs/calib-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
echo "[calib] start $(date)"
$HOME/.conda/envs/syldrift/bin/python calib_official.py
echo "[calib] exit=$? $(date)"
