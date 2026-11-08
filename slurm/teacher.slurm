#!/bin/bash
#SBATCH --job-name=syldrift_teacher
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/teacher-%j.out
#SBATCH --error=logs/teacher-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
echo "[teacher] start $(date)"
$HOME/.conda/envs/syldrift/bin/python make_teacher.py "${LIMIT:-0}"
echo "[teacher] exit=$? $(date)"
