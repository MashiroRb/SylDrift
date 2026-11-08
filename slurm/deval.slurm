#!/bin/bash
#SBATCH --job-name=syldrift_deval
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=logs/deval-%j.out
#SBATCH --error=logs/deval-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
echo "[deval] start $(date)"
$HOME/.conda/envs/syldrift/bin/python -m syldrift.f5_eval_distilled --n "${N:-4}" --ckpt "${CKPT:-}" --tag "${TAG:-base}" --out "runs/deval_${TAG:-base}.json"
echo "[deval] exit=$? $(date)"
