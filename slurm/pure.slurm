#!/bin/bash
#SBATCH --job-name=syldrift_pure
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pure-%j.out
#SBATCH --error=logs/pure-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
echo "[pure] start $(date) nfe=${NFE}"
$HOME/.conda/envs/syldrift/bin/python -m syldrift.f5_pure_drift --steps "${STEPS:-3000}" \
  --batch "${BATCH:-4}" --lr "${LR:-1e-5}" --beta "${BETA:-0.5}" --nfe "${NFE:-4}" \
  --seed "${SEED:-0}" --out "${OUT:-runs/pure_nfe4}"
echo "[pure] exit=$? $(date)"
