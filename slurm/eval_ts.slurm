#!/bin/bash
#SBATCH --job-name=syldrift_evts
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evts-%j.out
#SBATCH --error=logs/evts-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[evts] start $(date)"
$HOME/.conda/envs/syldrift/bin/python -m syldrift.eval_testset --tag "${TAG:-base}" --nfe ${NFE:-32 4 2} --n "${N:-30}" --ckpt "${CKPT:-}" --asr "${ASR:-}" --out "${OUT:-}"
echo "[evts] exit=$? $(date)"
