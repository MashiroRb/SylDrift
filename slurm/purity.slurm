#!/bin/bash
#SBATCH --job-name=syldrift_purity
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/purity-%j.out
#SBATCH --error=logs/purity-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
$HOME/.conda/envs/syldrift/bin/python purity_whisper.py "${N_UTT:-1500}"
echo "[purity] exit=$?"
