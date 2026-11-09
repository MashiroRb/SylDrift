#!/bin/bash
#SBATCH --job-name=syldrift_peval
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/peval-%j.out
#SBATCH --error=logs/peval-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p runs/pure_eval
for S in 1 2 3; do
  $HOME/.conda/envs/syldrift/bin/python -m syldrift.eval_testset \
    --tag "${TAG}_s${S}" --nfe "${NFE}" --n "${N:-10}" --seed $S \
    --asr openai/whisper-large-v3 --ckpt "${CKPT}" \
    --out "runs/pure_eval/${TAG}_s${S}.json" 2>&1 | grep -E '^\[nfe' | tail -1
done
echo "[peval] exit=$?"
