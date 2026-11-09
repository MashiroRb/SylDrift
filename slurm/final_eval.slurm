#!/bin/bash
#SBATCH --job-name=syldrift_fev
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fev-%j.out
#SBATCH --error=logs/fev-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PWD}/third_party/F5-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p runs/final_eval
for S in 1 2 3 4 5; do
  $HOME/.conda/envs/syldrift/bin/python -m syldrift.eval_testset \
    --tag "${TAG}_s${S}" --nfe "${NFE}" --n 30 --seed $S --sim 1 \
    --asr openai/whisper-large-v3 --ckpt "${CKPT}" \
    --out "runs/final_eval/${TAG}_s${S}.json" 2>&1 | tail -2
done
echo "[fev] done ${TAG}"
