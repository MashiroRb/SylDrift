#!/bin/bash
#SBATCH --job-name=syldrift_cer
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cer-%j.out
#SBATCH --error=logs/cer-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PY=$HOME/.conda/envs/syldrift/bin/python
echo "[cer] start $(date)"
$PY -m syldrift.eval_set --ckpt "${CKPT:-runs/base/ckpt-last.pt}" --n "${N:-10}" --manifest "${MANIFEST:-data/split_val.jsonl}" --oracle_dur "${ORACLE_DUR:-0}" --out "${OUT:-runs/eval_cer.json}"
echo "[cer] exit=$? $(date)"
