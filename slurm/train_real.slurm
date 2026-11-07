#!/bin/bash
#SBATCH --job-name=syldrift_train
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PY=$HOME/.conda/envs/syldrift/bin/python
echo "[slurm] job=$SLURM_JOB_ID $(date) $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY -m syldrift.train_real --manifest "${MANIFEST:-data/manifest_test.jsonl}" --steps "${STEPS:-100}" --batch "${BATCH:-4}" --drift_w "${DRIFT_W:-0.1}" --dim "${DIM:-384}" --depth "${DEPTH:-6}" --heads "${HEADS:-6}" --lr "${LR:-2e-4}" --out "${OUT:-runs/mini}" --log_every 20
echo "[slurm] exit=$? $(date)"
