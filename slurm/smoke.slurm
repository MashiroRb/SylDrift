#!/bin/bash
#SBATCH --job-name=syldrift_smoke
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$HOME/.conda/envs/syldrift/bin/python -m syldrift.train --synthetic --steps 30 --batch 4 --log_every 5
echo "[slurm] exit=$? $(date)"
