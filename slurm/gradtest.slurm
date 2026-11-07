#!/bin/bash
#SBATCH --job-name=syldrift_gradtest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/gradtest-%j.out
#SBATCH --error=logs/gradtest-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
$HOME/.conda/envs/syldrift/bin/python tests/test_drift_grad.py
echo "[gradtest] exit=$?"
