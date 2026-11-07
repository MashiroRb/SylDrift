#!/bin/bash
#SBATCH --job-name=syldrift_test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
$HOME/.conda/envs/syldrift/bin/python tests/test_drift.py
echo "[test] exit=$?"
