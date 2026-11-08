#!/bin/bash
#SBATCH --job-name=syldrift_refchk
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/refchk-%j.out
#SBATCH --error=logs/refchk-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/hf_cache"
export HF_HUB_OFFLINE=1
$HOME/.conda/envs/syldrift/bin/python calib_ref_check.py
echo "[refchk] exit=$?"
