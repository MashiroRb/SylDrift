#!/bin/bash
#SBATCH --job-name=syldrift_manifest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/manifest-%j.out
#SBATCH --error=logs/manifest-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
PY=$HOME/.conda/envs/syldrift/bin/python
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1

echo "[manifest] start $(date) host=$(hostname) cores=${SLURM_CPUS_PER_TASK}"
$PY -u -m syldrift.make_manifest_aishell3 \
    --root "${ROOT:-data/partial}" \
    --out "${OUT:-data/manifest_test.jsonl}" \
    --workers "${WORKERS:-12}" \
    --limit "${LIMIT:-0}"
echo "[manifest] exit=$? $(date)"
