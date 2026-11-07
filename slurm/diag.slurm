#!/bin/bash
#SBATCH --job-name=syldrift_diag
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/diag-%j.out
#SBATCH --error=logs/diag-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
PY=$HOME/.conda/envs/syldrift/bin/python
REF=$($PY -c "import json;print(json.loads(open('data/manifest_test.jsonl').readline())['wav'])")
echo "[diag] ref=$REF"
$PY -m syldrift.diag --ckpt "${CKPT:-runs/long/ckpt-63000.pt}" --ref "$REF"
echo "[diag] exit=$?"
