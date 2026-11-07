#!/bin/bash
#SBATCH --job-name=syldrift_synth
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/synth-%j.out
#SBATCH --error=logs/synth-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
PY=$HOME/.conda/envs/syldrift/bin/python

$PY -m syldrift.synth --ckpt runs/mini/ckpt-last.pt \
  --text "今天的天气很好" --steps 1 --out runs/mini/sample.wav
ls -lh runs/mini/sample.wav 2>/dev/null
echo "[slurm] exit=$?"
