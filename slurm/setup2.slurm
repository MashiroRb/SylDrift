#!/bin/bash
#SBATCH --job-name=syldrift_setup2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup2-%j.out
#SBATCH --error=logs/setup2-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export no_proxy=localhost,127.0.0.1

CONDA=/apps/miniconda3/bin/conda
ENV=$HOME/.conda/envs/syldrift

echo "[setup2] start $(date)"
if [ ! -x "$ENV/bin/python" ]; then
  echo "[setup2] cloning aias-env (local, no network)"
  $CONDA create -y -n syldrift --clone aias-env
fi
PY="$ENV/bin/python"
$PY -V

echo "[setup2] installing torch from PyPI"
$PY -m pip install -q --upgrade pip --timeout 60 --retries 10
$PY -m pip install "torch>=2.4" torchaudio --timeout 120 --retries 20
$PY -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available())"

echo "[setup2] installing utils"
$PY -m pip install -q numpy soundfile librosa jiwer --timeout 120 --retries 20
echo "[setup2] done $(date)"
