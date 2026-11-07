#!/bin/bash
#SBATCH --job-name=syldrift_setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export no_proxy=localhost,127.0.0.1

CONDA=/apps/miniconda3/bin/conda
TUNA=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main

echo "[setup] start $(date) host=$(hostname)"
if [ ! -x "$HOME/.conda/envs/syldrift/bin/python" ]; then
  $CONDA create -y -n syldrift python=3.10 --override-channels -c "$TUNA"
fi
PY="$HOME/.conda/envs/syldrift/bin/python"
$PY -V

$PY -m pip install -q --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "[setup] installing torch via aliyun pytorch-wheels"
$PY -m pip install torch==2.5.1 torchaudio==2.5.1 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu121/torch_stable.html \
    || $PY -m pip install torch==2.5.1 torchaudio==2.5.1 \
       --index-url https://download.pytorch.org/whl/cu121
$PY -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available())"

echo "[setup] done $(date)"
