#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export no_proxy=localhost,127.0.0.1
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
cd /home/liudianru26/project/SylDrift

$PY -m pip install -q --upgrade pip
$PY -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$PY -c "import torch;print('TORCH_OK',torch.__version__)"
$PY -m pip install -q soundfile librosa jiwer numpy
$PY -m pip install -q modelscope
echo "SETUP_EXIT=$?"
