#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export no_proxy=localhost,127.0.0.1
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
cd /home/liudianru26/project/SylDrift || exit 1
mkdir -p wheels

TORCH=https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl
TAUDIO=https://download.pytorch.org/whl/cu121/torchaudio-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl

for i in 1 2 3 4 5 6; do
  curl -L -C - --retry 30 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
       -o wheels/torch.whl "$TORCH" && break
  echo "torch retry $i"
done
for i in 1 2 3 4 5; do
  curl -L -C - --retry 30 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
       -o wheels/torchaudio.whl "$TAUDIO" && break
  echo "torchaudio retry $i"
done
ls -lh wheels/

$PY -m pip install --no-deps wheels/torch.whl wheels/torchaudio.whl
$PY -m pip install -q filelock sympy networkx jinja2 fsspec typing-extensions
$PY -c "import torch;print('TORCH_OK', torch.__version__, torch.cuda.is_available())"

$PY -m pip install -q numpy soundfile librosa jiwer
$PY -m pip install -q modelscope
echo "SETUP2_EXIT=$?"
