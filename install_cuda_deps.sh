#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
cd /home/liudianru26/project/SylDrift || exit 1
for i in 1 2 3 4 5; do
  $PY -m pip install --retries 20 --timeout 120 \
    nvidia-cudnn-cu12==9.1.0.70 nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-cusolver-cu12==11.4.5.107 nvidia-nccl-cu12==2.21.5 \
    nvidia-nvjitlink-cu12 nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 \
    nvidia-nvtx-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 \
    && break
  echo "RETRY $i"
done
$PY -c "import torch;print('TORCH_OK',torch.__version__,torch.cuda.is_available())"
echo "DEPS_EXIT=$?"
