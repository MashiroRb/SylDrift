#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
cd /home/liudianru26/project/SylDrift || exit 1
git clone --depth 1 https://github.com/SWivid/F5-TTS.git third_party/F5-TTS 2>&1 | tail -3
/home/liudianru26/.conda/envs/syldrift/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("SWivid/F5-TTS")
print("F5_OK", p)
PY
