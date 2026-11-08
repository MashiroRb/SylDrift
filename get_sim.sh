#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
/home/liudianru26/.conda/envs/syldrift/bin/python - <<'PY'
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("microsoft/wavlm-base-plus-sv",
                          allow_patterns=["*.json","*.txt","*.bin","*.safetensors","*.model"])
    print("SIM_OK", p)
except Exception as e:
    print("SIM_FAIL", type(e).__name__, str(e)[:150])
PY
