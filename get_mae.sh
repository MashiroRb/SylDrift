#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
$PY - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["hance-ai/audiomae_as2m", "facebook/audiomae"]:
    try:
        p = snapshot_download(repo, allow_patterns=["*.json","*.bin","*.safetensors","*.txt","*.pt"])
        print("MAE_OK", repo, p)
        break
    except Exception as e:
        print("MAE_FAIL", repo, type(e).__name__, str(e)[:120])
PY
