#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
$PY - <<'PY'
from huggingface_hub import snapshot_download
for repo in ["openai/whisper-small", "microsoft/wavlm-base-plus-sv"]:
    try:
        p = snapshot_download(repo, allow_patterns=["*.json","*.txt","*.bin","*.safetensors","*.model"])
        print("OK", repo, p)
    except Exception as e:
        print("FAIL", repo, e)
PY
echo "MODELS_EXIT=$?"
