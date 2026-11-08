#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
$PY - <<'PY'
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("openai/whisper-large-v3",
                          allow_patterns=["*.json","*.txt","*.safetensors","*.model"])
    print("JUDGE_OK", p)
except Exception as e:
    print("JUDGE_FAIL", type(e).__name__, str(e)[:200])
PY
