#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
$PY - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("openai/whisper-small",
                      allow_patterns=["*.json","*.txt","*.safetensors","*.bin","*.model"])
print("SNAP", p)
import glob
print("FILES", [(f.split("/")[-1], __import__("os").path.getsize(f)) for f in glob.glob(p+"/*")])
PY
echo "WHISPER_DONE"
