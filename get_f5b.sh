#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
for i in 1 2 3 4 5; do
  $PY - <<'PY'
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("SWivid/F5-TTS", allow_patterns=["F5TTS_v1_Base/*", "*.txt", "*.md"],
                          max_workers=2)
    print("F5_OK", p)
except Exception as e:
    print("F5_RETRY", type(e).__name__, str(e)[:200])
PY
  if [ -f /home/liudianru26/project/SylDrift/hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_1200000.safetensors ]; then
    echo "WEIGHTS_PRESENT"; break
  fi
  sleep 5
done
ls -lh /home/liudianru26/project/SylDrift/hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/ 2>/dev/null
