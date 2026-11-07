#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
cd /home/liudianru26/project/SylDrift || exit 1
SNAP=hf_cache/hub/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base
mkdir -p "$SNAP"
rm -f "$SNAP/model_1200000.safetensors"
URL=https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/model_1250000.safetensors
for i in 1 2 3 4 5 6 7 8; do
  curl -L -C - --retry 20 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
    -o "$SNAP/model_1250000.safetensors" "$URL" && break
  echo "RETRY $i"
  sleep 3
done
ls -lh "$SNAP"
echo "F5D_DONE"
