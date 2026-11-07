#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
cd /home/liudianru26/project/SylDrift/third_party/F5-TTS || exit 1
$PY -m pip install . 2>&1 | tail -20
echo "PIP_EXIT=$?"
$PY -c "import f5_tts; print('F5 OK')" 2>&1 | tail -5
