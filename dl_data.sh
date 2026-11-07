#!/bin/bash
set -x
cd /home/liudianru26/project/SylDrift
python3 -m pip install --user -q modelscope 2>&1 | tail -3
export PATH=$HOME/.local/bin:$PATH
modelscope download --dataset speech_tts/AISHELL-3 --local_dir data/aishell3
echo "EXIT=$?"
