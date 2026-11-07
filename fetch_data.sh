#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
cd /home/liudianru26/project/SylDrift || exit 1
MS=/home/liudianru26/.conda/envs/syldrift/bin/modelscope
mkdir -p data
$MS download --dataset speech_tts/AISHELL-3 --local_dir data/aishell3
echo "DL_EXIT=$?"
du -sh data/aishell3 2>/dev/null
