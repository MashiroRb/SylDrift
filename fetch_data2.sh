#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
cd /home/liudianru26/project/SylDrift || exit 1
MS=/home/liudianru26/.conda/envs/syldrift/bin/modelscope
$MS download --dataset OmniData/AISHELL-3 --local_dir data/aishell3_full
echo "DL_EXIT=$?"
du -sh data/aishell3_full
