#!/bin/bash
set -x
export http_proxy=http://127.0.0.1:7891
export https_proxy=http://127.0.0.1:7891
export HF_HOME=/home/liudianru26/project/SylDrift/hf_cache
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
$PY -m pip install -q funasr 2>&1 | tail -5
$PY -c "import funasr; print('FUNASR_OK', funasr.__version__)" 2>&1 | tail -3
