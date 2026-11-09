#!/bin/bash
# usage: multiseed.sh <tag> <ckpt|""> <nfe>
set -uo pipefail
TAG=$1; CKPT=$2; NFE=$3
for S in 1 2 3 4 5; do
  $HOME/.conda/envs/syldrift/bin/python -m syldrift.eval_testset \
    --tag "${TAG}_s${S}" --nfe $NFE --n 30 --seed $S \
    --asr openai/whisper-large-v3 --ckpt "$CKPT" \
    --out "runs/calib/ms_${TAG}_s${S}.json" 2>&1 | grep -E '^\[nfe' | tail -1
  echo "done seed $S"
done
