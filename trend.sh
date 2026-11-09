#!/bin/bash
# quick trend: 3 seeds x 10 utts at 4 NFE
CK=$1; TAG=$2
for S in 1 2 3; do
  $HOME/.conda/envs/syldrift/bin/python -m syldrift.eval_testset \
    --tag "${TAG}_s${S}" --nfe 4 --n 10 --seed $S --asr openai/whisper-large-v3 \
    --ckpt "$CK" --out "runs/trend/${TAG}_s${S}.json" 2>&1 | grep -E '^\[nfe' | tail -1
done
