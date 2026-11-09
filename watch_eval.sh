#!/bin/bash
# Evaluate each 500-step checkpoint as it appears (3 seeds x 10 utts, 4 NFE).
DIR=$1; NFE=$2; TAG=$3
for K in 500 1000 1500 2000 2500 3000; do
  CK="$DIR/model_${K}.pt"
  for i in $(seq 1 240); do [ -f "$CK" ] && break; sleep 30; done
  [ -f "$CK" ] || { echo "skip $K (never appeared)"; continue; }
  for S in 1 2 3; do
    $HOME/.conda/envs/syldrift/bin/python -m syldrift.eval_testset \
      --tag "${TAG}_${K}_s${S}" --nfe $NFE --n 10 --seed $S \
      --asr openai/whisper-large-v3 --ckpt "$CK" \
      --out "runs/pure_eval/${TAG}_${K}_s${S}.json" 2>&1 | grep -E '^\[nfe' | tail -1
  done
  echo "evaluated $TAG @ $K"
done
