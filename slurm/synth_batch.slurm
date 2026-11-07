#!/bin/bash
#SBATCH --job-name=syldrift_synthb
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/synthb-%j.out
#SBATCH --error=logs/synthb-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
PY=$HOME/.conda/envs/syldrift/bin/python
CKPT=${CKPT:-runs/base/ckpt-last.pt}
OUTDIR=${OUTDIR:-runs/base/samples}
mkdir -p "$OUTDIR"

REF=$($PY - <<'EOF'
import json
r=[json.loads(l) for l in open("data/manifest_test.jsonl",encoding="utf-8")][:1]
print(r[0]["wav"])
EOF
)
echo "[synth] ref=$REF"
i=0
while IFS=$'\t' read -r utt syl; do
  i=$((i+1))
  $PY -m syldrift.synth --ckpt "$CKPT" --text "$syl" --ref "$REF" \
      --steps 1 --out "$OUTDIR/utt$i.wav" 2>&1 | grep synth
done <<'LIST'
u1	jin1 tian1 tian1 qi4 hen3 hao3
u2	wo3 men5 qu4 gong1 yuan2 san4 bu4 ba5
u3	zhe4 shi4 yi1 ci4 zhong1 wen2 yu3 yin1 he2 cheng2 ce4 shi4
LIST
ls -lh "$OUTDIR"
$PY - <<'EOF'
import glob, soundfile as sf, numpy as np
for p in sorted(glob.glob("runs/base/samples/*.wav")):
    d,sr=sf.read(p); print(p, "dur", round(len(d)/sr,2), "rms", round(float(np.sqrt((d**2).mean())),4))
EOF
echo "[synth] exit=$?"
