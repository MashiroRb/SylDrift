#!/bin/bash
set -x
cd /home/liudianru26/project/SylDrift || exit 1
PY=/home/liudianru26/.conda/envs/syldrift/bin/python
TGZ=data/aishell3_full/raw/data_aishell3.tgz

echo "[wait] $(date) waiting for download to finish"
while [ ! -f "$TGZ" ] || [ -f "$TGZ.incomplete" ]; do sleep 60; done
echo "[wait] done $(date)"
ls -lh "$TGZ"

echo "[extract] $(date)"
mkdir -p data/raw_extract
tar xzf "$TGZ" -C data/raw_extract
echo "[extract] done $(date)"
ls data/raw_extract

ROOT=$(dirname "$(find data/raw_extract -name content.txt | head -1)")
echo "[manifest] root=$ROOT"
$PY -m syldrift.make_manifest --root "$ROOT" --out data/manifest.jsonl --limit 20000
wc -l data/manifest.jsonl
echo "POST_DOWNLOAD_DONE $(date)"
