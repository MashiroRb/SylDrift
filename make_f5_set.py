"""Resample N training utterances to 24 kHz and write F5 metadata."""
import json, os, sys
import numpy as np
import soundfile as sf
import torch
import torchaudio

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
out_dir = "data/f5_ft"
os.makedirs(os.path.join(out_dir, "wavs"), exist_ok=True)
hz = json.load(open("data/hanzi.json", encoding="utf-8"))
rows = [json.loads(l) for l in open("data/split_train.jsonl", encoding="utf-8")]

lines, i = [], 0
for r in rows:
    if i >= N:
        break
    t = hz.get(r["utt"])
    if not t:
        continue
    name = f"{i:06d}.wav"
    dst = os.path.join(out_dir, "wavs", name)
    if not os.path.exists(dst):
        try:
            w, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        except Exception:
            continue
        w = w.mean(1)
        if sr != 24000:
            w = torchaudio.functional.resample(torch.from_numpy(w), sr, 24000).numpy()
        sf.write(dst, w, 24000)
    lines.append(f"{name}|{t}")
    i += 1

with open(os.path.join(out_dir, "raw.arrow.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("prepared", len(lines))
