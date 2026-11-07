"""Prepare a 24 kHz fine-tuning set for F5 from our splits."""
import json, os, sys, subprocess

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
out_wav = "data/f5_ft/wavs"
os.makedirs(out_wav, exist_ok=True)

hz = json.load(open("data/hanzi.json", encoding="utf-8"))
rows = [json.loads(l) for l in open("data/split_train.jsonl", encoding="utf-8")][:N]

lines = []
for i, r in enumerate(rows):
    t = hz.get(r["utt"])
    if not t:
        continue
    dst = os.path.join(out_wav, f"{i:06d}.wav")
    if not os.path.exists(dst):
        # 16k -> 24k resample, mono
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", r["wav"], "-ar", "24000", "-ac", "1", dst]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            continue
    lines.append(f"{i:06d}.wav|{t}")

with open("data/f5_ft/metadata.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("prepared", len(lines), "utts")
