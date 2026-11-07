"""Build a SylDrift manifest from AISHELL-3 style content.txt.

content.txt line:  <utt>.wav \t <hanzi> <pinyin> <hanzi> <pinyin> ...
We keep the toned pinyin tokens: they are the syllable inventory the drift
field is conditioned on (~1300 keys), so homophones share their pool.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syldrift.audio import load_wav, log_mel  # noqa: E402
from syldrift.data import uniform_durations  # noqa: E402

SYL = re.compile(r"^[a-z]+[1-5]$")


def parse_line(text):
    return [t for t in text.split() if SYL.match(t)]


def read_content(paths):
    table = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            syl = parse_line(parts[1])
            if len(syl) >= 2:
                table[parts[0].strip().replace(".wav", "")] = syl
    return table


def _work(args_chunk):
    import json as _json
    out = []
    for w, syl, mel_dir in args_chunk:
        try:
            wav = load_wav(w)
        except Exception:
            continue
        sec = wav.shape[0] / 22050.0
        if not (0.8 <= sec <= 12.0):
            continue
        mel = log_mel(wav)
        T = mel.shape[1]
        if T < len(syl) * 3:
            continue
        utt = os.path.splitext(os.path.basename(w))[0]
        mp = os.path.join(mel_dir, utt + ".npy")
        np.save(mp, mel.numpy().astype(np.float16))
        out.append({"wav": w, "utt": utt, "spk": os.path.basename(os.path.dirname(w)),
                    "syl": syl, "dur": uniform_durations(len(syl), T),
                    "n_frames": int(T), "mel": mp})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/manifest.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--min_sec", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    content = read_content(glob.glob(os.path.join(args.root, "**", "content.txt"), recursive=True))
    print(f"[manifest] {len(content)} transcripts")

    mel_dir = os.path.join(os.path.dirname(args.out), "mel")
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(args.root, "**", "*.wav"), recursive=True))
    print(f"[manifest] {len(wavs)} wavs")

    jobs = []
    for w in wavs:
        utt = os.path.splitext(os.path.basename(w))[0]
        syl = content.get(utt)
        if syl:
            jobs.append((w, syl, mel_dir))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[manifest] {len(jobs)} jobs, workers={args.workers}", flush=True)

    from multiprocessing import Pool
    nshard = max(1, args.workers * 4)
    shards = [jobs[i::nshard] for i in range(nshard)]
    NL = chr(10)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f, Pool(args.workers) as pool:
        for k, rows in enumerate(pool.imap_unordered(_work, shards), 1):
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + NL)
            n += len(rows)
            print(f"  shard {k}/{nshard} -> {n}", flush=True)
    print(f"[manifest] wrote {n} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
