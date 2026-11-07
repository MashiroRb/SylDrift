"""Build a SylDrift manifest from AISHELL-3 (or any <spk>/*.wav + content.txt set).

content.txt columns are tab-separated: utt_id, hanzi, pinyin(with tone marks)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syldrift.audio import load_wav, log_mel  # noqa: E402
from syldrift.data import split_syllables, uniform_durations  # noqa: E402


def read_content(paths):
    table = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                table[parts[0].strip()] = parts[1].strip()
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing train/ and test/")
    ap.add_argument("--out", default="data/manifest.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_sec", type=float, default=12.0)
    ap.add_argument("--min_sec", type=float, default=0.8)
    args = ap.parse_args()

    content = read_content([
        os.path.join(args.root, "train", "content.txt"),
        os.path.join(args.root, "test", "content.txt"),
        os.path.join(args.root, "content.txt"),
    ])
    print(f"[manifest] {len(content)} transcripts")

    mel_dir = os.path.join(os.path.dirname(args.out), "mel")
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(args.root, "**", "*.wav"), recursive=True))
    print(f"[manifest] {len(wavs)} wavs")

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for w in wavs:
            utt = os.path.splitext(os.path.basename(w))[0]
            text = content.get(utt)
            if not text:
                continue
            syl = split_syllables(text)
            if len(syl) < 2:
                continue
            try:
                wav = load_wav(w)
            except Exception:
                continue
            sec = wav.shape[0] / 22050.0
            if sec < args.min_sec or sec > args.max_sec:
                continue
            mel = log_mel(wav)                      # [80, T]
            T = mel.shape[1]
            dur = uniform_durations(len(syl), T)
            spk = os.path.basename(os.path.dirname(w))
            mp = os.path.join(mel_dir, utt + ".npy")
            np.save(mp, mel.numpy().astype(np.float16))
            f.write(json.dumps({"wav": w, "utt": utt, "spk": spk, "syl": syl,
                                "dur": dur, "n_frames": int(T), "mel": mp},
                               ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  {n}", flush=True)
            if args.limit and n >= args.limit:
                break
    print(f"[manifest] wrote {n} -> {args.out}")


if __name__ == "__main__":
    main()
