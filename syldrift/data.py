"""Manifest building and dataset for SylDrift.

Two alignment sources:
  * mfa: read TextGrid (later, highest quality)
  * uniform: split by syllable count using a pinyin duration prior
    (fallback so the pipeline can run before MFA is installed)

Manifest line (jsonl):
  {"wav":..., "spk":..., "syl":["ni3","hao3"], "dur":[8,6], "n_frames":..., "mel":"..."}
"""
import glob
import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import load_wav, log_mel, mel_frames, N_MELS

# 常用汉字 -> 拼音声母韵母的粗略时长先验（帧，hop=256 @22.05k => 86 fps）
DUR_PRIOR = {
    "a": 9, "o": 9, "e": 9, "i": 8, "u": 8, "v": 9,
    "ai": 10, "ei": 10, "ao": 11, "ou": 10, "an": 10, "en": 9, "ang": 11, "eng": 10, "ong": 11,
}


def split_syllables(text):
    """Character-level syllable split for Mandarin; non-Chinese kept as one chunk."""
    text = re.sub(r"\s+", "", text)
    out = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
        elif ch.strip():
            out.append(ch)
    return out


def uniform_durations(n_syl, n_frames, per_syl=9):
    """Even split with the last syllable absorbing the remainder."""
    if n_syl <= 0:
        return []
    base = max(3, n_frames // n_syl)
    d = [base] * n_syl
    d[-1] = max(3, n_frames - base * (n_syl - 1))
    return d


def build_manifest(wav_dir, transcripts, out_path, limit=None, cache_mel=True):
    """transcripts: dict utt_id -> text"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mel_dir = os.path.join(os.path.dirname(out_path), "mel")
    if cache_mel:
        os.makedirs(mel_dir, exist_ok=True)

    rows = []
    files = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))
    for p in files:
        utt = os.path.splitext(os.path.basename(p))[0]
        text = transcripts.get(utt)
        if not text:
            continue
        syl = split_syllables(text)
        if len(syl) < 2:
            continue
        try:
            wav = load_wav(p)
        except Exception:
            continue
        mel = log_mel(wav)
        T = mel.shape[1] if mel.dim() == 2 else mel.shape[0]
        dur = uniform_durations(len(syl), T)
        mel_path = ""
        if cache_mel:
            mel_path = os.path.join(mel_dir, utt + ".npy")
            np.save(mel_path, mel.numpy().astype(np.float16))
        rows.append({"wav": p, "utt": utt, "spk": utt.split("_")[0] if "_" in utt else "spk0",
                     "syl": syl, "dur": dur, "n_frames": int(T), "mel": mel_path})
        if limit and len(rows) >= limit:
            break

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[manifest] {len(rows)} utts -> {out_path}")
    return len(rows)


class MelDataset(Dataset):
    def __init__(self, manifest, max_frames=800):
        self.rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
        self.max_frames = max_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if r["mel"]:
            m = torch.from_numpy(np.load(r["mel"]).astype(np.float32))
        else:
            m = log_mel(load_wav(r["wav"])).transpose(0, 1)
        if m.dim() == 2 and m.shape[0] == N_MELS:
            m = m.transpose(0, 1)
        m = m[: self.max_frames]
        return {
            "mel": m,
            "syl": r["syl"][: m.shape[0] // 3 + 1],
            "dur": r["dur"][: m.shape[0] // 3 + 1],
            "spk": r["spk"],
            "utt": r["utt"],
        }


def collate(batch):
    T = max(b["mel"].shape[0] for b in batch)
    mel = torch.zeros(len(batch), T, batch[0]["mel"].shape[1])
    for i, b in enumerate(batch):
        mel[i, : b["mel"].shape[0]] = b["mel"]
    return {"mel": mel, "syl": [b["syl"] for b in batch], "dur": [b["dur"] for b in batch]}
