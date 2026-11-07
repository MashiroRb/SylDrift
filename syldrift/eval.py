"""Evaluate synthesized audio: CER (ASR round-trip) and speaker similarity.

Models are loaded lazily; any missing dependency is reported, not fatal.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

CER_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-small")
SIM_MODEL = os.environ.get("SIM_MODEL", "microsoft/wavlm-base-plus-sv")


def norm_text(s):
    s = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9]", "", s)
    return s.lower()


def edit_distance(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    ref, hyp = norm_text(ref), norm_text(hyp)
    return edit_distance(ref, hyp) / max(1, len(ref))


class ASR:
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=CER_MODEL,
                             device=0 if torch.cuda.is_available() else -1)

    def __call__(self, wav_path):
        out = self.pipe(wav_path, generate_kwargs={"language": "chinese"})
        return out["text"]


class SpeakerSim:
    def __init__(self):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        self.feat = Wav2Vec2FeatureExtractor.from_pretrained(SIM_MODEL)
        self.model = WavLMForXVector.from_pretrained(SIM_MODEL)
        self.model.eval()

    @torch.no_grad()
    def embed(self, wav_path):
        import soundfile as sf
        w, sr = sf.read(wav_path)
        if w.ndim > 1:
            w = w.mean(1)
        inp = self.feat(w, sampling_rate=sr, return_tensors="pt")
        return self.model(**inp).embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="jsonl with text + reference wav")
    ap.add_argument("--gen_dir", required=True, help="dir with <utt>.wav")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="runs/eval.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")][: args.limit]
    res = {"n": 0, "cer": [], "sim": [], "missing": 0}

    asr = sim = None
    try:
        asr = ASR()
    except Exception as e:
        print("[warn] ASR unavailable:", e)
    try:
        sim = SpeakerSim()
    except Exception as e:
        print("[warn] speaker model unavailable:", e)

    for r in rows:
        p = os.path.join(args.gen_dir, r["utt"] + ".wav")
        if not os.path.exists(p):
            res["missing"] += 1
            continue
        if asr:
            hyp = asr(p)
            res["cer"].append(cer("".join(r["syl"]), hyp))
        if sim:
            e1, e2 = sim.embed(p), sim.embed(r["wav"])
            res["sim"].append(float(torch.cosine_similarity(e1, e2).item()))
        res["n"] += 1
        if res["n"] % 10 == 0:
            print(f"  {res['n']}", flush=True)

    summary = {
        "n": res["n"],
        "missing": res["missing"],
        "cer": float(np.mean(res["cer"])) if res["cer"] else None,
        "sim": float(np.mean(res["sim"])) if res["sim"] else None,
    }
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
