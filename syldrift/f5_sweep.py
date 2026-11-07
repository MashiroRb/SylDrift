"""NFE sweep for F5-TTS: CER/SIM vs number of function evaluations.

This is the baseline curve our one-step fine-tuning has to move.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch

SYL = re.compile(r"^[a-z]+[1-5]$")


def hanzi_map(content_paths):
    m = {}
    for p in content_paths:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                utt = parts[0].replace(".wav", "")
                hz = "".join(t for t in parts[1].split() if not SYL.match(t))
                if hz:
                    m[utt] = hz
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--out", default="runs/f5_nfe_sweep.json")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()

    from f5_tts.api import F5TTS
    from syldrift.eval import ASR, cer

    ckpt = args.ckpt or glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt,
               device="cuda" if torch.cuda.is_available() else "cpu")
    asr = ASR()

    hz = hanzi_map(["data/raw_extract/test/content.txt"])
    rows = [json.loads(l) for l in open("data/manifest_test.jsonl", encoding="utf-8")][: args.n]
    wavmap = {}
    for root, _, files in os.walk("data/raw_extract/test/wav"):
        for f in files:
            if f.endswith(".wav"):
                wavmap[f[:-4]] = os.path.join(root, f)

    # reference = first utterance, target = the rest of the texts
    ref_utt = rows[0]["utt"]
    ref_wav, ref_text = wavmap[ref_utt], hz.get(ref_utt, "")
    results = {nfe: [] for nfe in args.nfe}
    os.makedirs("runs/f5_sweep", exist_ok=True)

    for i, r in enumerate(rows[1:], 1):
        text = hz.get(r["utt"])
        if not text:
            continue
        for nfe in args.nfe:
            p = f"runs/f5_sweep/{args.tag}_u{i}_nfe{nfe}.wav"
            wav, sr, _ = f5.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=text,
                                  nfe_step=nfe, cfg_strength=2.0, sway_sampling_coef=-1.0)
            sf.write(p, wav, sr)
            hyp = asr(p)
            results[nfe].append(cer(text, hyp))
            print(f"[{i}] nfe={nfe} cer={results[nfe][-1]:.3f}", flush=True)

    summary = {str(k): {"cer": float(np.mean(v)) if v else None, "n": len(v)}
               for k, v in results.items()}
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
