"""F5-TTS baseline: synthesize Chinese with a reference clip from AISHELL-3.

Establishes the quality ceiling our drift fine-tuning must match or beat.
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


def hanzi_of(content_path, utt):
    for line in open(content_path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0].replace(".wav", "") == utt:
            return "".join(t for t in parts[1].split() if not SYL.match(t))
    return ""


def find_wav(root, utt):
    hits = glob.glob(os.path.join(root, "**", utt + ".wav"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--content", default="data/raw_extract/test/content.txt")
    ap.add_argument("--wavroot", default="data/raw_extract/test/wav")
    ap.add_argument("--gen_text", default="今天天气很好，我们一起去公园散步吧。")
    ap.add_argument("--out", default="runs/f5_baseline.wav")
    ap.add_argument("--nfe", type=int, default=32)
    args = ap.parse_args()

    from f5_tts.api import F5TTS
    ckpt = args.ckpt or glob.glob(
        "hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    print("[f5] ckpt:", ckpt, flush=True)
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, device="cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in open("data/manifest_test.jsonl", encoding="utf-8")][:1]
    utt = rows[0]["utt"]
    ref = find_wav(args.wavroot, utt)
    ref_text = hanzi_of(args.content, utt)
    print(f"[f5] ref={ref} text={ref_text}", flush=True)

    wav, sr, _ = f5.infer(ref_file=ref, ref_text=ref_text, gen_text=args.gen_text,
                          nfe_step=args.nfe, cfg_strength=2.0, sway_sampling_coef=-1.0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, wav, sr)
    print(f"[f5] saved {args.out} dur={len(wav)/sr:.2f}s sr={sr}", flush=True)

    try:
        from syldrift.eval import ASR, cer
        hyp = ASR()(args.out)
        print(f"[f5] asr={hyp}  cer={cer(args.gen_text, hyp):.3f}", flush=True)
    except Exception as e:
        print("[f5] asr skipped:", e, flush=True)


if __name__ == "__main__":
    main()
