"""Fixed test-set evaluation: CER / length ratio / (optional) SIM at several NFE.

Uses data/testset30.jsonl built by build_testset.py: 30 utterances from unseen
speakers, each with a reference clip from the same speaker but different text.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "third_party/F5-TTS/src")
sys.path.insert(0, ".")
from f5_tts.api import F5TTS  # noqa: E402
from syldrift.eval import ASR, cer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--nfe", nargs="+", type=int, default=[32, 4, 2])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="")
    ap.add_argument("--asr", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = args.ckpt or glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device=dev)
    asr = ASR(model=args.asr or None, language="chinese")

    rows = [json.loads(l) for l in open("data/testset30.jsonl", encoding="utf-8")][: args.n]
    outdir = f"runs/eval_{args.tag}"
    os.makedirs(outdir, exist_ok=True)

    results = {}
    for nfe in args.nfe:
        cers, ratios = [], []
        for i, r in enumerate(rows):
            if not r.get("ref_wav"):
                continue
            wav, sr, _ = f5.infer(ref_file=r["ref_wav"], ref_text=r["ref_text"],
                                  gen_text=r["text"], nfe_step=nfe,
                                  cfg_strength=2.0, sway_sampling_coef=-1.0)
            p = os.path.join(outdir, f"nfe{nfe}_{i:02d}.wav")
            sf.write(p, wav, sr)
            hyp = asr(p)
            c = cer(r["text"], hyp)
            cers.append(c)
            ratios.append(len(hyp) / max(1, len(r["text"])))
            print(f"[nfe{nfe}][{i}] cer={c:.3f} len={ratios[-1]:.2f}", flush=True)
        results[str(nfe)] = {
            "n": len(cers),
            "cer": float(np.mean(cers)) if cers else None,
            "cer_std": float(np.std(cers)) if cers else None,
            "len_ratio": float(np.mean(ratios)) if ratios else None,
        }
        print(json.dumps({k: results[str(nfe)] for k in ["n", "cer", "len_ratio"]}), flush=True)

    results["tag"] = args.tag
    results["asr"] = args.asr or "openai/whisper-small"
    dst = args.out or f"runs/calib/report_{args.tag}.json"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(results, open(dst, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
