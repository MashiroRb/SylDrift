"""Evaluate a checkpoint with the SAME one-step procedure used in distillation.

Using the utterance's own first 86 frames as the prompt isolates "did distill
training reduce error" from "does the prompt transfer".
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
from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402
from syldrift.eval import ASR, cer  # noqa: E402

P = 86


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="student checkpoint (.pt), empty = base")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=base_ck, device=dev)
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location=dev, weights_only=False)
        state = blob.get("ema_model_state_dict", blob)
        missing, unexpected = f5.ema_model.load_state_dict(state, strict=False)
        print(f"[eval] loaded {args.ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    cfm = f5.ema_model.float().eval()
    voc = f5.vocoder

    # evaluation set = utterances that already have cached mels (f5_ft build)
    meta = [l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8")]
    rows = [{"utt": os.path.splitext(n)[0], "text": t} for n, t in meta][: args.n * 3]
    hz = {r["utt"]: r["text"] for r in rows}
    asr = ASR()
    scores = []
    os.makedirs("runs/dist_eval", exist_ok=True)
    for i, r in enumerate(rows):
        mp = os.path.join("data/f5_ft/mel", r["utt"] + ".npy")
        if not os.path.exists(mp):
            continue
        mel = torch.from_numpy(np.load(mp).astype(np.float32)).unsqueeze(0).to(dev)
        if mel.shape[1] <= P + 20:
            continue
        dur = mel.shape[1] - P
        text = hz.get(r["utt"], "")
        if not text:
            continue
        toks = convert_char_to_pinyin([text])[0]
        if isinstance(toks, str):
            toks = list(toks)
        ids = torch.tensor([[cfm.vocab_char_map.get(c, cfm.vocab_char_map.get("<unk>", 0))
                             for c in toks]], device=dev)

        cond = mel[:, :P]
        x = torch.randn(1, P + dur, mel.shape[-1], device=dev)
        mask = torch.zeros(1, P + dur, 1, device=dev, dtype=torch.bool)
        mask[:, :P] = True
        step_cond = cond.new_zeros(1, P + dur, mel.shape[-1])
        step_cond[:, :P] = cond
        x = torch.where(mask, step_cond, x)
        with torch.no_grad():
            for k in range(args.steps):
                t = torch.full((1,), k / args.steps, device=dev)
                v = cfm.transformer(x=x, cond=step_cond, text=ids, time=t, mask=None,
                                    drop_audio_cond=False, drop_text=False, cache=False)
                x = torch.where(mask, step_cond, x + v / args.steps)
            gen = x[:, -dur:]
            wav = voc.decode(gen.transpose(1, 2))
        p = f"runs/dist_eval/{args.tag}_{i}.wav"
        sf.write(p, wav.squeeze().float().cpu().numpy(), 24000)
        hyp = asr(p)
        c = cer(text, hyp)
        scores.append(c)
        print(f"[{i}] cer={c:.3f} ref={text[:18]} hyp={hyp[:18]}", flush=True)

    summary = {"tag": args.tag, "n": len(scores), "cer": float(np.mean(scores)) if scores else None}
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
