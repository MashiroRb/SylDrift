"""Intelligibility check: synthesize from pinyin, transcribe with Whisper, compare
against the reference hanzi.

CER is computed on characters after stripping punctuation/space, which is the
convention used by the Chinese TTS leaderboards.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syldrift.model import FlowTTS  # noqa: E402
from syldrift.synth import mel_to_wav  # noqa: E402
from syldrift.audio import load_wav, SAMPLE_RATE, N_FFT, HOP, N_MELS, FMIN, FMAX  # noqa: E402
from syldrift.eval import cer, ASR  # noqa: E402

SYL = re.compile(r"^[a-z]+[1-5]$")


def pairs_from_content(path, n):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        toks = parts[1].split()
        hanzi = "".join(t for t in toks if not SYL.match(t))
        syl = [t for t in toks if SYL.match(t)]
        if len(syl) >= 4 and hanzi:
            out.append((parts[0].replace(".wav", ""), hanzi, syl))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--content", default="data/partial/test/content.txt")
    ap.add_argument("--hanzi_json", default="data/hanzi.json")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--oracle_dur", type=int, default=0)
    ap.add_argument("--wavdir", default="data/partial/test/wav")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="runs/eval_cer.json")
    args = ap.parse_args()

    device = "cpu"
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = blob["args"]
    vocab = blob.get("vocab") or json.load(
        open(os.path.join(os.path.dirname(args.ckpt), "vocab.json")))
    model = FlowTTS(dim=a["dim"], depth=a["depth"], text_vocab=max(256, len(vocab))).to(device).eval()
    model.load_state_dict(blob["model"])

    if args.manifest:
        hz = json.load(open(args.hanzi_json, encoding="utf-8")) if os.path.exists(args.hanzi_json) else {}
        rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")][: args.n]
        samples = [(r["utt"], hz.get(r["utt"], ""), r["syl"], r["wav"], r.get("n_frames", 0))
                   for r in rows if hz.get(r["utt"])]
    else:
        samples = [(u, h, s, None, 0) for (u, h, s) in pairs_from_content(args.content, args.n)]
    asr = ASR()
    scores, details = [], []
    for i, (utt, hanzi, syl, ref_direct, nf) in enumerate(samples, 1):
        ref = ref_direct
        if ref is None:
            for root, _, files in os.walk(args.wavdir):
                if utt + ".wav" in files:
                    ref = os.path.join(root, utt + ".wav")
                    break
        if ref is None:
            continue
        wav = load_wav(ref)[: SAMPLE_RATE].unsqueeze(0)
        m = torchaudio_mel(wav)
        spk = torch.log(m.clamp_min(1e-5)).mean(-1).unsqueeze(0)

        ids = [vocab.get(s, 1) for s in syl]
        # duration oracle when the manifest provides it: separates "did the model
        # learn text->speech" from "can we estimate duration"
        T = int(nf) if (args.oracle_dur and nf) else max(20, int(len(ids) * 12))
        ids_t = torch.tensor(ids).unsqueeze(0)
        with torch.no_grad():
            mel = model.sample(ids_t, spk, T, steps=1, frame_ids=uniform_frames(ids_t, T))
        w = mel_to_wav(mel.squeeze(0).transpose(0, 1))
        p = f"/tmp/cer_{i}.wav"
        sf.write(p, w.numpy(), SAMPLE_RATE)

        hyp = asr(p)
        c = cer(hanzi, hyp)
        scores.append(c)
        details.append({"utt": utt, "ref": hanzi, "hyp": hyp, "cer": round(c, 4)})
        print(f"[{i}/{len(samples)}] cer={c:.3f} ref={hanzi[:20]} hyp={hyp[:20]}", flush=True)

    summary = {"n": len(scores), "cer_mean": float(np.mean(scores)) if scores else None,
               "details": details}
    json.dump(summary, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))


def uniform_frames(ids, T):
    B, L = ids.shape
    out = torch.zeros(B, T, dtype=torch.long)
    per = T // L
    for k in range(L):
        a = k * per
        b = T if k == L - 1 else (k + 1) * per
        out[:, a:b] = ids[:, k:k + 1]
    return out


def torchaudio_mel(wav):
    import torchaudio
    t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=FMIN, f_max=FMAX)
    return t(wav).squeeze(0)


if __name__ == "__main__":
    main()
