"""Diagnostic: is the text actually influencing the output?

Generates the same length from different texts with a fixed seed and measures
the mel distance. If the distance is near zero, the conditioning has collapsed.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from syldrift.model import FlowTTS  # noqa: E402
from syldrift.audio import load_wav, N_MELS, SAMPLE_RATE, N_FFT, HOP, FMIN, FMAX  # noqa: E402


def uniform_frames(ids, T):
    B, L = ids.shape
    out = torch.zeros(B, T, dtype=torch.long)
    per = T // L
    for k in range(L):
        a = k * per
        b = T if k == L - 1 else (k + 1) * per
        out[:, a:b] = ids[:, k:k + 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = blob["args"]
    vocab = blob.get("vocab") or json.load(open(os.path.join(os.path.dirname(args.ckpt), "vocab.json")))
    model = FlowTTS(dim=a["dim"], depth=a["depth"], text_vocab=max(256, len(vocab))).eval()
    model.load_state_dict(blob["model"])

    import torchaudio
    wav = load_wav(args.ref)[: SAMPLE_RATE].unsqueeze(0)
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        f_min=FMIN, f_max=FMAX)
    spk = torch.log(mel_t(wav).squeeze(0).clamp_min(1e-5)).mean(-1, keepdim=True).transpose(0, 1)

    texts = {
        "A": "jin1 tian1 tian1 qi4 hen3 hao3".split(),
        "B": "wo3 men5 qu4 gong1 yuan2 san4 bu4 ba5".split(),
        "C": "zhe4 shi4 yi1 ci4 zhong1 wen2 yu3 yin1 he2 cheng2".split(),
    }
    torch.set_grad_enabled(False)
    outs = {}
    for k, syl in texts.items():
        ids = torch.tensor([[vocab.get(s, 1) for s in syl]])
        torch.manual_seed(0)
        fi = uniform_frames(ids, args.frames)
        outs[k] = model.sample(ids, spk, args.frames, steps=1, frame_ids=fi)

    for x, y in [("A", "B"), ("A", "C"), ("B", "C")]:
        d = F.mse_loss(outs[x], outs[y]).item()
        cos = F.cosine_similarity(outs[x].flatten(), outs[y].flatten(), dim=0).item()
        print(f"text {x} vs {y}: mse={d:.4f} cos={cos:.4f}")

    torch.manual_seed(1)
    ids = torch.tensor([[vocab.get(s, 1) for s in texts["A"]]])
    alt = model.sample(ids, spk, args.frames, steps=1, frame_ids=uniform_frames(ids, args.frames))
    print(f"same text, different seed: mse={F.mse_loss(outs['A'], alt).item():.4f}")

    print(f"output std={outs['A'].std().item():.4f} mean={outs['A'].mean().item():.4f}")


if __name__ == "__main__":
    main()
