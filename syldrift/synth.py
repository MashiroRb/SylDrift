"""Synthesize with a trained SylDrift checkpoint (1-NFE by default)."""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .model import FlowTTS
from .audio import SAMPLE_RATE, N_FFT, HOP, N_MELS, FMIN, FMAX


def mel_to_wav(mel):
    """Griffin-Lim fallback so the pipeline produces audio without a vocoder."""
    inv = torchaudio.transforms.InverseMelScale(
        n_stft=N_FFT // 2 + 1, n_mels=mel.shape[0],
        sample_rate=SAMPLE_RATE, f_min=FMIN, f_max=FMAX)
    spec = inv(torch.exp(mel.clamp(min=-12)))
    gl = torchaudio.transforms.GriffinLim(n_fft=N_FFT, hop_length=HOP, n_iter=64)
    return gl(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="runs/sample.wav")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--frames", type=int, default=0, help="0 = estimate from text")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device)
    a = blob["args"]
    model = FlowTTS(dim=a["dim"], depth=a["depth"]).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    vocab = blob.get("vocab") or json.load(open(os.path.join(os.path.dirname(args.ckpt), "vocab.json")))
    ids = [vocab.get(ch, 1) for ch in args.text]
    text_ids = torch.tensor(ids).unsqueeze(0).to(device)
    T = args.frames or max(20, int(len(ids) * 12))
    torch.manual_seed(args.seed)
    spk = torch.zeros(1, N_MELS, device=device)

    mel = model.sample(text_ids, spk, T, steps=args.steps).squeeze(0).transpose(0, 1)
    wav = mel_to_wav(mel.cpu())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sf.write(args.out, wav.numpy(), SAMPLE_RATE)
    print(f"[synth] {args.out}  frames={T} steps={args.steps} "
          f"dur={wav.shape[-1]/SAMPLE_RATE:.2f}s")


if __name__ == "__main__":
    main()
