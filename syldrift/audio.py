"""Audio front-end: wav -> log-mel, shared by data prep and evaluation."""
import numpy as np
import torch
import torchaudio


SAMPLE_RATE = 22050
N_FFT = 1024
HOP = 256
N_MELS = 80
FMIN, FMAX = 0.0, 8000.0


def load_wav(path, sr=SAMPLE_RATE):
    wav, orig_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav.squeeze(0)


def log_mel(wav, n_mels=N_MELS):
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
        n_mels=n_mels, f_min=FMIN, f_max=FMAX, power=2.0,
    )
    m = spec(wav.unsqueeze(0)).squeeze(0)
    return torch.log(m.clamp_min(1e-5))


def mel_frames(n_samples):
    return int(np.ceil(n_samples / HOP))
