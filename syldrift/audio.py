"""Audio front-end: wav -> log-mel, shared by data prep and evaluation."""
import numpy as np
import torch
import torchaudio


SAMPLE_RATE = 22050
N_FFT = 1024
HOP = 256
N_MELS = 80
FMIN, FMAX = 0.0, 8000.0


_MEL = None


def _mel():
    global _MEL
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0)
    return _MEL


def load_wav(path, sr=SAMPLE_RATE):
    import soundfile as sf
    data, orig_sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.mean(axis=1))
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav


def log_mel(wav, n_mels=N_MELS):
    m = _mel()(wav.unsqueeze(0)).squeeze(0)
    return torch.log(m.clamp_min(1e-5))


def mel_frames(n_samples):
    return int(np.ceil(n_samples / HOP))
