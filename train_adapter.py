"""Closed-form linear adapter: mel frame -> whisper-large-v3 frame feature.

Why: the syllable pool lives in whisper space (mel space was 1.1x chance, i.e.
useless). The generator emits mel, so we need a differentiable, frozen map into
whisper space to compute the drift. Fitted by least squares on real pairs.
"""
import glob, json, os, sys
import numpy as np, torch, soundfile as sf
from transformers import WhisperModel, WhisperFeatureExtractor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
dev = "cuda" if torch.cuda.is_available() else "cpu"
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
enc = WhisperModel.from_pretrained("openai/whisper-large-v3").encoder.to(dev).eval()

meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
wavs = sorted(glob.glob("data/f5_ft/wavs/*.wav"))[:N]

Xs, Ys = [], []
with torch.no_grad():
    for w in wavs:
        base = os.path.splitext(os.path.basename(w))[0]
        mp = f"data/f5_ft/mel/{base}.npy"
        if not os.path.exists(mp):
            continue
        mel = torch.from_numpy(np.load(mp).astype(np.float32))       # [T,100]
        x, sr = sf.read(w, dtype="float32", always_2d=True); x = x.mean(1)
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        h = enc(fe(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev)).last_hidden_state[0]
        T = mel.shape[0]
        h2 = torch.nn.functional.interpolate(h.T.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0).T
        Xs.append(mel); Ys.append(h2.cpu())
        if len(Xs) % 100 == 0:
            print("pair", len(Xs), flush=True)

X = torch.cat(Xs); Y = torch.cat(Ys)
mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-5)
Xn = (X - mu) / sd
# solve Xn @ W = Y  (ridge)
lam = 1e-2
A = Xn.T @ Xn + lam * torch.eye(Xn.shape[1])
B = Xn.T @ Y
W = torch.linalg.solve(A, B)
pred = Xn @ W
r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
torch.save({"W": W, "mu": mu, "sd": sd}, "data/mel2whisper.pt")
print(json.dumps({"pairs": len(Xs), "frames": int(X.shape[0]), "r2": round(float(r2), 4)}))
