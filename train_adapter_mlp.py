"""Non-linear adapter mel->whisper (linear one had R2=-1.24, i.e. useless)."""
import glob, json, os, sys
import numpy as np, torch, torch.nn as nn, soundfile as sf
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
        mel = torch.from_numpy(np.load(mp).astype(np.float32))
        x, sr = sf.read(w, dtype="float32", always_2d=True); x = x.mean(1)
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        h = enc(fe(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev)).last_hidden_state[0]
        T = mel.shape[0]
        h2 = torch.nn.functional.interpolate(h.T.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0).T
        Xs.append(mel); Ys.append(h2.cpu())
X = torch.cat(Xs); Y = torch.cat(Ys)
mu, sd = X.mean(0), X.std(0).clamp_min(1e-5)
X = (X - mu) / sd
ymu, ysd = Y.mean(0), Y.std(0).clamp_min(1e-5)
Yn = (Y - ymu) / ysd

net = nn.Sequential(nn.Linear(100, 512), nn.GELU(), nn.Linear(512, 1280)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
X, Yn = X.to(dev), Yn.to(dev)
n = X.shape[0]
for ep in range(30):
    perm = torch.randperm(n, device=dev)
    tot = 0.0
    for i in range(0, n, 8192):
        idx = perm[i:i + 8192]
        pred = net(X[idx])
        loss = ((pred - Yn[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    if ep % 10 == 9:
        with torch.no_grad():
            pred = net(X)
            r2 = 1 - ((Yn - pred) ** 2).sum() / ((Yn - Yn.mean(0)) ** 2).sum()
        print(f"epoch {ep+1} mse {tot/n:.4f} r2 {r2.item():.4f}", flush=True)
torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd}, "data/mel2whisper_mlp.pt")
print("saved data/mel2whisper_mlp.pt")
