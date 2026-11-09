"""Learn a discriminative syllable encoder on mel segments (metric for drift).

mel -> whisper mapping is not learnable (linear R2=-1.24, MLP R2=0.09), so the
feature space is learned directly with a syllable-classification objective.
This does not touch the F5 backbone.
"""
import glob, json, os, sys
from collections import Counter
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
dev = "cuda" if torch.cuda.is_available() else "cpu"
meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))

segs, labs = [], []
for w in sorted(glob.glob("data/f5_ft/mel/*.npy"))[:N]:
    base = os.path.splitext(os.path.basename(w))[0]
    text = meta.get(base + ".wav")
    if not text:
        continue
    mel = np.load(w).astype(np.float32)
    toks = convert_char_to_pinyin([text])[0]
    if isinstance(toks, str):
        toks = list(toks)
    T = mel.shape[0]
    if len(toks) < 2 or T < len(toks) * 3:
        continue
    per = T // len(toks)
    for k, lab in enumerate(toks):
        a, b = k * per, (k + 1) * per
        if b - a < 4:
            continue
        segs.append(mel[a:b]); labs.append(lab)

cnt = Counter(labs)
keep = {k for k, v in cnt.items() if v >= 12}
idx = {k: i for i, k in enumerate(sorted(keep))}
pairs = [(s, idx[l]) for s, l in zip(segs, labs) if l in keep]
print("segments", len(pairs), "classes", len(idx), flush=True)

L = 24
def pad(s):
    if s.shape[0] >= L:
        c = s.shape[0] // 2
        return s[max(0, c - L // 2): max(0, c - L // 2) + L]
    return np.pad(s, ((0, L - s.shape[0]), (0, 0)), mode="edge")

X = torch.tensor(np.stack([pad(s) for s, _ in pairs])).float()
Y = torch.tensor([y for _, y in pairs])
print("X", X.shape, flush=True)

class Enc(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(100, 256, 3, padding=1), nn.GELU(),
            nn.Conv1d(256, d, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)

m = Enc().to(dev)
head = nn.Linear(256, len(idx)).to(dev)
opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=1e-3)
nb = max(1, len(X) // 512)
for ep in range(12):
    perm = torch.randperm(len(X))
    tot = acc = 0
    for i in range(0, len(X), 512):
        b = perm[i:i + 512]
        xb, yb = X[b].to(dev), Y[b].to(dev)
        emb = m(xb)
        logits = head(emb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        tot += loss.item() * len(b); acc += (logits.argmax(1) == yb).sum().item()
    print(f"ep {ep+1} loss {tot/len(X):.3f} acc {acc/len(X):.3f}", flush=True)

torch.save({"net": m.state_dict(), "classes": sorted(idx)}, "data/sylenc.pt")
# purity: retrieval on the encoder space
m.eval()
with torch.no_grad():
    E = torch.cat([m(X[i:i+2048].to(dev)).cpu() for i in range(0, len(X), 2048)])
E = F.normalize(E, dim=-1)
cent = F.normalize(torch.stack([E[Y == c].mean(0) for c in Y.unique()]), dim=-1)
pred = (E @ cent.T).argmax(1)
acc = (pred == Y).sum().item() / len(Y)
K = len(Y.unique())
print(json.dumps({"classes": int(K), "probes": int(len(Y)),
                  "top1_acc": round(acc, 4), "chance": round(1 / K, 5),
                  "ratio": round(acc * K, 2)}))
json.dump({"classes": int(K), "probes": int(len(Y)), "top1_acc": round(acc, 4),
           "chance": round(1 / K, 5), "ratio": round(acc * K, 2)},
          open("runs/calib/purity_sylenc.json", "w"), indent=2)
