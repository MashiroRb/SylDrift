"""Build the syllable pool in the learned-encoder space (frozen sylenc)."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa

dev = "cuda" if torch.cuda.is_available() else "cpu"
WIN = 24
class Enc(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(100, 256, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(256, d, 3, padding=1), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(1))
    def forward(self, x): return self.net(x.transpose(1, 2)).squeeze(-1)

enc = Enc().to(dev)
enc.load_state_dict(torch.load("data/sylenc.pt", map_location=dev)["net"]); enc.eval()
meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))

pool = defaultdict(list)
files = sorted(glob.glob("data/f5_ft/mel/*.npy"))[:4000]
for fi, f in enumerate(files):
    base = os.path.splitext(os.path.basename(f))[0]
    text = meta.get(base + ".wav")
    if not text: continue
    mel = np.load(f).astype(np.float32); T = mel.shape[0]
    toks = convert_char_to_pinyin([text])[0]
    if isinstance(toks, str): toks = list(toks)
    if len(toks) < 2 or T < len(toks) * 3: continue
    per = T // len(toks)
    for k, lab in enumerate(toks):
        a, b = k * per, (k + 1) * per
        if b - a < 4: continue
        seg = mel[a:b]
        seg = torch.tensor(seg).T.unsqueeze(0)          # [1,100,T]
        seg = F.interpolate(seg, size=WIN, mode="linear", align_corners=False).transpose(1, 2)  # [1,WIN,100]
        with torch.no_grad():
            e = enc(seg.to(dev))[0].float().cpu().numpy()
        pool[lab].append(e)
    if (fi + 1) % 500 == 0: print("pool", fi + 1, flush=True)

keys = [k for k, v in pool.items() if len(v) >= 16]
arr = {k: np.stack(pool[k][:256]).astype(np.float16) for k in keys}
np.savez_compressed("data/syl_pool_enc.npz", **arr)
print(json.dumps({"syllables": len(pool), "kept": len(keys),
                  "vectors": int(sum(v.shape[0] for v in arr.values()))}))
