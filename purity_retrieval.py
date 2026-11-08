"""Purity as top-1 retrieval accuracy over syllable classes (chance = 1/K)."""
import glob, json, os, sys, random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa

meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
files = sorted(glob.glob("data/f5_ft/mel/*.npy"))[:4000]
pool = defaultdict(list)
for f in files:
    base = os.path.splitext(os.path.basename(f))[0]
    text = meta.get(base + ".wav")
    if not text: continue
    mel = np.load(f).astype(np.float32)
    labs = convert_char_to_pinyin([text])[0]
    if isinstance(labs, str): labs = list(labs)
    T = mel.shape[0]
    if len(labs) < 2 or T < len(labs)*3: continue
    per = T // len(labs)
    for k, lab in enumerate(labs):
        a, b = k*per, (k+1)*per
        if b-a < 3: continue
        seg = mel[a:b]
        seg = (seg - seg.mean())/(seg.std()+1e-5)
        pool[lab].append(np.concatenate([seg.mean(0), seg.std(0)]))
keys = [k for k,v in pool.items() if len(v) >= 40]
cent = torch.tensor(np.stack([np.stack(pool[k]).mean(0) for k in keys]))
cent = F.normalize(cent, dim=-1)
random.seed(0)
hit = tot = 0
for i,k in enumerate(keys):
    v = torch.tensor(np.stack(pool[k]))
    v = (v - v.mean(0, keepdim=True))/ (v.std(0, keepdim=True)+1e-6)
    v = F.normalize(v, dim=-1)
    sims = v @ cent.T
    pred = sims.argmax(1)
    hit += int((pred == i).sum()); tot += len(v)
rep = {"classes": len(keys), "probes": tot, "top1_acc": round(hit/tot, 4), "chance": round(1/len(keys), 5)}
print(json.dumps(rep, indent=2))
json.dump(rep, open("runs/calib/purity_retrieval.json","w"), indent=2)
