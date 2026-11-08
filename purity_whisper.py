"""Purity check with a pretrained speech encoder (Whisper) instead of raw mel.

Raw mel gave top-1 retrieval == chance (0.0035 vs 0.00317), so the drift field
had no signal. Whisper's encoder is a strong phonetic feature extractor already
in our cache.
"""
import glob, json, os, random, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa

N_UTT = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
dev = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import WhisperModel, WhisperFeatureExtractor
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
enc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(dev).eval()

meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
wavs = sorted(glob.glob("data/f5_ft/wavs/*.wav"))[:N_UTT]

pool = defaultdict(list)
with torch.no_grad():
    for i, w in enumerate(wavs):
        base = os.path.splitext(os.path.basename(w))[0]
        text = meta.get(base + ".wav")
        if not text:
            continue
        labs = convert_char_to_pinyin([text])[0]
        if isinstance(labs, str):
            labs = list(labs)
        if len(labs) < 2:
            continue
        x, sr = sf.read(w, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
            sr = 16000
        inp = fe(x, sampling_rate=16000, return_tensors="pt")
        h = enc(inp.input_features.to(dev)).last_hidden_state[0]      # [T', 768]
        T = h.shape[0]
        per = T // len(labs)
        if per < 2:
            continue
        for k, lab in enumerate(labs):
            a, b = k * per, (k + 1) * per
            pool[lab].append(h[a:b].mean(0).float().cpu().numpy())
        if (i + 1) % 200 == 0:
            print("enc", i + 1, flush=True)

keys = [k for k, v in pool.items() if len(v) >= 20]
print(f"[pool] syllables={len(pool)} with>=20={len(keys)}", flush=True)
random.seed(0)
sample = random.sample(keys, min(100, len(keys)))
within, cross = [], []
for lab in sample:
    v = F.normalize(torch.tensor(np.stack(pool[lab])), dim=-1)
    a = v[torch.randint(0, len(v), (64,))]
    b = v[torch.randint(0, len(v), (64,))]
    within.append(float((a * b).sum(-1).mean()))
for i in range(len(sample) - 1):
    va = F.normalize(torch.tensor(np.stack(pool[sample[i]])), dim=-1)
    vb = F.normalize(torch.tensor(np.stack(pool[sample[i + 1]])), dim=-1)
    m = min(64, len(va), len(vb))
    cross.append(float((va[:m] * vb[:m]).sum(-1).mean()))
cent = F.normalize(torch.tensor(np.stack([np.stack(pool[k]).mean(0) for k in keys])), dim=-1)
hit = tot = 0
for i, k in enumerate(keys):
    v = F.normalize(torch.tensor(np.stack(pool[k])), dim=-1)
    pred = (v @ cent.T).argmax(1)
    hit += int((pred == i).sum()); tot += len(v)
rep = {"feature": "whisper-small encoder", "classes": len(keys), "probes": tot,
       "within": round(float(np.mean(within)), 4), "cross": round(float(np.mean(cross)), 4),
       "top1_acc": round(hit / tot, 4), "chance": round(1 / len(keys), 5)}
print(json.dumps(rep, indent=2))
os.makedirs("runs/calib", exist_ok=True)
json.dump(rep, open("runs/calib/purity_whisper.json", "w"), indent=2)
