"""Persist Whisper-encoder syllable pool (the only feature space that beat chance)."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
import torch
import soundfile as sf
sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa
from transformers import WhisperModel, WhisperFeatureExtractor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
dev = "cuda" if torch.cuda.is_available() else "cpu"
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
enc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(dev).eval()

meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
wavs = sorted(glob.glob("data/f5_ft/wavs/*.wav"))[:N]
pool = defaultdict(list)
with torch.no_grad():
    for i, w in enumerate(wavs):
        base = os.path.splitext(os.path.basename(w))[0]
        text = meta.get(base + ".wav")
        if not text: continue
        labs = convert_char_to_pinyin([text])[0]
        if isinstance(labs, str): labs = list(labs)
        if len(labs) < 2: continue
        x, sr = sf.read(w, dtype="float32", always_2d=True); x = x.mean(1)
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        h = enc(fe(x, sampling_rate=16000, return_tensors="pt").input_features.to(dev)).last_hidden_state[0]
        T = h.shape[0]; per = T // len(labs)
        if per < 2: continue
        for k, lab in enumerate(labs):
            pool[lab].append(h[k*per:(k+1)*per].mean(0).float().cpu().numpy())
        if (i+1) % 500 == 0: print("enc", i+1, flush=True)

keys = sorted([k for k, v in pool.items() if len(v) >= 16])
arr = {k: np.stack(pool[k][:256]).astype(np.float16) for k in keys}
np.savez_compressed("data/syl_pool_w.npz", **arr)
json.dump({k: i for i, k in enumerate(keys)}, open("data/syl_pool_w_index.json", "w"), ensure_ascii=False)
print(json.dumps({"syllables": len(pool), "kept": len(keys),
                  "vectors": int(sum(v.shape[0] for v in arr.values())),
                  "dim": int(arr[keys[0]].shape[1])}, indent=2))
