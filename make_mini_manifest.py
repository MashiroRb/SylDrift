import glob, json, os, random, sys
sys.path.insert(0, "/home/liudianru26/project/SylDrift")
import numpy as np
from syldrift.audio import load_wav, log_mel
from syldrift.data import uniform_durations

root = "data/aishell3_full/sample/audio"
out = "data/mini.jsonl"
mel_dir = "data/mini_mel"
os.makedirs(mel_dir, exist_ok=True)
wavs = sorted(glob.glob(os.path.join(root, "*.wav")))
random.seed(0)
fake = "今天的天气很好我们去公园散步吧"
n = 0
with open(out, "w", encoding="utf-8") as f:
    for w in wavs:
        utt = os.path.splitext(os.path.basename(w))[0]
        try:
            wav = load_wav(w)
        except Exception:
            continue
        mel = log_mel(wav)
        T = mel.shape[1]
        syl = list(fake[: max(3, T // 12)])
        dur = uniform_durations(len(syl), T)
        mp = os.path.join(mel_dir, utt + ".npy")
        np.save(mp, mel.numpy().astype(np.float16))
        f.write(json.dumps({"wav": w, "utt": utt, "spk": "s0", "syl": syl,
                            "dur": dur, "n_frames": int(T), "mel": mp}, ensure_ascii=False) + "\n")
        n += 1
print("mini manifest", n)
