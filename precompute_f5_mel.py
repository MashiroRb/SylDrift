"""Precompute F5-vocos log-mels for the fine-tuning set (uses the model's own spec)."""
import glob, os, sys
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.api import F5TTS  # noqa: E402

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device="cpu")
    ms = f5.ema_model.mel_spec
    wavs = sorted(glob.glob("data/f5_ft/wavs/*.wav"))
    if limit:
        wavs = wavs[:limit]
    out = "data/f5_ft/mel"
    os.makedirs(out, exist_ok=True)
    n = 0
    for w in wavs:
        base = os.path.splitext(os.path.basename(w))[0]
        dst = os.path.join(out, base + ".npy")
        if os.path.exists(dst):
            n += 1
            continue
        x, sr = sf.read(w, dtype="float32", always_2d=True)
        t = torch.from_numpy(x.mean(1)).unsqueeze(0)
        with torch.no_grad():
            mel = ms(t)[0].transpose(0, 1)      # [T, n_mel]
        np.save(dst, mel.float().numpy().astype(np.float16))
        n += 1
        if n % 500 == 0:
            print("done", n, flush=True)
    print("total", n)

if __name__ == "__main__":
    main()
