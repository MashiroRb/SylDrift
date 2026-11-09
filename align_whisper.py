"""Line B step 1 fallback: whisper word timestamps -> character time spans.

FunASR fa-zh was installed but its input contract could not be satisfied within
the 1h budget, so we fall back to whisper timestamps (allowed by the task).
"""
import glob, json, os, sys
import numpy as np, torch, soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
dev = "cuda" if torch.cuda.is_available() else "cpu"
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(dev).eval()
model.generation_config.language = "zh"
model.generation_config.task = "transcribe"
model.generation_config.return_timestamps = True
model.generation_config.forced_decoder_ids = None
meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))

out = {}
wavs = sorted(glob.glob("data/f5_ft/wavs/*.wav"))[:N]
for i, w in enumerate(wavs):
    base = os.path.splitext(os.path.basename(w))[0]
    text = meta.get(base + ".wav")
    if not text:
        continue
    x, sr = sf.read(w, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        import torchaudio
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        sr = 16000
    feats = proc(x, sampling_rate=sr, return_tensors="pt").input_features.to(dev)
    with torch.no_grad():
        ids = model.generate(feats, return_timestamps=True, max_new_tokens=200)
    toks = proc.batch_decode(ids, skip_special_tokens=False)[0]
    out[base] = {"text": text, "tokens": toks, "dur": len(x) / sr}
    if (i + 1) % 100 == 0:
        print("align", i + 1, flush=True)
json.dump(out, open("data/whisper_align_raw.json", "w"), ensure_ascii=False)
print("saved", len(out))
