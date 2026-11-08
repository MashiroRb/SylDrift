"""Precompute F5 teacher output (32 NFE) for distillation targets."""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.api import F5TTS
from f5_tts.model.utils import convert_char_to_pinyin

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SHARD = os.environ.get("SHARD", "0/1")
_k, _n = [int(x) for x in SHARD.split("/")]
OUT = "data/f5_ft/teacher"
os.makedirs(OUT, exist_ok=True)
P = 86

ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device="cuda")
cfm = f5.ema_model.float()
vocab = cfm.vocab_char_map

meta = [l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8")]
if LIMIT:
    meta = meta[:LIMIT]
if _n > 1:
    meta = meta[_k::_n]
n = 0
for name, text in meta:
    base = os.path.splitext(name)[0]
    dst = os.path.join(OUT, base + ".npy")
    if os.path.exists(dst):
        n += 1
        continue
    mp = os.path.join("data/f5_ft/mel", base + ".npy")
    if not os.path.exists(mp):
        continue
    mel = torch.from_numpy(np.load(mp).astype(np.float32)).unsqueeze(0)
    if mel.shape[1] <= P + 10:
        continue
    toks = convert_char_to_pinyin([text])[0]
    if isinstance(toks, str):
        toks = list(toks)
    ids = torch.tensor([[vocab.get(c, vocab.get("<unk>", 0)) for c in toks]], device="cuda")
    cond = mel[:, :P].cuda()
    dur = mel.shape[1] - P
    with torch.no_grad():
        out, _traj = cfm.sample(cond=cond, text=ids, duration=dur, steps=32, cfg_strength=1.0)
    np.save(dst, out[0, -dur:].float().cpu().numpy().astype(np.float16))
    n += 1
    if n % 100 == 0:
        print("teacher", n, flush=True)
print("total", n, "shard", SHARD)
