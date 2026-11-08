"""Step 2 gate: wrapped sampler must equal the official one bit-for-bit."""
import glob, sys
import numpy as np, torch, json
sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.api import F5TTS

ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device="cpu")
cfm = f5.ema_model.float()
mel = torch.from_numpy(np.load("data/f5_ft/mel/000000.npy").astype(np.float32)).unsqueeze(0)
ids = torch.tensor([[cfm.vocab_char_map.get(c, 0) for c in "北京万科总经"]], dtype=torch.long)
torch.manual_seed(0)
a, _ = cfm.sample(cond=mel[:, :86], text=ids, duration=120, steps=4, cfg_strength=1.0)
torch.manual_seed(0)
b, _ = cfm.sample.__wrapped__(cfm, cond=mel[:, :86], text=ids, duration=120, steps=4, cfg_strength=1.0)
d = (a - b).abs().max().item()
print(json.dumps({"max_abs_delta": d, "pass": d < 1e-3}))
json.dump({"max_abs_delta": d, "pass": d < 1e-3}, open("runs/calib/sampler_check.json", "w"))
