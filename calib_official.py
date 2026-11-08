"""Step 0a: run F5's own example to prove the pipeline before blaming the model."""
import glob, os, sys, json
import soundfile as sf
import torch

sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.api import F5TTS
from syldrift.eval import ASR, cer

ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device="cuda")
ex = "third_party/F5-TTS/src/f5_tts/infer/examples/basic"
ref = os.path.join(ex, "basic_ref_en.wav")
ref_text = "Some call me nature, others call me mother nature."
gen = "I don't really care what you call me. I've been a silent spectator, watching species evolve."
os.makedirs("runs/calib", exist_ok=True)
wav, sr, _ = f5.infer(ref_file=ref, ref_text=ref_text, gen_text=gen, nfe_step=32,
                      cfg_strength=2.0, sway_sampling_coef=-1.0)
p = "runs/calib/f5_official_en.wav"
sf.write(p, wav, sr)
asr = ASR(language='english')
hyp = asr(p)
print("REF:", gen)
print("HYP:", hyp)
print("WER:", round(cer(gen, hyp), 4))
json.dump({"ref": gen, "hyp": hyp, "wer": cer(gen, hyp)}, open("runs/calib/official.json", "w"), ensure_ascii=False)
