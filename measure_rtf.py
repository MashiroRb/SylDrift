"""Measured RTF: single core, batch 1, same runtime, for several NFE."""
import json, os, sys, time
import soundfile as sf, torch
sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.api import F5TTS
import glob

ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device="cpu")
rows = [json.loads(l) for l in open("data/testset30.jsonl", encoding="utf-8")][:3]
res = {}
for nfe in (4, 8, 16, 32):
    times, durs = [], []
    for r in rows:
        t0 = time.perf_counter()
        wav, sr, _ = f5.infer(ref_file=r["ref_wav"], ref_text=r["ref_text"],
                              gen_text=r["text"], nfe_step=nfe,
                              cfg_strength=2.0, sway_sampling_coef=-1.0, seed=1234)
        dt = time.perf_counter() - t0
        times.append(dt); durs.append(len(wav) / sr)
    rtf = sum(times) / sum(durs)
    res[str(nfe)] = {"rtf": round(rtf, 3), "wall_s": round(sum(times), 2),
                     "audio_s": round(sum(durs), 2), "speedup_vs_32": None}
    print(f"nfe {nfe}: rtf {rtf:.3f} (wall {sum(times):.1f}s / audio {sum(durs):.1f}s)", flush=True)
base = res["32"]["rtf"]
for k in res:
    res[k]["speedup_vs_32"] = round(base / res[k]["rtf"], 2)
print(json.dumps(res, indent=2))
json.dump(res, open("runs/calib/rtf.json", "w"), indent=2)
