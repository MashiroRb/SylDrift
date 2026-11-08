"""Step 0.3: verify the reference audio really says what we pass as ref_text."""
import json, os, sys
import soundfile as sf
sys.path.insert(0, ".")
from syldrift.eval import ASR, cer

hz = json.load(open("data/hanzi.json", encoding="utf-8"))
rows = [json.loads(l) for l in open("data/split_test.jsonl", encoding="utf-8")][:6]
asr = ASR(language="chinese")
for r in rows:
    ref = hz.get(r["utt"], "")
    hyp = asr(r["wav"])
    print(f"{r['utt']}  CER={cer(ref, hyp):.3f}")
    print("  ref_text :", ref[:40])
    print("  asr(audio):", hyp[:40])
