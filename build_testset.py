"""Step 0: build a proper fixed test set (10-40 hanzi, one reference per speaker)."""
import json, os, random, collections

hz = json.load(open("data/hanzi.json", encoding="utf-8"))
rows = [json.loads(l) for l in open("data/split_test.jsonl", encoding="utf-8")]
by_spk = collections.defaultdict(list)
for r in rows:
    t = hz.get(r["utt"], "")
    if 10 <= len(t) <= 40:
        by_spk[r["spk"]].append((r, t))

random.seed(7)
picked = []
for spk, items in by_spk.items():
    random.shuffle(items)
    picked.extend(items[:1])
random.shuffle(picked)
picked = picked[:30]

# a reference clip per speaker: another utterance by the same speaker (different text)
ref_for = {}
for spk in {p[0]["spk"] for p in picked}:
    cands = [r for r in rows if r["spk"] == spk and hz.get(r["utt"]) and 6 <= len(hz[r["utt"]]) <= 30]
    if cands:
        ref_for[spk] = random.choice(cands)

out = []
for r, t in picked:
    ref = ref_for.get(r["spk"])
    out.append({"utt": r["utt"], "text": t, "wav": r["wav"], "n_frames": r["n_frames"],
                "ref_wav": ref["wav"] if ref else "", "ref_text": hz.get(ref["utt"], "") if ref else ""})
with open("data/testset30.jsonl", "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("testset", len(out), "speakers", len({o['utt'][:7] for o in out}))
print("len dist", sorted(len(o["text"]) for o in out)[:5], "...", sorted(len(o["text"]) for o in out)[-5:])
