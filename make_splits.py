"""Disjoint train/val/test manifests (speaker-level split for val)."""
import json, random, sys, os
src_train, src_test = sys.argv[1], sys.argv[2]
out_dir = sys.argv[3] if len(sys.argv) > 3 else "data"
val_frac = 0.05

rows = [json.loads(l) for l in open(src_train, encoding="utf-8")]
spks = sorted({r["spk"] for r in rows})
random.seed(13); random.shuffle(spks)
n_val = max(1, int(len(spks) * val_frac))
val_spks = set(spks[:n_val])
train = [r for r in rows if r["spk"] not in val_spks]
val = [r for r in rows if r["spk"] in val_spks]
test = [json.loads(l) for l in open(src_test, encoding="utf-8")]

def dump(rs, name):
    p = os.path.join(out_dir, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rs))

dump(train, "split_train.jsonl")
dump(val, "split_val.jsonl")
dump(test, "split_test.jsonl")

tr_spk = {r["spk"] for r in train}
va_spk = {r["spk"] for r in val}
te_spk = {r["spk"] for r in test}
print("speakers train/val/test:", len(tr_spk), len(va_spk), len(te_spk))
print("overlap train&test utts:", len({r["utt"] for r in train} & {r["utt"] for r in test}))
print("overlap val&test utts:", len({r["utt"] for r in val} & {r["utt"] for r in test}))
