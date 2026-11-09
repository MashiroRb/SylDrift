"""Rebuild the syllable pool using whisper-derived character boundaries (line B)."""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "third_party/F5-TTS/src")

raw = json.load(open("data/whisper_align_raw.json", encoding="utf-8"))
TS = re.compile(r"<\|(\d+\.\d+)\|>")

def char_spans(tokens_str, text, dur):
    """Walk timestamp tokens, attach a start time to each character of `text`."""
    _, *rest = tokens_str.split("<|startoftranscript|>") if "<|startoftranscript|>" in tokens_str else ([""], [tokens_str])[1][0] if False else (None, [tokens_str])
    parts = re.split(r"(<\|[^|]*\|>)", tokens_str)
    cur_t = None
    seq = []      # (t, piece)
    for p in parts:
        m = TS.fullmatch(p.strip()) if p.strip() else None
        if m:
            cur_t = float(m.group(1)); continue
        if p.startswith("<|") and p.endswith("|>"):
            continue
        if p and cur_t is not None:
            for ch in p:
                if "\u4e00" <= ch <= "\u9fff":
                    seq.append((cur_t, ch))
    if not seq:
        return None
    # align seq chars to our text chars in order
    spans, j = {}, 0
    for ch in text:
        if j < len(seq) and seq[j][1] == ch:
            spans.setdefault(ch, [])
            spans[ch].append(seq[j][0]); j += 1
    times = [t for t, _ in seq]
    return times

out = {}
n_ok = 0
for base, rec in raw.items():
    times = char_spans(rec["tokens"], rec["text"], rec["dur"])
    mp = f"data/f5_ft/mel/{base}.npy"
    if not times or not os.path.exists(mp):
        continue
    out[base] = {"text": rec["text"], "times": times, "dur": rec["dur"]}
    n_ok += 1
json.dump(out, open("data/char_spans.json", "w"), ensure_ascii=False)
print("utterances with spans:", n_ok, "/", len(raw))

# pool on real boundaries
from f5_tts.model.utils import convert_char_to_pinyin  # noqa
pool = defaultdict(list)
FPS = 24000 / 256.0
for base, rec in out.items():
    mel = np.load(f"data/f5_ft/mel/{base}.npy").astype(np.float32)
    T = mel.shape[0]
    labs = convert_char_to_pinyin([rec["text"]])[0]
    if isinstance(labs, str): labs = list(labs)
    ts = rec["times"]
    if len(ts) < 2:
        continue
    # merge consecutive char times into syllable boundaries by proportional split
    edges = np.linspace(0, T, len(labs) + 1)
    # map time -> frame using detected char times scaled to mel length
    tmax = max(ts[-1], 1e-3)
    scale = T / tmax
    edges = np.array([0] + [min(T, int(round(t * scale))) for t in ts] + [T])
    # resample edges to syllable count via interpolation
    syl_edges = np.interp(np.linspace(0, len(edges) - 1, len(labs) + 1), np.arange(len(edges)), edges).astype(int)
    for k, lab in enumerate(labs):
        a, b = syl_edges[k], syl_edges[k + 1]
        if b - a < 3:
            continue
        seg = mel[a:b]
        seg = (seg - seg.mean()) / (seg.std() + 1e-5)
        pool[lab].append(np.concatenate([seg.mean(0), seg.std(0)]))

keys = [k for k, v in pool.items() if len(v) >= 10]
cent = F.normalize(torch.tensor(np.stack([np.stack(pool[k]).mean(0) for k in keys])), dim=-1)
hit = tot = 0
for i, k in enumerate(keys):
    v = F.normalize(torch.tensor(np.stack(pool[k])), dim=-1)
    pred = (v @ cent.T).argmax(1)
    hit += int((pred == i).sum()); tot += len(v)
rep = {"boundary": "whisper word timestamps", "classes": len(keys), "probes": tot,
       "top1_acc": round(hit / tot, 4), "chance": round(1 / len(keys), 5),
       "ratio": round((hit / tot) / (1 / len(keys)), 2) if keys else None}
print(json.dumps(rep, indent=2))
json.dump(rep, open("runs/calib/purity_aligned_mel.json", "w"), indent=2)
