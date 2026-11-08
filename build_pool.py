"""Step 1: syllable segment pool + purity self-check.

Boundaries: uniform split over characters (MFA unavailable in the time budget) --
this is flagged explicitly in the report. Purity is measured, not assumed.
"""
import glob, json, os, random, sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402

MEL = "data/f5_ft/mel"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000


def syl_labels(text):
    toks = convert_char_to_pinyin([text])[0]
    if isinstance(toks, str):
        toks = list(toks)
    return toks


def main():
    meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(MEL, "*.npy")))[:N]
    pool = defaultdict(list)
    segs = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        text = meta.get(base + ".wav")
        if not text:
            continue
        mel = np.load(f).astype(np.float32)          # [T, 100]
        # per-utterance CMVN: raw mel means are dominated by speaker/channel
        # (measured: within-syllable cosine 0.027 vs cross 0.019 -> useless)
        mel = (mel - mel.mean(0, keepdims=True)) / (mel.std(0, keepdims=True) + 1e-5)
        labs = syl_labels(text)
        if len(labs) < 2 or mel.shape[0] < len(labs) * 3:
            continue
        T = mel.shape[0]
        per = T // len(labs)
        for k, lab in enumerate(labs):
            a, b = k * per, (k + 1) * per
            if b - a < 3:
                continue
            # per-segment standardisation keeps spectral shape, drops level
            seg_frames = mel[a:b]
            seg_frames = (seg_frames - seg_frames.mean()) / (seg_frames.std() + 1e-5)
            seg = np.concatenate([seg_frames.mean(0), seg_frames.std(0)])
            pool[lab].append(seg)
            segs.append((lab, seg))

    keys = [k for k, v in pool.items() if len(v) >= 32]
    print(f"[pool] syllables={len(pool)} with>=32 samples={len(keys)} segments={len(segs)}", flush=True)

    random.seed(0)
    sample = random.sample(keys, min(100, len(keys)))
    within, cross = [], []
    for lab in sample:
        v = torch.tensor(np.stack(pool[lab]))
        v = F.normalize(v - v.mean(0, keepdim=True), dim=-1)
        a = v[torch.randint(0, len(v), (64,))]
        b = v[torch.randint(0, len(v), (64,))]
        within.append(float((a * b).sum(-1).mean()))
    for i in range(len(sample) - 1):
        va = torch.tensor(np.stack(pool[sample[i]]))
        vb = torch.tensor(np.stack(pool[sample[i + 1]]))
        va = F.normalize(va - va.mean(0, keepdim=True), dim=-1)
        vb = F.normalize(vb - vb.mean(0, keepdim=True), dim=-1)
        m = min(64, len(va), len(vb))
        cross.append(float((va[:m] * vb[:m]).sum(-1).mean()))

    import statistics as st
    rep = {
        "boundary_method": "uniform-split-by-character (MFA not available in budget; FLAGGED)",
        "syllables_total": len(pool),
        "syllables_ge32": len(keys),
        "segments": len(segs),
        "within_syl_sim": round(st.mean(within), 4),
        "cross_syl_sim": round(st.mean(cross), 4),
        "margin": round(st.mean(within) - st.mean(cross), 4),
        "n_probe_syllables": len(sample),
    }
    os.makedirs("runs/calib", exist_ok=True)
    json.dump(rep, open("runs/calib/pool_report.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))

    # persist a compact pool for training
    idx = {k: i for i, k in enumerate(keys)}
    arr = {k: np.stack([s for s in pool[k][:256]]).astype(np.float16) for k in keys}
    np.savez_compressed("data/syl_pool.npz", **arr)
    json.dump(idx, open("data/syl_pool_index.json", "w"), ensure_ascii=False)
    print("[pool] saved data/syl_pool.npz", sum(v.shape[0] for v in arr.values()))


if __name__ == "__main__":
    main()
