"""Step 2: SylDrift loss on top of F5 distillation.

L = 1.0*||gen - teacher||^2 + lambda * L_drift

Drift field lives in frozen Whisper-encoder space (mel space measured at chance).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402
from syldrift.drift import soft_assignment, pool_by_syllable, drift_loss  # noqa: E402

P = 86


class Set(Dataset):
    def __init__(self, items, vocab):
        self.items, self.vocab = items, vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        mel_p, teach_p, text, wav = self.items[i]
        toks = convert_char_to_pinyin([text])[0]
        if isinstance(toks, str):
            toks = list(toks)
        return {"mel": torch.from_numpy(np.load(mel_p).astype(np.float32)),
                "teach": torch.from_numpy(np.load(teach_p).astype(np.float32)),
                "ids": torch.tensor([self.vocab.get(c, 0) for c in toks]),
                "wav": wav, "text": text, "nsyl": len(toks)}


def collate(b):
    B = len(b)
    T = max(x["mel"].shape[0] for x in b)
    Tt = max(x["teach"].shape[0] for x in b)
    L = max(x["ids"].numel() for x in b)
    mel = torch.zeros(B, T, 100); teach = torch.zeros(B, Tt, 100)
    ids = torch.zeros(B, L, dtype=torch.long); ns = torch.zeros(B, dtype=torch.long)
    for i, x in enumerate(b):
        mel[i, :x["mel"].shape[0]] = x["mel"]
        teach[i, :x["teach"].shape[0]] = x["teach"]
        ids[i, :x["ids"].numel()] = x["ids"]
        ns[i] = x["nsyl"]
    return {"mel": mel, "teach": teach, "ids": ids, "nsyl": ns,
            "wav": [x["wav"] for x in b], "text": [x["text"] for x in b]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lambda_drift", type=float, default=0.1)
    ap.add_argument("--lam_rep", type=float, default=0.5)
    ap.add_argument("--out", default="runs/f5_drift")
    ap.add_argument("--selfcheck", type=int, default=1)
    args = ap.parse_args()

    from f5_tts.api import F5TTS
    from transformers import WhisperModel, WhisperFeatureExtractor
    import soundfile as sf

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device=dev)
    cfm = f5.ema_model.float()
    for p in cfm.parameters():
        p.requires_grad_(True)

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    wenc = WhisperModel.from_pretrained("openai/whisper-small").encoder.to(dev).eval()
    for p in wenc.parameters():
        p.requires_grad_(False)

    pool_npz = np.load("data/syl_pool_w.npz")
    pool_idx = json.load(open("data/syl_pool_w_index.json", encoding="utf-8"))
    print(f"[drift] pool syllables {len(pool_idx)}", flush=True)

    meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
    items = []
    for tp in sorted(glob.glob("data/f5_ft/teacher/*.npy")):
        base = os.path.splitext(os.path.basename(tp))[0]
        mp = os.path.join("data/f5_ft/mel", base + ".npy")
        wp = os.path.join("data/f5_ft/wavs", base + ".wav")
        t = meta.get(base + ".wav")
        if os.path.exists(mp) and t and os.path.exists(wp):
            items.append((mp, tp, t, wp))
    print(f"[drift] items {len(items)}", flush=True)
    ds = Set(items, cfm.vocab_char_map)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)

    def feats(wavs):
        xs = []
        for w in wavs:
            x, sr = sf.read(w, dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 16000:
                import torchaudio
                x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
            xs.append(fe(x, sampling_rate=16000, return_tensors="pt").input_features[0])
        n = min(t.shape[-1] for t in xs)
        batch = torch.stack([t[:, :n] for t in xs]).to(dev)
        with torch.no_grad():
            return wenc(batch).last_hidden_state          # [B, T', 768] frozen

    tau_raw = torch.tensor(0.0, device=dev, requires_grad=True)
    opt = AdamW(list(cfm.parameters()) + [tau_raw], lr=args.lr, betas=(0.9, 0.98))
    warm = max(50, int(0.05 * args.steps))
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
    ema = {k: v.detach().clone().float() for k, v in cfm.state_dict().items()}
    os.makedirs(args.out, exist_ok=True)

    step, bank = 0, None
    while step < args.steps:
        for b in dl:
            step += 1
            mel = b["mel"].to(dev); teach = b["teach"].to(dev); ids = b["ids"].to(dev)
            dur = teach.shape[1]
            cond = mel[:, :P]

            out, _ = cfm.sample.__wrapped__(cfm, cond=cond, text=ids, duration=dur,
                                            steps=1, cfg_strength=1.0)
            gen = out[:, -dur:]
            l_rec = F.mse_loss(gen, teach)

            # ---- drift ---------------------------------------------------- #
            h = feats(b["wav"])                             # frozen features
            with torch.no_grad():
                h_ref = h
            # syllable-pooled generated features via soft assignment on a uniform grid
            B = gen.shape[0]
            K = int(b["nsyl"].min().item())
            bounds = []
            for i in range(B):
                d = torch.full((K,), dur / K, device=dev)
                bounds.append(torch.cat([torch.zeros(1, device=dev), d.cumsum(0)]))
            bounds = torch.stack(bounds)
            A = soft_assignment(bounds, gen.shape[1], tau=1.5)      # [B,K,T_mel]
            # project mel-frame assignment onto whisper frames by linear interpolation
            A_w = F.interpolate(A, size=h.shape[1], mode="linear", align_corners=False)
            pooled = pool_by_syllable(h, A_w)                        # [B,K,768] (h frozen)
            flat = pooled.reshape(B * K, -1).detach().requires_grad_(False)

            tau = tau_raw.exp().clamp(0.05, 5.0)
            pos = []
            for i in range(B):
                lab = b["text"][i]
                toks = convert_char_to_pinyin([lab])[0]
                if isinstance(toks, str):
                    toks = list(toks)
                for t_ in toks[:K]:
                    arr = pool_npz[t_] if t_ in pool_npz.files else None
                    if arr is None:
                        arr = np.random.randn(32, 768).astype(np.float16)
                    if arr.shape[0] < 32:
                        reps = int(np.ceil(32 / arr.shape[0]))
                        arr = np.tile(arr, (reps, 1))
                    pos.append(arr[:32])
            pos = torch.tensor(np.stack(pos), device=dev, dtype=torch.float32)
            # rolling negative bank: keep a fixed-size set of past generated vectors
            if bank is None:
                bank = flat.detach().clone()
            else:
                bank = torch.cat([bank, flat.detach()], 0)[-256:]
            negb = bank.unsqueeze(0).expand(flat.shape[0], -1, -1)
            l_drift = drift_loss(flat, pos, negb, tau)

            loss = l_rec + args.lambda_drift * l_drift
            if not torch.isfinite(loss):
                print(f"[skip] non-finite at {step}", flush=True)
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.selfcheck and step == 1:
                gn = sum((p.grad ** 2).sum().item() for p in cfm.parameters() if p.grad is not None) ** 0.5
                print(f"[selfcheck] grad_norm={gn:.4e} tau_grad={tau_raw.grad.item():.4e} "
                      f"l_drift={l_drift.item():.4f} l_rec={l_rec.item():.4f}", flush=True)
                assert gn > 0 and abs(tau_raw.grad.item()) > 0, "drift/tau gradient is zero"
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for k, v in cfm.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.999).add_(v.float(), alpha=0.001)
                    else:
                        ema[k] = v.clone()
            if step % 25 == 0:
                print(f"[step {step}] rec={l_rec.item():.4f} drift={l_drift.item():.4f} "
                      f"tau={tau.item():.3f} lam={args.lambda_drift}", flush=True)
            if step % 200 == 0:
                torch.save({"ema_model_state_dict": {k: v.to(cfm.state_dict()[k].dtype) for k, v in ema.items()}},
                           os.path.join(args.out, f"model_{step}.pt"))
            if step >= args.steps:
                break
    torch.save({"ema_model_state_dict": {k: v.to(cfm.state_dict()[k].dtype) for k, v in ema.items()}},
               os.path.join(args.out, "model_last.pt"))
    print("[drift] done", flush=True)


if __name__ == "__main__":
    main()
