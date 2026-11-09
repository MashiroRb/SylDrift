"""PURE DRIFT training on F5 (no teacher, no regression term).

L = L_syl + beta * L_stat,  lambda_fm = 0, beta = 0.5

Feature space: a frozen discriminative syllable encoder trained on mel segments
(retrieval 9.0x chance). Mel->whisper mapping was not learnable (linear R2=-1.24,
MLP R2=0.09), so the space is learned directly; F5 backbone is untouched.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402
from syldrift.drift import soft_assignment, drift_loss  # noqa: E402

P = 86
WIN = 24


class SylEnc(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(100, 256, 3, padding=1), nn.GELU(),
            nn.Conv1d(256, d, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


class Set(Dataset):
    def __init__(self, items, vocab):
        self.items, self.vocab = items, vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        mel_p, text = self.items[i]
        toks = convert_char_to_pinyin([text])[0]
        if isinstance(toks, str):
            toks = list(toks)
        return {"mel": torch.from_numpy(np.load(mel_p).astype(np.float32)),
                "ids": torch.tensor([self.vocab.get(c, 0) for c in toks]),
                "nsyl": len(toks)}


def collate(b):
    B = len(b)
    T = max(x["mel"].shape[0] for x in b)
    L = max(x["ids"].numel() for x in b)
    mel = torch.zeros(B, T, 100)
    ids = torch.zeros(B, L, dtype=torch.long)
    ns = torch.zeros(B, dtype=torch.long)
    for i, x in enumerate(b):
        mel[i, :x["mel"].shape[0]] = x["mel"]
        ids[i, :x["ids"].numel()] = x["ids"]
        ns[i] = x["nsyl"]
    return {"mel": mel, "ids": ids, "nsyl": ns}


def mel_stats(m):
    """4 global statistics: energy mean/std, roughness, spectral tilt."""
    e = m.mean(-1)                                    # [B,T]
    e_mu = e.mean(1)
    e_std = e.std(1)
    rough = (m[:, 1:] - m[:, :-1]).abs().mean(-1).mean(1)
    freqs = torch.linspace(0, 1, m.shape[-1], device=m.device)
    cen = (m.clamp_min(-12).exp() * freqs).sum(-1) / m.clamp_min(-12).exp().sum(-1).clamp_min(1e-6)
    tilt = cen.mean(1)
    return torch.stack([e_mu, e_std, rough, tilt], 1)   # [B,4]


def l_stat(gen, ref):
    vg, vr = mel_stats(gen), mel_stats(ref)
    sig = vr.std(0).clamp_min(1e-6)
    return (((vg - vr) / sig) ** 2).sum(1).mean()


def syl_windows(mel, nsyl):
    """uniform syllable windows -> resampled [B,K,WIN,100]"""
    B, T, D = mel.shape
    K = int(nsyl.min().item())
    outs = []
    for b in range(B):
        edges = torch.linspace(0, T, K + 1, device=mel.device).round().long()
        segs = [mel[b, edges[k]:max(edges[k] + 2, edges[k + 1])] for k in range(K)]
        segs = [F.interpolate(s.T.unsqueeze(0), size=WIN, mode="linear",
                              align_corners=False).squeeze(0).T for s in segs]
        outs.append(torch.stack(segs))
    return torch.stack(outs)          # [B,K,WIN,100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lam_rep", type=float, default=0.5)
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pure_drift")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from f5_tts.api import F5TTS
    ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device=dev)
    cfm = f5.ema_model.float()
    # NFE-step sampling keeps N forward graphs alive; without checkpointing a
    # 24G card OOMs at batch 4 (measured).
    try:
        cfm.transformer.checkpoint_activations = True
        print("[mem] gradient checkpointing ON", flush=True)
    except Exception as e:
        print("[mem] checkpointing unavailable:", e, flush=True)
    for p in cfm.parameters():
        p.requires_grad_(True)

    enc = SylEnc().to(dev)
    enc.load_state_dict(torch.load("data/sylenc.pt", map_location=dev)["net"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    classes = torch.load("data/sylenc.pt", map_location="cpu")["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    print(f"# pure-drift, no teacher, no regression term | nfe={args.nfe} "
          f"beta={args.beta} lr={args.lr} steps={args.steps}", flush=True)

    # build the real syllable pool in encoder space
    pool_npz = {}
    pool_path = "data/syl_pool_enc.npz"
    if os.path.exists(pool_path):
        z = np.load(pool_path)
        pool_npz = {k: z[k] for k in z.files}
        print(f"[pool] cached {len(pool_npz)} syllables", flush=True)
    else:
        print("[pool] missing data/syl_pool_enc.npz", flush=True)
        return

    meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
    items = []
    for mp in sorted(glob.glob("data/f5_ft/mel/*.npy")):
        base = os.path.splitext(os.path.basename(mp))[0]
        t = meta.get(base + ".wav")
        if t:
            items.append((mp, t))
    ds = Set(items, cfm.vocab_char_map)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)
    print(f"[data] {len(ds)} items", flush=True)

    tau_raw = torch.tensor(0.0, device=dev, requires_grad=True)
    opt = AdamW(list(cfm.parameters()) + [tau_raw], lr=args.lr, betas=(0.9, 0.98))
    warm = max(50, int(0.05 * args.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm))))
    ema = {k: v.detach().clone().float() for k, v in cfm.state_dict().items()}
    os.makedirs(args.out, exist_ok=True)

    step, bank = 0, None
    while step < args.steps:
        for b in dl:
            step += 1
            mel = b["mel"].to(dev); ids = b["ids"].to(dev); ns = b["nsyl"].to(dev)
            cond = mel[:, :P]
            dur = mel.shape[1] - P

            out, _ = cfm.sample.__wrapped__(cfm, cond=cond, text=ids, duration=dur,
                                            steps=args.nfe, cfg_strength=1.0)
            gen = out[:, -dur:]                                  # [B,Tg,100]
            Tg = gen.shape[1]

            # ---- L_syl: syllable-level drift in encoder space ----
            K = int(ns.min().item())
            gw = syl_windows(gen, ns)                            # [B,K,WIN,100]
            gfeat = enc(gw.reshape(-1, WIN, 100)).reshape(gen.shape[0], K, -1)
            flat = gfeat.reshape(-1, gfeat.shape[-1])

            # positives: this batch's syllable strings -> cached real pool
            inv = {v: k for k, v in cfm.vocab_char_map.items()}
            ids_list = ids[:, :K].tolist()
            pos = []
            for i in range(gfeat.shape[0]):
                for k in range(K):
                    cname = inv.get(ids_list[i][k])
                    arr = pool_npz.get(cname) if cname else None
                    if arr is None:
                        arr = np.random.randn(32, gfeat.shape[-1]).astype(np.float16)
                    if arr.shape[0] < 32:
                        arr = np.tile(arr, (int(np.ceil(32 / arr.shape[0])), 1))
                    pos.append(arr[:32])
            pos = torch.tensor(np.stack(pos), device=dev, dtype=torch.float32)

            if bank is None:
                bank = flat.detach().clone()
            else:
                bank = torch.cat([bank, flat.detach()], 0)[-256:]
            negb = bank.unsqueeze(0).expand(flat.shape[0], -1, -1)
            tau = tau_raw.exp().clamp(0.05, 5.0)
            l_syl = drift_loss(flat, pos, negb, tau)

            # ---- L_stat ----
            l_st = l_stat(gen, mel)

            loss = l_syl + args.beta * l_st
            if not torch.isfinite(loss):
                print(f"[skip] non-finite at {step}", flush=True)
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                for k, v in cfm.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.999).add_(v.float(), alpha=0.001)
                    else:
                        ema[k] = v.clone()

            if step % 50 == 0:
                lr_ratio = float(Tg) / float(mel.shape[1])
                print(f"[step {step}] L_syl={l_syl.item():.4f} L_stat={l_st.item():.4f} "
                      f"tau={tau.item():.3f} len_ratio(mel)={lr_ratio:.2f}", flush=True)
            if step % 500 == 0:
                torch.save({"ema_model_state_dict": {k: v.to(cfm.state_dict()[k].dtype) for k, v in ema.items()}},
                           os.path.join(args.out, f"model_{step}.pt"))
                print("[ckpt]", step, flush=True)
            if step >= args.steps:
                break
    torch.save({"ema_model_state_dict": {k: v.to(cfm.state_dict()[k].dtype) for k, v in ema.items()}},
               os.path.join(args.out, "model_last.pt"))
    print("[pure-drift] done", flush=True)


if __name__ == "__main__":
    main()
