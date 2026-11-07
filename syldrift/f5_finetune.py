"""Fine-tune F5-TTS with the SylDrift objective.

Adds a syllable-conditioned drifting term on the 1-step sample to the standard
flow-matching loss, aiming to recover low-NFE quality.

Text is the raw hanzi string (F5 uses a character vocabulary), so the syllable
pool is built from characters.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "third_party/F5-TTS/src")
SYL = re.compile(r"^[a-z]+[1-5]$")


def hanzi_of(content_paths):
    m = {}
    for p in content_paths:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                hz = "".join(t for t in parts[1].split() if not SYL.match(t))
                if hz:
                    m[parts[0].replace(".wav", "")] = hz
    return m


class MelText(Dataset):
    def __init__(self, rows, mel_spec, vocab, max_sec=8.0):
        self.rows = rows
        self.mel = mel_spec
        self.vocab = vocab
        self.max_sec = max_sec

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        w = torch.from_numpy(wav.mean(1))[: int(self.max_sec * sr)]
        mel = self.mel(w.unsqueeze(0))[0].transpose(0, 1)   # [T, n_mel]
        ids = [self.vocab.get(c, self.vocab.get("<unk>", 0)) for c in r["text"]]
        return {"mel": mel, "ids": torch.tensor(ids, dtype=torch.long), "syl": r["text"]}


def collate(batch, pad_id=0):
    T = max(b["mel"].shape[0] for b in batch)
    L = max(len(b["ids"]) for b in batch)
    mel = torch.zeros(len(batch), T, batch[0]["mel"].shape[1])
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        mel[i, : b["mel"].shape[0]] = b["mel"]
        ids[i, : len(b["ids"])] = b["ids"]
    return {"mel": mel, "ids": ids, "syl": [b["syl"] for b in batch]}


def one_step(cfm, mel_shape, ids, device):
    x = torch.randn(mel_shape, device=device)
    t = torch.zeros(mel_shape[0], device=device)
    v = cfm.transformer(x, t, text=ids)
    return x + v


def drift_terms(pred, target, syls):
    """Cheap corpus-internal drift: pull each generated frame-set toward its own
    target frames, push away from other samples in the batch."""
    B, T, D = pred.shape
    idx = torch.randint(0, T, (B, 16), device=pred.device)
    pos = torch.gather(target, 1, idx[..., None].expand(-1, -1, D))
    flat = pred.reshape(B, 1, D)
    d_pos = torch.cdist(flat, pos).squeeze(1)
    w = torch.softmax(-d_pos / 0.5, dim=-1)
    attract = torch.einsum("bm,bmd->bd", w, pos) - flat.squeeze(1)
    neg = pred.roll(1, 0)[:, ::4].reshape(B, -1, D)
    d_neg = torch.cdist(flat, neg).squeeze(1)
    wn = torch.softmax(-d_neg / 0.5, dim=-1)
    repel = torch.einsum("bn,bnd->bd", wn, neg) - flat.squeeze(1)
    V = attract - repel
    return F.mse_loss(pred.mean(1), (pred.mean(1) + V).detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--drift_w", type=float, default=0.05)
    ap.add_argument("--out", default="runs/f5_ft")
    ap.add_argument("--manifest", default="data/manifest_ft.jsonl")
    args = ap.parse_args()

    from f5_tts.api import F5TTS
    ckpt = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, device=device)
    cfm = f5.ema_model
    mdtype = next(cfm.parameters()).dtype
    print("[ft] model dtype", mdtype, flush=True)
    for p in cfm.parameters():
        p.requires_grad_(True)
    print("[ft] params", sum(p.numel() for p in cfm.parameters()) / 1e6, "M", flush=True)

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    ds = MelText(rows, cfm.mel_spec, cfm.vocab_char_map)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)
    opt = AdamW(cfm.parameters(), lr=args.lr, betas=(0.9, 0.98))
    os.makedirs(args.out, exist_ok=True)

    step = 0
    while step < args.steps:
        for b in dl:
            step += 1
            mel = b["mel"].to(device=device, dtype=mdtype)
            ids = b["ids"].to(device)
            fm, _cond, _pred = cfm(mel, ids)
            if args.drift_w > 0:
                x1 = one_step(cfm, mel.shape, ids, device)
                dl_loss = drift_terms(x1, mel, b["syl"])
            else:
                dl_loss = torch.zeros((), device=device)
            loss = fm + args.drift_w * dl_loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                print(f"[step {step}] fm={fm.item():.4f} drift={dl_loss.item():.4f}", flush=True)
            if step % 500 == 0:
                torch.save({"ema_model_state_dict": cfm.state_dict()},
                           os.path.join(args.out, f"model_{step}.safetensors"))
                print("[ckpt]", step, flush=True)
            if step >= args.steps:
                break
    torch.save({"ema_model_state_dict": cfm.state_dict()},
               os.path.join(args.out, "model_last.safetensors"))
    print("[ft] done", flush=True)


if __name__ == "__main__":
    main()
