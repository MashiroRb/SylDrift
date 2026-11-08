"""One-step distillation of F5-TTS with the SylDrift objective.

Student = F5 CFM initialised from the base checkpoint. For each utterance:
  * cond  = first 86 mel frames (prompt)
  * gen   = one CFM step from noise, duration = T - 86
  * loss  = L2(gen, teacher_32NFE) + drift(gen pooled by syllable, real frames)
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "third_party/F5-TTS/src")
from f5_tts.model.utils import convert_char_to_pinyin  # noqa: E402

P = 86


class DistillSet(Dataset):
    def __init__(self, items, vocab):
        self.items = items
        self.vocab = vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        mel_p, teach_p, text = self.items[i]
        mel = torch.from_numpy(np.load(mel_p).astype(np.float32))
        teach = torch.from_numpy(np.load(teach_p).astype(np.float32))
        toks = convert_char_to_pinyin([text])[0]
        if isinstance(toks, str):
            toks = list(toks)
        ids = torch.tensor([self.vocab.get(c, self.vocab.get("<unk>", 0)) for c in toks])
        return {"mel": mel, "teach": teach, "ids": ids}


def collate(batch):
    B = len(batch)
    T = max(b["mel"].shape[0] for b in batch)
    Tt = max(b["teach"].shape[0] for b in batch)
    L = max(b["ids"].numel() for b in batch)
    mel = torch.zeros(B, T, 100)
    teach = torch.zeros(B, Tt, 100)
    ids = torch.zeros(B, L, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        mel[i, : b["mel"].shape[0]] = b["mel"]
        teach[i, : b["teach"].shape[0]] = b["teach"]
        ids[i, : b["ids"].numel()] = b["ids"]
        lens[i] = b["teach"].shape[0]
    return {"mel": mel, "teach": teach, "ids": ids, "lens": lens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--drift_w", type=float, default=0.0)
    ap.add_argument("--out", default="runs/f5_distill")
    args = ap.parse_args()

    from f5_tts.api import F5TTS
    ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device=dev)
    cfm = f5.ema_model.float()
    for p in cfm.parameters():
        p.requires_grad_(True)
    print("[distill] params %.1fM" % (sum(p.numel() for p in cfm.parameters()) / 1e6), flush=True)

    meta = dict(l.strip().split("|") for l in open("data/f5_ft/raw.arrow.txt", encoding="utf-8"))
    items = []
    for tp in sorted(glob.glob("data/f5_ft/teacher/*.npy")):
        base = os.path.splitext(os.path.basename(tp))[0]
        mp = os.path.join("data/f5_ft/mel", base + ".npy")
        text = meta.get(base + ".wav")
        if os.path.exists(mp) and text:
            items.append((mp, tp, text))
    print("[distill] items", len(items), flush=True)

    ds = DistillSet(items, cfm.vocab_char_map)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)
    opt = AdamW(cfm.parameters(), lr=args.lr, betas=(0.9, 0.98))
    os.makedirs(args.out, exist_ok=True)

    step = 0
    while step < args.steps:
        for b in dl:
            step += 1
            mel = b["mel"].to(dev)
            teach = b["teach"].to(dev)
            ids = b["ids"].to(dev)
            # cfm.sample is wrapped in @torch.no_grad(), so the one-step pass is
            # replicated here to keep the graph alive for distillation.
            cond = mel[:, :P]
            dur = int(teach.shape[1])
            B = cond.shape[0]
            total = P + dur
            x = torch.randn(B, total, cond.shape[-1], device=dev)
            cond_mask = torch.zeros(B, total, 1, device=dev, dtype=torch.bool)
            cond_mask[:, :P] = True
            step_cond = cond.new_zeros(B, total, cond.shape[-1])
            step_cond[:, :P] = cond
            x = torch.where(cond_mask, step_cond, x)
            t = torch.zeros(B, device=dev)
            v = cfm.transformer(x=x, cond=step_cond, text=ids, time=t,
                                mask=None, drop_audio_cond=False, drop_text=False,
                                cache=False)
            gen = (x + v)[:, -dur:]
            loss = F.mse_loss(gen, teach)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
            opt.step()
            if step % 25 == 0:
                print(f"[step {step}] distill={loss.item():.4f}", flush=True)
            if step % 250 == 0:
                torch.save({"ema_model_state_dict": cfm.state_dict()},
                           os.path.join(args.out, f"model_{step}.pt"))
                print("[ckpt]", step, flush=True)
            if step >= args.steps:
                break
    torch.save({"ema_model_state_dict": cfm.state_dict()},
               os.path.join(args.out, "model_last.pt"))
    print("[distill] done", flush=True)


if __name__ == "__main__":
    main()
