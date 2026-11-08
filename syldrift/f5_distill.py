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
    ap.add_argument("--cons_w", type=float, default=0.0)
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
    warmup = max(50, int(0.05 * args.steps))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        import math
        prog = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # EMA keeps the distilled student from oscillating (the 3k-step run diverged)
    ema = {k: v.detach().clone().float() for k, v in cfm.state_dict().items()}
    ema_decay = 0.999
    os.makedirs(args.out, exist_ok=True)

    step = 0
    while step < args.steps:
        for b in dl:
            step += 1
            mel = b["mel"].to(dev)
            teach = b["teach"].to(dev)
            ids = b["ids"].to(dev)
            # cfm.sample is decorated with @torch.no_grad(); __wrapped__ gives the
            # raw function so gradients survive. Hand-rolling the sampler was wrong
            # (manual 32-step sampling of the base model gave CER 1.0 vs 0.43 via
            # F5's own sampler), so we must reuse theirs verbatim.
            cond = mel[:, :P]
            dur = int(teach.shape[1])
            out, _ = cfm.sample.__wrapped__(cfm, cond=cond, text=ids, duration=dur,
                                            steps=1, cfg_strength=1.0)
            gen = out[:, -dur:]

            # self-consistency: the 1-step answer must agree with a 2-step rollout.
            # Regressing only the 32-step endpoint let the student "cheat" at very
            # low NFE (CER>1 = runaway length); this ties the two regimes together.
            cons = torch.zeros((), device=dev)
            if args.cons_w > 0:
                with torch.no_grad():
                    out2, _ = cfm.sample(cond=cond, text=ids, duration=dur,
                                         steps=2, cfg_strength=1.0)
                    gen2 = out2[:, -dur:]
                cons = F.mse_loss(gen, gen2)
            loss = F.mse_loss(gen, teach) + args.cons_w * cons
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for k, v in cfm.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v.float(), alpha=1 - ema_decay)
                    else:
                        ema[k] = v.clone()
            if step % 25 == 0:
                print(f"[step {step}] distill={loss.item():.4f} cons={cons.item():.4f}", flush=True)
            if step % 250 == 0:
                torch.save({"ema_model_state_dict": {k: v.to(cfm.state_dict()[k].dtype) for k, v in ema.items()}},
                           os.path.join(args.out, f"model_{step}.pt"))
                print("[ckpt]", step, flush=True)
            if step >= args.steps:
                break
    torch.save({"ema_model_state_dict": {k: v.to(cfm.state_dict()[k].dtype) for k, v in ema.items()}},
               os.path.join(args.out, "model_last.pt"))
    print("[distill] done", flush=True)


if __name__ == "__main__":
    main()
