"""Real-data SylDrift training: flow matching + syllable-conditioned drift.

Stage-1 scope: short utterances, in-context prompt = first second of the same
clip, target = the rest. This keeps the pipeline runnable before MFA lands.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model import FlowTTS
from .drift import soft_assignment, pool_by_syllable, drift_loss, anneal_tau
from .data import MelDataset, collate, split_syllables


def build_vocab(rows, path, min_count=2):
    from collections import Counter
    c = Counter(ch for r in rows for ch in "".join(r["syl"]))
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, n in c.most_common():
        if n >= min_count:
            vocab[ch] = len(vocab)
    json.dump(vocab, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return vocab


def encode(syl, vocab, max_len=200):
    ids = [vocab.get(ch, 1) for ch in syl][:max_len]
    return torch.tensor(ids or [1], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--prompt_frames", type=int, default=86)
    ap.add_argument("--drift_w", type=float, default=0.1)
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    ds = MelDataset(args.manifest)
    rows = ds.rows
    vocab = build_vocab(rows, os.path.join(args.out, "vocab.json"))
    print(f"[data] {len(ds)} utts, vocab={len(vocab)}, device={device}", flush=True)

    model = FlowTTS(dim=args.dim, depth=args.depth).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    tau_kernel = torch.tensor(0.5, device=device, requires_grad=True)
    opt = AdamW(list(model.parameters()) + [tau_kernel], lr=args.lr, betas=(0.9, 0.98))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)

    t0 = time.time()
    step = 0
    while step < args.steps:
        for batch in loader:
            step += 1
            mel = batch["mel"].to(device)
            B, T, D = mel.shape
            P = min(args.prompt_frames, T // 3)
            prompt, target = mel[:, :P], mel[:, P:]
            if target.shape[1] < 10:
                continue

            text = [encode(s, vocab) for s in batch["syl"]]
            L = max(len(t) for t in text)
            text_ids = torch.zeros(B, L, dtype=torch.long)
            for i, t in enumerate(text):
                text_ids[i, : len(t)] = t
            text_ids = text_ids.to(device)
            spk_ref = prompt.mean(1)

            # flow matching
            t = torch.rand(B, device=device)
            noise = torch.randn_like(target)
            x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
            v = model(x_t, t, text_ids, spk_ref)
            fm = F.mse_loss(v, target - noise)

            # drift on 1-NFE sample
            x1 = model.sample(text_ids, spk_ref, target.shape[1], steps=1)
            bounds = []
            for i, d in enumerate(batch["dur"]):
                dd = torch.tensor(d, device=device, dtype=torch.float32)
                dd = dd * (target.shape[1] / max(dd.sum().item(), 1))
                bounds.append(torch.cat([torch.zeros(1, device=device), dd.cumsum(0)]))
            K = min(b.shape[0] - 1 for b in bounds)
            bounds = torch.stack([b[: K + 1] for b in bounds])
            A = soft_assignment(bounds, target.shape[1], anneal_tau(step, args.steps))
            pooled = pool_by_syllable(x1, A)
            flat = pooled.reshape(B * K, D)

            idx = torch.randint(0, target.shape[1], (B * K, 8), device=device)
            base = target.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, target.shape[1], D)
            pos = torch.gather(base, 1, idx[..., None].expand(-1, -1, D))
            neg = flat.detach().unsqueeze(1).expand(-1, 4, -1)
            dl = drift_loss(flat, pos, neg, tau_kernel.abs() + 1e-3)

            loss = fm + args.drift_w * dl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                print(f"[step {step:5d}] fm={fm.item():.4f} drift={dl.item():.4f} "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
            if step % args.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args)},
                           os.path.join(args.out, f"ckpt-{step}.pt"))
                print(f"[ckpt] {step}", flush=True)
            if step >= args.steps:
                break

    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args)},
               os.path.join(args.out, "ckpt-last.pt"))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
