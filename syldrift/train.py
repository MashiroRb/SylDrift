"""SylDrift training entry point.

Usage (synthetic smoke test):
    python -m syldrift.train --synthetic --steps 20 --batch 4
"""
import argparse
import json
import os
import time

import torch
from torch.optim import AdamW

from .model import FlowTTS
from .drift import soft_assignment, pool_by_syllable, drift_loss, anneal_tau


def synthetic_batch(bs, n_syl, frames_per_syl, mel_dim, vocab, device):
    dur = torch.full((bs, n_syl), frames_per_syl, device=device)
    T = int(dur.sum(1).max().item())
    boundaries = torch.cat([torch.zeros(bs, 1, device=device), dur.cumsum(1)], dim=1)
    mel = torch.randn(bs, T, mel_dim, device=device)
    text_ids = torch.randint(0, vocab, (bs, n_syl), device=device)
    spk = torch.randn(bs, mel_dim, device=device)
    return mel, text_ids, spk, boundaries, T


def sample_positives(mel, K, M, mel_dim, device):
    """Positive pool: M random real frames per syllable (identity feature space)."""
    B, T, _ = mel.shape
    base = mel.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T, mel_dim)
    idx = torch.randint(0, T, (B * K, M), device=device)
    return torch.gather(base, 1, idx[..., None].expand(-1, -1, mel_dim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--mel_dim", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="runs/smoke")
    ap.add_argument("--log_every", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    model = FlowTTS(dim=args.dim, depth=args.depth, mel_dim=args.mel_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params/1e6:.1f}M device={device} "
          f"gpu={torch.cuda.get_device_name(0) if device=='cuda' else '-'}", flush=True)

    tau_kernel = torch.tensor(0.5, device=device, requires_grad=True)
    opt = AdamW(list(model.parameters()) + [tau_kernel], lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        mel, text_ids, spk, boundaries, T = synthetic_batch(
            args.batch, 6, 8, args.mel_dim, 256, device)

        t = torch.rand(args.batch, device=device)
        noise = torch.randn_like(mel)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * mel
        v = model(x_t, t, text_ids, spk)
        fm_loss = torch.nn.functional.mse_loss(v, mel - noise)

        x1 = model.sample(text_ids, spk, T, steps=1)
        A = soft_assignment(boundaries, T, anneal_tau(step, args.steps))
        pooled = pool_by_syllable(x1, A)                    # [B,K,D]
        B, K, D = pooled.shape
        flat = pooled.reshape(B * K, D)

        pos = sample_positives(mel, K, 8, D, device)        # [B*K,8,D]
        neg = flat.detach().unsqueeze(1).expand(-1, 4, -1)
        d_loss = drift_loss(flat, pos, neg, tau_kernel.abs() + 1e-3)

        loss = fm_loss + 0.1 * d_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"[step {step:4d}] fm={fm_loss.item():.4f} drift={d_loss.item():.4f} "
                  f"tau_k={tau_kernel.item():.3f} "
                  f"{(time.time()-t0)/step:.2f}s/step", flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(args.out, "ckpt.pt"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"params": n_params, "steps": args.steps,
                   "wall_s": round(time.time() - t0, 2)}, f, indent=2)
    print("[done] saved", os.path.join(args.out, "ckpt.pt"), flush=True)


if __name__ == "__main__":
    main()
