"""Fine-tune F5-TTS on precomputed 24 kHz mels (fast path).

Loss = standard flow-matching (F5's own CFM) + optional drifting term.
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


class MelSet(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        mel_path, text = self.pairs[i]
        mel = torch.from_numpy(np.load(mel_path).astype(np.float32))
        # the checkpoint was loaded with the pinyin/custom vocab, so hanzi must be
        # converted before lookup, otherwise every character maps to <unk>
        toks = convert_char_to_pinyin([text])[0]
        if isinstance(toks, str):
            toks = list(toks)
        ids = [self.vocab.get(c, self.vocab.get("<unk>", 0)) for c in toks]
        return {"mel": mel, "ids": torch.tensor(ids, dtype=torch.long)}


def collate(batch):
    T = max(b["mel"].shape[0] for b in batch)
    L = max(len(b["ids"]) for b in batch)
    mel = torch.zeros(len(batch), T, batch[0]["mel"].shape[1])
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    for i, b in enumerate(batch):
        mel[i, : b["mel"].shape[0]] = b["mel"]
        ids[i, : len(b["ids"])] = b["ids"]
    return {"mel": mel, "ids": ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="runs/f5_ft")
    ap.add_argument("--ftdir", default="data/f5_ft")
    args = ap.parse_args()

    from f5_tts.api import F5TTS
    ck = glob.glob("hf_cache/hub/models--SWivid--F5-TTS/snapshots/*/F5TTS_v1_Base/model_*.safetensors")[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ck, device=dev)
    cfm = f5.ema_model
    # fp16 + AdamW without GradScaler overflows (loss went NaN every step and the
    # loop silently skipped them). Train in fp32 instead.
    cfm.float()
    mdtype = torch.float32
    print("[ft] dtype", mdtype, flush=True)
    for p in cfm.parameters():
        p.requires_grad_(True)
    print("[ft] params %.1fM" % (sum(p.numel() for p in cfm.parameters()) / 1e6), flush=True)

    meta = [l.strip().split("|") for l in open(os.path.join(args.ftdir, "raw.arrow.txt"), encoding="utf-8")]
    pairs = []
    for name, text in meta:
        p = os.path.join(args.ftdir, "mel", os.path.splitext(name)[0] + ".npy")
        if os.path.exists(p):
            pairs.append((p, text))
    print("[ft] pairs", len(pairs), flush=True)

    ds = MelSet(pairs, cfm.vocab_char_map)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True, num_workers=2)
    opt = AdamW(cfm.parameters(), lr=args.lr, betas=(0.9, 0.98))
    os.makedirs(args.out, exist_ok=True)

    step = 0
    while step < args.steps:
        for b in dl:
            step += 1
            mel = b["mel"].to(dev, dtype=mdtype)
            ids = b["ids"].to(dev)
            loss, _cond, _pred = cfm(mel, ids)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    print(f"[step {step}] NON-FINITE loss skipped", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                print(f"[step {step}] fm={loss.item():.4f}", flush=True)
            if step % 500 == 0:
                torch.save({"ema_model_state_dict": cfm.state_dict()},
                           os.path.join(args.out, f"model_{step}.pt"))
                print("[ckpt]", step, flush=True)
            if step >= args.steps:
                break
    torch.save({"ema_model_state_dict": cfm.state_dict()},
               os.path.join(args.out, "model_last.pt"))
    print("[ft] done", flush=True)


if __name__ == "__main__":
    main()
