"""Minimal flow-matching TTS backbone used for SylDrift.

Design notes
------------
* One forward pass == one NFE. Text conditions the frames through
  cross-attention; the reference-speaker embedding modulates every block
  through adaLN, which is what keeps timbre out of the drifting field.
* Sizes are deliberately small (default ~18M) so that a single 24G card can
  train it; scaling is a config change, not a code change.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None] * 1000.0
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


class Block(nn.Module):
    def __init__(self, dim, heads, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, text, text_mask=None):
        s1, sc1, g1, s2, sc2, g2 = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), s1, sc1)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = modulate(self.norm2(x), s2, sc2)
        h, _ = self.cross_attn(h, text, text, key_padding_mask=text_mask, need_weights=False)
        x = x + g2.unsqueeze(1) * h
        x = x + self.mlp(self.norm3(x))
        return x


class FlowTTS(nn.Module):
    def __init__(self, vocab=256, dim=384, depth=6, heads=6, mel_dim=80, text_vocab=256):
        super().__init__()
        self.mel_dim = mel_dim
        self.text_emb = nn.Embedding(text_vocab, dim)
        self.text_conv = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2), nn.GELU(), nn.Conv1d(dim, dim, 5, padding=2)
        )
        self.in_proj = nn.Linear(mel_dim, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 4096, dim) * 0.02)
        self.t_embed = TimestepEmbed(dim)
        self.spk_proj = nn.Linear(mel_dim, dim)
        self.cond_proj = nn.Sequential(nn.Linear(dim * 2, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, dim) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(dim, mel_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, text_ids, spk_ref, text_mask=None, frame_ids=None):
        """x: [B,T,mel] noisy mel at time t; t: [B]; text_ids: [B,L]; spk_ref: [B,mel]

        frame_ids: [B,T] syllable index for every frame. Without it the model has
        no way to know which syllable occupies which frames and learns to ignore
        the text entirely (measured: different texts produced cos=0.9999 output).
        """
        h = self.in_proj(x)
        T = h.shape[1]
        h = h + self.pos_emb[:, :T]
        if frame_ids is not None:
            h = h + self.text_emb(frame_ids)
        txt = self.text_emb(text_ids).transpose(1, 2)
        txt = (txt + self.text_conv(txt)).transpose(1, 2)
        spk = self.spk_proj(spk_ref)
        cond = self.cond_proj(torch.cat([self.t_embed(t), spk], dim=-1))
        for blk in self.blocks:
            h = blk(h, cond, txt, text_mask)
        return self.out(self.final_norm(h))

    def sample(self, text_ids, spk_ref, n_frames, steps=1, text_mask=None, frame_ids=None):
        """Differentiable: the drift term needs gradients to reach the generator.

        Callers that only want audio must wrap this in torch.no_grad().
        """
        B = text_ids.shape[0]
        x = torch.randn(B, n_frames, self.mel_dim, device=text_ids.device)
        for i in range(steps):
            t = torch.full((B,), i / steps, device=text_ids.device)
            v = self.forward(x, t, text_ids, spk_ref, text_mask, frame_ids)
            x = x + v / steps
        return x
