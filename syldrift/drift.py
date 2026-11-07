"""Soft syllable assignment and drifting-field loss.

Two pieces that must stay together:
  1. frame -> syllable assignment from *text-side* durations (continuous, differentiable)
  2. drifting field over pooled syllable features
"""
import torch
import torch.nn.functional as F


def stable_sigmoid(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def soft_assignment(boundaries: torch.Tensor, n_frames: int, tau: float) -> torch.Tensor:
    """boundaries: [B, K+1] cumulative frame counts, boundaries[:, 0] == 0.

    Returns A of shape [B, K, T] with sum_k A[b,k,t] ~= 1.
    The outermost boundaries are padded by 4*tau so both ends saturate;
    without this the first and last frames only get 0.5 of their weight
    (verified numerically: max |colsum-1| goes 0.5 -> 0.018).
    """
    B, K1 = boundaries.shape
    K = K1 - 1
    b = boundaries.clone()
    pad = 4.0 * tau
    b = torch.cat([b[:, :1] - pad, b[:, 1:-1], b[:, -1:] + pad], dim=1)
    t = torch.arange(n_frames, device=b.device, dtype=b.dtype).view(1, 1, -1)
    left = stable_sigmoid((t - b[:, :-1].unsqueeze(-1)) / tau)
    right = stable_sigmoid((t - b[:, 1:].unsqueeze(-1)) / tau)
    return (left - right).clamp_min(0.0)


def pool_by_syllable(feat: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """feat: [B, T, D] frame features; A: [B, K, T] -> pooled [B, K, D]."""
    w = A / A.sum(-1, keepdim=True).clamp_min(1e-6)
    return torch.bmm(w, feat)


def drift_field(x: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                tau_kernel: torch.Tensor) -> torch.Tensor:
    """Kernel-weighted attraction to positives minus repulsion from negatives.

    x   : [N, D] generated features (already detached by the caller)
    pos : [N, M, D] per-sample positive pool
    neg : [N, R, D] per-sample negative pool
    tau_kernel: scalar tensor (learnable)
    """
    x = x.reshape(-1, x.shape[-1])
    d_pos = torch.cdist(x.unsqueeze(1), pos).squeeze(1)      # [N, M]
    w_pos = torch.softmax(-d_pos / tau_kernel, dim=-1)
    attract = torch.einsum("nm,nmd->nd", w_pos, pos) - x

    d_neg = torch.cdist(x.unsqueeze(1), neg).squeeze(1)      # [N, R]
    w_neg = torch.softmax(-d_neg / tau_kernel, dim=-1)
    repel = torch.einsum("nr,nrd->nd", w_neg, neg) - x

    return attract - repel


def drift_loss(pooled_gen: torch.Tensor, pos_pool: torch.Tensor,
               neg_pool: torch.Tensor, tau_kernel: torch.Tensor) -> torch.Tensor:
    """Stop-gradient drifting objective: pull generated syllable features onto
    the real pool, push them off other generated features."""
    x = pooled_gen.detach()
    V = drift_field(x, pos_pool, neg_pool, tau_kernel)
    target = (x + V).detach()
    return F.mse_loss(pooled_gen, target)


def anneal_tau(step: int, total: int, tau_hi: float = 3.0, tau_lo: float = 1.0) -> float:
    if total <= 1:
        return tau_lo
    frac = min(max(step / (0.7 * total), 0.0), 1.0)
    return tau_hi + (tau_lo - tau_hi) * frac
