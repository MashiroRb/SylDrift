"""Regression: the drift term must actually update the generator.

Bug history: FlowTTS.sample() carried @torch.no_grad(), so x1 had no grad_fn,
the drift MSE only trained the kernel temperature, and every "with vs without
drift" comparison was meaningless.
"""
import torch
from syldrift.model import FlowTTS
from syldrift.drift import soft_assignment, pool_by_syllable, drift_loss


def build():
    torch.manual_seed(0)
    m = FlowTTS(dim=64, depth=2, heads=4, text_vocab=32)
    return m


def grads(m, use_drift):
    m.zero_grad(set_to_none=True)
    B, K, T, D = 2, 3, 24, 80
    ids = torch.randint(1, 31, (B, 4))
    spk = torch.randn(B, D)
    bounds = torch.tensor([[0., 8, 16, 24], [0., 8, 16, 24]])
    x1 = m.sample(ids, spk, T, steps=1, frame_ids=torch.randint(1, 31, (B, T)))
    A = soft_assignment(bounds, T, 1.5)
    pooled = pool_by_syllable(x1, A).reshape(B * K, D)
    if use_drift:
        pos = torch.randn(B * K, 8, D)
        neg = pooled.detach().unsqueeze(1).expand(-1, 4, -1)
        loss = drift_loss(pooled, pos, neg, torch.tensor(0.5))
    else:
        loss = pooled.square().mean()
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    return g, pooled.requires_grad


if __name__ == "__main__":
    m = build()
    _, rg = grads(m, use_drift=True)
    print("[check] pooled.requires_grad =", rg)
    assert rg, "sample() produced a tensor without grad_fn -> drift cannot train the generator"

    m.zero_grad(set_to_none=True)
    g0, _ = grads(m, use_drift=False)
    m.zero_grad(set_to_none=True)
    g1, _ = grads(m, use_drift=True)
    same = torch.allclose(g0, g1)
    print(f"[check] grad norm without drift={g0.norm():.4f} with drift={g1.norm():.4f} identical={same}")
    assert not same, "drift produced the same gradients -> it is not influencing the model"
    print("ALL PASS")
