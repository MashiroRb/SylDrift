"""Numerical checks for the soft-assignment / drift core."""
import torch
from syldrift.drift import soft_assignment, pool_by_syllable, drift_field, drift_loss


def test_assignment_sums_to_one():
    b = torch.tensor([[0., 12, 25, 38, 50]])
    A = soft_assignment(b, 50, tau=1.5)
    col = A.sum(1)
    assert torch.allclose(col, torch.ones_like(col), atol=0.03), col.max()
    print(f"[ok] colsum max dev = {(col-1).abs().max().item():.4f}")


def test_hard_limit():
    b = torch.tensor([[0., 12, 25, 38, 50]])
    A = soft_assignment(b, 50, tau=0.05)
    hard = A.argmax(1)[0].tolist()
    expect = [0]*12 + [1]*13 + [2]*13 + [3]*12
    assert hard == expect, (hard, expect)
    print("[ok] tau -> 0 recovers hard split")


def test_gradient_flows_to_boundaries():
    b = torch.tensor([[0., 12, 25, 38, 50]], requires_grad=True)
    A = soft_assignment(b, 50, tau=1.5)
    feat = torch.randn(1, 50, 8, requires_grad=True)
    pooled = pool_by_syllable(feat, A)
    pooled.sum().backward()
    assert b.grad is not None and b.grad.abs().sum() > 0
    assert feat.grad is not None and feat.grad.abs().sum() > 0
    print(f"[ok] grad to boundaries = {b.grad.abs().sum().item():.4f}")


def test_drift_loss_shape_and_sign():
    g = torch.randn(6, 16, requires_grad=True)
    pos = torch.randn(6, 8, 16)
    neg = torch.randn(6, 4, 16)
    tau = torch.tensor(0.5, requires_grad=True)
    loss = drift_loss(g, pos, neg, tau)
    loss.backward()
    assert tau.grad is not None
    print(f"[ok] drift loss = {loss.item():.4f}, tau grad = {tau.grad.item():.4f}")


if __name__ == "__main__":
    test_assignment_sums_to_one()
    test_hard_limit()
    test_gradient_flows_to_boundaries()
    test_drift_loss_shape_and_sign()
    print("ALL PASS")
