#!/usr/bin/env python3
"""ARM 10's loss, asserted rather than trusted.

The arm's entire premise is that the objective stops requiring MAGNITUDES.
If the loss can be reduced by moving the eval's level rather than its order,
the arm is not testing what it claims. So shift-invariance is not a nicety
here -- it is the property under test, and it gets an assertion.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import listwise_rank_loss, rank_top1   # noqa: E402

K, G = 16, 64
torch.manual_seed(0)
pred = torch.randn(G * K) * 200.0

# 1. SHIFT-INVARIANCE, per group and globally: adding any constant to every
#    child in a group must not change the loss at all.
base = listwise_rank_loss(pred, K, 160.0)
for shift in (1.0, -50.0, 1000.0):
    got = listwise_rank_loss(pred + shift, K, 160.0)
    assert abs(got.item() - base.item()) < 1e-9, (shift, got.item(), base.item())
per_group = pred.view(G, K) + torch.randn(G, 1) * 500.0
got = listwise_rank_loss(per_group.reshape(-1), K, 160.0)
assert abs(got.item() - base.item()) < 1e-9, ("per-group shift", got.item())
print("shift-invariance OK (global and per-group)")

# 2. A PERFECT ORDER drives the loss toward zero; the best child is local 0,
#    and the parent's preference is -pred, so index 0 must be the SMALLEST.
perfect = torch.arange(K, dtype=torch.float32).repeat(G) * 100.0
lo = listwise_rank_loss(perfect, K, 1.0)
assert lo.item() < 1e-6, lo.item()
assert rank_top1(perfect, K).item() == 1.0
print("perfect order: loss %.3e, top1 %.3f" % (lo.item(), rank_top1(perfect, K)))

# 3. The WORST order (best child ranked last) must cost more than random.
worst = (K - 1 - torch.arange(K, dtype=torch.float32)).repeat(G) * 100.0
assert listwise_rank_loss(worst, K, 1.0).item() > base.item()
assert rank_top1(worst, K).item() == 0.0
print("worst order: loss %.3f, top1 %.3f"
      % (listwise_rank_loss(worst, K, 1.0), rank_top1(worst, K)))

# 4. A gradient must exist and must be ORDER-directed: pushing the best
#    child's eval DOWN (more negative = better for the parent) must help.
x = pred.clone().requires_grad_(True)
listwise_rank_loss(x, K, 160.0).backward()
g0 = x.grad.view(G, K)[:, 0]
assert (g0 > 0).float().mean() > 0.9, g0[:8]
print("gradient pushes the searched child's eval down in %.0f%% of groups"
      % (100 * (g0 > 0).float().mean()))
print("ALL ARM 10 LOSS ASSERTIONS PASSED")
