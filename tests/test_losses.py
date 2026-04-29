"""Loss sanity checks."""
import torch

from hetubv_gcl.utils.losses import bpr_loss, infonce_user_alignment


def test_bpr_loss_finite():
    torch.manual_seed(0)
    d = 16
    b = 32
    u = torch.randn(b, d)
    pos = torch.randn(b, d)
    neg = torch.randn(b, d)
    loss = bpr_loss(u, pos, neg)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_infonce_finite():
    torch.manual_seed(0)
    b, d = 8, 32
    a = torch.randn(b, d)
    p = torch.randn(b, d)
    loss = infonce_user_alignment(a, p, temperature=0.2)
    assert torch.isfinite(loss)
