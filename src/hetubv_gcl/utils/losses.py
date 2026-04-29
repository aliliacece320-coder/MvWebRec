"""BPR loss for implicit ranking; InfoNCE aligns user embeddings across views (in-batch negatives)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
) -> torch.Tensor:
    pos = (user_emb * pos_item_emb).sum(dim=-1)
    neg = (user_emb * neg_item_emb).sum(dim=-1)
    return -F.logsigmoid(pos - neg).mean()


def infonce_user_alignment(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE: rows are paired; other rows in batch act as negatives."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = (a @ p.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_p = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_a + loss_p)
