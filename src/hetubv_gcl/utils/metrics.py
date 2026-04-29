"""Recall@K and NDCG@K on held-out items (masked train positives)."""
from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import torch


@torch.no_grad()
def recall_ndcg_at_k(
    model: torch.nn.Module,
    adj_global: torch.Tensor,
    train_pos: Dict[int, Set[int]],
    eval_dict: Dict[int, List[int]],
    num_items: int,
    ks: List[int],
    batch_users: int = 256,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Evaluates users that appear in eval_dict with at least one target item."""
    device = device or next(model.parameters()).device
    model.eval()
    users_emb, items_emb = model.forward_from_adj(adj_global)
    users_emb = users_emb.to(device)
    items_emb = items_emb.to(device)

    user_ids = [u for u, items in eval_dict.items() if items]
    if not user_ids:
        return {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}

    hits = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    for start in range(0, len(user_ids), batch_users):
        batch_u = user_ids[start : start + batch_users]
        u_vec = users_emb[batch_u]
        scores = u_vec @ items_emb.T

        for row, u in enumerate(batch_u):
            mask_items = train_pos.get(u, set())
            s = scores[row].clone()
            for j in mask_items:
                if 0 <= j < num_items:
                    s[j] = -1e9
            _, topk = torch.topk(s, k=max(ks))

            targets = set(eval_dict[u])
            ranked = topk.cpu().tolist()
            for k in ks:
                top = ranked[:k]
                hit = int(any(x in targets for x in top))
                hits[k].append(hit)
                if hit:
                    first = next(i for i, x in enumerate(top) if x in targets)
                    ndcgs[k].append(1.0 / np.log2(first + 2))
                else:
                    ndcgs[k].append(0.0)

    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
        out[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return out


@torch.no_grad()
def topk_items(
    user_emb_row: torch.Tensor,
    item_emb: torch.Tensor,
    train_pos: Set[int],
    k: int,
    num_items: int,
) -> List[int]:
    scores = user_emb_row @ item_emb.T
    s = scores.clone()
    for j in train_pos:
        if 0 <= j < num_items:
            s[j] = -1e9
    _, idx = torch.topk(s, k=k)
    return idx.cpu().tolist()
