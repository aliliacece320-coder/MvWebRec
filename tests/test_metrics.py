"""Toy ranking: stub model with fixed embeddings to verify Recall/NDCG."""
from __future__ import annotations

import torch

from mvwebrec.utils.metrics import recall_ndcg_at_k


class _StubRecModel(torch.nn.Module):
    """Ignores adjacency; returns fixed user/item embeddings."""

    def __init__(self, users_emb: torch.Tensor, items_emb: torch.Tensor):
        super().__init__()
        self.register_buffer("users_emb", users_emb)
        self.register_buffer("items_emb", items_emb)

    def forward_from_adj(self, adj):
        return self.users_emb, self.items_emb


def test_recall_hits_masked_negatives():
    # 1 user, 3 items; mask train item 1; target test item 0 should rank first after mask
    u = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    i = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    model = _StubRecModel(u, i)
    adj = torch.eye(5, dtype=torch.float32).to_sparse()
    train_pos = {0: {1}}
    eval_dict = {0: [0]}
    num_items = 3
    ks = [2]
    m = recall_ndcg_at_k(
        model, adj, train_pos, eval_dict, num_items, ks, device=torch.device("cpu")
    )
    assert m["recall@2"] == 1.0
    assert m["ndcg@2"] > 0.99
