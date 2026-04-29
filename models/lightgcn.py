"""
Shared-weight LightGCN on a user-item bipartite graph (symmetric normalized adjacency).
Multiple views = same embeddings, different adjacency matrices.
Final node representation: mean of embeddings across layers (0..L).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def scipy_csr_to_torch_sparse_float(csr: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = csr.astype(np.float32).tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    shape = coo.shape
    t = torch.sparse_coo_tensor(indices, values, torch.Size(shape), device=device).coalesce()
    return t


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int, n_layers: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_nodes = num_users + num_items
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.n_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward_from_adj(
        self, adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (user_emb, item_emb) from one view."""
        emb_list: List[torch.Tensor] = [self.embedding.weight]
        x = self.embedding.weight
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            emb_list.append(x)
        final = torch.stack(emb_list, dim=0).mean(dim=0)
        users = final[: self.num_users]
        items = final[self.num_users :]
        return users, items

    def forward_views(
        self, adj_by_view: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, adj in adj_by_view.items():
            out[name] = self.forward_from_adj(adj)
        return out
