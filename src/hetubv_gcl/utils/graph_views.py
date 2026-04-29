"""
Multi-view bipartite graphs from train interactions:
- Global: all train edges (long-term / full history in train split)
- Recent: last k events per user by timestamp (short-term)
- Frequency: edges to popular items only (stable / mainstream preference proxy; ML-100K often has one edge per pair, so we use global item popularity threshold)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass
class GraphStats:
    name: str
    nnz: int
    avg_user_degree: float


def _bipartite_normalized_adjacency(
    n_users: int, n_items: int, user_ids: np.ndarray, item_ids: np.ndarray
) -> sp.csr_matrix:
    if len(user_ids) == 0:
        raise ValueError("Empty edge list for graph.")
    n_nodes = n_users + n_items
    row = np.concatenate([user_ids, item_ids + n_users])
    col = np.concatenate([item_ids + n_users, user_ids])
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float64)
    pos = rowsum > 0
    d_inv_sqrt[pos] = np.power(rowsum[pos], -0.5)
    d_mat = sp.diags(d_inv_sqrt.astype(np.float32))
    return d_mat @ adj @ d_mat


def build_global_view(
    train_pairs: List[Tuple[int, int]], n_users: int, n_items: int
) -> sp.csr_matrix:
    u = np.array([p[0] for p in train_pairs], dtype=np.int64)
    v = np.array([p[1] for p in train_pairs], dtype=np.int64)
    return _bipartite_normalized_adjacency(n_users, n_items, u, v)


def build_recent_view(
    user_train: Dict[int, List[Tuple[int, int]]],
    n_users: int,
    n_items: int,
    recent_k: int,
) -> sp.csr_matrix:
    u_list: List[int] = []
    i_list: List[int] = []
    for u, seq in user_train.items():
        tail = seq[-recent_k:] if recent_k > 0 else seq
        for item_id, _ts in tail:
            u_list.append(u)
            i_list.append(item_id)
    u = np.array(u_list, dtype=np.int64)
    v = np.array(i_list, dtype=np.int64)
    return _bipartite_normalized_adjacency(n_users, n_items, u, v)


def build_frequency_view(
    train_pairs: List[Tuple[int, int]],
    n_users: int,
    n_items: int,
    item_popularity: Dict[int, int],
    quantile: float,
) -> sp.csr_matrix:
    if not item_popularity:
        return build_global_view(train_pairs, n_users, n_items)
    degs = np.array(list(item_popularity.values()), dtype=np.float64)
    thr = float(np.quantile(degs, quantile))
    kept_items: Set[int] = {i for i, c in item_popularity.items() if c >= thr}
    pairs = [(u, i) for u, i in train_pairs if i in kept_items]
    if not pairs:
        return build_global_view(train_pairs, n_users, n_items)
    u = np.array([p[0] for p in pairs], dtype=np.int64)
    v = np.array([p[1] for p in pairs], dtype=np.int64)
    return _bipartite_normalized_adjacency(n_users, n_items, u, v)


def graph_stats(name: str, adj: sp.csr_matrix, n_users: int) -> GraphStats:
    nnz = int(adj.nnz)
    sub = adj[:n_users, n_users:]
    row_deg = np.diff(sub.indptr)
    avg_ud = float(row_deg.mean()) if len(row_deg) else 0.0
    return GraphStats(name=name, nnz=nnz, avg_user_degree=avg_ud)


def print_view_summary(stats: List[GraphStats]) -> None:
    print("=== Multi-view graph stats ===")
    for s in stats:
        print(f"{s.name}: nnz={s.nnz}, avg train user degree ~ {s.avg_user_degree:.2f}")


def print_user_view_example(
    user_id: int,
    user_train: Dict[int, List[Tuple[int, int]]],
    item_popularity: Dict[int, int],
    recent_k: int,
    freq_quantile: float,
) -> None:
    print(f"\n=== Example user {user_id} (multi-view neighbors) ===")
    seq = user_train.get(user_id, [])
    if not seq:
        print("User has no train interactions.")
        return
    print("Train history (item_id, timestamp):", seq[:15], "..." if len(seq) > 15 else "")
    tail = seq[-recent_k:] if recent_k > 0 else seq
    print(f"Recent view (last {recent_k}):", [t[0] for t in tail])
    if item_popularity:
        degs = np.array(list(item_popularity.values()), dtype=np.float64)
        thr = float(np.quantile(degs, freq_quantile))
        freq_items = {i for i, c in item_popularity.items() if c >= thr}
        user_freq = [t[0] for t in seq if t[0] in freq_items]
        print(
            f"Frequency view (items with pop>=q{freq_quantile:.2f}, thr={thr:.0f}):",
            user_freq[:20],
            "..." if len(user_freq) > 20 else "",
        )


def build_all_views(
    bundle,
    recent_k: int,
    freq_quantile: float,
    mode: str,
) -> Dict[str, sp.csr_matrix]:
    n_users, n_items = bundle.num_users, bundle.num_items
    global_a = build_global_view(bundle.train_pairs, n_users, n_items)
    out: Dict[str, sp.csr_matrix] = {"global": global_a}

    if mode == "baseline":
        return out

    if mode == "ablation_no_recent":
        out["frequency"] = build_frequency_view(
            bundle.train_pairs,
            n_users,
            n_items,
            bundle.item_popularity,
            freq_quantile,
        )
        return out

    out["recent"] = build_recent_view(
        bundle.user_train, n_users, n_items, recent_k
    )
    out["frequency"] = build_frequency_view(
        bundle.train_pairs,
        n_users,
        n_items,
        bundle.item_popularity,
        freq_quantile,
    )
    return out
