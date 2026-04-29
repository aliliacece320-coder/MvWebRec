import numpy as np

from utils.graph_views import build_global_view, build_recent_view


def test_global_view_shape_and_nnz():
    pairs = [(0, 0), (0, 1), (1, 2)]
    adj = build_global_view(pairs, n_users=2, n_items=3)
    n_nodes = 2 + 3
    assert adj.shape == (n_nodes, n_nodes)
    assert adj.nnz > 0


def test_recent_view_respects_tail():
    user_train = {
        0: [(0, 1), (1, 2), (2, 3)],
    }
    adj = build_recent_view(user_train, n_users=1, n_items=4, recent_k=2)
    sub = adj[:1, 1:].toarray()
    # last two interactions use items 1 and 2
    assert sub[0, 1] > 0 and sub[0, 2] > 0
