"""
Load MovieLens (100K) as implicit Web usage logs: rating >= threshold => positive event.
Time-based split on positive interactions (sorted by timestamp).
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DatasetBundle:
    num_users: int
    num_items: int
    user_train: Dict[int, List[Tuple[int, int]]]
    user_val: Dict[int, List[int]]
    user_test: Dict[int, List[int]]
    train_pairs: List[Tuple[int, int]]
    train_triplets: List[Tuple[int, int, int]]
    item_popularity: Dict[int, int]


def _read_ml100k_ratings(data_dir: str) -> List[Tuple[int, int, int, int]]:
    path = os.path.join(data_dir, "u.data")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {path}. Run: bash scripts/download_ml100k.sh"
        )
    rows: List[Tuple[int, int, int, int]] = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            u, i, rating, ts = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            rows.append((u, i, rating, ts))
    return rows


def build_dataset(
    data_dir: str,
    implicit_threshold: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> DatasetBundle:
    raw = _read_ml100k_ratings(data_dir)
    positives = [(u, i, ts) for u, i, r, ts in raw if r >= implicit_threshold]
    positives.sort(key=lambda x: x[2])

    n = len(positives)
    if n == 0:
        raise ValueError("No positive interactions after thresholding.")

    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))
    train_trip = positives[:t_end]
    val_trip = positives[t_end:v_end]
    test_trip = positives[v_end:]

    users = {u for u, _, _ in positives}
    items = {i for _, i, _ in positives}
    user_map = {u: idx for idx, u in enumerate(sorted(users))}
    item_map = {i: idx for idx, i in enumerate(sorted(items))}

    def remap_triples(trip: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        return [(user_map[u], item_map[i], ts) for u, i, ts in trip]

    train_trip = remap_triples(train_trip)
    val_trip = remap_triples(val_trip)
    test_trip = remap_triples(test_trip)

    num_users = len(user_map)
    num_items = len(item_map)

    user_train: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for u, i, ts in train_trip:
        user_train[u].append((i, ts))

    for u in user_train:
        user_train[u].sort(key=lambda x: x[1])

    train_pairs = [(u, i) for u, i, _ in train_trip]
    item_pop: Dict[int, int] = defaultdict(int)
    for u, i in train_pairs:
        item_pop[i] += 1

    user_val: Dict[int, List[int]] = defaultdict(list)
    for u, i, _ in val_trip:
        user_val[u].append(i)

    user_test: Dict[int, List[int]] = defaultdict(list)
    for u, i, _ in test_trip:
        user_test[u].append(i)

    return DatasetBundle(
        num_users=num_users,
        num_items=num_items,
        user_train=dict(user_train),
        user_val=dict(user_val),
        user_test=dict(user_test),
        train_pairs=train_pairs,
        train_triplets=train_trip,
        item_popularity=dict(item_pop),
    )


def print_stats(bundle: DatasetBundle) -> None:
    nu, ni = bundle.num_users, bundle.num_items
    n_train = len(bundle.train_pairs)
    n_val = sum(len(v) for v in bundle.user_val.values())
    n_test = sum(len(v) for v in bundle.user_test.values())
    print("=== Data (implicit Web usage narrative) ===")
    print(f"Users: {nu}, Items: {ni}")
    print(f"Train interactions: {n_train}, Val: {n_val}, Test: {n_test}")
    covered_test_users = len(bundle.user_test)
    print(f"Users with test events: {covered_test_users}")
