"""
Train LightGCN with optional multi-view InfoNCE alignment.
Run from repo root: python train.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.preprocessing import build_dataset, print_stats  # noqa: E402
from models.lightgcn import LightGCN, scipy_csr_to_torch_sparse_float  # noqa: E402
from utils.graph_views import (  # noqa: E402
    build_all_views,
    graph_stats,
    print_user_view_example,
    print_view_summary,
)
from utils.losses import bpr_loss, infonce_user_alignment  # noqa: E402
from utils.metrics import recall_ndcg_at_k  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def _train_pos_dict(bundle) -> Dict[int, Set[int]]:
    out: Dict[int, Set[int]] = {}
    for u, i in bundle.train_pairs:
        out.setdefault(u, set()).add(i)
    return out


def _sample_bpr_batch(
    train_arr: np.ndarray,
    user_pos_sets: List[Set[int]],
    num_items: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = train_arr.shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    users = train_arr[idx, 0]
    pos = train_arr[idx, 1]
    neg = np.empty(batch_size, dtype=np.int64)
    for t in range(batch_size):
        u = int(users[t])
        s = user_pos_sets[u]
        while True:
            j = random.randint(0, num_items - 1)
            if j not in s:
                neg[t] = j
                break
    return (
        torch.from_numpy(users.astype(np.int64)),
        torch.from_numpy(pos.astype(np.int64)),
        torch.from_numpy(neg),
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_logging(cfg: dict) -> None:
    log_cfg = cfg.get("logging") or {}
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    log_file = (log_cfg.get("file") or "").strip()
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)


def _metrics_csv_path(cfg: dict, ckpt_path: str) -> str:
    log_cfg = cfg.get("logging") or {}
    explicit = (log_cfg.get("metrics_csv") or "").strip()
    if explicit:
        return explicit
    d = os.path.dirname(os.path.abspath(ckpt_path))
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join(d, f"{stem}.metrics.csv")


def _append_metrics_row(path: str, row: Dict[str, object]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    fieldnames = list(row.keys())
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(ROOT, "config.yaml"))
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        help="Override config mode: full | baseline | ablation_no_recent",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Override checkpoint_path in config.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
    if args.checkpoint:
        cfg["checkpoint_path"] = args.checkpoint

    _setup_logging(cfg)
    log = logging.getLogger("train")

    mode = str(cfg.get("mode", "full"))
    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    device = torch.device(cfg.get("device", "cpu"))
    data_dir = str(cfg["data_dir"])

    bundle = build_dataset(
        data_dir=data_dir,
        implicit_threshold=int(cfg["implicit_threshold"]),
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
    )
    print_stats(bundle)

    views_sp = build_all_views(
        bundle,
        recent_k=int(cfg["recent_k"]),
        freq_quantile=float(cfg["frequency"]["quantile"]),
        mode=mode,
    )
    stats = []
    for name, adj in views_sp.items():
        stats.append(graph_stats(name, adj, bundle.num_users))
    print_view_summary(stats)
    example_u = next(iter(bundle.user_train.keys()))
    print_user_view_example(
        example_u,
        bundle.user_train,
        bundle.item_popularity,
        int(cfg["recent_k"]),
        float(cfg["frequency"]["quantile"]),
    )

    adj_torch: Dict[str, torch.Tensor] = {
        k: scipy_csr_to_torch_sparse_float(v, device) for k, v in views_sp.items()
    }

    model = LightGCN(
        bundle.num_users,
        bundle.num_items,
        int(cfg["model"]["embed_dim"]),
        int(cfg["model"]["n_layers"]),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    train_pos = _train_pos_dict(bundle)
    user_pos_sets: List[Set[int]] = [
        train_pos.get(u, set()) for u in range(bundle.num_users)
    ]
    train_arr = np.asarray(bundle.train_pairs, dtype=np.int64)
    n_batches = max(1, len(train_arr) // int(cfg["train"]["batch_size"]))

    lambda_cl = float(cfg["train"]["lambda_cl"])
    tau = float(cfg["train"]["tau"])
    ks = [int(k) for k in cfg["train"]["eval_ks"]]

    if mode == "baseline":
        cl_keys = None
    elif mode == "ablation_no_recent":
        cl_keys = ("global", "frequency")
    else:
        cl_keys = ("global", "recent")

    best_val = -1.0
    ckpt_path = str(cfg.get("checkpoint_path", "./checkpoints/model.pt"))
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)) or ".", exist_ok=True)
    metrics_csv = _metrics_csv_path(cfg, ckpt_path)
    if os.path.isfile(metrics_csv):
        os.remove(metrics_csv)
    patience = int(cfg["train"].get("early_stop_patience", 0))
    no_improve = 0

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        losses_bpr = []
        losses_cl = []
        pbar = tqdm(range(n_batches), desc=f"epoch {epoch+1}")
        for _ in pbar:
            u_b, pos_b, neg_b = _sample_bpr_batch(
                train_arr,
                user_pos_sets,
                bundle.num_items,
                int(cfg["train"]["batch_size"]),
            )
            u_b, pos_b, neg_b = u_b.to(device), pos_b.to(device), neg_b.to(device)

            opt.zero_grad(set_to_none=True)
            emb_views = model.forward_views(adj_torch)
            u_g, items_g = emb_views["global"]
            loss_b = bpr_loss(u_g[u_b], items_g[pos_b], items_g[neg_b])

            loss_c = torch.zeros((), device=device)
            if cl_keys is not None and lambda_cl > 0:
                a_name, p_name = cl_keys
                ua = emb_views[a_name][0][u_b]
                up = emb_views[p_name][0][u_b]
                loss_c = infonce_user_alignment(ua, up, tau)

            loss = loss_b + lambda_cl * loss_c
            loss.backward()
            opt.step()

            losses_bpr.append(float(loss_b.detach().cpu()))
            losses_cl.append(float(loss_c.detach().cpu()))
            pbar.set_postfix(bpr=np.mean(losses_bpr), nce=np.mean(losses_cl))

        model.eval()
        val_metrics = recall_ndcg_at_k(
            model,
            adj_torch["global"],
            train_pos,
            bundle.user_val,
            bundle.num_items,
            ks,
            device=device,
        )
        rm = val_metrics.get(f"recall@{ks[0]}", 0.0)
        mean_bpr = float(np.mean(losses_bpr)) if losses_bpr else 0.0
        mean_cl = float(np.mean(losses_cl)) if losses_cl else 0.0
        row: Dict[str, object] = {
            "epoch": epoch + 1,
            "train_bpr": round(mean_bpr, 6),
            "train_cl": round(mean_cl, 6),
        }
        row.update(val_metrics)
        _append_metrics_row(metrics_csv, row)
        log.info("Val epoch=%s metrics=%s", epoch + 1, val_metrics)

        improved = rm > best_val
        if improved:
            best_val = rm
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "val_metrics": val_metrics,
                    "epoch": epoch + 1,
                    "train_pos": {k: list(v) for k, v in train_pos.items()},
                },
                ckpt_path,
            )
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            log.info(
                "Early stopping: val recall@%s did not improve for %s epochs",
                ks[0],
                patience,
            )
            break

    log.info("Best checkpoint (by val recall@%s): %s", ks[0], ckpt_path)
    if not os.path.isfile(ckpt_path):
        log.error("No checkpoint was saved; cannot evaluate on test.")
        return

    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"])
    test_metrics = recall_ndcg_at_k(
        model,
        adj_torch["global"],
        train_pos,
        bundle.user_test,
        bundle.num_items,
        ks,
        device=device,
    )
    log.info("Test metrics: %s", test_metrics)


if __name__ == "__main__":
    main()
