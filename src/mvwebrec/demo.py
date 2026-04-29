"""Case study demo, multi-view recommendation comparison, t-SNE, metrics table."""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from mvwebrec.datasets.preprocessing import build_dataset, print_stats
from mvwebrec.models.lightgcn import LightGCN, scipy_csr_to_torch_sparse_float
from mvwebrec.paths import repo_root
from mvwebrec.utils.graph_views import build_all_views, print_user_view_example
from mvwebrec.utils.metrics import recall_ndcg_at_k, topk_items
from mvwebrec.visualization.plot_tsne import plot_tsne_two_models
from mvwebrec.visualization.plot_views import plot_pipeline_flowchart, plot_view_degrees


def load_config_from_ckpt(ck: dict) -> dict:
    return ck["config"]


def _as_train_pos(raw) -> Dict[int, Set[int]]:
    if not raw:
        return {}
    return {int(u): set(map(int, items)) for u, items in raw.items()}


def _metrics_for_checkpoint(
    ckpt_path: str,
    bundle,
    train_pos: Dict[int, Set[int]],
    ks: List[int],
    device: torch.device,
) -> Tuple[Dict[str, float], LightGCN, dict, dict]:
    ck = torch.load(ckpt_path, map_location=device)
    cfg_l = ck["config"]
    model_l = LightGCN(
        bundle.num_users,
        bundle.num_items,
        int(cfg_l["model"]["embed_dim"]),
        int(cfg_l["model"]["n_layers"]),
    )
    model_l.load_state_dict(ck["model"])
    model_l.to(device)
    model_l.eval()
    views_l = build_all_views(
        bundle,
        recent_k=int(cfg_l["recent_k"]),
        freq_quantile=float(cfg_l["frequency"]["quantile"]),
        mode=str(cfg_l.get("mode", "full")),
    )
    adj_l = {
        k: scipy_csr_to_torch_sparse_float(v, device) for k, v in views_l.items()
    }
    with torch.no_grad():
        metrics = recall_ndcg_at_k(
            model_l,
            adj_l["global"],
            train_pos,
            bundle.user_test,
            bundle.num_items,
            ks,
            device=device,
        )
    return metrics, model_l, adj_l, cfg_l


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/model.pt")
    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default="",
        help="Optional second checkpoint for t-SNE comparison (LightGCN global-only).",
    )
    parser.add_argument(
        "--ablation_ckpt",
        type=str,
        default="",
        help="Optional w/o-Recent checkpoint (Global+Frequency + InfoNCE) for metric table.",
    )
    parser.add_argument("--user_id", type=int, default=-1, help="Demo user index.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--tsne_users", type=int, default=400)
    args = parser.parse_args()

    repo = str(repo_root())

    device = torch.device("cpu")
    ck = torch.load(args.checkpoint, map_location=device)
    cfg = load_config_from_ckpt(ck)
    mode = str(cfg.get("mode", "full"))
    data_dir = str(cfg["data_dir"])

    bundle = build_dataset(
        data_dir=data_dir,
        implicit_threshold=int(cfg["implicit_threshold"]),
        train_ratio=float(cfg["split"]["train_ratio"]),
        val_ratio=float(cfg["split"]["val_ratio"]),
        test_ratio=float(cfg["split"]["test_ratio"]),
    )
    print_stats(bundle)

    train_pos = _as_train_pos(ck.get("train_pos"))
    if not train_pos:
        train_pos = {u: set() for u in range(bundle.num_users)}
        for u, i in bundle.train_pairs:
            train_pos.setdefault(u, set()).add(i)

    views_sp = build_all_views(
        bundle,
        recent_k=int(cfg["recent_k"]),
        freq_quantile=float(cfg["frequency"]["quantile"]),
        mode=mode,
    )
    adj_torch = {
        k: scipy_csr_to_torch_sparse_float(v, device) for k, v in views_sp.items()
    }

    model = LightGCN(
        bundle.num_users,
        bundle.num_items,
        int(cfg["model"]["embed_dim"]),
        int(cfg["model"]["n_layers"]),
    )
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()

    ks = [int(k) for k in cfg["train"]["eval_ks"]]
    test_metrics = recall_ndcg_at_k(
        model,
        adj_torch["global"],
        train_pos,
        bundle.user_test,
        bundle.num_items,
        ks,
        device=device,
    )
    print("This checkpoint — Test metrics:", test_metrics)

    os.makedirs(os.path.join(repo, "visualization"), exist_ok=True)
    flow_out = os.path.join(repo, "visualization", "pipeline_flowchart.png")
    plot_pipeline_flowchart(flow_out)
    print("Saved flowchart:", flow_out)

    deg_path = os.path.join(repo, "visualization", "view_degrees.png")
    plot_view_degrees(views_sp, bundle.num_users, deg_path)
    print("Saved view degree plot:", deg_path)

    u = args.user_id
    if u < 0:
        u = next(iter(sorted(bundle.user_train.keys())))
    print_user_view_example(
        u,
        bundle.user_train,
        bundle.item_popularity,
        int(cfg["recent_k"]),
        float(cfg["frequency"]["quantile"]),
    )

    with torch.no_grad():
        embs = model.forward_views(adj_torch)

    hist = bundle.user_train.get(u, [])
    print("\n=== Case study: history (train, item, ts) ===")
    print(hist[-20:])

    k = args.topk
    blocks: List[str] = []
    pairs = [
        ("global", "Rec using Global-view user emb (long-term)"),
        ("recent", "Rec using Recent-view user emb (short-term)"),
        ("frequency", "Rec using Frequency-view user emb (popular/stable)"),
    ]
    with torch.no_grad():
        for key, title in pairs:
            if key not in embs:
                continue
            ue, ie = embs[key]
            rec = topk_items(ue[u], ie, train_pos.get(u, set()), k, bundle.num_items)
            blocks.append(f"\n{title}\n  top-{k}: {rec}")

    print("\n=== Recommendation comparison (same items space, different user embedding) ===")
    print("\n".join(blocks))

    table_rows: List[Tuple[str, Dict[str, float]]] = [
        ("Multi-view + InfoNCE (full)", test_metrics),
    ]
    model_b = None
    adj_b = None

    if args.ablation_ckpt and os.path.isfile(args.ablation_ckpt):
        test_ab, _, _, _ = _metrics_for_checkpoint(
            args.ablation_ckpt, bundle, train_pos, ks, device
        )
        table_rows.insert(0, ("w/o Recent (Global+Freq+CL)", test_ab))

    if args.baseline_ckpt and os.path.isfile(args.baseline_ckpt):
        test_b, model_b, adj_b, _ = _metrics_for_checkpoint(
            args.baseline_ckpt, bundle, train_pos, ks, device
        )
        table_rows.insert(0, ("LightGCN Global only (baseline)", test_b))

    if len(table_rows) > 1:
        print("\n=== Metric table (Recall@K / NDCG@K on test) ===")
        print(f"{'Method':<36} " + " | ".join(test_metrics.keys()))
        for name, met in table_rows:
            vals = " | ".join(f"{met[k]:.4f}" for k in test_metrics)
            print(f"{name:<36} {vals}")
    elif args.baseline_ckpt:
        print("\n(Set valid paths for --baseline_ckpt / --ablation_ckpt to expand the table.)")

    if args.baseline_ckpt and os.path.isfile(args.baseline_ckpt) and model_b is not None:
        rng = np.random.RandomState(int(cfg["train"]["seed"]))
        cand = [x for x in bundle.user_test if bundle.user_test[x]]
        if len(cand) > args.tsne_users:
            cand = list(rng.choice(cand, size=args.tsne_users, replace=False))
        else:
            cand = cand[: args.tsne_users]

        with torch.no_grad():
            u_o, _ = model.forward_from_adj(adj_torch["global"])
            u_b, _ = model_b.forward_from_adj(adj_b["global"])
        emb_o = u_o[cand].cpu().numpy()
        emb_b = u_b[cand].cpu().numpy()
        tsne_out = os.path.join(repo, "visualization", "tsne_compare.png")
        plot_tsne_two_models(emb_b, emb_o, tsne_out)
        print("Saved t-SNE:", tsne_out)
    else:
        print("\n(Pass --baseline_ckpt to save visualization/tsne_compare.png.)")

