"""t-SNE for user embedding comparison (baseline vs multi-view model)."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne_two_models(emb_baseline: np.ndarray, emb_ours: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    n = min(len(emb_baseline), len(emb_ours))
    emb_baseline = emb_baseline[:n]
    emb_ours = emb_ours[:n]
    ts_b = TSNE(n_components=2, random_state=0, init="random", learning_rate="auto")
    ts_o = TSNE(n_components=2, random_state=1, init="random", learning_rate="auto")
    zb = ts_b.fit_transform(emb_baseline)
    zo = ts_o.fit_transform(emb_ours)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(zb[:, 0], zb[:, 1], s=10, alpha=0.7, c="#4477aa")
    axes[0].set_title("Baseline (Global LightGCN)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].scatter(zo[:, 0], zo[:, 1], s=10, alpha=0.7, c="#cc6677")
    axes[1].set_title("Ours (multi-view + InfoNCE)")
    axes[1].set_xticks([])
    axes[1].yaxis.set_visible(False)
    fig.suptitle("User embedding t-SNE (same sampled users)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
