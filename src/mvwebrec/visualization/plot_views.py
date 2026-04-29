"""Bar chart of approximate user degrees per view; pipeline flowchart."""
from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


def plot_view_degrees(
    views_sp: Dict[str, sp.csr_matrix],
    n_users: int,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    names = list(views_sp.keys())
    means = []
    for name in names:
        adj = views_sp[name]
        sub = adj[:n_users, n_users:]
        row_deg = np.diff(sub.indptr)
        means.append(float(row_deg.mean()) if len(row_deg) else 0.0)
    plt.figure(figsize=(6, 4))
    plt.bar(names, means, color=["#4477aa", "#228833", "#ccbb44"][: len(names)])
    plt.ylabel("Avg user degree (train bipartite)")
    plt.title("Multi-view graphs: connectivity summary")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pipeline_flowchart(out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.2, 0.5, 1.4, 1.0, "Usage logs\n(MovieLens+time)"),
        (2.0, 0.5, 1.6, 1.0, "Views\nGlobal/Recent/Freq"),
        (4.2, 0.5, 1.3, 1.0, "LightGCN\n(shared)"),
        (6.0, 0.5, 1.3, 1.0, "Train\nBPR+InfoNCE"),
        (7.8, 0.5, 1.4, 1.0, "Demo\nRec+t-SNE"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=True, facecolor="#e8f0fe", edgecolor="#333")
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        x0, y0, w0, _, _ = boxes[i]
        x1, _, _, _, _ = boxes[i + 1]
        ax.annotate(
            "",
            xy=(x1, 1.0),
            xytext=(x0 + w0, 1.0),
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
