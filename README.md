# MvWebRec

Multi-view graph recommendation on **MovieLens** treated as implicit, time-stamped **usage logs**. Three bipartite views are built from training interactions: **global** (full history), **recent** (last-`k` events per user), and **frequency** (edges to popular items). A **shared-weight LightGCN** encodes users and items; optimization uses **BPR** on the global view plus **InfoNCE** to align user embeddings across two views (global–recent in the full setup, global–frequency in the ablation). Evaluation reports **Recall@K** and **NDCG@K**; `demo.py` compares recommendations by view and can plot **t-SNE** when baseline checkpoints are provided.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.10+** recommended.

## Data

Download **MovieLens-100K** into `data/ml-100k/`:

```bash
bash data/download_ml100k.sh
```

Configure path and options in [`config.yaml`](config.yaml) (`data_dir`, `implicit_threshold`, `split`, `recent_k`, `frequency.quantile`, etc.).

## Train

| Config | Role |
|--------|------|
| [`config_baseline.yaml`](config_baseline.yaml) | Global graph only, BPR |
| [`config_ablation.yaml`](config_ablation.yaml) | Global + frequency views, InfoNCE (no recent view) |
| [`config.yaml`](config.yaml) | Full: global + recent + frequency, BPR + InfoNCE |

```bash
python train.py --config config_baseline.yaml
python train.py --config config_ablation.yaml
python train.py --config config.yaml
```

Optional: `--mode …`, `--checkpoint …`. Checkpoints and per-run metrics CSV go under `checkpoints/` (ignored by git). **Early stopping** uses `train.early_stop_patience` on validation `recall@K` (`K` = first value in `eval_ks`). Logging: `logging` section in YAML.

## Demo

```bash
python demo.py \
  --checkpoint ./checkpoints/model.pt \
  --baseline_ckpt ./checkpoints/model_baseline.pt \
  --ablation_ckpt ./checkpoints/model_ablation.pt \
  --user_id 5
```

Writes figures under `visualization/` (e.g. flowchart, view degrees, t-SNE comparison).

## Tests

```bash
pytest tests/ -q
```

## References

- **MovieLens** — [GroupLens](https://grouplens.org/datasets/movielens/)
- **LightGCN** — He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*
