# MvWebRec

Multi-view graph recommendation on **MovieLens** treated as implicit, time-stamped **usage logs**. Three bipartite views are built from training interactions: **global** (full history), **recent** (last-`k` events per user), and **frequency** (edges to popular items). A **shared-weight LightGCN** encodes users and items; optimization uses **BPR** on the global view plus **InfoNCE** to align user embeddings across two views. Evaluation reports **Recall@K** and **NDCG@K**; `demo.py` compares recommendations by view and can plot **t-SNE** when baseline checkpoints are provided.

## Layout

```text
.
├── configs/              # YAML experiment configs
├── data/                 # runtime data (ml-100k/ gitignored)
├── docs/                 # notes (e.g. improvement ideas)
├── scripts/              # download helpers
├── src/mvwebrec/         # Python package (datasets, models, utils, visualization)
├── tests/
├── train.py              # CLI entry → mvwebrec.train
├── demo.py               # CLI entry → mvwebrec.demo
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"    # editable install + pytest
```

Alternatively install dependencies only (root `train.py` / `demo.py` add `src/` to `sys.path` so the package runs without `-e`):

```bash
pip install -r requirements.txt
```

Python **3.10+** recommended.

## Data

```bash
bash scripts/download_ml100k.sh
```

Produces `data/ml-100k/u.data`. Tune paths and flags in [`configs/config.yaml`](configs/config.yaml).

## Train

| Config | Role |
|--------|------|
| [`configs/config_baseline.yaml`](configs/config_baseline.yaml) | Global graph only, BPR |
| [`configs/config_ablation.yaml`](configs/config_ablation.yaml) | Global + frequency, InfoNCE (no recent view) |
| [`configs/config.yaml`](configs/config.yaml) | Full model |

```bash
python train.py --config configs/config_baseline.yaml
python train.py --config configs/config_ablation.yaml
python train.py --config configs/config.yaml
```

Default `--config` is `configs/config.yaml`. Optional: `--mode …`, `--checkpoint …`. Outputs go to `checkpoints/` and `visualization/` (see `.gitignore`).

## Demo

```bash
python demo.py \
  --checkpoint ./checkpoints/model.pt \
  --baseline_ckpt ./checkpoints/model_baseline.pt \
  --ablation_ckpt ./checkpoints/model_ablation.pt \
  --user_id 5
```

## Tests

```bash
pytest tests/ -q
```

## References

- **MovieLens** — [GroupLens](https://grouplens.org/datasets/movielens/)
- **LightGCN** — He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*
