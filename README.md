# MvWebRec

**MvWebRec** (Multi-View Web Recommendation) is a compact, demo-oriented recommendation project in a **Web Usage Mining** spirit: MovieLens ratings and timestamps are treated as implicit, time-stamped behavior. We build three bipartite **views**—full history (**global**), each user’s recent tail (**recent**), and a popularity-heavy subgraph (**frequency**). A single **LightGCN** with **shared weights** encodes users and items; **BPR** trains link-prediction-style ranking, and a light **InfoNCE** term aligns user embeddings across views. The focus is clear pipelines and visualization (Top-K and t-SNE), not novel architecture engineering.

**Repository name on GitHub:** `MvWebRec` (this project).

## Requirements

- Python 3.10+
- Dependencies: [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Reproducible installs:** To capture exact versions on your machine:

```bash
pip freeze > requirements-lock.txt
```

`requirements-lock.txt` is **gitignored** in this repo; keep it locally or attach it to a release if you need pinned environments for others.

## Data

```bash
bash data/download_ml100k.sh
```

You should get `data/ml-100k/u.data`. The `data/ml-100k/` directory is **gitignored** so large data is not committed.

## Training

Three configs: baseline / ablation / full model.

```bash
python train.py --config config_baseline.yaml
python train.py --config config_ablation.yaml
python train.py --config config.yaml
```

Optional overrides: `--mode baseline|full|ablation_no_recent`, `--checkpoint ./checkpoints/custom.pt`.

Logs go to stderr by default. If `logging.file` is set in the YAML, logs are also written there. Per-epoch metrics are written to **`{checkpoint_basename}.metrics.csv`** next to the checkpoint (or path in `logging.metrics_csv`). Each training run **replaces** that CSV for the given checkpoint stem.

**Early stopping:** if `train.early_stop_patience > 0`, training stops when validation `recall@K` (first `K` in `eval_ks`) does not improve for that many epochs. The saved checkpoint is still the best validation checkpoint.

## Demo

```bash
python demo.py \
  --checkpoint ./checkpoints/model.pt \
  --baseline_ckpt ./checkpoints/model_baseline.pt \
  --ablation_ckpt ./checkpoints/model_ablation.pt \
  --user_id 5
```

Figures (e.g. flowchart, degree plot, t-SNE) are written under `visualization/` when you run `demo.py`; **PNG outputs are not tracked in Git** (regenerate locally).

## Tests

```bash
pytest tests/ -q
```

## Credits

- **MovieLens:** [GroupLens](https://grouplens.org/datasets/movielens/) — use per their terms.
- **LightGCN:** He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* — this repo is a simplified teaching-style implementation, not an official reproduction.

## Further ideas

See [`improvement.md`](improvement.md).
