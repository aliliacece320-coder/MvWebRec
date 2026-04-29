# HetUBV-GCL

[English](README.md) | **简体中文**

*Modeling Heterogeneous User Behavior Views in Web Usage Mining via Graph Contrastive Learning.*（通过图对比学习在 Web 使用挖掘中对异构用户行为视角进行建模。）

将 **MovieLens** 视为带时间戳的**隐式使用日志**，在多二部图上做推荐。由训练交互构建三个二部图视角：**全局**（完整历史）、**近期**（每用户最近 `k` 条）、**频次**（连向热门物品的边）。采用**共享权重的 LightGCN** 编码用户与物品；目标为在全局图上 **BPR**，并用 **InfoNCE** 对齐两个视角上的用户嵌入。评估输出 **Recall@K**、**NDCG@K**；`demo.py` 可按视角对比推荐，并在提供 baseline 检查点时可绘制 **t-SNE**。

## 目录结构

```text
.
├── configs/              # YAML 实验配置
├── data/                 # 运行时数据（ml-100k/ 已 gitignore）
├── docs/                 # 笔记（如改进思路）
├── scripts/              # 下载等辅助脚本
├── src/hetubv_gcl/       # Python 包（datasets、models、utils、visualization）
├── tests/
├── train.py              # CLI 入口 → hetubv_gcl.train
├── demo.py               # CLI 入口 → hetubv_gcl.demo
└── pyproject.toml
```

## 环境配置

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"    # 可编辑安装 + pytest
```

也可只安装依赖（根目录 `train.py` / `demo.py` 会把 `src/` 加入 `sys.path`，无需 `-e` 亦可运行）：

```bash
pip install -r requirements.txt
```

建议使用 **Python 3.10+**。

## 数据

```bash
bash scripts/download_ml100k.sh
```

生成 `data/ml-100k/u.data`。路径与开关可在 [`configs/config.yaml`](configs/config.yaml) 中调整。

## 训练

| 配置文件 | 说明 |
|----------|------|
| [`configs/config_baseline.yaml`](configs/config_baseline.yaml) | 仅全局图 + BPR |
| [`configs/config_ablation.yaml`](configs/config_ablation.yaml) | 全局 + 频次，InfoNCE（无近期视角） |
| [`configs/config.yaml`](configs/config.yaml) | 完整模型 |

```bash
python train.py --config configs/config_baseline.yaml
python train.py --config configs/config_ablation.yaml
python train.py --config configs/config.yaml
```

默认 `--config` 为 `configs/config.yaml`。可选：`--mode …`、`--checkpoint …`。产出写入 `checkpoints/` 与 `visualization/`（见 `.gitignore`）。

## 演示

```bash
python demo.py \
  --checkpoint ./checkpoints/model.pt \
  --baseline_ckpt ./checkpoints/model_baseline.pt \
  --ablation_ckpt ./checkpoints/model_ablation.pt \
  --user_id 5
```

## 测试

```bash
pytest tests/ -q
```

## 仓库维护

有实质改动且影响使用方式、目录结构或安装步骤时：**提交**并尽量推送到 **GitHub**；同时更新 [`README.md`](README.md) 与 [`README.zh-CN.md`](README.zh-CN.md)，保持中英一致。

**GitHub 仓库名：**项目展示名为 **HetUBV-GCL**，可在仓库 *Settings → General* 中将名称改为 `hetubv-gcl`，然后在本地执行 `git remote set-url origin https://github.com/PatoMc320/hetubv-gcl.git`（若用户名有变请替换）。

## 参考文献

- **MovieLens** — [GroupLens](https://grouplens.org/datasets/movielens/)
- **LightGCN** — He 等，*LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*
