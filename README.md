# MvWebRec

**MvWebRec**（Multi-View Web Recommendation）— 面向 Web Usage Mining 场景的多视角图学习推荐：将 MovieLens 评分与时间戳视为隐式「点击/浏览」日志，构建 Global / Recent / Frequency 三种用户–物品二部图视角，使用**共享参数** LightGCN 编码，BPR 主任务 + 轻量 InfoNCE 对齐不同视角下的用户表示，用于 Top-K 推荐与可视化展示。

- **建议 GitHub 仓库名**：`mv-web-rec`（全小写、连字符，便于 URL 与引用）

## 环境

- Python 3.10+
- 依赖见 [`requirements.txt`](requirements.txt)。**可复现锁定**：安装后在同一环境中执行  
  `pip freeze > requirements-lock.txt`  
  提交或附带该文件便于他人对齐版本（说明见文末「依赖锁定」）。

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 数据

```bash
bash data/download_ml100k.sh
```

解压后应存在 `data/ml-100k/u.data`（已在 [`.gitignore`](.gitignore) 中忽略数据目录，避免将大数据提交进 Git）。

## 训练

三种配置对应 baseline / 消融 / 完整方法：

```bash
python train.py --config config_baseline.yaml
python train.py --config config_ablation.yaml
python train.py --config config.yaml
```

可选覆盖：`--mode baseline|full|ablation_no_recent`、`--checkpoint ./checkpoints/custom.pt`。

训练日志默认输出到 stderr；若 `config.yaml` 中 `logging.file` 非空，会额外写入该文件。每个 epoch 指标会追加写入 **`{checkpoint 文件名不含后缀}.metrics.csv`**（与 checkpoint 同目录），或通过 `logging.metrics_csv` 指定路径。

**早停**：在 `train.early_stop_patience` 设为大于 0 时，当验证集 `recall@K`（`K` 为 `eval_ks` 首个元素）连续若干 epoch 未提升则停止；最佳权重仍以验证集最优为准，保存路径不变。

## 展示脚本（demo）

```bash
python demo.py \
  --checkpoint ./checkpoints/model.pt \
  --baseline_ckpt ./checkpoints/model_baseline.pt \
  --ablation_ckpt ./checkpoints/model_ablation.pt \
  --user_id 5
```

产出示例：`visualization/pipeline_flowchart.png`、`view_degrees.png`、`tsne_compare.png`（需传入 `--baseline_ckpt`）。

## 测试

```bash
pip install pytest
pytest tests/ -q
```

## 引用与许可

- **MovieLens**：[GroupLens](https://grouplens.org/datasets/movielens/)，使用请遵循其许可条款。
- **LightGCN 思想**：He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*（实现为本仓库简化授课版，非官方复现）。

## 后续改进

见 [`improvement.md`](improvement.md)。

## 依赖锁定（工作流）

1. 在干净 venv 中 `pip install -r requirements.txt`。
2. 运行 `pip freeze > requirements-lock.txt`。
3. 复现时：`pip install -r requirements-lock.txt`。

根目录下的 [`requirements-lock.txt`](requirements-lock.txt) 为在某一固定环境中导出的示例；若你的平台或 Python 版本不同，请在本机重新生成。

每次启动训练会**删除并重写**与本次 run 对应的 `*.metrics.csv`（与 checkpoint 同目录），避免多次运行混在同一文件。

## 推送到 GitHub

1. 在 [GitHub 新建仓库](https://github.com/new)，名称填 **`mv-web-rec`**（或你喜欢的名称），**不要**勾选「Add a README」（本地已有）。
2. 在本项目根目录执行（将 `YOUR_USER` 换成你的 GitHub 用户名）：

```bash
git remote add origin https://github.com/YOUR_USER/mv-web-rec.git
git branch -M main
git push -u origin main
```

若已安装 [GitHub CLI](https://cli.github.com/) 且已登录，可改为：

```bash
gh repo create mv-web-rec --private --source=. --remote=origin --push
```

将 `--private` 改为 `--public` 可创建公开仓库。
