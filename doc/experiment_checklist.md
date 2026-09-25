# TA-SID 实验清单 & 遗留问题
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

> 生成日期：2026-07-17
> 用途：在新服务器上继续实验时对照排查

---

## 一、已完成实验（无需重跑）

### 1.1 核心对比（3 数据集 × 2 种子）

| 数据集 | Baseline (seed=2025) | TA-SID (seed=2025) | TA-SID (seed=42) | 提升 |
|:------|:-------------------:|:------------------:|:----------------:|:----:|
| Beauty | 0.0379 | **0.04181** | 0.04035 | **+10.3%** |
| Sport | 0.0207 | **0.0276** | 0.02759 | **+33.3%** |
| Toys | 0.0305 | **0.0317** | 0.03118 | **+3.9%** |

**结论：** TA-SID 在三数据集上全部正向，种间差异仅 1~3%。

### 1.2 已有记录

| 文件 | 位置 |
|:----|:-----|
| TA-SID 三轮完整日志 | `ai-log.md` |
| RQ-VAE 基线结果 | `rqvae/ckpt/{Beauty,Sports,Toys}/` |
| RQ-VAE TA-SID 结果 | `rqvae/ckpt/{Beauty,Sport,Toys}_TA_SID/` |
| T5 训练结果 | `model/logs/` |
| 论文方案文档 | `doc/TA-SID_PLAN.md`, `doc/scheme_a_summary.md` |

---

## 二、待完成实验（在新服务器跑）

### 🔴 [P0] 消融实验：Random Noise（核心，审稿人必问）

验证 TA-SID 的提升来自 behavior 信号，而非额外参数。

| 数据集 | 命令 | 预计时间 |
|:------|:-----|:--------|
| **Toys** | `./tiger_migration.sh run_ablation_toys` | ~8h |
| **Sport** | `./tiger_migration.sh run_ablation_sport` | ~8h（可选） |

**判断标准：**
- Random Noise 结果 ≈ Baseline → 证明 behavior 信号有效 ✅
- Random Noise 结果 ≈ TA-SID → 说明提升来自额外参数，方法无效 ❌

**注意：** 消融和 baseline 在同一代码结构内跑，`run_both_datasets.sh --random` 会生成随机 128d 行为嵌入。

### 🟡 [P1] Beauty Baseline seed=42

| 数据集 | 命令 | 预计时间 |
|:------|:-----|:--------|
| Beauty | `./tiger_migration.sh run_baseline` | ~2h |

**条件：** `data/Beauty/Beauty_t5_rqvae.npy` 必须存在（已打包在迁移文件中）。

**作用：** 完成配对 t 检验，排除种子对结果的影响。

### 🟢 [P2] 定性分析（0 GPU，随时可做）

不需要 GPU，可和新服务器实验并行进行：

| 任务 | 方法 | 工具 |
|:----|:-----|:----|
| t-SNE 可视化 | 对 content vs. fused embeddings 降维绘图 | `rqvae/fuse_embeddings.py` + sklearn |
| Code 变化示例表 | 列出 RQ-VAE 前后码本分配不同的 items | 对比 `.npy` 码文件 |
| 用户行为图 | 展示过渡图的 PPMI 矩阵热力图 | `build_transition_graph.py` 产物 |

### 🔵 [P3] 超参数扫描（可选，论文 C 类）

| 参数 | 默认值 | 扫描范围 | 时间 |
|:----|:------|:---------|:----|
| `--svd_dim` | 128 | [64, 128, 256] | ~3h/轮 |
| `--window_size` | 3 | [2, 3, 5] | ~1h/轮 |

---

## 三、遗留问题（排查用）

### 🚩 3.1 目录命名不一致：`Sport` vs `Sports`

```
rqvae/ckpt/Sports/        ← 基线 RQ-VAE（有 's'）
rqvae/ckpt/Sport_TA_SID/  ← TA-SID RQ-VAE（无 's'）
data/Sport/               ← 数据目录（无 's'）
```

**影响：** 基线 RQ-VAE 的 hardcoded `data_path` 指向 `../data/Sports/`（带 s），但数据目录是 `data/Sport/`。运行 `generate_code_Sport.py` 时需确认 `ckpt_path` 中的目录名与实际一致。

### 🚩 3.2 Beauty 缺少 `item_emb.parquet`

```bash
# Beauty 只有以下 parquet 文件：
data/Beauty/behavior_emb.parquet
data/Beauty/fused_item_emb.parquet
data/Beauty/train.parquet
data/Beauty/valid.parquet
data/Beauty/test.parquet

# 没有：
data/Beauty/item_emb.parquet         ← 缺失
data/Beauty/Beauty_t5_rqvae.npy     ← 已从 TIGER 项目迁移过来
```

**影响：** 无法在 Beauty 上做消融实验（需要原始 content embedding）。Beauty 的 baseline seed=42 靠的是迁移过来的 `Beauty_t5_rqvae.npy`，不需要 `item_emb.parquet`。

### 🚩 3.3 Sport/Toys 768d 内容嵌入已被覆盖

- 原始 768d `item_emb.parquet`（Sentence-T5-base）被 Sentence-T5-xl（4096d）覆盖
- 导致**无法在这两个数据集上跑 baseline seed=42**
- 当前只剩 Beauty 能做 baseline seed=42

### 🚩 3.4 硬编码的绝对路径

| 文件 | 硬编码的内容 |
|:----|:------------|
| `rqvae/generate_code_Sport.py:18` | `ckpt_path = "/data/gtx/project/code/TIGER_minilm/rqvae/ckpt/..."` |
| `rqvae/generate_code_Toys.py:18` | `ckpt_path = "/data/gtx/project/code/TIGER_minilm/rqvae/ckpt/..."` |
| `rqvae/generate_code_Sport.py:19` | `output_file = "/data/gtx/project/code/TIGER_minilm/data/Sport/..."` |
| `rqvae/generate_code_Toys.py:19` | `output_file = "/data/gtx/project/code/TIGER_minilm/data/Toys/..."` |

**影响：** 新服务器上项目路径不同会导致这些文件报错。使用迁移脚本的 `setup` 命令会自动适配，如果手动运行需要修改路径。

### 🚩 3.5 RQ-VAE epoch 快照未打包

迁移包只包含 `best_collision_model.pth`，不包含 `epoch_*.pth` 文件。

**影响：** 无法恢复某个特定 epoch 的 RQ-VAE 模型。如果需要中间结果，必须在旧服务器单独打包。

### 🚩 3.6 相对路径执行位置

`model/dataset.py` 和 `model/main.py` 中的默认路径使用 `../data/{DATASET}/` 格式，**必须在 `model/` 目录下运行**。迁移脚本中已经在各命令中处理了目录切换。

---

## 四、在新服务器上的标准操作流程

```bash
# === 0. 初始化 ===
cd ~
tar xzf tiger_ta_sid_migration.tar.gz
cd TIGER_minilm

# === 1. 环境验证 ===
./tiger_migration.sh setup

# === 2. 跑补实验（推荐 tmux 后台） ===
tmux new -s experiment
./tiger_migration.sh run_all
# Ctrl+B, D 脱离

# === 3. 查看某个日志 ===
tail -f model/logs/tiger_ta_sid_toys_random.log

# === 4. 查看结果汇总 ===
./tiger_migration.sh report

# === 5. 单独跑 ===
./tiger_migration.sh run_ablation_toys    # Toys 消融
./tiger_migration.sh run_baseline          # Beauty baseline seed=42
./tiger_migration.sh run_sweep             # 超参扫描（可选）
```

---

## 五、论文所需材料清单

| 素材 | 状态 | 优先级 |
|:----|:----|:------|
| 主实验结果表（3 数据集） | ✅ 已有 | P0 |
| NDCG@5/10/20 详细表 | ✅ 从日志提取 | P0 |
| 种子鲁棒性（seed=42 验证） | ✅ 已有 | P0 |
| **消融实验**（Random Noise） | ⬜ 待跑 | **P0** |
| t-SNE 对比图 | ⬜ 待做（0 GPU） | P1 |
| Code 分配变化示例 | ⬜ 待做（0 GPU） | P1 |
| 过渡图/行为矩阵可视化 | ⬜ 待做（0 GPU） | P1 |
| Baseline seed=42（Beauty 配对） | ⬜ 待跑 | P1 |
| 超参数敏感性分析 | ⬜ 可选 | P2 |
| 与相关方法对比（方案 B~E） | ⬜ 下一篇文章 | P3 |
