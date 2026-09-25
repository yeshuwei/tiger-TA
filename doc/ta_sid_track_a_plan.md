# 线 A 执行方案：基于现有代码的补充实验
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

> 生成日期：2026-07-28
> 用途：线 A 的具体执行步骤、命令、新建脚本说明
> 前置条件：conda 环境 `tiger_t5`（已验证：torch 2.4.0+cu124, 所有模块可导入）

---

## 一、环境准备

```bash
# 每次新开终端先激活：
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tiger_t5
cd /data/gtx/project/code/TIGER_minilm
```

已安装依赖：torch 2.4.0, numpy 2.2.6, pandas 2.3.3, scikit-learn, scipy, tqdm, pyarrow

---

## 二、执行总览

```
时间线                     GPU 任务                         CPU 任务
Day 1 (启动)              ┌──────────────────┐            ┌──────────────────┐
                         │ A-1: Random Noise │            │ 创建 4 个分析脚本 │
                         │     消融 (~8h)    │            │                  │
                         │ A-2: Beauty       │            │ 创建冷启动脚本    │
                         │     seed=42 (~2h) │            │                  │
                         └──────────────────┘            └──────────────────┘

Day 2 (GPU 跑完后)        GPU 空闲                          CPU 分析并行
                                                          ┌──────────────────┐
                                                         │ 跑长尾分析       │
                                                         │ 跑 SID 质量分析   │
                                                         │ 跑指标提取       │
                                                         │ 跑 t-SNE        │
                                                         └──────────────────┘

Day 3+ (GPU 空闲)         ┌──────────────────┐
                         │ A-3: 冷启动实验    │
                         │     (~4h)         │
                         └──────────────────┘
```

---

## 三、GPU 任务执行命令

### A-1：Random Noise 消融（最高优先级，~8h）

**作用：** 验证 TA-SID 的提升来自 behavior 信号，而非维度增加。

**命令：**

```bash
# 终端 1：启动 Random Noise 消融
conda activate tiger_t5
cd /data/gtx/project/code/TIGER_minilm

# 方式一：用迁移脚本（推荐）
./tiger_migration.sh run_ablation

# 如果迁移脚本报路径错，用手动分步执行：
```

<details>
<summary>手动分步执行（如果脚本报错时备用）</summary>

```bash
# Step 1: 生成随机行为嵌入
python3 -c "
import numpy as np, pandas as pd
orig = pd.read_parquet('data/Toys/behavior_emb.parquet')
n, d = orig.shape[0], orig.shape[1]
np.random.seed(2026)
random_emb = np.random.randn(n, d).astype(np.float32)
df = pd.DataFrame({'ItemID': orig['ItemID'].values, 'embedding': [random_emb[i] for i in range(n)]})
df.to_parquet('data/Toys/random_behavior_emb.parquet', index=False)
print(f'✓ {n}×{d} random embedding saved')
"

# Step 2: 融合 Content + Random Behavior
python3 -c "
import numpy as np, pandas as pd, torch, torch.nn as nn

content_df = pd.read_parquet('data/Toys/item_emb.parquet')
c_ids = content_df['ItemID'].values
content_emb = np.stack(content_df['embedding'].values)

random_df = pd.read_parquet('data/Toys/random_behavior_emb.parquet')
rand_emb = np.stack(random_df['embedding'].values)

ct = torch.from_numpy(content_emb).float()
rt = torch.from_numpy(rand_emb).float()

ln_c = nn.LayerNorm(ct.shape[-1])
ln_r = nn.LayerNorm(rt.shape[-1])
fused = torch.cat([ln_c(ct), ln_r(rt)], dim=-1).numpy().astype(np.float32)

emb_list = [fused[i] for i in range(fused.shape[0])]
out_df = pd.DataFrame({'ItemID': c_ids, 'embedding': emb_list})
out_df.to_parquet('data/Toys/fused_random_emb.parquet', index=False)
print(f'✓ fused random emb: {fused.shape}')
"

# Step 3: RQ-VAE 训练（cd 到 rqvae/ 目录执行）
cd rqvae
mkdir -p ckpt/Toys_Random
python3 main.py \
    --data_path \"../data/Toys/fused_random_emb.parquet\" \
    --ckpt_dir \"./ckpt/Toys_Random\" \
    --epochs 3000 \
    --batch_size 1024 \
    --eval_step 50 \
    2>&1 | tee ../rqvae/logs/Toys_Random_train.log

# Step 4: 生成 Random SID 码（需要找到最佳 checkpoint 后用 generate_code.py）
# Step 5: T5 训练
cd ../model
python3 main.py \
    --dataset_path \"../data/Toys\" \
    --code_path \"../data/Toys/Toys_random_sid.npy\" \
    --save_path \"../model/ckpt/toys_random_sid.pth\" \
    --log_path \"../model/logs/toys_random_sid.log\" \
    --seed 2025 \
    2>&1 | tee \"../model/logs/toys_random_t5_train.log\"
```
</details>

### A-2：Beauty baseline seed=42（~2h）

**作用：** 完成 Beauty 的配对比较，排除种子影响。

**命令：**

```bash
# 终端 2：启动 Beauty baseline（与 A-1 同时跑）
conda activate tiger_t5
cd /data/gtx/project/code/TIGER_minilm
./tiger_migration.sh run_baseline
```

### A-3：冷启动实验（P0-C，需新建脚本，~4h）

在 A-1 和 A-2 跑完后（或 GPU 空闲时），用新建的冷启动脚本执行：

```bash
# 新建脚本：tools/run_coldstart.py
python3 tools/run_coldstart.py \
    --dataset Toys \
    --ratios 0.05,0.10,0.20 \
    --rqvae_ckpt rqvae/ckpt/Toys_TA_SID/*/best_collision_model.pth \
    --t5_model model/ckpt/tiger_ta_sid_toys.pth
```

将新建专门的脚本说明（详见下方"四、需要新建的脚本"）。

---

## 四、需要新建的 5 个脚本

### 脚本 1：补充指标提取（P1-A）

**文件：** `tools/extract_metrics.py`

**作用：** 从已有的 T5 日志中提取所有 Test Dataset 指标，整理成论文表格。

**核心逻辑：**

```python
# 日志中已有格式（每种模型×数据集 的日志都有）：
# Test Dataset: {'Recall@5': 0.037, 'Recall@10': 0.060, 'Recall@20': 0.090}
# Test Dataset: {'NDCG@5': 0.023, 'NDCG@10': 0.030, 'NDCG@20': 0.038}

# 遍历 log 文件列表，提取每个的最后一次 Test 结果
# 输出：LaTeX 表格 + Markdown 表格
```

**素材清单（日志中已存在）：**

| 实验 | 日志文件 | 有 Test 指标 |
|:----|:---------|:------------:|
| Beauty baseline seed=2025 | `model/logs/tiger.log` | ✅ Recall+NDCG |
| Beauty TA-SID seed=2025 | `model/logs/tiger_ta_sid.log` | ✅ NDCG |
| Beauty TA-SID seed=42 | `model/logs/tiger_ta_sid_beauty_seed42.log` | ✅ NDCG |
| Sport baseline | `model/logs/sports.log` | ✅ NDCG |
| Sport TA-SID | `model/logs/tiger_ta_sid_sport.log` | ✅ NDCG |
| Toys baseline | `model/logs/toys.log` | ✅ NDCG |
| Toys TA-SID | `model/logs/tiger_ta_sid_toys.log` | ✅ NDCG |

**输出示例：**

```markdown
| Dataset | Method | Seed | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | NDCG@20 |
|---------|--------|:----:|:--------:|:---------:|:------:|:-------:|:-------:|
| Beauty  | TIGER  | 2025 | 0.0139   | 0.0254    | 0.0246 | 0.0307  | 0.0379  |
| Beauty  | TA-SID | 2025 | 0.0159   | 0.0297    | 0.0264 | 0.0336  | 0.0418  |
```

### 脚本 2：长尾分析（P0-D）

**文件：** `tools/longtail_analysis.py`

**作用：** 按 item 出现频率分三组，统计每组 Recall/NDCG。

**输入：**
- `data/{ds}/train.parquet` — 计算 item 频率
- T5 推理日志或 T5 model 做逐 item 评估

**实现思路：**

```python
# 从 train.parquet 统计 item 出现次数
# 按频率排序，分三组：
#   Head: top 20%（高频）
#   Mid: 中间 60%
#   Tail: bottom 20%（低频）
# 对每组计算平均 Recall@10 / NDCG@10
# 输出：分组对比表
```

**难点：** 需要从 T5 模型输出中解析每个 item 的预测排名。有两种方式：
- **方式 A（简单）：** 从 beam search 输出文件解析（如果保存了每步输出）
- **方式 B（复杂）：** 重新跑推理，逐 item 记录

**优先采用方式 A**，先看日志中是否有足够信息。

### 脚本 3：SID 质量分析（P0-E）

**文件：** `tools/sid_quality.py`

**作用：** 计算四个质量指标。

| 指标 | 计算方式 |
|:----|:---------|
| **Codebook utilization** | 加载 RQ-VAE checkpoint → 对全部 item 做编码 → 统计每层 codebook 有多少 codeword 被命中 |
| **Collision rate** | 1 - (unique SID 数量 / item 总数) — 从 `*_ta_sid.npy` 文件直接算 |
| **Invalid ID rate** | T5 beam search 输出中，不在 valid SID 集合中的比例 — 从 T5 日志解析 |
| **Transition consistency** | 高频 transition pair 的量化向量余弦相似度均值 |

### 脚本 4：t-SNE 可视化（P1-C）

**文件：** `tools/tsne_visualize.py`

**作用：** 生成 3 张对比图。

**输入：**
- `data/{ds}/item_emb.parquet` — Content embedding
- `data/{ds}/behavior_emb.parquet` — Behavior embedding
- `data/{ds}/fused_item_emb.parquet` — Fused embedding
- `data/{ds}/train.parquet` — 提取 category/popularity 信息用于着色

**输出：** 3 张 `.png` 图（content / behavior / fused 的 t-SNE 散点图）

### 脚本 5：冷启动实验（P0-C）

**文件：** `tools/run_coldstart.py`

**作用：** 按比例移除 test items，验证冷启动能力。

**核心逻辑：**

```python
for ratio in [0.05, 0.10, 0.20]:
    # 1. 从 test.parquet 中随机移除 ratio 的 items
    # 2. 从 T5 训练集中移除这些 items（mask 掉）
    # 3. 用已有 RQ-VAE checkpoint 为 unseen items 生成 SID
    #    （不需要重训 RQ-VAE，因为 RQ-VAE 只需要 fused_emb）
    # 4. 用已训好的 T5 模型做推理，统计 cold-start Recall/NDCG
```

注意：冷启动不需要重新训练 T5，只需要在推理时把 unseen items 的 SID 加入候选集。

---

## 五、执行顺序（推荐并行策略）

### Phase 1：立即启动（耗时 0）

```bash
# 修好 conda 环境（仅首次）
conda activate tiger_t5
pip install scikit-learn scipy

# 确认 GPU 可用
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Phase 2：GPU 跑实验 + 我创建脚本（并行，Day 1）

```bash
# 终端 1（GPU）：启动 Random Noise 消融
conda activate tiger_t5
cd /data/gtx/project/code/TIGER_minilm
./tiger_migration.sh run_ablation

# 终端 2（GPU）：启动 Beauty baseline seed=42
conda activate tiger_t5
cd /data/gtx/project/code/TIGER_minilm
./tiger_migration.sh run_baseline

# 终端 3（CPU）：在我创建好脚本后，立即开始 CPU 分析
```

GPU 跑实验期间（需要 8-10h），CPU 分析脚本全部可以写完并执行完毕。

### Phase 3：CPU 分析（与 Phase 2 完全并行，Day 1）

| 脚本 | 预计耗时 | 产出 |
|:----|:--------:|:-----|
| `tools/extract_metrics.py` | 10min | 指标汇总表 |
| `tools/longtail_analysis.py` | 1-2h | Head/Mid/Tail 分组对比 |
| `tools/sid_quality.py` | 1-2h | 四个质量指标 |
| `tools/tsne_visualize.py` | 30min | 3 张 .png 图 |

### Phase 4：冷启动实验（Day 2，GPU 空闲后）

```bash
# 在 Phase 2 的 GPU 任务跑完后执行
conda activate tiger_t5
python3 tools/run_coldstart.py --dataset Toys --ratios 0.05,0.10,0.20
```

---

## 六、需要我接下来创建的清单

按创建顺序：

1. **`tools/extract_metrics.py`** — 补充指标提取（最简单，先出成果）
2. **`tools/sid_quality.py`** — SID 质量分析（数据全在本地）
3. **`tools/longtail_analysis.py`** — 长尾分析（依赖 item 频率统计）
4. **`tools/tsne_visualize.py`** — t-SNE 可视化（需要 matplotlib）
5. **`tools/run_coldstart.py`** — 冷启动实验（最复杂，最后做）

---

## 七、执行清单（打勾用）

### GPU 任务
- [ ] `conda activate tiger_t5` + `pip install scikit-learn scipy`（环境准备）
- [ ] A-1：Random Noise 消融启动（`./tiger_migration.sh run_ablation`）
- [ ] A-2：Beauty baseline seed=42 启动（`./tiger_migration.sh run_baseline`）
- [ ] A-3：冷启动实验（Phase 4）

### CPU 分析
- [ ] `tools/extract_metrics.py` — 跑出指标汇总表
- [ ] `tools/sid_quality.py` — 跑出四指标
- [ ] `tools/longtail_analysis.py` — 跑出长尾分组表
- [ ] `tools/tsne_visualize.py` — 跑出 t-SNE 图

### 论文素材产出
- [ ] 主实验指标表（含 Rec@5/10, NDCG@5/10/20）
- [ ] 消融实验表（Baseline / TA-SID / Random Noise）
- [ ] 长尾分组表
- [ ] SID 质量四指标
- [ ] t-SNE 对比图
- [ ] 冷启动结果表
