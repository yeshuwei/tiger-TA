# TA-SID 实施方案 (Plan A: Concat)

## 项目结构

```
TIGER_minilm/
├── data/{DATASET}/          # 数据集目录 (Beauty/Sports/Toys)
│   ├── item_emb.parquet          # [已有] Content Embedding (768维)
│   ├── behavior_emb.parquet      # [新增] Behavior Embedding (128维)
│   ├── fused_item_emb.parquet    # [新增] Concat融合嵌入 (896维)
│   └── {DATASET}_t5_rqvae.npy    # [替换] TA-SID离散码
├── rqvae/
│   ├── main.py                   # [不修改] RQ-VAE训练入口
│   ├── models/                   # [不修改] RQ-VAE模型代码
│   ├── datasets.py               # [不修改] 嵌入数据集
│   ├── generate_code.py          # [改路径] 生成离散码
│   ├── build_transition_graph.py # [新增] Step 1: 构建Transition图 → Behavior Embedding
│   └── fuse_embeddings.py        # [新增] Step 2: 融合Content+Behavior
└── model/
    └── main.py                   # [不修改] T5生成模型
```

---

## 修改原则

- **不修改任何已有代码文件的逻辑** —— 全部通过新增文件和改路径/配置实现
- T5 生成模型（`model/main.py`）**完全不动**
- RQ-VAE 模型代码（`rqvae/models/`）**完全不动**

---

## Step 1: 构建 Item Transition Graph → Behavior Embedding

### 文件：`rqvae/build_transition_graph.py`

### 输入
- `data/{DATASET}/train.parquet` — 用户行为序列

### 输出
- `data/{DATASET}/behavior_emb.parquet` — Behavior Embedding（128维）

### 格式规范
- 与 `item_emb.parquet` 格式一致：包含 `ItemID` 和 `embedding` 两列
- `embedding` 为 128 维 float 向量

### 核心逻辑

```
1. 读取 train.parquet，获取每个用户的物品序列
2. 滑动窗口(item-level)统计共现
   - 窗口大小 = 3（左右各1，即当前物品与其前后相邻物品配对）
   - 对每对物品 (i, j)，共现矩阵 C[i][j] += 1
3. PPMI 变换
   - sum_all = ΣC
   - row_sum[i] = ΣC[i][:]
   - col_sum[j] = ΣC[:][j]
   - pmi[i][j] = log(C[i][j] * sum_all / (row_sum[i] * col_sum[j]))
   - ppmi[i][j] = max(pmi[i][j], 0)
4. Truncated SVD (k=128)
   - U, S, Vt = svd(PPMI, k=128)
   - behavior_emb = U @ diag(sqrt(S))     # shape: (N, 128)
5. 保存为 data/{DATASET}/behavior_emb.parquet
```

### 边界情况处理
- **冷启动物品**（不在 train 序列中）：behavior_emb 设为全 0 向量
- **低频物品**（共现次数 < 阈值）：保留 PPMI 值，不做额外处理
- **物品索引**：严格与 process_script.py 中的 itemID_mapping 对齐

### 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| svd_dim | 128 | Behavior Embedding 维度 |
| window_size | 3 | 共现窗口大小 |
| min_cooccur | 1 | 最小共现阈值 |

### 验证方法
- 输出 embedding 的形状检查：`(num_items, 128)`
- 打印 top-5 最相似物品对，通过人工判断是否合理（同类目物品应为高频共现）

---

## Step 2: 融合 Content + Behavior → Recommendation-aware Embedding

### 文件：`rqvae/fuse_embeddings.py`

### 输入
- `data/{DATASET}/item_emb.parquet` — Content Embedding（768维）
- `data/{DATASET}/behavior_emb.parquet` — Behavior Embedding（128维）

### 输出
- `data/{DATASET}/fused_item_emb.parquet` — 融合嵌入（896维）

### 核心逻辑

```
1. 加载 Content Embedding (N, 768) 和 Behavior Embedding (N, 128)
2. -------------- 关键！-----------------
   分别对两种 embedding 做 Layer Normalization
   （防止 Behavior 信号因尺度小而被 Content 淹没）
   - content_emb = LayerNorm(content_emb)
   - behavior_emb = LayerNorm(behavior_emb)
3. fused = torch.cat([content_emb, behavior_emb], dim=-1)  # (N, 896)
4. 保存为 data/{DATASET}/fused_item_emb.parquet
```

### 为什么需要 LayerNorm（重点）

| 信号 | 来源 | 典型范数 | 风险 |
|------|------|---------|------|
| Content | Sentence-T5 语义嵌入 | 较大 | 主导融合 |
| Behavior | PPMI+SVD | 较小 | 被淹没 |

**不做归一化 → RQ-VAE 主要量化 Content 信号 → TA-SID 退化为普通 Semantic ID**

### 输出格式
- 与 `item_emb.parquet` 完全一致的格式，后续 RQ-VAE 直接替换数据路径即可

---

## Step 3: 用融合嵌入重新训练 RQ-VAE

### 操作

```bash
cd /data/gtx/project/code/TIGER_minilm

# 直接用原 main.py，只改数据路径参数
python rqvae/main.py \
    --data_path ../data/{DATASET}/fused_item_emb.parquet \
    --ckpt_dir ./rqvae/ckpt/{DATASET}_TA_SID \
    [其他参数保持默认]
```

### 修改说明
| 涉及文件 | 操作 | 原因 |
|---------|------|------|
| `rqvae/main.py` | **不修改** | `EmbDataset` 自动读取 `item_emb.parquet` 格式，`in_dim` 自动检测为 896 |
| `rqvae/datasets.py` | **不修改** | `pd.read_parquet()` 读取任意维度的 `embedding` 列 |
| `rqvae/models/rqvae.py` | **不修改** | `in_dim` 自动适配；Encoder MLP `768→512→256→128→64` 的工作维度取决于实际输入，896自动进入第一层线性层 |

### 自动生效的原因
`EmbDataset` 读取 embedding 维度是自动推导的：
```python
# datasets.py 第14行
self.dim = self.embeddings.shape[-1]  # 768 → 896，自动变化
```

### 训练监控

| 指标 | 预期变化 | 说明 |
|------|---------|------|
| Reconstruction Loss | 可能略高于基线 | 896维重建比768维更难 |
| Collision Rate | 预期降低 | 融入 Transition 信号后区分度更高 |
| Training 收敛速度 | 相近 | 参数规模未变 |

---

## Step 4: 生成 TA-SID 离散码

### 修改文件：`rqvae/generate_code.py`

### 需要修改的行

| 行号（当前） | 当前值 | 改为 |
|------------|--------|------|
| ~46 | `DATASET = "Toys"` | 设为对应数据集 |
| ~48 | `ckpt_path = "./ckpt/{DATASET}/.../best_collision_model.pth"` | `"./ckpt/{DATASET}_TA_SID/.../best_collision_model.pth"` |
| ~49 | `output_file = f"../data/{DATASET}/{DATASET}_t5_rqvae.npy"` | `f"../data/{DATASET}/{DATASET}_ta_sid.npy"` |
| ~57 | `data = EmbDataset(args.data_path)` | args.data_path 需指向 `fused_item_emb.parquet` |

### 输出
- `data/{DATASET}/{DATASET}_ta_sid.npy` — TA-SID 离散码
- 格式与原 `*_t5_rqvae.npy` 完全一致：`(num_items, 4)` 的 int array
- 第4维仍是碰撞解消用的 padding 位

---

## Step 5: 训练 T5 评估效果

### 操作

```bash
python model/main.py \
    --dataset_path ../data/{DATASET} \
    --code_path ../data/{DATASET}/{DATASET}_ta_sid.npy \
    --save_path ./model/ckpt/tiger_ta_sid.pth \
    --log_path ./model/logs/tiger_ta_sid.log \
    [其他参数保持默认]
```

### 修改说明

| 涉及文件 | 操作 | 原因 |
|---------|------|------|
| `model/main.py` | **不修改** | 通过命令行参数 `--code_path` 指定新 code 文件即可 |
| `model/dataset.py` | 确认 `vocab_size=1025` 是否兼容 | 若 TA-SID 的 code 值范围不变（4×256=1024），则无需修改 |

### 对照实验设置

| 实验 | code_path | ckpt 保存路径 | 说明 |
|------|-----------|--------------|------|
| **Baseline** | `{DATASET}_t5_rqvae.npy` | `./ckpt/tiger.pth` | 原 TIGER 复现 |
| **TA-SID** | `{DATASET}_ta_sid.npy` | `./ckpt/tiger_ta_sid.pth` | 本方案 |

### 预期对比

| 指标 | 基线 (TIGER) | TA-SID (预期) |
|------|-------------|--------------|
| Recall@10 | Beauty: 0.0594 | 稳定提升 |
| NDCG@10 | Beauty: 0.0321 | 稳定提升 |
| Collision Rate | 0.01~0.09 | 预期改善 |
| Invalid ID Rate | 基准值 | 预期不变或改善 |

---

## 完整执行流程

### 一条命令跑通（以 Beauty 为例）

```bash
PROJECT_DIR="/data/gtx/project/code/TIGER_minilm"
DATASET="Beauty"

cd $PROJECT_DIR

# ========== Step 1: Behavior Embedding ==========
echo "[Step 1] Building Item Transition Graph..."
python rqvae/build_transition_graph.py \
    --dataset $DATASET \
    --svd_dim 128 \
    --window_size 3

# ========== Step 2: Fusion ==========
echo "[Step 2] Fusing Content + Behavior Embeddings..."
python rqvae/fuse_embeddings.py \
    --dataset $DATASET

# ========== Step 3: Train RQ-VAE ==========
echo "[Step 3] Training RQ-VAE on fused embeddings..."
python rqvae/main.py \
    --data_path ../data/${DATASET}/fused_item_emb.parquet \
    --ckpt_dir ./rqvae/ckpt/${DATASET}_TA_SID

# ========== Step 4: Generate TA-SID Codes ==========
echo "[Step 4] Generating TA-SID discrete codes..."
# 先修改 generate_code.py 中的路径，再执行
python rqvae/generate_code.py

# ========== Step 5: Train T5 ==========
echo "[Step 5] Training T5 with TA-SID..."
python model/main.py \
    --dataset_path ../data/${DATASET} \
    --code_path ../data/${DATASET}/${DATASET}_ta_sid.npy \
    --save_path ./model/ckpt/tiger_ta_sid.pth \
    --log_path ./model/logs/tiger_ta_sid.log
```

---

## 新增文件清单（待实现）

| # | 文件 | 预估行数 | 复杂度 | 依赖 |
|---|------|---------|--------|------|
| 1 | `rqvae/build_transition_graph.py` | ~120行 | ⭐⭐ | sklearn, numpy, pandas |
| 2 | `rqvae/fuse_embeddings.py` | ~50行 | ⭐ | torch, pandas |

## 修改文件清单

| # | 文件 | 修改内容 | 复杂度 |
|---|------|---------|--------|
| 1 | `rqvae/generate_code.py` | 修改 ckpt_path, output_file, data_path | ⭐ |

## 不动文件（确认不修改）

| 文件 | 原因 |
|------|------|
| `rqvae/main.py` | `in_dim` 自动适配 fused embedding 维度 |
| `rqvae/models/*.py` | 模型结构完全不变 |
| `rqvae/datasets.py` | `EmbDataset.__getitem__` 返回 tensor，与维度无关 |
| `rqvae/trainer.py` | 训练逻辑完全不变 |
| `model/main.py` | 通过 `--code_path` 参数指定新 code 文件 |
| `model/dataset.py` | vocab_size=1025 与 TA-SID code 范围兼容 |
| `model/dataloader.py` | 数据加载逻辑不变 |
| `data/process_script.py` | 数据预处理完全不变 |

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Behavior Embedding 被 Content 淹没 | 中 | 高 | LayerNorm 归一化 |
| 冷启动物品无行为信号 | 高 | 中 | behavior_emb 全0，退化为纯内容 |
| 896维重建损失升高 | 中 | 低 | 监控 collision rate 而非重建损失 |
| PPMI+SVD 在稀疏数据上不稳定 | 低 | 中 | 增大窗口或 min_cooccur 阈值 |
| 训练时间增加 | 低 | 低 | RQ-VAE 参数量未显著增加 |

---

## 实验扩展路线（后续版本）

若方案 A 有效，后续可探索：

1. **方案 B**：Linear(896, 768) 可学习投影，保持输出 768 维
2. **方案 C**：MLP 融合，更复杂的非线性融合
3. **Codebook Balance Loss**：引入 entropy loss 提高 codebook 利用率
4. **Transition Loss**：在 RQ-VAE 训练中加入序列预测辅助损失
5. **不同 Behavior Emb 方法**：Node2Vec / LINE / DeepWalk 对比 PPMI+SVD

---

> 创建日期：2026-07-15
> 基线论文：Recommender Systems with Generative Retrieval (TIGER)
> 改进方案：TA-SID — Transition-aware Semantic ID
