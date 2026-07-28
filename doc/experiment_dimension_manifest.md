# TA-SID 实验维度清单 (Experiment Dimension Manifest)

> 维护目的：记录所有实验的内容嵌入维度，确保论文数据可追溯
> 更新日期：2026-07-28

## 命名规则

所有文件统一使用 `_{dim}d` 后缀标记维度：
- `_768d` = Sentence-T5-base（768 维）
- `_4096d` = Sentence-T5-xl（4096 维）

无后缀的文件为**历史遗留**，需对照此表确认维度。

---

## 一、主实验结果总表

### 1.1 768d Content (Sentence-T5-base) — 论文主实验

| 数据集 | Baseline N@20 | TA-SID N@20 | 提升 | RQ-VAE 碰撞率 | 状态 |
|:------|:-------------:|:-----------:|:----:|:-------------:|:----:|
| Beauty | 0.0379 | **0.0418** | **+10.3%** | 1.09% | ✅ 已完成 |
| Sport | 0.0207 | **?**（运行中） | **?** | 0.65% | 🔄 T5 epoch 9/200 |
| Toys | 0.0305 | **?**（待跑） | **?** | ? | ⏳ 等待 GPU |

### 1.2 4096d Content (Sentence-T5-xl) — Robustness 对照

| 数据集 | TA-SID N@20 | 提升 vs Baseline | RQ-VAE 碰撞率 | 状态 |
|:------|:-----------:|:----------------:|:-------------:|:----:|
| Beauty | **0.0382** | +0.8% | 8.21% | ✅ 已完成 |
| Sport | **0.0276** | +33.3% | ? | ✅ 已完成 |
| Toys | **0.0317** | +3.9% | ? | ✅ 已完成 |

---

## 二、Content Embedding 文件索引

### 2.1 原始内容嵌入 (raw data)

| 数据集 | 768d (Sentence-T5-base) | 4096d (Sentence-T5-xl) |
|:------|:----------------------|:----------------------|
| Beauty | ❌ 未生成 | `{raw_dir}/Beauty/content_embeddings.pkl` |
| Sport | `{raw_dir}/Sport/content_embeddings_768d.pkl` ✅ | `{raw_dir}/Sport/content_embeddings.pkl` |
| Toys | `{raw_dir}/Toys/content_embeddings_768d.pkl` ✅ | `{raw_dir}/Toys/content_embeddings.pkl` |

> `{raw_dir}` = `/data/gtx/project/datasets/tiger_beauty`

### 2.2 格式化 item_emb (data 目录下)

| 数据集 | 768d | 4096d |
|:------|:-----|:------|
| Beauty | ❌（原始 768d 被覆盖，但 fused 768d 仍存） | `data/Beauty/item_emb_4096d.parquet` (cp from item_emb.parquet) |
| Sport | `data/Sport/item_emb_768d.parquet` ✅ | `data/Sport/item_emb_4096d.parquet` (cp from item_emb.parquet) |
| Toys | `data/Toys/item_emb_768d.parquet` ✅ | `data/Toys/item_emb_4096d.parquet` (cp from item_emb.parquet) |

### 2.3 融合嵌入 fused_item_emb

| 数据集 | 768d track (768+128=896d) | 4096d track (4096+128=4224d) |
|:------|:------------------------:|:----------------------------:|
| Beauty | `data/Beauty/fused_item_emb.parquet` ✅ (896d, 原始遗留) | `data/Beauty/fused_item_emb_4096d.parquet` |
| Sport | `data/Sport/fused_item_emb_768d.parquet` ✅ | `data/Sport/fused_item_emb_4096d.parquet` (cp from fused_item_emb.parquet) |
| Toys | `data/Toys/fused_item_emb_768d.parquet` ✅ | `data/Toys/fused_item_emb_4096d.parquet` (cp from fused_item_emb.parquet) |

---

## 三、RQ-VAE 模型文件

| 实验 | 输入维度 | 碰撞率 | Checkpoint 路径 |
|:----|:-------:|:------:|:---------------|
| Beauty TA-SID 768d | 896 | 1.09% | `rqvae/ckpt/Beauty_TA_SID/` |
| Beauty TA-SID 4096d (验证) | 4224 | 8.13% | `rqvae/ckpt/Beauty_4096d/` |
| Beauty TA-SID 4096d (全量) | 4224 | 8.21% | `rqvae/ckpt/Beauty_4096d_full/` |
| Sport TA-SID 768d | 896 | 0.65% ✅ | `rqvae/ckpt/Sport_TA_SID_768d/` |
| Sport TA-SID 4096d | 4224 | ? | `rqvae/ckpt/Sport_TA_SID_4096d/` (cp from Sport_TA_SID/) |
| Toys TA-SID 768d | 896 | ? | `rqvae/ckpt/Toys_TA_SID_768d/` (待跑) |
| Toys TA-SID 4096d | 4224 | ? | `rqvae/ckpt/Toys_TA_SID_4096d/` (cp from Toys_TA_SID/) |

---

## 四、SID Codes 文件

| 实验 | 文件 | 维度来源 |
|:----|:-----|:--------:|
| Beauty TA-SID 768d | `data/Beauty/Beauty_ta_sid.npy` | 768d (遗留) |
| Beauty TA-SID 4096d (验证) | `data/Beauty/Beauty_ta_sid_4096d.npy` | 4096d |
| Beauty TA-SID 4096d (全量) | `data/Beauty/Beauty_ta_sid_4096d_full.npy` | 4096d |
| Sport TA-SID 768d | `data/Sport/Sport_sid_codes_768d.npy` ✅ | 768d |
| Sport TA-SID 4096d | `data/Sport/Sport_sid_codes_4096d.npy` (cp from Sport_ta_sid.npy) | 4096d |
| Toys TA-SID 768d | `data/Toys/Toys_sid_codes_768d.npy` (待跑) | 768d |
| Toys TA-SID 4096d | `data/Toys/Toys_sid_codes_4096d.npy` (cp from Toys_ta_sid.npy) | 4096d |

---

## 五、T5 模型与日志

| 实验 | NDCG@20 | 日志 | Checkpoint |
|:----|:-------:|:-----|:-----------|
| Beauty TA-SID 768d (seed=2025) | **0.0418** | `model/logs/tiger_ta_sid.log` | `model/ckpt/tiger_ta_sid_768d.pth` |
| Beauty TA-SID 768d (seed=42) | **0.0404** | `model/logs/tiger_ta_sid_beauty_seed42.log` | `model/ckpt/tiger_ta_sid_beauty_seed42.pth` |
| Beauty TA-SID 4096d (全量) | **0.0382** | `model/logs/beauty_ta_sid_4096d_full.log` | — |
| Sport TA-SID 768d (seed=2025) | **?（运行中）** | `model/logs/tiger_ta_sid_sport_768d.log` | `model/ckpt/tiger_ta_sid_sport_768d.pth` |
| Sport TA-SID 4096d (seed=2025) | **0.0276** | `model/logs/tiger_ta_sid_sport_4096d.log` (cp) | `model/ckpt/tiger_ta_sid_sport_4096d.pth` (cp) |
| Toys TA-SID 768d (seed=2025) | **?（待跑）** | — | — |
| Toys TA-SID 4096d (seed=2025) | **0.0317** | `model/logs/tiger_ta_sid_toys_4096d.log` (cp) | `model/ckpt/tiger_ta_sid_toys_4096d.pth` (cp) |

---

## 六、维度对实验结果的影响分析

### 6.1 RQ-VAE 碰撞率对比

| 数据集 | 768d 碰撞率 | 4096d 碰撞率 |
|:------|:-----------:|:------------:|
| Beauty | 1.09% | 8.21% |
| Sport | **0.65%** ✅ | ? |
| Toys | ?（待跑） | ? |

> **结论：** 768d 内容的碰撞率显著低于 4096d（0.65-1.09% vs 8%+），说明 768d 融合嵌入（896 维）更适合当前 RQ-VAE 架构。

### 6.2 维度对 NDCG 的影响

| 数据集 | 768d TA-SID vs Baseline | 4096d TA-SID vs Baseline |
|:------|:----------------------:|:------------------------:|
| Beauty | +10.3% ✅ | +0.8% ❌ |
| Sport | ?（运行中） | +33.3% |
| Toys | ?（待跑） | +3.9% |

### 6.3 论文用途

| 部分 | 内容 | 使用的维度 |
|:----|:-----|:----------|
| **Main Results** | 三数据集统一 768d 对比 | 768d |
| **Ablation: Random Noise** | 行为信号有效性验证 | 768d (推荐) |
| **Robustness Analysis** | 不同内容维度下的方法效果 | 768d vs 4096d 对比 |
| **Discussion** | RQ-VAE 架构对高维输入的限制 | 4096d 碰撞率分析 |

---

## 七、原始数据信息

| 数据集 | 商品数 | 用户数 | 评论数 | 原始 JSON |
|:------|:-----:|:-----:|:-----:|:----------|
| Beauty | 12,101 | 22,363 | 198,502 | `Beauty_5.json` |
| Sport | 18,357 | 35,598 | 296,337 | `Sports_and_Outdoors_5.json` |
| Toys | 11,924 | 19,412 | 167,597 | `Toys_and_Games_5.json` |

> 原始数据路径: `/data/gtx/project/datasets/tiger_beauty/{Dataset}/`

---

## 附录：文件迁移安全策略

所有备份通过 `cp -n`（不覆盖已有文件）执行。
备份文件命名规则：`{原文件名}_{dim}d.{ext}`，如 `item_emb_4096d.parquet`。
原文件保留不动，确保旧脚本兼容。
