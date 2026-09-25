# TA-SID 实验维度清单 (Experiment Dimension Manifest)
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

> 维护目的：记录所有实验的内容嵌入维度，确保论文数据可追溯
> 更新日期：2026-07-29

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
| Sport | 0.0207 | **0.0327** | **+58.0%** 🎯 | 10.66% | ✅ 已完成 |
| Toys | 0.0305 | **0.0360** | **+18.0%** | 8.68% | ✅ 已完成 |

### 1.2 4096d Content (Sentence-T5-xl) — Robustness 对照

| 数据集 | TA-SID N@20 | 提升 vs Baseline | RQ-VAE 碰撞率 | 状态 |
|:------|:-----------:|:----------------:|:-------------:|:----:|
| Beauty | **0.0382** | +0.8% | 8.21% | ✅ 已完成 |
| Sport | **0.0276** | +33.3% | ⚠️ 早期实验未记录 | ✅ 已完成 |
| Toys | **0.0317** | +3.9% | ⚠️ 早期实验未记录 | ✅ 已完成 |

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
| Sport TA-SID 768d | 896 | 10.66% ✅ | `rqvae/ckpt/Sport_TA_SID_768d/` |
| Sport TA-SID 4096d | 4224 | — | `rqvae/ckpt/Sport_TA_SID_4096d/` (cp from Sport_TA_SID/) |
| Toys TA-SID 768d | 896 | 8.68% ✅ | `rqvae/ckpt/Toys_TA_SID_768d/` |
| Toys TA-SID 4096d | 4224 | — | `rqvae/ckpt/Toys_TA_SID_4096d/` (cp from Toys_TA_SID/) |
| Toys Random Noise | 896 | 8.71% | `rqvae/ckpt/Toys_Random/` |

---

## 四、SID Codes 文件

| 实验 | 文件 | 维度来源 |
|:----|:-----|:--------:|
| Beauty TA-SID 768d | `data/Beauty/Beauty_ta_sid_768d_sid.npy` ✅ | 768d |
| Beauty TA-SID 4096d (验证) | `data/Beauty/Beauty_ta_sid_4096d.npy` | 4096d |
| Beauty TA-SID 4096d (全量) | `data/Beauty/Beauty_ta_sid_4096d_full.npy` | 4096d |
| Sport TA-SID 768d | `data/Sport/Sport_ta_sid_768d_sid.npy` ✅ | 768d |
| Sport TA-SID 4096d | `data/Sport/Sport_sid_codes_4096d.npy` | 4096d |
| Toys TA-SID 768d | `data/Toys/Toys_ta_sid_768d_sid.npy` ✅ | 768d |
| Toys TA-SID 4096d | `data/Toys/Toys_sid_codes_4096d.npy` | 4096d |
| Toys Random Noise | `data/Toys/Toys_random_sid.npy` ✅ | 768d (随机噪声) |

---

## 五、T5 模型与日志

| 实验 | NDCG@20 | 日志 | Checkpoint |
|:----|:-------:|:-----|:-----------|
| Beauty TA-SID 768d (seed=2025) | **0.0418** | `model/logs/tiger_ta_sid.log` | `model/ckpt/tiger_ta_sid_768d_beauty_seed2025.pth` ✅ |
| Beauty TA-SID 768d (seed=42) | **0.0404** | `model/logs/tiger_ta_sid_beauty_seed42.log` | `model/ckpt/tiger_ta_sid_beauty_seed42.pth` |
| Beauty TA-SID 4096d (全量) | **0.0382** | `model/logs/beauty_ta_sid_4096d_full.log` | — |
| Sport TA-SID 768d (seed=2025) | **0.0327** 🎯 | `model/logs/tiger_ta_sid_sport_768d.log` | `model/ckpt/tiger_ta_sid_768d_sport_seed2025.pth` ✅ |
| Sport TA-SID 4096d (seed=2025) | **0.0276** | `model/logs/tiger_ta_sid_sport_4096d.log` | `model/ckpt/tiger_ta_sid_sport_4096d.pth` |
| Toys TA-SID 768d (seed=2025) | **0.0360** | `model/logs/tiger_ta_sid_Toys_768d.log` | `model/ckpt/tiger_ta_sid_768d_toys_seed2025.pth` ✅ |
| Toys TA-SID 4096d (seed=2025) | **0.0317** | `model/logs/tiger_ta_sid_toys_4096d.log` | `model/ckpt/tiger_ta_sid_toys_4096d.pth` |

---

## 六、维度对实验结果的影响分析

### 6.1 RQ-VAE 碰撞率对比

| 数据集 | 768d 碰撞率 | 4096d 碰撞率 |
|:------|:-----------:|:------------:|
| Beauty | 1.09% | 8.21% |
| Sport | 10.66% ✅ | ⚠️ 早期实验未记录 |
| Toys | 8.68% ✅ | ⚠️ 早期实验未记录 |

> **768d 碰撞率说明：** Beauty 最低（1.09%），Sport 最高（10.66%）。碰撞率高不代表效果差——Sport 碰撞率最高但 NDCG 提升也最大（+58%），说明碰撞率不是唯一决定因素。
>
> **4096d 碰撞率缺失原因：** 4096d 实验在 768d 分析之前运行（早期阶段），使用的是项目初始版本的 RQ-VAE 训练，其训练日志未被纳入 `rqvae/logs/` 系统。当时未系统记录碰撞率。**影响不大：** 4096d 在论文中只做 Robustness 对照，768d 才是主实验结果，主实验的碰撞率数据完整可追溯。

### 6.2 维度对 NDCG 的影响

| 数据集 | 768d TA-SID vs Baseline | 4096d TA-SID vs Baseline |
|:------|:----------------------:|:------------------------:|
| Beauty | +10.3% ✅ | +0.8% |
| Sport | **+58.0%** 🎯 | +33.3% |
| Toys | **+18.0%** | +3.9% |

> **结论：** 768d 在三个数据集上均显著优于 4096d，说明 768d 嵌入与 RQ-VAE 架构更匹配。Sport 提升最大（+58%），且 768d 版本比 4096d 版本额外提升 24.7 个百分点。

### 6.3 论文用途

| 部分 | 内容 | 使用的维度 |
|:----|:-----|:----------|
| **Main Results** | 三数据集统一 768d 对比 | 768d |
| **Ablation: Random Noise** | 行为信号有效性验证 | 768d (推荐) |
| **Robustness Analysis** | 不同内容维度下的方法效果 | 768d vs 4096d 对比 |
| **Discussion** | RQ-VAE 架构对高维输入的限制 | 4096d 碰撞率分析 |

---

## 七、Track B 过渡对文件

| 数据集 | 文件 | 过渡对数 | Unique Items | 状态 |
|:------|:-----|:--------:|:-----------:|:----:|
| Beauty | `data/Beauty/transition_pairs.npy` | 131,413 | — | ✅ 已就绪 |
| Sport | `data/Sport/transition_pairs_768d.npy` | 189,543 | 18,257 | ✅ 已就绪 |
| Toys | `data/Toys/transition_pairs_768d.npy` | 109,361 | 11,900 | ✅ 已就绪 |

## 八、Track B 实验结果

| 实验 | 配置 | RQ-VAE 碰撞率 | Test NDCG@20 | 对比 Track A | 状态 |
|:----|:----|:-------------:|:-----------:|:-----------:|:----:|
| B-1: Sport 768d | λ_trans=0.1, τ=0.1, λ_balance=0 | 0.34% (vs 10.66%) | **0.03046** | -6.8% ❌ | ✅ 完成 |
| B-2: Sport 768d | λ_trans=0.01, τ=0.1, λ_balance=0 | 0.43% (vs 10.66%) | — | — | 🔄 T5 训练中 |

> **B-1 结论：** λ=0.1 的 L_trans 虽使碰撞率从 10.66% 降至 0.34%，但 T5 NDCG@20 反降 6.8%，说明对比损失过强，可能扭曲了量化空间。
> **B-2 方向：** λ 调小 10 倍至 0.01，减弱对比损失干扰，观察是否在保持低碰撞率的同时提升 NDCG。

---

## 九、原始数据信息

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
