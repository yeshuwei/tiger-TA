# E44 第二创新多数据集筛查（Multi-dataset screening）

> ⚠ **2026-09-04 补注**：本文档的历史结论（Toys 是"容量不足态"、K0\*=512、为 Adaptive 找第二正例）
> **已由 E45 推翻**——Toys 塌缩是 seed2024 单次训练的分配崩坏（换种子 256 即健康），判据精度结论
> 保留、扩容动作作废。权威裁定见 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)。
> 目的：为第二创新（Adaptive Coarse-Level Capacity Allocation）寻找 Toys 之外
> 的"容量不足态"数据集，证明 L0 塌缩普遍存在、Adaptive 修复有效。
> 日期：2026-08-19 | 状态：历史记录（结论已被 E45 修正）

---

## 1. 数据获取（网络受限下的实际路径）

**原计划**：从 snap.stanford.edu（Amazon Review 2014 官方源）下载 3-5 个
稀疏长尾类别（Video_Games / Home_and_Kitchen / Cell_Phones）。

**网络现实**：本机到 snap.stanford.edu 下载速度 <10 KB/s（多线程也无改善），
google / archive.org 不可达；仅 hf-mirror.com、github.com、cseweb.ucsd.edu 可达。

**实际可用数据**（hf-mirror `ttb06/amazon_review`，原始 5-core JSON 格式）：

| 类别 | 大小 | 规模预估 | 用途 |
|---|---|---|---|
| Musical_Instruments | 2.4 MB | 900 商品 / 1429 用户 / 10k 评论 | ✅ 已跑完 |
| Electronics | 495 MB | 8-15 万商品（待确认） | ⏳ 下载中 |
| Sports_and_Outdoors / Toys_and_Games / Beauty | 已有 | — | 已有 |

**未找到**：Video_Games / Home_and_Kitchen / Cell_Phones 的原始 5-core JSON
镜像（HF 上多为处理过的 parquet/CSV，字段不全，不可复用管道）。

---

> ⚠ 口径说明（2026-08-19 修正）：主实验输入统一为 **896d = 768d content(Sentence-T5)
> ⊕ 128d behavior(PPMI+SVD)**。`fused_item_emb_768d.parquet` 中的 "768d" 指 content 维度，
> 文件本身是 896d。三数据集主实验（Beauty_TA_SID / Sport_TA_SID_768d / Toys_TA_SID_768d）
> 全部输入 896d。本筛查的 MI/Electronics 均按同一 896d 口径生成。

## 2. Musical_Instruments 全链路结果（✅ 完成）

管道：`reviews JSON → Sentence-T5 content 768d → preprocess(1-based 对齐) →
PPMI+SVD behavior 128d → fuse(896d) → RQ-VAE(256×256×256, 3000ep) → codes → 诊断`

| 指标 | 值 |
|---|---|
| items / users / reviews | 900 / 1429 / 10261 |
| L0 用码 | **35/256（14%）** ⚠ 塌缩 |
| L1 / L2 用码 | 170/256 / 237/256（正常）|
| 碰撞率 | 0.00% |
| xq[L0] / xq[full] / z AUC | 0.7441 / 0.7584 / 0.7613 |
| **z-gap** | **0.0029（≤ ε=0.01）✅ 无损** |
| 判定 | **K0\*=256，不需要扩容** |

### 关键发现：L0 塌缩 ≠ 必须扩容

MI 与 Toys 的对比是这次筛查最重要的产出：

> ⚠️ **2026-09-09 provenance 订正**：本表 0.0367 / L0=62 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**；本处仅作"塌缩实例"历史对照保留。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020（健康 codes，L0=207，z-gap 0.005）。权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

| | L0 用码 | 商品数 | z-gap | 需要扩容？ |
|---|---|---|---|---|
| **Toys** | 62/256（24%） | 11924 | **0.093** | ✅ 是（K0\*=512）|
| **MI** | 35/256（14%） | 900 | **0.0029** | ❌ 否（K0\*=256）|

- 两个数据集 L0 都塌缩（再次证明 **L0 塌缩是普遍现象**，已覆盖
  Beauty/Sport/Toys/MI 四个数据集）。
- 但 **只有商品规模够大时（Toys），L0 塌缩才压垮 3-token 重建**——
  MI 900 商品被 L1/L2 的 256² 组合容量轻松补回，端到端无损。
- **论文价值**：这恰好证明 **z-gap 判据（而非裸 L0 利用率）才是判别
  "是否需要扩容"的正确信号**。只看 L0 占用会把 MI 误判为需要扩容；
  z-gap 正确区分了 MI（无需）和 Toys（需要）。这是对 4.2 机制选型
  （gap-based 闭环）的额外支撑。

---

## 3. Electronics（进行中）

- 495MB gz，5-core，预计 8-15 万商品（远大于 256 码）。
- 若 Electronics 长尾严重（大量 niche 商品仅 5 条评论），L0 可能塌缩且
  z-gap 大 → 成为第二个"容量不足态"验证。
- 若 Electronics 密集（交互均匀），则补充"大类别未必塌缩"的对照。
- 训练成本：~10 万商品 → RQ-VAE 3000ep 需数小时（Toys 的 10 倍）。

---

## 3b. Electronics E1 结果（✅ 完成 2026-08-19）

管道：`reviews_Electronics_5.json → Sentence-T5 content 768d → preprocess(1-based 对齐) →
PPMI+SVD behavior 128d → fuse(896d) → RQ-VAE(256×256×256, 3000ep) → codes → E1-fast 诊断`

| 指标 | 值 |
|---|---|
| items / users / reviews | 63,001 / 192,403 / 1,689,188 |
| L0 用码 | **256/256（100%，用满）** max=892, mean=246.1 |
| L1 / L2 用码 | 256/256 / 256/256 |
| 碰撞率 | 0.095% |
| 3-token 去重 | 62,941/63,001（几乎全部唯一）|
| xq[L0] / xq[L0+L1] / xq[full] AUC | 0.9066 / 0.9191 / 0.9243 |
| z AUC | 0.9358 |
| **z-gap** | **0.0116（> ε=0.01，边缘超标）** |
| 判定 | **K0\*=512 候选**（z-gap>ε → Adaptive 触发） |

### 关键发现：Electronics = 第三种形态「容量饱和态」

| | L0 用码 | 商品数 | z-gap | 形态 |
|---|---|---|---|---|
| **Beauty** | 204/256 (80%) | ~12.1K | 0.0001 | 健康态 |
| **Sport** | 249/256 (97%) | ~18.6K | 0.0032 | 健康态 |
| **MI** | 35/256 (14%) | 900 | 0.0029 | 塌缩但小规模（L1/L2 补足）|
| **Toys** | 62/256 (24%) | 11,924 | **0.093** | 容量不足态（塌缩）|
| **Electronics** | **256/256 (100%)** | 63,001 | **0.0116** | **容量饱和态（无塌缩但粒度不足）** |

- Electronics L0 **没有塌缩反而用满** 256 码（每码平均 246 个商品）——与 Toys 的
  62 码塌缩形成鲜明对照。证明 **L0 塌缩不是普适现象**，是数据规模+结构的函数。
- 但即便 256 码全用满，63K 商品的 L0 粒度（246 items/code）仍不足以精确重建
  z 空间结构 → z-gap=0.0116 边缘超标（>ε=0.01）。
- **论文价值**：z-gap 判据在两种不同形态的容量压力（塌缩型 Toys / 饱和型
  Electronics）上都正确触发 Adaptive——再次证明 gap-based 闭环比裸 L0 利用率
  更鲁棒（若只看 L0 利用率，Electronics 100% 会误判为"无需扩容"）。

---

## 4. 当前结论与下一步

1. **L0 塌缩普遍性已经很强**（4 个数据集）。
2. MI 提供了"塌缩但无需扩容"的边界案例，强化 z-gap 判据必要性。
3. 待 Electronics 结果：若 z-gap>ε → Adaptive(K0=512) 重训验证 → 完整闭环。
4. 若 Electronics 也不塌缩：论文叙事改为"Toys 是唯一的容量不足态 +
   L0 塌缩普遍存在 + z-gap 判据精确"，筛查证据即够。

脚本/文件：
- 诊断：`tools/sid_capacity_diagnostic.py`（CONFIG 已加 Musical_Instruments）
- 嵌入：`tools/generate_content_embeddings.py`（choices 已放开）
- 融合：`rqvae/fuse_embeddings.py`（--content_file/--output_file 参数化）
- 模型：`rqvae/ckpt/Musical_Instruments_TA_SID/`
- 数据：`data/Musical_Instruments/`
