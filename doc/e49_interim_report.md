# E49 行为消融：已执行部分总结（中途快照）

> ⚠️ **这是中途快照，不是最终报告。** 截止 **2026-09-22 22:50**，P3 已完成、P4 正在跑（Toys Raw-CoOcc
> seed2025 训练中）。**36 路目标中已完成 20 路、剩余 16 路**。所有"待跑"行在本文档中一律留空或标注，
> **不作预判**。最终数字以 P6 自动生成的 `doc/e49_behavior_ablation_results.md` 为准。
>
> 工作文档（实时状态）：[e49_behavior_ablation.md](e49_behavior_ablation.md)　计划全文：`~/.claude/plans/tidy-painting-church.md`
> 配套：[e49_behavior_space_gate.md](e49_behavior_space_gate.md)（诊断原始表）·
> [e49_behavior_space_analysis.md](e49_behavior_space_analysis.md)（解读）·
> [e49_p2_gate_verdict.md](e49_p2_gate_verdict.md)（决策门判定）

---

## 0. 一句话

**消融链已能证明「多加一个向量 / 任意统计噪声都不涨」，但回答不了两个更尖锐的问题——"PPMI 是否必要"
和"TA-SID 是不是只是普通 ItemCF"。E49 新增 `Raw-CoOcc-SVD` 与 `ItemCF-SVD` 两个变体回答它们，
并把已有对照行补齐到统一 3-seed 口径。目前 Sport 侧 3-seed 已全部出齐，结论是：
两个新变体都显著高于 content-only 与 ZeroPad/Shuffled，但仍都低于 TA-SID，
且**排序与行为空间诊断的前置预测完全一致**。**

---

## 1. 本轮要回答的两个问题（审稿人视角）

| # | 问题 | 对应变体 |
|:-:|:--|:--|
| 1 | **PPMI 这一步是不是必要的？** 会不会"任何共现矩阵做 SVD"都行？ | Raw-CoOcc-SVD（严格单变量：只去掉 PPMI） |
| 2 | **TA-SID 是不是只是普通的 item-level 协同过滤（CF）表示？** | ItemCF-SVD（局部序列上下文 vs 全局协同信号） |

> **术语约定（本轮起统一）**：不使用 "directional transition" / "转移概率"。Transition 一律定义为
> **"sequential behavioral association patterns extracted from user interaction sequences"**——
> 强调 TA-SID 建模的是**序列行为关联结构**，而**不是**显式的 P(j|i) 转移概率。

---

## 2. 实验设计：7 个变体

所有变体共用同一条下游链路：同一 `train.parquet` 序列、同一 canonical ItemID 顺序、同一 `svd_dim=128`、
同一 `U·√S` 变换、同一冷启动物品置零、同一 `LN(content 768d) ⊕ LN(behavior 128d) → 896d` 融合、
同一 RQ-VAE 配置、同一 T5 配置。**唯一被改动的是"行为向量从哪来"。**

| # | 变体 | 共现作用域 | 权重变换 | vs TA-SID 的改动 | 定位 |
|:-:|:--|:--|:--|:--|:--|
| 1 | Content-only | — | — | 无 behavior | 主 baseline |
| 2 | ZeroPad | — | 128 个零 | 仅维度 | 是否只是"加维度"的功劳 |
| 3 | Random | — | i.i.d. 高斯 | 任意向量 | 辅助消融（**单 seed**，不参与显著性比较） |
| 4 | Shuffled | 窗口=3 相邻 | PPMI（**行级打乱**） | item↔向量对应 | 是否只需统计特性 |
| 5 | **Raw-CoOcc-SVD** | **窗口=3 相邻（与 TA-SID 逐位相同）** | **无（直接 SVD 原始计数 C）** | **仅去掉 PPMI** | **① PPMI 是否必要（严格单变量）** |
| 6 | **ItemCF-SVD** | **整篮（同序列内所有商品对）** | **cosine 归一** | **作用域 + 归一化（两项）** | **② 局部序列上下文 vs 全局协同信号** |
| 7 | TA-SID (PPMI-SVD) | 窗口=3 相邻 | PPMI | 主方法 | — |

**两条设计声明（写进论文，避免过度声称）**

- **Raw-CoOcc 是严格单变量对照**：调用与 TA-SID **完全相同**的 `build_cooccurrence_matrix`
  （window=3, min_cooccur=1），得到**同一个 C**（已用 nnz + 逐元素校验和断言，见 §3.2），
  仅跳过 `compute_ppmi`。唯一变量 = PPMI 的有无。
- **ItemCF 不是严格单变量实验**：它同时改变了 ①共现作用域（相邻窗口 → 整篮全对）
  ②归一化（PPMI → cosine）③关联方向性（有序近邻 → 无序共购）。因此**本变体不用于论证
  "TA-SID 优于 CF"或"排除 CF 解释"**，其定位是回答一个方法学问题：**局部序列上下文共现结构是否比
  全局 item-level collaborative signal 更适合作为 SID augmentation？** 两处同时变更将在论文中
  显式声明为局限。
- **不引入 LightGCN / PureSVD**：目标是验证 SID augmentation 的**行为表示来源**，不是比较推荐模型强弱。

---

## 3. 怎么做的：完整流水线

### 3.1 统一协议（与主表逐位一致）

> **一个变体 = 一次 RQ-VAE（默认 seed 2024）→ 一个 codes 文件 → T5 用 seed 2025/2026/2027 各训一次。**

这是主表既有产物（Sport `Sport_sid_codes_768d.npy` + 3 T5 seed）已经采用的协议，本轮全部沿用，
保证消融表内部 seed 口径统一。

### 3.2 端到端驱动（[rqvae/run_behavior_ablation.sh](../rqvae/run_behavior_ablation.sh)，205 行，幂等）

每个 stage 都是 `if [ ! -e ... ]` 跳过已完成产物，**绝不覆盖、绝不重命名/复制**（e48 orphan 事故的直接教训）：

| Stage | 动作 | 实现 |
|:--|:--|:--|
| **A** | 生成行为向量 | 新变体走 [behavior_variants.py](../rqvae/behavior_variants.py)；`shuffled`/`zeropad` 走 [ablation_data.py](../rqvae/ablation_data.py) |
| **B** | 融合 | [fuse_embeddings.py](../rqvae/fuse_embeddings.py)（新增 `--behavior_file`，融合逻辑一行未改） |
| **C1** | 训练 RQ-VAE | `rqvae/main.py --num_emb_list 256 256 256`（**不传 `--seed`**，即默认 2024） |
| **C2** | codegen | `generate_sid_codes_param.py --ckpt <best_collision_model.pth>` |
| **D** | 容量诊断（留档，失败不阻断） | [sid_capacity_diagnostic.py](../tools/sid_capacity_diagnostic.py) → `doc/e49_{dsl}_{tag}.md` |
| **E** | T5 ×3 seed + 评测 | `model/main.py --seed $S` → `tools/eval_trained_model.py --beam_size 30 --topk_list 5,10,20,50,100` |

**复现命令**

```bash
bash rqvae/run_behavior_ablation.sh Sport rawcooc 2025 2026 2027   # 全链路
bash rqvae/run_behavior_ablation.sh Toys itemcf  2025 2026 2027
bash rqvae/run_behavior_ablation.sh Toys t5-only Toys_zeropad_seed2024_sid.npy zeropad_ablation 2025 2026 2027
```

### 3.3 行为向量怎么算（[rqvae/behavior_variants.py](../rqvae/behavior_variants.py)，260 行，CPU）

**复用而非复制**：从 [build_transition_graph.py](../rqvae/build_transition_graph.py) 导入
`build_cooccurrence_matrix` / `compute_ppmi` / `_save_output`。

```python
# rawcooc —— 与 TA-SID 同一个 C，只是不做 PPMI
M = build_cooccurrence_matrix(sequences, num_items, window_size=3)
# itemcf —— 整篮无向共现 + cosine 归一
M = build_basket_matrix(sequences, num_items)   # 同序列内所有商品对，无窗口
M = to_cosine(M)                                 # S_ij / sqrt(d_i·d_j)，对角置零
# 两者共用同一条 SVD 管线（与 _run_behavior_emb L273-294 一致）
U, S, _ = svds(M, k=128, v0=rng.standard_normal(min(M.shape)))  # 显式 v0 → 可复现
emb = U * sqrt(max(S, 0))        # U·√S
emb[~seen_items] = 0.0           # 冷启动物品置零
```

关键实现细节：

- **canonical 行序**：ItemID 顺序取自 `item_emb_768d.parquet`，与融合端断言逐位相等。
- **序列构造**：`list(row["history"]) + [row["target"]]`，与主链路逐行一致。
- **确定性**：`svds` 默认随机起始向量会让结果不可复现，故显式传 `v0 = rng.standard_normal(..., seed)`。
- **矩阵指纹**：`matrix_checksum()` 输出 `nnz + sum + 排序后 (row,col,data) 的 md5`，用于断言 rawcooc
  路径复用了与 TA-SID **逐位相同**的 C。

### 3.4 融合怎么算（768 + 128 → 896）

```python
content_norm  = LayerNorm(content_emb)    # (N, 768)  独立 LN
behavior_norm = LayerNorm(behavior_emb)   # (N, 128)  独立 LN
fused = cat([content_norm, behavior_norm], dim=-1)     # (N, 896)
```

两个模态**各自独立**做 LayerNorm 再拼接——这是主链路的既有实现，本轮一行未改。

### 3.5 P1 离线自检（开 GPU 之前，全部通过，零成本）

| 检查 | 结果 |
|:--|:--|
| **C 一致性**（rawcooc vs PPMI 路径） | ✅ 校验和完全相同：Sport `nnz=360722, sum=758172.0, md5=b025cbfff9ccf94c`；Toys `nnz=204302, sum=437444.0, md5=2ac580d7ab964bea` |
| **融合链路等价** | ✅ 用新 `--behavior_file` 重算主表 fused，与既有产物 **max abs diff = 0.000e+00** |
| **PPMI 可复现** | ✅ 新 `behavior_emb_ppmi.parquet` 与 legacy 诊断指标一致（范数 0.5667/0.4743 vs 0.5658/0.4744） |
| **行序对齐** | ✅ 全部新 parquet 的 `ItemID` 与 `item_emb_768d.parquet` 逐位相等 |
| **冷启动物品置零** | ✅ rawcooc/itemcf/ppmi：冷启动物品（Sport 28 / Toys 24）behavior 全零，覆盖品全非零 |
| **内容段逐位相同** | ✅ `fused_item_emb_{rawcooc,itemcf}` 前 768 维与主表 fused **maxdiff = 0.0e+00** → 唯一变量确实只在 behavior 侧 |

**这一组自检的意义**：证明新变体骑在**同一条**下游链路上，观测到的任何差异都只能来自 behavior 向量的
构造方式，而不是融合/RQ-VAE/T5 的配置漂移。

---

## 4. 执行进度：36 路中已完成 20 路

"路" = 一个 **变体 × 数据集 × seed** 的完整 T5 评测。目标 = 6 变体 × 2 数据集 × 3 seed = **36 路**
（Random 为辅助行，单 seed，不计入）。

| 数据集 | 变体 | 2025 | 2026 | 2027 | 状态 |
|:--|:--|:-:|:-:|:-:|:--|
| Sport | Content-only | 0.0274 | 0.0283 | 0.0269 | ✅ 3/3 |
| Sport | TA-SID | 0.0327 | 0.0317 | 0.0322 | ✅ 3/3 |
| Sport | **Raw-CoOcc** | 0.0311 | 0.0306 | 0.0299 | ✅ **3/3** |
| Sport | **ItemCF** | 0.0298 | 0.0296 | 0.0290 | ✅ **3/3** |
| Sport | Shuffled | 0.0262 | — | — | ⏳ 1/3（legacy 单 seed） |
| Sport | ZeroPad | 0.0286 | — | — | ⏳ 1/3（legacy 单 seed） |
| Toys | Content-only | 0.0341 | 0.0338 | 0.0328 | ✅ 3/3 |
| Toys | TA-SID | 0.0369 | 0.0383 | 0.0380 | ✅ 3/3（健康 codes） |
| Toys | **Raw-CoOcc** | 🔄 | — | — | ⏳ 0/3（seed2025 训练中） |
| Toys | **ItemCF** | — | — | — | ⏳ 0/3 |
| Toys | Shuffled | — | — | — | ⏳ 0/3（需全链路） |
| Toys | ZeroPad | — | — | — | ⏳ 0/3（codes 已备） |
| （辅助）| Sport Random | 0.0281 | — | — | ✅ 1 seed，按设计不补 |

**剩余 16 路与预计耗时**（实测单次 T5 = 61–66 min，RQ-VAE ≈ 8 min）：

| 阶段 | 内容 | 路数 |
|:--|:--|:-:|
| P4 | Toys Raw-CoOcc ×3（**进行中**）+ ItemCF ×3 | 6 |
| P5-a | Sport Shuffled 2026/2027 + ZeroPad 2026/2027 | 4 |
| P5-b | Toys Shuffled ×3（含 RQ-VAE） | 3 |
| P5-c | Toys ZeroPad ×3 | 3 |

---

## 5. 结果与结论

### 5.1 Sport：3-seed 完整（NDCG@20）

| 变体 | 2025 | 2026 | 2027 | **均值±std** | vs content-only |
|:--|:--:|:--:|:--:|:--:|:--:|
| Content-only | 0.0274 | 0.0283 | 0.0269 | 0.0275±0.0007 | — |
| Shuffled | 0.0262 | — | — | 0.0262（单 seed） | −4.4% † |
| ZeroPad | 0.0286 | — | — | 0.0286（单 seed） | +4.4% † |
| **ItemCF** | 0.0298 | 0.0296 | 0.0290 | **0.0295±0.0004** | **+7.0%** |
| **Raw-CoOcc** | 0.0311 | 0.0306 | 0.0299 | **0.0305±0.0006** | **+10.9%** |
| **TA-SID** | 0.0327 | 0.0317 | 0.0322 | **0.0322±0.0005** | **+16.9%** ‡ |

> † Shuffled / ZeroPad 目前只有 seed2025 一路，故其"vs content-only"按 **seed 匹配**（对 content-only
> 的 seed2025 = 0.0274）计算，与其余各行的"3-seed 均值 vs 3-seed 均值"口径不同，**不可横向并列比较**。
> 待 P5-a 补齐 3-seed 后改用均值口径。
> ‡ 本列的百分比由**逐 seed 原值**算出的均值导出；[experiment_results_summary.md](experiment_results_summary.md)
> §1.5 用**四舍五入后的均值**（0.0322 / 0.0275）计算得 **+17.1%**。差异纯属舍入，非口径分歧。

**两组核心对比**

| 对比 | Δ | 相对 |
|:--|:--:|:--:|
| **TA-SID vs Raw-CoOcc**（PPMI 的净贡献） | +0.0017 | **+5.5%** |
| **TA-SID vs ItemCF** | +0.0027 | **+9.3%** |
| Raw-CoOcc vs ItemCF | +0.0011 | +3.6% |

**一个值得注意的强证据：四个变体的 seed 区间完全不重叠**

```
content [0.0269, 0.0283] < ItemCF [0.0290, 0.0298] < Raw-CoOcc [0.0299, 0.0311] < TA-SID [0.0317, 0.0327]
```

即 **TA-SID 在每一个 seed 上都高于 Raw-CoOcc 与 ItemCF**，不存在"靠某个 seed 拉高均值"的情况。
（⚠️ 但 Raw-CoOcc 下限 0.0299 与 ItemCF 上限 0.0298 仅差 0.0001，这一对属**临界可分**，不宜强声称。）

### 5.2 行为空间诊断：前置预测与下游收益同序（**闭环在 Sport 成立**）

**`cons_auc_local`** = P(cos(相邻对) > cos(同篮非相邻对))，0.5 = 不区分。
它把"泛共现"从参照系里扣掉，只问行为空间**是否区分局部序列上下文与同篮远距离共现**。

| 数据集 | TA-SID | Raw-CoOcc | ItemCF | Random/Shuffled |
|:--|:--:|:--:|:--:|:--:|
| Sport | **0.6237** | 0.5730 | 0.4093 | ~0.4995 |
| Toys | **0.7157** | 0.6297 | 0.3494 | ~0.4995 |

**下游 NDCG 的排序与它完全一致（Sport）**：

```
cons_auc_local :  TA-SID 0.6237  >  Raw-CoOcc 0.5730  >  ItemCF 0.4093
NDCG@20        :  TA-SID 0.0322  >  Raw-CoOcc 0.0305  >  ItemCF 0.0295
```

→ 形成 **"行为空间可分性 → 下游推荐收益"** 的闭环。
⚠️ **但这是 Sport 单数据集的结果**，Toys 两变体尚未跑（§4），**闭环主张须待 P4 齐备后才能复核**。

`cons_auc_local` 的三点可直接写进论文：

1. **单调序两数据集一致**：PPMI > Raw-CoOcc > ItemCF，且随机/打乱 ≈ 0.50（无信号）。
2. **PPMI 相对原始计数有一致增益**（Sport +0.051 / Toys +0.086）→ 去掉 PPMI 确实损失局部序列结构。
3. **ItemCF 显著低于 0.5**（0.41 / 0.35）——整篮 cosine 归一后行为空间**主动反向**：同篮非相邻对
   （泛共现）的余弦**高于**相邻对。可解释：整篮矩阵里"共购次数多"直接抬高相似度，而局部顺序
   信息在 cosine 归一中被抹平，于是空间按"泛热门共购"而非"序列邻接"组织。

> ⚠️ **`cons_auc`（相邻对 vs 随机对）不可单独引用**：ItemCF 在该列高达 0.9055/0.9078，看似最好，
> 但那是**定义优势**（相邻对本身是整篮对的子集，必然被整篮矩阵满足）。必须用 `cons_auc_local`
> 才能把"泛共现"扣掉。这条已写进诊断表表头，防止误引。

### 5.3 必须如实报告的负向结果

**原计划预期**："Raw-CoOcc 的 frequency leakage 显著高于 PPMI 版"。
**实测是两个数据集一致地相反**：

| 数据集 | TA-SID | Raw-CoOcc |
|:--|:--:|:--:|
| `freq_leak` = Spearman(‖emb‖, log1p(freq)) Sport | **0.8069** | 0.7430 |
| `freq_leak` Toys | **0.8288** | 0.7525 |

即 **PPMI 之后 embedding 范数与流行度的耦合更强，不是更弱**——与"PPMI 抑制流行度偏差"的朴素预期相悖。

**但"流行度主导"并非只有这一种度量**，另两个指标给出相反信号：

- `top1_energy`（σ₁²/Σσ²）：Raw-CoOcc **一致更高**（Sport 0.0298 vs 0.0192；Toys 0.0265 vs 0.0186）
  → 原始计数矩阵的方差更集中在单一主方向。
- `pc1_cos_freq`（|cos(主方向, 流行度方向)|）：Sport Raw-CoOcc 0.6947 ≫ TA-SID 0.3639
  （**原始计数的第一主方向几乎就是流行度方向**）；**Toys 反向**，两数据集不一致。

**自洽解释（写作按此表述，不外推）**：两个量测的是**不同的东西**——`freq_leak` 量的是范数**序**与
流行度**序**的单调一致性；`pc1_cos_freq` / `top1_energy` 量的是方差**方向**被流行度占据的程度。
二者不必同向，实测也不同向。PPMI 改变的是**方差在方向上的分配**（把流行度从"第一主方向"降级为
众多方向之一），而非"范数与流行度的耦合强弱"。

**对本轮主张的影响**：**不影响主结论**，但**必须改口径**——

- ❌ 论文中**不以"PPMI 去流行度偏差"作为卖点**（证据不支持，且 Toys 的 `pc1_cos_freq` 反向）；
- ✅ 改用：**"PPMI 的作用是重塑行为空间的方向结构——降低主方向的流行度成分、提升对局部序列
  上下文的区分度——而非降低范数与流行度的耦合。"**

（`eff_rank` 在四者间几乎无差别，120–126 / 满秩 128，**不具区分度**，仅留档不作论据。）

### 5.4 Toys 基线（只有基线，四个变体全部未跑）

Toys 侧本轮**只有两个基线行可用**，四变体（Raw-CoOcc / ItemCF / Shuffled / ZeroPad）**一路未跑**，
故此处**不给出任何 Toys 变体结论**，仅登记基线口径供 P4/P5 对照：

| 行 | 2025 | 2026 | 2027 | 均值±std |
|:--|:--:|:--:|:--:|:--:|
| Content-only | 0.0341 | 0.0338 | 0.0328 | 0.0336±0.0006 |
| TA-SID（**健康 codes**，L0=207） | 0.0369 | 0.0383 | 0.0380 | **0.0377±0.0007** |
| ~~TA-SID（孤儿 codes，L0=62）~~ | ~~0.0360~~ | ~~0.0366~~ | ~~0.0375~~ | ~~0.0367~~ ❌ **已作废** |

→ Toys 基线提升 **+12.2%**（与 [e48 订正](e48_toys_main_codes_provenance_fix.md) 一致）。
⚠️ **关键提醒**：Toys TA-SID 的权威值是 **0.0377**，来自 `tiger_ta_sid_768d_toys_rq2024_seed*_eval.log`；
盘上另有三份 `Toys_ta_sid_768d_seed*_eval.log` 给出 0.0367，那是**孤儿 codes 上的作废数字**，
不得引用（详见 §7.1）。

### 5.5 现在能下 / 不能下的结论

**✅ 可以下的结论（Sport 侧，3-seed 完整）**

1. **"共现结构 + SVD"本身携带信息**：Raw-CoOcc（+10.9%）与 ItemCF（+7.0%）都明显高于 content-only，
   也高于 ZeroPad（0.0286）与 Shuffled（0.0262）——行为侧的信息不是"加维度"或"随便加个向量"能替代的。
2. **PPMI 不是可省略的一步**：去掉 PPMI 一致掉 5.5%，且行为空间 `cons_auc_local` 掉 0.051。
3. **局部序列上下文优于全局协同信号**：ItemCF（整篮 cosine）排序垫底，且在行为空间上表现为"反向"。
4. **行为空间诊断对下游收益有前置预测力**：排序在两个独立层面（behavior 空间 / T5 下游）完全一致。

**⚠️ 不能下的结论（须等 Toys 与剩余路数）**

1. **`≪` 还是 `≈`**：TA-SID 相对 Raw-CoOcc 仅 **+5.5%**（Δ=0.0017）。§2 预注册的两种措辞
   （"PPMI 有效重塑行为空间" vs "PPMI 非必要，主张收缩为共现+SVD"）指向**相反的论文主张**，
   **必须等 Toys 数据齐备才能定夺**。当前证据倾向"PPMI 有增量但不是主要收益来源，收益大头来自
   相邻共现结构本身"——若 Toys 复现该幅度，**论文措辞需要收缩**。
2. **闭环主张**：目前只有 Sport 一个数据集的支持，Toys 未跑。
3. **任何 Toys 侧的变体结论**：Toys Raw-CoOcc / ItemCF / Shuffled / ZeroPad 全部未跑。

---

## 6. 工程与口径教训（本轮踩过的坑，记录备查）

1. **`Test Dataset` 行只在验证集 NDCG@20 创新高时才打印**（[main.py:285-291](../model/main.py#L285-L291)）
   → `grep -c "Test Dataset"` 数的是**改进次数，不是 epoch 数**（实测真实 83 epoch vs 数出 72）。
   曾被误导以为训练卡死。真实 epoch 须数 `Epoch N/200` 行。
2. **`tee` 在评测*开始*时就创建 eval 日志，而 NDCG 表格在评测*结束*才写入**
   → 判断"评测是否就绪"**必须判"能否解析出 NDCG 表格"，不能判"文件是否存在"**。否则会读到空文件
   并误判"训练失败"。2026-09-22 18:33 实际发生过一次，导致决策门假阴性、链路白停。
   `run_e49_p3_p6.sh` 已改为 `metric()` 判可解析 + 门内 15s×20 重试双层防护。
3. **md5 不是数据指纹**：跨 env 会让同一份数据产生不同 md5（arrow 22 vs 24）。比对须用数据级指纹
   或下游产物复现。
4. **实测耗时**：单次 T5（含训练+评测）**61–66 min**（~54 s/epoch，约 83 epoch 触发早停）；
   单变体全链路 ≈ 75 min（RQ-VAE 8 + codegen/诊断 3 + T5 63）。计划文档里假设的 45–55 min 偏乐观。

---

## 7. ⚠️ 已发现但尚未修复的缺陷（接手须知）

### 7.1 P6 汇总脚本会为 Toys TA-SID 写出**作废数字**

[e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md) 已裁定：
Toys 主表的 codes 文件 `Toys_ta_sid_768d_sid.npy`（L0=62）是**孤儿**，其上的
**NDCG@20 = 0.0367 已作废**；权威数字来自健康 codes（L0=207）的订正重跑
（`tiger_ta_sid_768d_toys_rq2024_seed*_eval.log`）→ **0.0377±0.0007**。

但 [run_e49_p3_p6.sh](../rqvae/run_e49_p3_p6.sh) 的 P6 段：

```python
VARIANTS = [... ("TA-SID", "Toys_ta_sid_768d"), ...]      # ← 匹配到孤儿日志
REF = {"Toys": {"TA-SID": "Toys_ta_sid_768d_seed2025_eval.log"}}   # ← 同理
```

`collect("Toys_ta_sid_768d_seed*_eval.log")` 匹配的是**作废的孤儿日志**
（0.0360/0.0366/0.0375 → 0.0367），而订正日志名为 `tiger_ta_sid_768d_toys_rq2024_seed*_eval.log`，
**不在匹配范围内**。→ **P6 自动生成的 `e49_behavior_ablation_results.md` 会把 Toys TA-SID 写成 0.0367。**

| 日志 | codes | L0 distinct | NDCG@20 | 状态 |
|:--|:--|:-:|:--|:--|
| `Toys_ta_sid_768d_seed{2025,26,27}_eval.log` | `Toys_ta_sid_768d_sid.npy` | 62 | 0.0360/0.0366/0.0375 → **0.0367** | ❌ 作废 |
| `tiger_ta_sid_768d_toys_rq2024_seed{2025,26,27}_eval.log` | `Toys_ta_sid_768d_seed2024_sid.npy` | 207 | 0.0369/0.0383/0.0380 → **0.0377** | ✅ 权威 |

**处置建议**：**不要现在改脚本**——`run_e49_p3_p6.sh` 正在运行，bash 按字节偏移惰性读取脚本文件，
运行期间编辑可能导致解释器读到错乱内容。安全做法二选一：
① 等 P6 跑完生成文档后，**按 e48 手工订正** Toys TA-SID 行；或
② 在链路结束、无进程持有该脚本时再改 `VARIANTS`/`REF` 后重跑 P6 段。

（另注：影响面仅限 Toys TA-SID 一行。Toys content-only 的匹配名 `Toys_content768d_baseline_seed*_eval.log`
正确，Sport 全部正确，Sport 主表 codes provenance 审计为 100% 干净。）

### 7.2 文档头部状态列已过时

[e49_behavior_ablation.md](e49_behavior_ablation.md) §6 的状态表仍把 P2 标为"🔄 运行中"、P3 标为"⏳"，
但实际 P2 早已通过（[e49_p2_gate_verdict.md](e49_p2_gate_verdict.md) 已生成）、P3 已完成、P4 进行中。
建议在 P6 收尾时一并刷新。

### 7.3 Toys Random 行缺口

Toys 只有 `model/logs/Toys_random_t5_train.log`（且其中无可解析的结果表），**没有 eval 日志**，
而 P6 的 `VARIANTS` 又把 Random 排除在表外、只在备注提一句。**若论文的 Toys 消融表需要与 Sport
一样列出 Random 行，该行目前是空的**——需决定：补跑 Toys Random 的 T5+eval，还是在表中注明
"Toys 不做 Random 行"。

---

## 8. 下一步

| 优先级 | 事项 |
|:--|:--|
| — | 等后台链 `rqvae/run_e49_p3_p6.sh` 自动跑完 P4→P6（剩余 16 路，预计 2026-09-23 16:00–18:00） |
| 1 | **修 P6 的 Toys TA-SID 引用**（§7.1），否则自动汇总表会写出作废数字 |
| 2 | Toys 两新变体出齐后，**复核"行为空间 → 下游收益"闭环**是否在第二个数据集上成立 |
| 3 | 依 Toys 数据**定夺 TA-SID vs Raw-CoOcc 的措辞**（§5.5：`≪` 还是 `≈`），必要时收缩创新点表述 |
| 4 | 全仓库术语校准：确认无 "directional transition"/"转移概率" 残余表述 |
| 5 | 决定 Toys Random 行如何处置（§7.3） |
| 6 | 补跑 Sport TA-SID 主行 codes 的**确定性重建与逐位比对**（[e49_behavior_ablation.md](e49_behavior_ablation.md) §8.1；需 GPU 空档） |

---

**相关文档**：[e49_behavior_ablation.md](e49_behavior_ablation.md)（工作文档）·
[e49_behavior_space_gate.md](e49_behavior_space_gate.md) · [e49_behavior_space_analysis.md](e49_behavior_space_analysis.md) ·
[e49_p2_gate_verdict.md](e49_p2_gate_verdict.md) · [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md) ·
[experiment_results_summary.md](experiment_results_summary.md) §1.5（主表口径）
