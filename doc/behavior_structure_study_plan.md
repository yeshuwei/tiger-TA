# TA-SID → Behavioral Structure Study：修正版研究方案 v0.2

> **性质**：v0.1 草案（用户提供）经与实验记录逐条核对后的**修正与可执行化**。
> **日期**：2026-10-07 ｜ **分支** `experiments-2026-09-25`
> **上游**：[prior_art_novelty_audit.md](prior_art_novelty_audit.md)（§8 复审计，判定"注入位置受控对照"已被 OGR 占）
> ｜ [e49_ablation_full_conclusion.md](e49_ablation_full_conclusion.md)（消融总汇）
> ｜ [ablation_main_table_3datasets.md](ablation_main_table_3datasets.md)（**唯一数字源**，脚本生成勿手改）
> ｜ [behavior_structure_probe_conclusion.md](behavior_structure_probe_conclusion.md)（结构保持主结论）
> ｜ [beauty_tier1_conclusion.md](beauty_tier1_conclusion.md)（Beauty 6 变体 AUC）
>
> **本文件用途**：作为 v0.2 的定稿依据。**默认不自动执行任何 GPU 实验**；§5 的新臂需用户明确授权。

---

## 0. 本版相对 v0.1 的修改清单

v0.1 的四条硬伤、三条软伤，逐条更正如下。**表中"v0.1 写法"均为错误，以"更正"列为准。**

| # | v0.1 位置 | v0.1 写法 | 问题 | 更正 |
|:--:|:--|:--|:--|:--|
| **硬伤 1** | §4 RQ1 表 | TIGER vs TA-SID：Beauty +49.6% / Sport +58.0% / Toys +18.0%，TA-SID 0.0404 / 0.0327 / 0.0360 | **整表是已作废数字**。+49.6/+58.0/+18.0 正是 [e49_ablation_full_conclusion.md](e49_ablation_full_conclusion.md) §4.4 列为"❌ 全部作废"的 4096d/64d 主实验 baseline；TA-SID 三个值分别是单 seed 值 / 孤儿值 / 非均值 | 见 §3.1 **修正主表**：Content-only 0.0275/0.0336/0.0358 → TA-SID 0.0322/0.0377/0.0415 = **+17.1% / +12.2% / +15.9%** |
| **硬伤 2** | §10 互补性 | "300k 随机 item pairs，Spearman ρ≈0.02；500 sampled items，Jaccard≈0.02" → "Content Relation ≉ Behavior Relation ⇒ 互补" | **该分析已于 2026-09-29 由我们自己删除作废**（[ta_sid_vs_adc_sid_discussion_record.md:177](ta_sid_vs_adc_sid_discussion_record.md#L177)、[paper_outline_ta_sid.md](paper_outline_ta_sid.md) §7.5）。作废理由：①数字未落盘不可溯源 ②"Jaccard" 实为 `overlap/k` ③基于已弃用的 4096d ④**"低相关⟹互补"逻辑不成立——随机 128 维向量同样 ρ≈0** | **§10 整节删除**。互补性改由 §3.1 的消融分化 + §3.2 的 content 地板对照背书（见 §6） |
| **硬伤 3** | §18 缺口 2 / §9 / §12 贡献 3 | "AUC↔NDCG 的系统分析还没有，是最值得补的一项" | **已经做完，且是全项目最完整的一份分析** | §3.3 已列出：3 数据集 × 6 变体，数据集内 ρ = **0.943 / 0.943 / 1.000**，z-score 合并 r = **0.9354**（p<0.0001）。它是**现成贡献**，不是缺口 |
| **硬伤 4** | §17 表 | "Random behavior → RQ2 → ✅ 已有" | 该行是 `Sport_random_ablation`（**谱匹配噪声，单 seed，已弃用**），54 路主表不含它、不得引用 | Random 标 ⚠️ **弃用**；结构破坏对照改用 **Shuffled**（行级打乱真实行为向量，3 seed × 3 数据集，✅ 已有） |
| **软伤 1** | §5 ④ | ItemCF 用于"对比 global collaborative similarity 与 local sequential structure" | [e49 预注册](e49_behavior_ablation.md) §1.2 明令 ItemCF 是**非单变量**（同时改作用域/归一化/方向性），"**不得用于论证 TA-SID 优于 CF**" | ItemCF 只作**描述性对照**；"local vs global" 需**新臂**（§5）。这是本方案唯一的新 GPU 实验 |
| **软伤 2** | §6 假设 | "局部序列共现是否比一般的全局协同关系更适合 Semantic ID？" | 现有数据**已部分证伪**：`Raw-CoOcc − ItemCF` = Sport **+0.0011** / Toys **−0.0015** / Beauty **−0.0004**，**仅 1/3 同向** | 假设改为 **"局部共现是否只有经 PPMI 重加权才压得过全局协同信号？"**（跨 3 数据集一致的只有 `TA-SID > ItemCF`，+0.0027/+0.0022/+0.0024） |
| **软伤 3** | §8 | "AUC 0.869~0.962" | 未钉出处。项目里有 ≥4 个口径不同的"类 AUC"，互相 mean\|Δ\|=0.013 | 统一到 `behavior_structure_probe` 的 **topk(≥1000 边) × degmatch** 口径。三数据集 × 6 变体真实区间 = **0.7974 – 0.9651** |

> ✅ **v0.1 中核对通过、原样保留的部分**：§1 方向判断、§7 的链条 `E_b→E_fused→z→z_q→SID`、
> §5 的 `window=3 / half-window=1 / undirected`（[build_transition_graph.py:93](rqvae/build_transition_graph.py#L93) `half_window = window_size // 2`）、
> §11 TA-SID 新定位、§13 保留/降级表、**§14 DO NOT CLAIM 全表**、§18 缺口 1/3 的方向、§19 E50 定位。

---

## 1. 研究定位（不变）

### 1.1 明确放弃

**"把行为/协同信息加入 SID、在量化前融合"本身不再是方法创新。** 占用者见 [prior_art_novelty_audit.md](prior_art_novelty_audit.md) §0 与 §8：

| 轴 | 占用者 |
|:--|:--|
| 信号轴（行为/共现 → SID） | LETTER (CIKM'24)、DAS、QARM、PLUM (WWW'26)、**H²Rec (KDD'26)** |
| 机制轴（量化前 LN + concat） | **DECOR (SIGIR'26)**、PRISM、**GrIS (CIKM'26)**、**OGR (投 AAAI'27)** |
| 注入位置对照（input vs loss） | **OGR Table 3**（已做头对头，concat 胜）、GrIS Table 5（第二处，方向相反） |

### 1.2 新研究问题

```
v0.1 问：How to inject behavioral information into Semantic IDs?          ← 已被占
v0.2 问：如何衡量、以及在什么条件下，SID 保留了行为结构？                  ← 本方案的落点
```

**核心问题（v0.2）**：

> **Semantic ID 在多大程度上、在哪一层保留了商品间的行为结构？这种保留能否解释下游序列推荐的差异，边界条件是什么？**

关键转向：从"**提出一个方法**"转为"**给出一份口径统一、可复现、并明确标注失效边界的结构保持研究**"。

---

## 2. 修正后的 RQ 结构与贡献层次

v0.1 把 RQ2（哪种行为结构更好）当核心。**修正：主轴换成 RQ3**，因为它是唯一"证据已齐、方法学最硬、且未被他作占满"的部分。

| | RQ | v0.2 定位 | 成色 |
|:--|:--|:--|:--|
| **RQ1** | 行为结构是否改善 SID 推荐？ | **Setup / 复制性基线**，用修正数字 | 低（常规增益，不作卖点） |
| **RQ2** | 哪种行为结构更适合？ | **降为"用哪种结构去测"的一个因子**；其核心对比需新臂（§5） | 中（结果可能为负，须诚实报告） |
| **RQ3** | 行为结构是否被量化保留、能否解释下游？ | **★ 论文主轴 ★** | **高**（现成、3 数据集、统一口径、含边界条件） |

### 三项贡献（v0.2 定稿版）

1. **贡献 1（主轴）——SID 行为结构保持的统一口径研究。**
   在**同一 RQ-VAE + T5 协议**下，用**度受控负采样**探针（`degmatch`）对 **3 数据集 × 6 变体**量化 SID 的行为结构保持打分，
   给出**数据集内**单调性（ρ = 0.943 / 0.943 / 1.000）并**显式标注三层边界**：
   ① content 地板 0.875–0.908；② 跨数据集绝对值不可比；③ 判据图与 TA-SID 同源带来的定义优势。
   *不主张 6/6，也不主张 4/6；只主张"分层 + 两端"跨数据集稳定。*

2. **贡献 2（新实验）——局部 vs 全局的单变量隔离。**
   现有 ItemCF 臂三项混杂，答不了 RQ2。本方案补一个 **2×2（作用域 × 加权）** 设计（§5），
   只**新增 2 个臂**即可把"局部序列结构是否特异有效"变成**单变量可证伪**的问题。
   *预期结果可能为负——见 §5.2 的预注册分支。*

3. **贡献 3（机制）——量化中性 + 端到端保序。**
   `z → xq[full]` 保序 **3/3（ρ=1.000）**；端到端 `A 输入 896d → xq[full]` 保序 **0.928 / 0.928 / 0.986**（2026-09-27 订正后）。
   即：**量化阶段对变体排序完全中性**，行为结构在离散化中不丢失——与主流"离散化必然丢信息"立论方向相反。

---

## 3. 已核实的现有证据（可直接写入论文）

> **数字源纪律**：以下全部取自 [ablation_main_table_3datasets.md](ablation_main_table_3datasets.md)（脚本生成，54/54 路）
> 与 [behavior_structure_probe_conclusion.md](behavior_structure_probe_conclusion.md) / [beauty_tier1_conclusion.md](beauty_tier1_conclusion.md)。
> **勿再引用任何 4096d、孤儿 codes(0.0367)、或 v0.1 §4 的数字。**

### 3.1 修正主表：RQ1（NDCG@20，3 seed 均值，同协议）

| 变体 | 行为信号 | Sport | Toys | Beauty |
|:--|:--|--:|--:|--:|
| Content-only（=TIGER 基线） | 无 | 0.0275±0.0007 | 0.0336±0.0007 | 0.0358±0.0009 |
| ZeroPad | 128 维零（维度对照） | 0.0280±0.0006 | 0.0334±0.0012 | 0.0374±0.0004 |
| Shuffled | 行级打乱（结构破坏） | 0.0264±0.0003 | 0.0284±0.0008 | 0.0346±0.0008 |
| ItemCF-SVD | 全局协同（**非单变量**） | 0.0295±0.0004 | 0.0355±0.0002 | 0.0391±0.0010 |
| Raw-CoOcc-SVD | 局部共现，无 PPMI | 0.0305±0.0006 | 0.0340±0.0015 | 0.0387±0.0011 |
| **TA-SID** | 局部共现 + PPMI | **0.0322±0.0005** | **0.0377±0.0007** | **0.0415±0.0011** |

**配对显著性（G2，2026-10-07 补，N@20，n=3/df=2，|d_z|>2.48 为门槛）——各变体 vs Content-only：**

| 变体 vs Content-only | Sport | Toys | Beauty |
|:--|:--|:--|:--|
| ZeroPad | 不显著 (d_z 0.62) | 不显著 (−0.07) | 未过线 3/3 (1.48) |
| Shuffled | **显著负** (−2.59) ✅ | **显著负** (−3.77) ✅ | 未过线 1/3 (−1.14) |
| ItemCF-SVD | **✅** (3.40) | 未过线 3/3 (2.36) | 未过线 3/3 (1.98) |
| Raw-CoOcc-SVD | **✅** (4.29) | ⚠️ 2/3 同向 (0.21) | 未过线 3/3 (1.63) |
| **TA-SID** | **✅** (4.25) | **✅** (3.38) | **✅** (4.22) |

> 全文见 [ablation_paired_significance.md](ablation_paired_significance.md)（+ `_r20.md`）。R@20 有 6 处判决翻转，
> 主指标只报 N@20；"未过线" ≠ "为零"（n=3 检出力弱）。**新增局限**：Beauty 上 `TA-SID − ItemCF` **不显著**（d_z 1.20）。

**RQ1 的正式表述**（用此，勿用 v0.1 表）：

| 数据集 | Content-only | TA-SID | 增益 |
|:--|--:|--:|--:|
| Sport | 0.0275 | 0.0322 | **+17.1%** |
| Toys | 0.0336 | 0.0377 | **+12.2%** |
| Beauty | 0.0358 | 0.0415 | **+15.9%** |

方向一致性（主表 §3.1）：`TA-SID − Content-only` **3/3 同向**；`TA-SID − Raw-CoOcc` **3/3**；`TA-SID − ItemCF` **3/3**。

**显著性升级（G2）**：`TA-SID − Content-only` 在 **3 数据集 × 2 指标 = 6/6 全部 p<0.05**（18/18 逐 seed 同向、零交叉）
⇒ `TA-SID` 那一行现在是**唯一在三个数据集上都显著优于 content-only 的变体**。
但 `TA-SID − Raw-CoOcc`/`− ItemCF` 的逐层分解**只有 Sport 两层独立显著** ⇒ **阶梯分解只能作点位估计，勿写"PPMI 层贡献 X%"。**

### 3.2 结构保持（Behavior AUC，`xq[full]`，topk × degmatch 口径）

| 变体 | Sport | Toys | Beauty |
|:--|--:|--:|--:|
| TA-SID | **0.9647** | **0.9651** | **0.9560** |
| Raw-CoOcc | 0.9469 | 0.9543 | 0.9377 |
| ItemCF | 0.9397 | 0.9455 | 0.9463 |
| Content-only | 0.9080 | 0.8751 | 0.8365 |
| ZeroPad | 0.9069 | 0.8716 | 0.8456 |
| Shuffled | 0.8763 | 0.8127 | 0.7974 |

> Beauty 六行来自 [beauty_tier1_conclusion.md](beauty_tier1_conclusion.md) §1，**口径与 Sport/Toys 逐位一致**（同脚本 `tools/behavior_structure_probe.py` 的 E 图/正负样本/余弦 AUC）。
> 三数据集 × 6 变体真实区间 = **0.7974 – 0.9651**（**不是** v0.1 §8 的 "0.869~0.962"）。

### 3.3 AUC ↔ NDCG@20（**RQ3 的核心证据，已存在**）

| 数据集 | Spearman ρ | 排序一致性 | 唯一倒置 |
|:--|--:|:--|:--|
| Sport | **+0.943** | 4/6 | Content ↔ ZeroPad（差 0.0011，噪声内；二者构造上应等价） |
| Toys | **+0.943** | 4/6 | Raw-CoOcc ↔ ItemCF（差 0.0015，相邻） |
| Beauty | **+1.000** | 6/6 | 无 |
| 合并（数据集内 z-score，12 点） | Pearson r = **0.9354**（p<0.0001） | — | — |

**三层边界（必须与上表同时报告）**：

1. **content 地板 0.875–0.908**：完全无行为信息的 Content-only 就拿到 0.908(Sport)/0.875(Toys)/0.837(Beauty)。
   行为向量的**增量**只有 **0.06–0.09**。不报 Content-only 基线，AUC 会被误读为"行为保持度"。
2. **跨数据集绝对值不可比**（Sport 0.876–0.965 / Toys 0.813–0.965 / Beauty 0.797–0.956）→ 必须数据集内标准化。
3. **定义优势**：判据图 E 用的就是 TA-SID/Raw-CoOcc 构造行为向量的那张 `window=3` 图 → 对二者有天然红利。
   ItemCF 用另一张（整篮）图却仍排第 3，是缓解但不消除。
4. **口径纪律**：对外统一表述为 **"数据集内各有 1 处相邻倒置，均在噪声量级；两端（TA-SID 恒第一、Shuffled 恒最后）跨数据集稳定"**。**不主张 6/6，也不主张 4/6。**

### 3.4 三级分解与量化中性（贡献 3）

| ρ | Sport | Toys | Beauty |
|:--|:--:|:--:|:--:|
| ρ(z, xq[full])——量化是否保序 | **+1.000** | **+1.000** | **+1.000** |
| ρ(A 输入 896d, z)——encoder 是否重排 | +0.928 | +0.928 | +0.986 |

⇒ **"量化保序"跨 3 数据集成立；encoder 基本不重排**。
（⚠️ 2026-09-27 订正：早期"encoder 建立排序"读数因 4224d 文件维度 bug 作废，勿引。）

### 3.5 注入位置对照（E50）——辅助，不作创新点

b0/b2/b4/b6 已跑完（[e50_b2_full_conclusion.md](e50_b2_full_conclusion.md)）：`b2−b4` 9/9 显著、`b2 vs b0` 三数据集全不显著。
**因 OGR/GrIS 已占 input-vs-loss 对照，此臂只作"行为信号无论注入输入还是损失都有效"的补充证据。**

---

## 4. 真实缺口（修正版）

v0.1 §18 的三个缺口，核对结果：

| v0.1 缺口 | 核对结果 |
|:--|:--|
| 缺口 1：变体是否在完全相同条件下比较？ | ✅ **已满足**。54/54 路，同 RQ-VAE(256³) + T5、同 train.parquet、同 ItemID 顺序、同 `svd_dim=128`。**唯一例外 Random（单 seed 弃用，本就不该用）** |
| 缺口 2：AUC↔NDCG 的系统分析 | ❌ **不是缺口**——已经做完（§3.3）。是**现成贡献** |
| 缺口 3：local vs global 的明确实验设计 | ✅ **是真缺口**，且是本方案唯一必需的新 GPU 实验（§5） |

### 本方案认定的真实缺口

| # | 缺口 | 性质 | 成本 |
|:--:|:--|:--|:--|
| **G1** | **单变量的 local vs global 臂**（ItemCF 三项混杂，答不了 RQ2） | GPU | 2 臂 ≈ 20–22 h（可只做 1 臂 ≈ 10–11 h） |
| **G2** ✅ | 主表 6 变体之间的**配对显著性**（现有只有 mean±std，无 p 值） | CPU，~0 | **已完成 2026-10-07** |
| **G3** | 三数据集 × 6 变体 AUC 的**统一成表**（Beauty 六行散在 Tier-1 文件里） | 文档 | 0 |
| **G4**（可选） | 链条止于 `xq`，**T5 token embedding 层未测** | CPU/探索 | 低–中，可行性待试 |
| ~~G5~~ | ~~互补性分析~~ | — | **不做**（§0 硬伤 2，已作废且逻辑不成立） |

---

## 5. 新实验设计：局部 vs 全局的单变量隔离（预注册）

### 5.1 臂定义

现有两个臂已构成 2×2 的**左列**；**补右列两臂**即得完整因子设计：

| | **raw counts** | **PPMI** |
|:--|:--|:--|
| **window=3（相邻对）** | Raw-CoOcc-SVD ✅ 已有 | **TA-SID** ✅ 已有 |
| **full basket（整篮）** | **A1：Basket-Raw-SVD ❌ 待跑** | **A2：Basket-PPMI-SVD ❌ 待跑** |

**构造规约（与 TA-SID 共用其余全部管线，只改作用域）**：

| 项 | 值 |
|:--|:--|
| 共现矩阵 | 每序列内**全部无序商品对**计数（替换 `window=3`），`min_cooccur=1` |
| A1 | 该矩阵直接 SVD → `U·√S` → 128d |
| A2 | 该矩阵 → `compute_ppmi` → SVD → `U·√S` → 128d |
| 融合 | 与 TA-SID 逐位相同：`LN(content 768d) ‖ LN(behavior 128d)` → 896d |
| RQ-VAE | `256×256×256`，固定 seed=**2024**（每数据集一份 codes，与主表协议一致） |
| T5 | `model/main.py`，seed **2025/2026/2027**，200 ep / early_stop 10 / beam 30 |
| 评测 | `tools/eval_trained_model.py --beam_size 30 --topk_list 5,10,20,50,100`，**按表头取 @20，勿用 `$NF`** |
| 门禁 | RQ-VAE 训完先查 **L0 unique ≥ 对应 b4 的 90%**，不达标记塌缩、跳过该数据集 T5 |
| 产物命名 | `data/{D}/{D}_basket{raw,ppmi}_768d_sid.npy`、`rqvae/ckpt/{D}_Basket{Raw,PPMI}_768d/`（**独立命名，绝不覆盖**） |

> **为何不用 ItemCF 顶替**：ItemCF 同时改了作用域（整篮）+ 归一化（cosine）+ 方向性三项（[e49 预注册](e49_behavior_ablation.md) §1.2），
> 不能作为单变量对照。A1/A2 与左列**只差作用域一项**。

### 5.2 预注册判据（**跑之前写死**）

统计口径（吸取 E50 §3.4 阈值冲突的教训）：**以配对检验为准，不以各臂独立 std 为准**。
3 数据集 × 3 seed 按 seed 配对，报 paired t + 95% CI + 逐 seed 符号一致性。

**核心对比 A2 vs TA-SID**（PPMI 侧的单变量局部效应）：

| 结果 | 论述 | 分支 |
|:--|:--|:--|
| `TA-SID > A2` 在 **3/3 数据集**且 p<0.05 | ✅ **"局部序列结构特异有效"**——本方案最好的结果 | 强化贡献 2 |
| `TA-SID ≈ A2`（配对不显著，量级 < 噪声） | ⚠️ **"任何共现作用域都行，局部性不是有效因子"**——**诚实负面结果**，论文退回纯 RQ3 主轴 | 主分支（预期） |
| `A2 > TA-SID` 在 3/3 | 全局更好，与 §5.2 假设相反 | 如实报告 |

**交互对比**（`TA-SID − Raw-CoOcc` vs `A2 − A1`）：检验 PPMI 的增益是否**依赖作用域**。

> **⚠️ 事前预期（诚实）**：现有 `Raw-CoOcc − ItemCF` 只有 **1/3 同向**，且 Toys 上全局反超局部。
> 因此 **`TA-SID ≈ A2` 是分支中的高频预期**。本方案**接受**该结果——见 §9 的 GO/NO-GO。

### 5.3 成本（实测口径，参照 E50 b2 的 9 路 ≈ 10.7 h）

| 数据集 | RQ-VAE | T5 × 3 seed | eval × 3 | 小计 / 臂 |
|:--|--:|--:|--:|--:|
| Sport | 9 min | 241 min | 12 min | ~262 min |
| Toys | 9 min | 146 min | 12 min | ~167 min |
| Beauty | 9 min | 197 min | 12 min | ~218 min |
| **1 臂全数据集** | | | | **~10.8 h** |
| **2 臂** | | | | **~21.6 h** |

**降级选项**：若预算受限，**只跑 A2**（约 **10.8 h**）即可回答核心对比 `A2 vs TA-SID`；A1 只服务于交互项，可延后。

---

## 6. 论文叙事（修正版逻辑链）

```
现有 SID 主要编码内容语义
        │
        ▼
已有工作已证明：行为信息可以进入 SID（LETTER/DAS/PLUM/DECOR/PRISM/GrIS/OGR）
        │  ← 本方案不在此处主张创新
        ▼
遗留问题：行为结构进入连续表示、再经 RQ-VAE 离散化之后，还剩多少？
        │
        ▼
RQ3（主轴）：用度受控探针测量结构保持，并检验其与下游 NDCG 的关系
        │  → 数据集内单调 (ρ=0.943/0.943/1.000)；量化中性 (ρ(z,xq)=1.000)
        │  → 边界：content 地板 0.875–0.908 / 跨数据集不可比 / 定义优势
        ▼
RQ2（因子）：换不同的行为结构去测，看结论是否稳健
        │  → 2×2 (作用域 × 加权)，回答"局部性是否特异"
        ▼
RQ1（setup）：行为结构确实改善推荐（+17.1% / +12.2% / +15.9%）
```

**贡献落点**：**"我们测了什么、用什么方法测、以及在什么条件下结论不成立"**——
**不是**"我们发现局部结构最好"（数据不支持，见 §5.2）。

---

## 7. 标题候选（去除 TA-SID）

| | 标题 | 说明 |
|:--|:--|:--|
| A | **How Much Behavioral Structure Survives Semantic ID Quantization?** | 聚焦 RQ3 主轴，问题导向 |
| B | **Measuring Behavioral Structure Preservation in Semantic IDs for Sequential Recommendation** | 方法/度量导向，最稳 |
| C | **What Behavioral Structure Should a Semantic ID Preserve?** | v0.1 方向 B，但需 §5 结果支撑；若结果为"≈"则不宜用 |

> v0.1 方向 C（保留 "TA-SID" 于标题）**不推荐**——会让审稿人预期 method paper。**同意 v0.1 的判断。**

---

## 8. DO NOT CLAIM（修正版，全表）

以下表述**一律不得出现**在摘要/引言/贡献列表：

- ❌ 首次将行为信息引入 Semantic ID（LETTER 2024 已做）
- ❌ 首次在 RQ-VAE 量化前融合行为结构（DECOR SIGIR'26 / PRISM / GrIS CIKM'26 / OGR 已做）
- ❌ 首次将 item co-occurrence 用于 Semantic ID
- ❌ 首次将 PPMI-SVD 行为表示用于 Semantic ID
- ❌ **首次比较 input-level 和 loss-level 行为注入**（OGR Table 3 已做，见 [prior_art_novelty_audit.md](prior_art_novelty_audit.md) §8.2）
- ❌ TA-SID 是一种全新的 behavior-aware Semantic ID architecture
- ❌ LayerNorm + concatenation 是方法创新
- ❌ **内容与行为嵌入"低相关 ⟹ 互补"**（该分析已作废；随机向量同样低相关）
- ❌ **"PPMI 有效"**（增量跨数据集不一致：[e49_ablation_full_conclusion.md](e49_ablation_full_conclusion.md) §3.2）
- ❌ **"AUC 诊断与 NDCG 6/6 同序"**（跨管线口径不一致，官方口径只主张"分层 + 两端"）
- ❌ **"共现层贡献 X% / PPMI 层贡献 Y%"**（G2：只有 Sport 在 N@20 下两层各自独立显著；其余为点位估计，n=3 检出力不足）
- ❌ **"Beauty 上 TA-SID 显著优于 ItemCF/通用 CF"**（G2：d_z=1.20, p=0.172，未过检验）
- ❌ **只报 "p<0.05" 而不交代 n=3/df=2**（且须提非参数符号检验在 n=3 的下界恒为 0.25）

---

## 9. 风险与 GO / NO-GO

### 9.1 风险登记

| # | 风险 | 概率 | 影响 | 缓解 |
|:--:|:--|:--:|:--|:--|
| R1 | §5 结果为"`TA-SID ≈ A2`"，局部性卖点不成立 | **中–高** | 贡献 2 弱化 | 主分支预期即如此；论文退回 RQ3 单主轴（仍可发表） |
| R2 | RQ3 是 n=6/数据集的相关性，检出力弱 | 高 | 审稿人称"anecdotal" | 用 3 数据集 × 6 变体 = 18 点 + 数据集内 z-score 合并（r=0.9354）；并明确只主张"分层+两端" |
| R3 | content 地板 0.875–0.908 ⇒ 审稿人称该指标"mostly content" | 高 | 贡献 1 被质疑 | **同一张表内永远并列 Content-only 基线**；报告"增量"而非绝对值 |
| R4 | 判据图与 TA-SID 同源 ⇒ 定义优势 | 中 | 贡献 1 被质疑 | 主动声明；补 ItemCF（异图）仍排第 3 作部分缓解；可试"跨图判据"作稳健性 |
| R5 | 该路线是纯实证/分析转向，**不解决方法新颖性被占** | — | 收益上限 = 一篇分析型论文 | 已在 §1.1 明确，属**止损**非翻盘 |

### 9.2 GO / NO-GO 门槛

**GO（推荐）**，当且仅当以下三条你都能接受：

1. **接受 RQ2 可能返回"无赢家"**——即 `TA-SID ≈ A2`，论文以 RQ3 单主轴成立。
   （回应用户 v0.1 的 Q5：**若必须证明 local 更好，则本方向 NO-GO**——那会退化成"为证明 TA-SID 好而组织实验"。）
2. **接受贡献是"度量 + 边界条件"而非"方法"**，收益上限为一篇分析/实证论文。
3. 授权 §5 的 GPU 实验（**1 臂 ~10.8 h** / 2 臂 ~21.6 h）。

**CONDITIONAL GO**：不跑 §5，仅以现有资产成文——成立，但贡献只剩 RQ3 一条，且无法回答"局部性是否特异"。

**NO-GO**：若要求以方法新颖性投稿，或要求结论必须是"局部结构最优"。

---

## 10. 执行清单（按依赖排序）

| 步骤 | 内容 | 成本 | 前置 | 状态 |
|:--:|:--|:--|:--|:--:|
| 0 | 修正各文档中的错误数字（§0 清单） | 0 | — | 🟡 本文件已列明 |
| 1 | G2：主表 6 变体配对显著性 | CPU 0.27s | 步骤 0 | ✅ **已完成 2026-10-07** |
| 2 | G3：三数据集 × 6 变体 AUC 统一成表（Sport/Toys 取自 probe，Beauty 取自 Tier-1） | 0 | — | ⬜ |
| 3 | **G1：跑 A2（Basket-PPMI-SVD）**，3 数据集 × 3 seed | ~10.8 h | 授权 | ⬜ |
| 4 | （可选）G1b：跑 A1（Basket-Raw-SVD）补交互项 | ~10.8 h | 步骤 3 | ⬜ |
| 5 | （可选）G4：T5 token embedding 层结构保持（先做可行性试探） | 低 | — | ⬜ |
| 6 | 按 §5.2 预注册判据出裁决，写 RQ2 结论 | 0 | 步骤 3 | ⬜ |

> **本文件不自动执行任何步骤**；步骤 3/4 需用户明确授权后再动 GPU。

---

## 附：相关文件索引

| 文件 | 用途 |
|:--|:--|
| [ablation_main_table_3datasets.md](ablation_main_table_3datasets.md) | **唯一数字源**（54/54 路，脚本生成勿手改） |
| [ablation_paired_significance.md](ablation_paired_significance.md) | **G2 配对显著性**（N@20；含写作红线 §3）+ `_r20.md` + 同名 `.csv` |
| [e49_ablation_full_conclusion.md](e49_ablation_full_conclusion.md) | 消融总汇 + §4.4 作废清单（v0.1 硬伤 1 的依据） |
| [behavior_structure_probe_conclusion.md](behavior_structure_probe_conclusion.md) | RQ3 主结论（Sport/Toys 6 变体 AUC + ρ） |
| [beauty_tier1_conclusion.md](beauty_tier1_conclusion.md) | Beauty 6 变体 AUC（同口径）+ 三级分解 |
| [beauty_tier2_conclusion.md](beauty_tier2_conclusion.md) | Beauty 6 变体 NDCG@20 |
| [e49grid_diag_summary.md](e49grid_diag_summary.md) | 量化端 12 格网格（量化中性证据） |
| [e50_b2_full_conclusion.md](e50_b2_full_conclusion.md) | E50 注入位置（辅助） |
| [prior_art_novelty_audit.md](prior_art_novelty_audit.md) | novelty 裁定（§8 复审计） |
| [e49_behavior_ablation.md](e49_behavior_ablation.md) | E49 预注册（ItemCF 非单变量的依据） |
