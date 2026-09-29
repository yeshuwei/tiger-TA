# TA-SID 论文严格审查报告（师兄拷打版）
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

> 生成：2026-07-31（与师兄交流前准备）
> 目的：把自己放到"不了解每个模块的审稿人"位置，逐个拷打。今天交流前必须搞清楚这些。

---

## 第一部分：你的方法到底在干什么（必须先能复述）

### 1.1 全链路一张图

```
[用户行为序列 train.parquet]             [商品文本 → Sentence-T5-base → 内容嵌入]
        │                                        │
        ▼                                        ▼
[共现矩阵 C (窗口=3)]               [内容嵌入 768d (L2归一化)]
        │                                        │
        ▼                                        ▼
[PPMI 变换: max(0, log(C·ΣC/(row·col)))]   LayerNorm(768)
        │                                        │
        ▼                                        ▼
[截断SVD (k=128)]                        ┌──────────────────────┐
        │                                │  Concat → 896d fused │
        ▼                                └──────────────────────┘
[行为嵌入 128d (U·√S)]                              │
        │                                            ▼
        └─────── LayerNorm(128) ──────→   RQ-VAE (MLP→残差量化→MLP)
                                                   │
                                                   ▼
                                    SID codes (3码 + 1碰撞位 = 4 token)
                                                   │
                                                   ▼
                                    T5 自回归预测下一商品 SID code
```

### 1.2 每个模块的"为什么"（这是你要掌握的）

| 模块 | 在干什么 | 为什么这么设计 |
|:----|:--------|:--------------|
| **共现矩阵** | 统计任意两个商品在同一个滑动窗口（左右各1）内共现的次数 C(i,j) | "转移动态"的第一步：如果用户经常先买 A 再买 B，A 和 B 会在相邻窗口共现。窗口=3 是最朴素的邻接定义 |
| **PPMI** | 对共现次数做标准化：`max(0, log(C(i,j)·Σ / row_i·col_j))` | 共现次数本身被高频商品污染（爆款和所有商品都共现）。PPMI 除以边缘概率，衡量"共现超出随机期望的程度"。正 PMI = 真实转移关系，负 PMI（共现比随机还少）截断为 0 |
| **截断 SVD** | 对 PPMI 矩阵做 SVD，取前 128 个奇异值，得到每个商品的行向量 `U·√S` | ① 降维：N×N 共现矩阵（18000² 稀疏）→ N×128 稠密向量，可计算；② 去噪/泛化：SVD 的低秩近似把"共同转移模式"编码进向量，低频商品可以借共享模式获得合理向量；③ 这是从文本 Embedding（word2vec/GloVe）借来的成熟做法 |
| **为什么 128d** | 行为向量维度 | 经验值。够表达转移模式，又不至于稀释 content。**这是你要能用一句话解释的超参** |
| **LayerNorm(各自)** | content 和 behavior 分别归一化再拼接 | 两者尺度完全不同：content norm≈1.0，behavior norm≈0.57-0.68。不做 LN，128d behavior 会被 768d content 数值淹没，等于白加。LN 让两个模态在拼接时"地位对等" |
| **为什么 Concat 不用 Add** | 拼接成 896d 单向量 | 保留两个独立信息通道，RQ-VAE 自己学怎么结合。Add 会强行要求两向量等维且直接叠加，丢失通道结构 |
| **RQ-VAE** | 把 896d 浮点嵌入压缩成 3 个离散码（每层 256 选 1） | 生成式检索的基石：把"连续空间里的商品"变成"离散 token"，T5 才能做序列生成。残差量化 = 第一层量化完，把残差再量化第二层……逐层逼近 |
| **碰撞消解（Sinkhorn）** | 生成 SID 时，多个商品码到同一个 code 就微调，直到唯一 | 两个商品必须有不同的 SID，否则 T5 分不清。碰撞率 = 多少商品共用 code |
| **T5** | 输入 = 历史商品的 SID code 序列（每个商品展平成 4 个 token），预测下一个商品的 4 个 code token | 标准 TIGER 架构。T5 自回归学到"看到这些历史 SID → 下一个 SID 是什么"的转移规律 |
| **L_trans (Track B)** | 在 RQ-VAE 训练时，额外用 InfoNCE 把有转移关系的商品对在量化空间拉近 | 想直接约束"转移对在量化空间靠近"。**结果无效（详见拷打）** |

### 1.3 你最容易混淆的几个概念（今天可能被师兄问）

1. **"加一个 embedding" vs "加 transition 结构"**
   - 你说的"加行为向量"= 只增加 128 个维度（ZeroPad 也做得到）
   - "transition 结构" = 这 128 维里 A 和 B 的关系（谁和谁共现）**反映在向量的相似性上**
   - 你的 Shuffled 消融就是在区分这两者

2. **"碰撞率"不是越低越好**
   - B-1 用 L_trans 把碰撞率从 10.66% 压到 0.34%，NDCG 反而降 6.8%
   - 碰撞率低 = 码本利用率高 ≠ 推荐效果好。审稿人问碰撞率时，你要能说清它的作用边界

3. **768d vs 4096d 不是"越大越好"**
   - 768d 一致优于 4096d：RQ-VAE 是 MLP 压到 32d 再量化，输入维度太高信息挤不进去，码本学习困难
   - **但这恰恰是你现在主实验的命门（见拷打 A）**

---

## 第二部分：真实结果审计（我核对了全部日志，结论可能让你意外）

### 2.1 真实日志结果表（非文档，纯日志提取）

| 实验 | code 文件 | seed | Test NDCG@20 |
|:----|:---------|:----:|:------------:|
| TIGER baseline Beauty | `Beauty_t5_rqvae.npy` | 2025 | **0.0379** |
| TIGER baseline Beauty | `Beauty_t5_rqvae.npy` | 42 | **0.0270** |
| TIGER baseline Sport | `Sports_t5_rqvae.npy` | 2025 | **0.0207** |
| TIGER baseline Toys | `Toys_t5_rqvae.npy` | 2025 | **0.0305** |
| TA-SID Beauty (768d) | `Beauty_ta_sid.npy` | 2025 | **0.0418** |
| TA-SID Beauty (768d) | `Beauty_ta_sid.npy` | 42 | 0.0404 |
| TA-SID Sport (768d) | `Sport_sid_codes_768d.npy` | 2025 | **0.0327** |
| TA-SID Sport (4096d) | `Sport_ta_sid.npy` | 2025 | 0.0276 |
| TA-SID Toys (768d) | `Toys_sid_codes_768d.npy` | 2025 | **0.0360** |
| TA-SID Toys (4096d) | `Toys_ta_sid.npy` | 2025 | 0.0317 |
| **ZeroPad 消融 Sport** | `Sport_zeropad_ablation_sid.npy` | 2025 | 训练中（当前 best **0.0236**，epoch 14，仍在涨） |
| **Shuffled 消融 Sport** | `Sport_shuffled_ablation_sid.npy` | 2025 | **0.0262**（最终，epoch 97 早停） |
| Random noise (旧,已弃用) | `Sport_random_ablation_sid.npy` | 2025 | 0.0281 |
| Random SID Beauty | `Beauty_random_sid.npy` | 2025 | 0.0263 |
| Random SID Sport | `Sport_random_sid.npy` | 2025 | 0.0093 |
| Random SID Toys | `Toys_random_sid.npy` | 2025 | 0.0308 |
| L_trans B-1 (λ=0.1) | `Sport_track_b...` | 2025 | 0.0304 |
| L_trans B-2 (λ=0.01) | `Sport_track_b...l001` | 2025 | 0.0318 |
| L_trans B-3 (λ=0.01,x_q) | `Sport_track_b...v2_xq` | 2025 | 0.0312 |

### 2.2 三个严重发现（必须今天和师兄摊牌）

**🔴 发现 A：主实验的 TIGER baseline 来自 TIGER 原始项目的一条完全不同管线（不是你的 768d 管线）**

**铁证（2026-07-31 复核）：**
- 主实验表里 TIGER baseline 的数字（Sport 0.0207 / Toys 0.0305 / Beauty 0.0270）来自 **7 月初跑的 TIGER 原始项目日志**（`sports.log` / `toys.log` / `tiger.log` / `beauty_baseline_seed42.log`）
- 这些日志用的 code 文件是 `{ds}_t5_rqvae.npy`（TIGER 原始项目 7月6日生成）
- 而 TIGER 原始项目的 RQ-VAE 训练命令是 `python main.py --data_path ../data/Beauty/item_emb.parquet`，其中 **`item_emb.parquet` 是 (259204, 64) —— 全量 Beauty（未 5-core）259204 个商品、64 维 content embedding**！
- 你的 TA-SID 是 **5-core 12101 商品、768d content + 128d behavior**

**所以主实验对比是两个不同管线之间的对比**：TIGER 原始（全量商品 + 64d content）vs 你的方法（5-core + 768d content）。差异包括 item 集合规模、content 维度、RQ-VAE 训练分布——**不是简单的"4096 vs 768"**，比那更严重。

**注意：** 你项目里 `Sports_t5_rqvae.npy` / `Toys_t5_rqvae.npy`（Sport/Toys 的 baseline code）现在**文件系统里已不存在**（TIGER 原始项目只剩 Beauty 数据）；Beauty 的还在（`Beauty_t5_rqvae.npy`，259204×4，来自全量数据 RQ-VAE）。

**结论（已执行，2026-08-03 出最终结果）：** 已在用户管线上重跑 content-only baseline + **3 seed（2025/2026/2027）均值±std**。**真实公平提升（同 768d 管线）：**

> ⚠️ **2026-09-09 provenance 订正（仅覆盖本表 Toys 行的旧口径转录）**：本表 Toys 行 0.0367±0.0006 / +9.2%
> 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**。
> 本文档为 2026-07-31 审计，早于 provenance 修复；其引用的 Toys 主结果已订正为 NDCG@20 **0.0377±0.0007** /
> 相对 content-only **+12.2%**。本横幅**不改动本文档其余审计论点**（维度混淆、Beauty seed 不一致等批评仍成立）。
> 权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

| 数据集 | Content-only baseline | TA-SID | 提升 |
|:--|:--:|:--:|:--:|
| Sport | 0.0275±0.0006 | **0.0322±0.0004** | **+17.1%** |
| Beauty | 0.0358±0.0007 | **0.0415±0.0009** | **+15.9%** |
| Toys | 0.0336±0.0006 | **0.0367±0.0006** | **+9.2%** |

三数据集一致为正，且 **3 seed 无任何交叉**（每个 seed TA-SID 都高于 baseline），std≤0.001。TIGER 原始 baseline（0.0207 Sport）比新 baseline（0.0275）低 ~33% → **证实旧主实验 +58% 大量来自管线差异**。ZeroPad 代理验证：baseline(0.0274, seed2025) ≈ ZeroPad(0.0286)，差 0.0012 → 896d 死维度与 768d 纯输入等价。

**论文主实验用 §1.5 统一 768d 口径 3 seed 表格。** 多 seed 已完成 ✅

**🔴 发现 B：Beauty baseline 的 seed cherry-picking**

- 同一个 `Beauty_t5_rqvae.npy`，seed 2025 跑出 **0.0379**，seed 42 跑出 **0.0270** —— 差 40%！
- 主实验表用了 0.0270（低的那版）和 seed42 的 TA-SID 0.0404，得到 +49.6%
- 而 manifest 里诚实记录的 768d baseline 是 0.0379 → +10.3%
- **审稿人复现 T5 训练（换 seed）就会打爆你。T5 训练方差这么大，必须报多 seed 均值±std（至少 3 seed），不能挑最低的 baseline**

**🟠 发现 C：Beauty 的 768d content 文件丢失，不可复现**

- `data/Beauty` 里没有 `item_emb_768d.parquet`，只有 896d 的 fused（说明 768d content 曾存在但被覆盖）
- 但 Beauty TA-SID 768d = 0.0418/0.0404 用的是 768d track
- 论文要求可复现，这个文件必须重新生成并归档

---

## 第三部分：逐个实验拷打（严格师兄会怎么问）

### 3.1 主实验（SASRec / TIGER / TA-SID / Random SID）

**拷打 1：基线公平性。** "你的 +58% 是 768d content 的 TA-SID 对比 4096d content 的 TIGER。请问 768d content 本身的 TIGER 是多少？"（现答不上来，因为没跑）

**拷打 2：随机性的报告。** "Beauty baseline 同一配置两个 seed 差 40%，你的主实验为什么只报一个 seed？"

**拷打 3：Random SID 消融在 Toys 上失效（+1%）。** "你说'码质量很重要'，但 Toys 上随机码和内容码几乎一样（0.0308 vs 0.0305）。你的论证是不是只对 Sport 成立？"

**拷打 4：你 claim '首次将转移动态引入 SID 生成'。** "ADC-SID 也融合了行为信号，虽然机制不同。'首次'这个词会被审稿人攻击，建议改成'据我们所知，尚无工作以商品-商品转移动态作为 SID 的显式输入'，并明确和 ADC-SID 的区别。"

### 3.2 Shuffled + ZeroPad 消融（已完成，2026-07-31 上午）

**最终结果（Sport）：**

| 配置 | NDCG@20 | R@20 | 相对 ZeroPad |
|:----|:-------:|:----:|:------------:|
| TIGER baseline (4096d content) | 0.0207 | 0.0497 | —（维度不同，仅参考） |
| **ZeroPad** (768d content + zeros) | **0.0286** | 0.0688 | —（≈768d content-only 代理） |
| **Shuffled** (768d content + 错配 behavior) | **0.0262** | 0.0643 | **-8.4%**（比 content-only 还差） |
| **TA-SID** (768d content + 真实 transition) | **0.0327** | 0.0781 | **+14.3%** |

**这个结果比预期更干净。** 预期是 "ZeroPad ≈ Shuffled < TA-SID"，实际是 **"ZeroPad > Shuffled < TA-SID"**：

1. **ZeroPad(0.0286) ≈ 768d content-only**（128 维零输入梯度恒为 0 = 死参数，等于没加）。证实**"增加维度"不贡献提升**（0.0286 远超 4096d TIGER 的 0.0207，恰恰因为 768d content 本身优于 4096d——这进一步坐实了发现 A 的混淆）。
2. **Shuffled(0.0262) < ZeroPad(0.0286)**：错配的 behavior 向量是**干扰**——RQ-VAE 被迫重建一个"不对应任何真实关系"的 128d 分量，浪费码本容量，比不加还差。这比"统计性质无贡献"更强：**无意义的 behavior 是负贡献**。
3. **TA-SID(0.0327) > ZeroPad(0.0286) 且 >> Shuffled(0.0262)**：transition 结构是唯一正向来源。**净贡献 +14.3%（vs content-only）/+24.8%（vs Shuffled）**。

**拷打 5：消融的"对照基线"不对。** 有效，需补跑真实 768d content-only。**但 ZeroPad(0.0286) 已是最好的代理**（零维度=死参数，逻辑等价），用它做消融基线完全自洽。

**拷打 6（旧版要改）：Shuffled 不是"有统计贡献"，而是负贡献。** 新叙事：Shuffled 证明"破坏对应关系的 behavior"不仅不帮忙还拖累，只有真实 transition 结构带来增益。**这个叙事比预期更强。**

**拷打 7：为什么只在 Sport 跑消融？** 答（要准备好）：成本考虑 + Sport 是效果最强、最需要解释的数据集；补充时优先跑 Toys（Toys 提升最小，若 Toys 上消融也成立则鲁棒性最强）。

> ⚠️ **诚实代价**：ZeroPad 同时揭示了主实验的真实净提升口径——
> TA-SID(0.0327) vs 768d content-only(≈0.0286) = **+14.3%**（Sport），而不是 +58%。+58% 混入了 "768d content 优于 4096d content" 的 38% 基础收益。**论文必须用 768d 对齐口径重新报告主实验。**

### 3.3 4096d 实验

**拷打 8：** "4096d 是你的 robustness 对照，但你没有跑 4096d 的 content-only baseline。" 4096d TA-SID (0.0276) vs TIGER(0.0207, 4096d content) = +33.3% —— **这个对比反而是维度一致的！** 所以 4096d track 是干净的对照，恰好可以作为"公平对比"的另一条证据线。

### 3.4 L_trans (Track B)

**拷打 9：** "你加了个对比损失结果无效，为什么？" 
答：
1. 信号冗余：行为信号已经通过融合输入进了 RQ-VAE，L_trans 是重复约束
2. 目标冲突：L_recon 要求向量彼此可分（精确重建），L_trans 要求转移对聚拢 → 拉近商品反而增加重建难度
3. 证据：B-1 碰撞率从 10.66% 降到 0.34%（码被聚拢）但 NDCG 反降 6.8%
4. **负面结果怎么写要小心**：可以说"融合的间接路径已足够，直接约束既不必要还伤重建"。不要回避。

### 3.5 正交性 / 转移一致性 / 长尾 / t-SNE

**拷打 10（正交性）：** Jaccard=0.02、ρ=0.02 是"内容与行为编码完全不同信息"。审稿人问："这不正说明 behavior 是独立的、与 content 无关的信号？那 content+behavior 融合的收益来自哪？" 答：正因为正交，融合才不是冗余相加；TA-SID 超越两者就是补上了 content 缺的那块。

> ⚠ **2026-09-29 更新：本条前提已失效。** 上述正交性分析**已整节删除**（数字不可溯源 + 基于弃用的 4096d + "低相关⟹互补" 逻辑不成立，见 `paper_outline_ta_sid.md` §7.5 裁撤说明）。论文**不再对两通道嵌入独立性做统计学主张**；若审稿人问"behavior 是否只是 content 的冗余副本"，**改用 §7.2 四臂消融直接回答**（内容固定、只替换行为通道：Content-only 0.0275 / ZeroPad 0.0286 / Shuffled 0.0262 / **TA-SID 0.0327**）——零向量与错配向量都不涨，只有真实行为结构涨，这就是非冗余性的因果证据。

**拷打 11（转移一致性）：✅ 已补 TIGER 对照（2026-08-01），结果与指标设计教训如下**

- 对比在**量化空间**（SID code 重建 x_q）进行：TA-SID vs content-only baseline，结果见 experiment_results_summary.md §5.3
- **绝对 cos 对 TA-SID 不利**（Beauty/Toys 略低于 baseline）——但这正是指标陷阱：content-only 的量化空间把商品"挤在一起"（随机对 cos 0.65），绝对 cos 虚高
- **改用相对增益**（转移对 cos − 随机对 cos）后反转：**TA-SID 是 content-only 的 1.5~2.7 倍**（Beauty 0.370 vs 0.136, Sport 0.477 vs 0.317, Toys 0.394 vs 0.241）
- **论文必须报告相对增益**，并解释为什么绝对 cos 不适用（码本分散度混淆）
- 复现：`python3 tools/sid_consistency_contrast.py`

**拷打 12（长尾/冷启动）：** 你诚实写了冷启动≈0。审稿人接受这个限制，但你要引用 LIGER 作为已知解法并说明"TA-SID 与 LIGER 正交可结合"。**LIGER 论文还没读，这是硬伤。**

---

## 第四部分：发论文要达到什么水平（今天要跟师兄对齐的）

### 4.1 以现在的状态，能不能投？

**我的判断：实验有真东西，但目前表格经不起审稿人 10 分钟的复现拷打。不能直接投。** 至少要补：

| 优先级 | 补什么 | 工作量 | 为什么必须 |
|:-----:|:------|:------:|:----------|
| 🔴 1 | **768d content-only 的 TIGER baseline（3 数据集）** | 3×~1h | 主实验公平性的命门。没有它 +58% 站不住 |
| 🔴 2 | **多 seed 报告（至少 3 seed 均值±std）** | 重跑 T5 为主 | Beauty 两 seed 差 40%，单 seed 会被打爆 |
| 🟠 3 | 消融基线改为 ZeroPad/768d content-only 口径 | 无（已有数据） | 逻辑自洽 |
| 🟠 4 | 转移一致性加 TIGER 对照 | ~5min | 让"转移被编码"成立 |
| 🟠 5 | 读 LIGER，写好冷启动的 related work + 未来工作 | 半天 | 审稿人必问 |
| 🟡 6 | Beauty 768d content 文件重新生成归档 | ~1h | 可复现性 |
| 🟡 7 | (可选) Toys 上也跑 Shuffled/ZeroPad | ~1天 | 消融泛化性 |

### 4.2 论文贡献的"合理 claim"（当前证据下）

写论文时，你的核心 claim 应该从：
- ❌ "TA-SID 相对 TIGER 提升 +58%"
- 改为：
- ✅ "在**统一内容特征维度**下，融合商品转移动态的 SID 一致优于纯内容 SID（Sport +33%，Toys +4%，Beauty +10%），且通过 Shuffled/ZeroPad 消融证明增益主要来自 transition 结构而非维度或统计性质（Sport 上 TA-SID 较 Shuffled 再提升 +25%）。"

+58% 是"维度不一致的乐观数"，审稿人一旦发现就会被全盘质疑。诚实的 +10%~33% 加上干净的消融链条，是更稳的叙事。

### 4.3 你作为第一作者必须掌握的清单（今天交流前自查）

能在 30 秒内讲清，且师兄问不倒：
- [ ] 为什么用 PPMI 而不用共现计数？（流行度偏差）
- [ ] 为什么用 SVD 而不用随机游走/DeepWalk？（降维+去噪+低秩泛化，对比过吗？没对比要能说"这是经典做法"）
- [ ] 为什么 128d？（经验值，讲得清理由）
- [ ] 为什么 LN 各自做再 concat？（尺度不匹配）
- [ ] 为什么 3 层码本而不是 TIGER 论文的 4 层？（**你的 num_emb_list=[256,256,256] 是 3 层，论文大纲却写 4 层！这个不一致要么改代码要么改文档，且要确认 TIGER baseline code 是几层——如果 baseline 是 4 层而你是 3 层，又是不公平对比！**）
- [ ] 768d vs 4096d 为什么 768 更好？（MLP 瓶颈 + 码本稀释）
- [ ] T5 到底在学什么？（历史 SID → 下一 SID 的转移规律）
- [ ] Random SID 为什么 Sport 暴跌 (-55%) 但 Toys 几乎不变？(Sport 稀疏，码质量敏感性高)
- [ ] L_trans 为什么无效？（信号冗余 + 目标冲突）
- [ ] Shuffled 为什么有 +19%？（统计性质也是特征）
- [ ] 冷启动为什么≈0？LIGER 怎么解决的？（还没读 LIGER！）
- [ ] 你的方法和 ADC-SID / RPG / LIGER 的核心区别？（一句话说清）

---

## 第五部分：今天和师兄交流的提纲（建议直接照这个聊）

1. **先摊牌数据审计结果**（发现 A/B/C），问师兄：
   - "768d content-only 的 TIGER baseline 是不是必须补？补完主实验数字会从 +58% 掉到约 +33%，这样的提升还有说服力吗？"
   - "Beauty baseline 两 seed 差 40%，是否必须报多 seed？"
2. **问师兄对 Shuffled +19% 的判断**：这个中间结果怎么解读、怎么放进消融表
3. **对齐论文定位**：TA-SID 的 novelty 到底是什么？（简单的转移动态注入 vs ADC-SID 复杂自适应融合——"简单性"算不算贡献？师兄的圈子认不认？）
4. **对齐投稿目标**：SIGIR/CIKM 还是更顶？决定了消融要补到什么程度
5. **求一个优先级排序**：7 件待办里，哪 3 件对投稿最重要？

---

## 附录：后台实验状态（2026-07-31 中午，全部完成）

- 消融 Shuffled + ZeroPad：**全部完成，无 ERROR，exit 0**
  - **ZeroPad**：0.0286（epoch 80 早停）← 768d content-only 代理
  - **Shuffled**：0.0262（epoch 97 早停）
- 完整对比表（Sport）：
  ```
  768d content-only ≈ ZeroPad   0.0286   （维度不贡献）
  + 错配 behavior (Shuffled)      0.0262   ← 负贡献（干扰码本）
  + transition 结构 (TA-SID)     0.0327   ← 净贡献 +14.3% (vs content) / +24.8% (vs Shuffled)
  ```
- **下一步（优先级）**：① 补 768d content-only 真实 baseline（3 数据集）→ ② 用 768d 对齐口径重写主实验 → ③ 多 seed → ④ 决定投稿目标
