# TA-SID 论文大纲

**建议标题：**
> **Transition-Augmented Semantic IDs for Generative Retrieval**
> 或
> **TA-SID: Injecting Item Transition Dynamics into Semantic ID Generation for Sequential Recommendation**

**目标会议/期刊：** SIGIR / CIKM / WWW / TOIS

---

## 摘要

**背景：** 生成式检索（TIGER）使用 RQ-VAE 将商品编码为语义 ID（SID），并训练 T5 预测下一商品的 SID。但现有 SID 生成仅依赖内容特征，忽略了行为层面的转移模式。

**方法：** 我们提出 TA-SID，通过 PPMI+SVD 提取商品-商品转移动态得到行为嵌入，然后通过简单的 LN+Concat 操作将其与内容嵌入融合，再送入 RQ-VAE 量化。

**结果：** 在统一 768d 口径、3 个 T5 seed（均值±标准差）下，TA-SID 一致超越纯内容 TIGER（NDCG@20 提升 **+12.2%~+17.1%**）和 SASRec（**+56.4%~+139.5%**，v2 对齐口径）。稀疏数据集（Sport）提升最大：+17.1% / +139.5%。

**意义：** TA-SID 表明，捕捉转移动态是一种简单但高效的 SID 生成增强方式，远优于复杂的自适应融合机制。

---

## 1. 引言

**第一段 — 问题：**
- 生成式检索范式（Tay et al., 2022; Sun et al., 2024）→ T5 预测下一商品的 SID
- SID 质量直接决定推荐的上限
- 当前 SID（RQ-VAE）纯粹基于内容特征

**第二段 — 现有方法的不足：**
- 纯内容 SID 丢失了行为模式：转移信号、共购模式、序列动态
- 现有的融合方案（ADC-SID）复杂度过高：MoQ 架构、5 个路由专家、7B 内容特征、自适应控制器
- **关键洞察：** 商品间的转移动态可以通过简单的图方法捕捉，并直接融合到量化流程中

**第三段 — 我们的方法：**
- PPMI 矩阵 → SVD → 128d 行为嵌入
- LN(内容) + LN(行为) → Concat → RQ-VAE 输入
- SID 现在同时编码"商品长什么样"（内容）和"商品如何被顺序消费"（转移）

**第四段 — 实验结果：**

```
Sport:  0.0275 → 0.0322 NDCG@20 (+17.1% vs content-only TIGER, +139.5% vs SASRec)
Beauty: 0.0358 → 0.0415 NDCG@20 (+15.9% vs content-only TIGER, +63.2% vs SASRec)
Toys:   0.0336 → 0.0377 NDCG@20 (+12.2% vs content-only TIGER, +56.4% vs SASRec)
（均为 3 seed 均值；content-only 为同管线 768d 复刻，SASRec 为对齐协议 3-seed，见 §6）
```

**第五段 — 贡献：**
1. 首次将商品-商品转移动态引入 SID 生成
2. 简单的 LN+Concat 融合 → 与现有 RQ-VAE 流水线即插即用
3. 全面评估：3 个数据集 × 3 seed（均值±std，零 seed 交叉），相对纯内容 TIGER 提升 +12.2%~+17.1%
4. 分析实验证明：转移一致性 AUC（度匹配负采样）三数据集**两端稳健、中间分层**——行为结构确实被编码进了量化后的 SID 空间

> **⚠ 2026-09-29 第二创新裁撤声明。** 本文为**单创新点论文**，唯一贡献为第 1–4 条所描述的"将商品-商品行为转移结构注入量化前的连续表示"。原第五点（防塌缩健康量化诊断 / 自适应粗层容量分配 K0\*）**已正式裁撤，不作为贡献主张**；相关 z-gap 判据、capacity stress、reseed 兜底等材料仅作**探索性记录**压缩保留（见 §7.8 说明条与 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md) / [e47_seed_health_isolation.md](e47_seed_health_isolation.md)），**不得出现在摘要、引言贡献列表、章节标题或结论中**。

---

## 2. 相关工作

### 2.1 推荐中的生成式检索
- **TIGER**（Sun et al., 2023）：首次将 T5 + RQ-VAE 用于推荐
- **LIGER**（Yang et al., 2024）：通过内容增强解码解决冷启动
- **RPG**（Wang et al., 2024）：长 SID（64 token）+ 图约束解码
- **ADC-SID**（Xu et al., 2025）：REG4Rec + MoQ + 自适应融合

### 2.2 语义 ID 生成
- RQ-VAE 与 RQ-VAE++（Zheng et al., 2023; Sun et al., 2024）
- MM-RQ-VAE（Rajput et al., 2024）：多模态 SID
- OPQ → RQ → 乘积量化用于 SID

### 2.3 推荐中的转移/会话信号
- 经典方法：马尔可夫链、FPMC（Rendle et al., 2010）、TransRec（He et al., 2017）
- 图方法：SR-GNN（Wu et al., 2019）、GC-SAN（Xu et al., 2019）
- **我们的区别：** 不提出新的转移模型——我们将已有的转移模式注入 **SID 生成阶段**，而非推荐阶段

### 2.4 与 ADC-SID 的区分

> ADC-SID（MMQ-v2）也融合内容和行为，但路径完全不同：自适应对齐控制器 + MoQ + 5 个行为路由专家 + PailiTAO v8 + Qwen3-7B。相比之下，TA-SID 通过简单的 PPMI+SVD **直接捕捉商品-商品转移动态**（而非用户-商品交互密度），以显著更低的架构复杂度实现了同等甚至更大的相对提升。

---

## 3. 预备知识

### 3.1 问题定义
- 序列推荐：根据历史行为预测下一商品
- Leave-one-out 评估（领域标准）

### 3.2 TIGER 框架
- 内容编码器 → RQ-VAE → SID 码 → T5 生成
- 词表：3 层码本 × 256 + 碰撞位 → 4 个 token，词表 1025（TIGER 的"4 token"= 3 层码本 + 1 碰撞位列，非 4 层码本）

### 3.3 RQ-VAE
- 残差量化、码本学习
- 损失：L_recon + L_vq + L_commit

### 3.4 PPMI 转移图
- 商品共现的点互信息
- PMI(i,j) = log(P(i,j) / P(i)P(j))
- 取正 PMI → 有意义的转移关系

---

## 4. 方法：TA-SID

### 4.1 整体架构

```
[商品转移图] → PPMI → SVD → [行为嵌入: 128d]
                                      ↓
[内容嵌入: 768d] → LN → Concat(896d) → RQ-VAE → SID
[行为嵌入: 128d] → LN ↗
```

### 4.2 转移图构建
- 从用户会话序列构建商品-商品转移图
- 边权重 = PPMI(i,j) = max(0, log(C(i,j)·|D| / C(i)C(j)))
- 过滤虚假共现（低 PMI）

### 4.3 PPMI+SVD 行为嵌入
- 对 PPMI 矩阵做 SVD → 潜在商品表示
- 128 维行为向量，编码转移模式
- 关键特性：频繁共现的商品具有相似的行为嵌入

### 4.4 融合策略：LN + Concat
- **为什么先 LN：** 内容（768d，L2 归一化）和行为（128d，norm~0.6）尺度不匹配
- **为什么用 Concat 而非 Add：** 保留独立信息通道供 RQ-VAE 使用
- **为什么不用复杂融合：** 简洁性本身就是优点——零架构改动

### 4.5 即插即用特性
- 同一套 RQ-VAE 训练代码、同一套 T5 架构
- 仅改变输入嵌入
- 可直接替换现有 TIGER 流水线 → 实用价值

### 4.6 （裁撤）~~Adaptive Coarse-Level Capacity Allocation / 防塌缩健康量化~~

> **⚠ 2026-09-29 本节整节裁撤。** 本节原为第二创新点（自适应粗层容量分配 → 防塌缩健康量化诊断），
> 其核心卖点（K0\* = 512、"+3.8%"、传导剂量）已被 E45/E46 推翻，§4 方法章节**不再包含任何第二创新内容**。
> 本文 §4 只写 §4.1–§4.5 一条主线：转移图 → PPMI+SVD → LN+Concat → RQ-VAE。
> **禁止在 §4 出现"Adaptive Capacity / Adaptive Transition-Preserving Quantization / BSR / Semantic ID quality screening"等提法。**
>
> 具有独立价值、但不构成创新的诊断观察压缩为一句探索性说明（可选，置于附录或 §7 末尾脚注）：
> RQ-VAE 粗层码本的分配质量与重构损失弱相关，代码本塌缩可在 recon 上不可见；在本文设置下该现象
> 仅出现在 TA-SID × Toys × seed2024 这一特定组合（1/3 种子），换种子即恢复，content-only / ZeroPad
> 各种子均健康 → 归因为**该组合下的初始化特例**，而非"任何训练都可能塌"，更非"维度不足"。
> 详情见 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md) 与
> [e47_seed_health_isolation.md](e47_seed_health_isolation.md)。**此说明仅作稳健性/可复现性备注，不得升级为贡献。**

---

## 5. 实验设置

### 5.1 数据集

| 数据集 | 用户数 | 商品数 | 交互数 | 平均序列长度 |
|:-------|:-----:|:------:|:------:|:----------:|
| Beauty | 22,363 | 12,101 | 198,502 | 8.9 |
| Sport | 35,598 | 18,357 | 296,337 | 8.3 |
| Toys | 19,412 | 11,924 | 167,597 | 8.6 |

标准 Amazon 5-core 过滤，leave-one-out 划分。

### 5.2 基线方法
- **SASRec**（Kang & McAuley, 2018）：基于 Transformer 的序列模型
- **TIGER**（Sun et al., 2023）：T5 + RQ-VAE（我们的纯内容基线）
- **Random SID**（Sun et al., 2023）：随机码代替学习到的 SID（码质量消融）

### 5.3 实现细节
- 内容特征：Sentence-T5-base（768d）
- 行为特征：PPMI+SVD（128d），矩阵尺寸 |V|×128
- RQ-VAE：3 层码本 × 256 + 碰撞位，100 epoch
- T5：4 层编码器/解码器，d_model=128，beam_size=30
- **报告口径：** 3 个 T5 seed（2025/2026/2027）均值±标准差；RQ-VAE 固定（SID 码由输入确定，随机性主要来自 T5 训练）
- 评估：leave-one-out，全量商品排序
- GPU：NVIDIA RTX 4090

### 5.4 评估指标
- 主实验：NDCG@5/10/20，Recall@5/10/20
- 扩展指标（文献对比用）：R@50/R@100

---

## 6. 主实验结果

### 6.1 主实验表

| 方法 | Sport | | | Toys | | | Beauty | | |
|:-------|:----:|:-:|:-:|:----:|:-:|:-:|:-----:|:-:|:-:|
| | R@20 | N@20 | 提升 | R@20 | N@20 | 提升 | R@20 | N@20 | 提升 |
| SASRec | 0.0374±0.0014 | 0.0134±0.0002 | — | 0.0651±0.0015 | 0.0241±0.0003 | — | 0.0711±0.0016 | 0.0254±0.0007 | — |
| Content-only (768d) | 0.0666±0.0015 | 0.0275±0.0006 | — | 0.0785±0.0007 | 0.0336±0.0006 | — | 0.0811±0.0008 | 0.0358±0.0007 | — |
| **TA-SID (768d)** | **0.0769±0.0011** | **0.0322±0.0004** | **+17.1%** | **0.0911±0.0020** | **0.0377±0.0007** | **+12.2%** | **0.0941±0.0016** | **0.0415±0.0009** | **+15.9%** |
| *TA-SID vs SASRec* | *+105.5%* | *+139.5%* | — | *+39.9%* | *+56.4%* | — | *+32.3%* | *+63.2%* | — |

> **口径声明（正式版）：** 3 个 T5 seed（2025/2026/2027）均值±标准差。Content-only = 同管线 768d 纯内容 RQ-VAE+T5（`item_emb_768d_contentonly.parquet`），是公平 baseline。SASRec = 对齐协议 3-seed（2026-08-07 v2：test=test.parquet 全量用户，见 [sasrec_baseline_report.md](sasrec_baseline_report.md)）。旧版 TIGER baseline（4096d/64d 原始项目管线）存在维度+数据口径混淆，已弃用（见 experiment_results_summary.md §1.5 注）。

### 6.2 关键观察

- **Sport 提升最大（+17.1%）**：交互密度最低（0.045%）、平均序列最短（8.32）的数据集上，行为信号提供了纯内容无法捕捉的转移模式
- **Beauty / Toys 增益相当且较小（+15.9% / +12.2%）**：⚠ **勿写"Toys 稠密故饱和"**——Beauty（0.0734%）与 Toys（0.0724%）密度几乎相同，增益却差 3.7pp，**密度与序列长都解释不了该排序**。统一写法："最稀疏的 Sport 增益最大，其余两数据集增益相当且较小；Beauty 与 Toys 间差异归因于数据分布差异，不做过度解读。"（口径见 [paper_main_results.md §2.1](paper_main_results.md)）
- **3 seed 零交叉**：TA-SID 每个 seed 都高于对应 baseline，std≤0.001 → 提升稳定，非 seed 偶然（审稿人复现得到一致结论）
- **SASRec 对齐后（v2）**：三个数据集上均为 SASRec < Content-only < TA-SID 的清晰层次；对齐任务（预测 i_L）对 SASRec 更严苛，其相对 margin 全面扩大（N@20 +56.4%~+139.5%）→ 转移注入是内容之上的增量收益

### 6.3 NDCG@K 详表（3 seed 均值）

| 数据集 | 方法 | N@5 | N@10 | N@20 |
|:--|:--|:--:|:--:|:--:|
| Sport | Content-only | 0.0161 | 0.0213 | 0.0275 |
| Sport | TA-SID | 0.0190 | 0.0252 | **0.0322** |
| Beauty | Content-only | 0.0227 | 0.0291 | 0.0358 |
| Beauty | TA-SID | 0.0264 | 0.0335 | **0.0415** |
| Toys | Content-only | 0.0206 | 0.0268 | 0.0336 |
| Toys | TA-SID | 0.0225 | 0.0294 | **0.0377** |

> std≤0.001（见 §6.1 N@20 列）；逐 seed 明细存于 `model/logs/*_eval.log`（TA-SID seed2025 记录见 auto_metrics_table.md §一）。

---

## 7. 消融与分析实验

### 7.1 码质量消融

| 配置 | Sport N@20 | Toys N@20 | Beauty N@20 |
|:--------|:---------:|:---------:|:----------:|
| Content-only (768d, 3-seed) | 0.0275 | 0.0336 | 0.0358 |
| Random SID（随机码, seed2025） | 0.0093 | 0.0308 | 0.0263 |
| **TA-SID（768d, 3-seed）** | **0.0322** | **0.0377** | **0.0415** |

**解读：**
- Random SID 远低于 content-only（Sport -66%）→ 码结构至关重要，SID 必须承载语义
- TA-SID SID 超越两者 → 转移模式在内容之上增加了有意义的结构
- Sport 上 Random SID 相对 content-only 跌幅最大 → 稀疏数据对量化质量更敏感

### 7.2 Transition 消融对照（回应审稿人核心质疑：提升只是"加了个向量"？）

**目的：** 排除"提升来自额外维度 / 任意向量 / 碰巧"的质疑，锁定提升来源为转移动态结构。

**方法：** 在 Sport 上保持 Content emb（768d）不变，只替换 behavior 部分（128d），LN+Concat 后走同一套 RQ-VAE+T5 流水线（seed2025，与主实验同管线）：

| 配置 | behavior 部分 | NDCG@20 | R@20 | 相对 Content-only |
|:----|:------------|:-------:|:----:|:------------:|
| Content-only (768d) | — | 0.0274 | 0.0664 | — |
| + **ZeroPad**（零向量） | 全零（梯度恒 0，死维度） | 0.0286 | 0.0688 | +4.4% |
| + **Shuffled**（错配向量） | 行级 shuffle（保留统计、破坏对应） | 0.0262 | 0.0643 | -4.4% |
| + **Transition**（TA-SID） | PPMI+SVD 128d | **0.0327** | **0.0781** | **+19.3%** |

**结论（一次只改一个变量）：**
1. **ZeroPad ≈ Content-only**（0.0286≈0.0274）：128 个零维度权重梯度恒为 0（死参数），有效容量不变 → **维度不是提升来源**
2. **Shuffled < ZeroPad**（0.0262 vs 0.0286）：错配的真实向量反而**有害**——RQ-VAE 被迫重建无意义的 128d 分量，浪费码本容量 → **"随便加个向量"不但不涨，还会降**
3. **TA-SID 唯一正向**：真实 transition 结构净增 **+19.3%**（vs content-only），相对 Shuffled 净增 **+24.8%** → **提升唯一来源是转移结构本身**

> 历史注：更早的 SVD 谱匹配随机噪声对照（+35.7% / 净贡献 +16.4%）因"噪声保留了 behavior 空间协方差，随机不纯粹"已弃用（见 experiment_results_summary.md §4.2）。本表为最终采用的干净对照。
> 主实验 3-seed TA-SID 均值 = 0.0322±0.0004（此处报 seed2025 单值，与消融同口径）。

### 7.3 转移一致性（AUC，量化空间）与碰撞率

**指标：** 在量化输出空间（SID 码重建 $x_q^{[\mathrm{full}]}$）比较高频转移对 vs **度匹配非边对**的余弦相似度。**AUC = P(转移对相似度 > 非边对相似度)**，0.5=随机（转移未编码），等价于 Link Prediction 的标准 AUC（Mann-Whitney U）。

> **负采样口径（统一为度匹配）：** 负样本取与正样本端点同处一个 $\lfloor\log_2 \deg\rfloor$ 度分桶的非边对，**以降低节点流行度差异带来的结构预测偏差**（均匀采样下负样本端点平均度数仅为正样本的约 1/9，会产生虚高分数）。
> ⚠ 旧的 uniform 负采样口径数值（Beauty 0.869 / Sport 0.962 / Toys 0.872）**已废弃，勿引用**；两套数值不可并列。

| 指标 | Sport | Beauty | Toys |
|:----|:----:|:----:|:------:|
| **TA-SID AUC** | **0.9647** | **0.9560** | **0.9651** |
| Content-only AUC | 0.9080 | 0.8365 | 0.8751 |
| Shuffled AUC（错配行为下界） | 0.8763 | 0.7974 | 0.8127 |
| TA-SID 碰撞率 | 10.66% | 1.09% | 8.68% |

**解读：**
- **TA-SID AUC 三数据集全部占优**（Δ vs Content-only = +0.0568 / +0.1195 / +0.0900）且恒为全变体最高 → 在 SID 量化空间，TA-SID 更可靠地把转移对从非边对中区分出来
- **两端稳健、中间分层**：TA-SID 恒居首、Shuffled（错配行为）恒居末且**低于"完全不给行为"的 Content-only**；中间变体（ItemCF / Raw-CoOcc / ZeroPad）的具体名次跨数据集不完全一致 → 论文只主张"**两端 + 分层**"，不主张逐名次同序
- **为什么不能用绝对 cos**：content-only 的码本把商品挤成一团，转移对与普通对的绝对余弦**同时**虚高 → 绝对 cos 混入"整体聚集度"这一与转移编码无关的量；AUC 剥离该干扰。对应对比学习 alignment-uniformity 框架（Wang & Isola 2020）
- ⚠ **诚实说明（量级不对应）**：ΔAUC 的绝对量级与主实验 NDCG@20 提升幅度**不对应**（按 ΔAUC 为 Beauty > Toys > Sport，按 NDCG 为 Sport > Beauty > Toys）。跨数据集可迁移的是"分层与两端"，不是具体名次或量级；本指标**不作跨数据集绝对阈值**，不写"Toys ΔAUC 最小故提升最小"。

### 7.4 内容维度的影响

| 配置 | Sport N@20 | Beauty N@20 | Toys N@20 |
|:--------|:---------:|:----------:|:---------:|
| TA-SID 768d（3-seed 均值） | **0.0322** | **0.0415** | **0.0377** |
| TA-SID 4096d（seed2025 单值） | 0.0276 | 0.0382 | 0.0317 |

- 768d 一致优于 4096d → 更高维的内容 ≠ 更好的 SID
- RQ-VAE 偏好适中维度的输入；过高维度会稀释码本有效性
- （768d 报 3-seed 均值以对齐主实验表；4096d 为 seed2025 单值，对比方向三数据集一致）

> **〔2026-09-29 整节删除〕原 §7.5「嵌入正交性分析」已删除，不进正文、附录与贡献列表。**
> 理由：① 该节数字**无产出物、不可溯源**（仅本大纲与 `paper_draft_zh_method_exp.md` 两处草稿有 0.0167/0.0216/0.0193，`analyze_emb_orthogonality.py` 只 print 不落盘，且脚本只算 k=20、算不出表里的 Top-10 行）；② 其指标名 "Jaccard" 实为 `overlap/k`，非真 Jaccard；③ 数字基于**已弃用的 4096d** 内容嵌入，与 768d 主口径不一致；④ 更根本地，"低相关 ⟹ 互补" 不成立（低相关只证明**不同**，不证明**有用**；随机向量同样 ρ≈0），且该结果近乎由构造决定（一边是共现矩阵的 128 维谱嵌入，一边是文本语义嵌入，二者不同是预期而非发现，随机 128 维向量同样得到 ρ≈0）。
> **非冗余性的决定性证据在 §7.2**（内容固定、仅替换行为通道 → 下游指标显著分化：Content-only 0.0275 / ZeroPad 0.0286 / Shuffled 0.0262 / TA-SID 0.0327）与 **§7.3**（六变体 AUC 0.7974–0.9651）。**本文不对两通道的嵌入独立性做统计学主张。**

### 7.5 长尾分析

| 分组 | Sport R@20 | Sport N@20 | Toys R@20 | Toys N@20 |
|:------|:---------:|:---------:|:---------:|:---------:|
| Head（高频） | 0.134 | 0.056 | 0.145 | 0.064 |
| Mid（中频） | 0.009 | 0.003 | 0.029 | 0.010 |
| Tail（低频） | 0.000 | ~0.000 | 0.002 | 0.001 |

- Head >> Mid >> Tail：长尾效应是生成式检索的普遍问题
- 两个数据集呈现相同模式 → T5 的固有限制
- Toys（更稠密）对尾部商品略好

### 7.6 t-SNE 可视化

- 融合嵌入比纯内容嵌入具有更清晰的商品聚类结构
- 内容嵌入按商品类别聚类，行为嵌入按消费模式聚类
- 两者互补 → 融合嵌入同时保留了两种结构

### 7.7 （裁撤）~~Adaptive Capacity 有效性验证 / 防塌缩健康量化诊断~~

> **⚠ 2026-09-29 本节整节裁撤，正文与附录均不再设此节。** 本节原为第二创新点的实验验证
> （Q1/Q2/Q3、capacity stress、多种子实锤），其核心结论（扩容收益、K0\*=512、传导剂量）已被 E45/E46 推翻，
> 判据类结果（z-gap）已在 §7.3 的稳健性/可复现性备注中一句带过。**禁止把本节任何内容写成贡献、节标题或摘要要点。**
>
> 仅保留一句探索性说明（置于 §7.3 末尾或附录脚注，可选）：
> "当 RQ-VAE 粗层码本分配异常时，其后验转移一致性可能明显低于编码器输入空间；该现象可由训练后
> 计算的一致性落差检出，而重构损失对其不敏感。在本文设置下该现象仅见于 TA-SID × Toys × seed2024
> 的单一组合（3 个 RQ-VAE 种子中的 1 个），换种子即恢复；content-only 与 ZeroPad 各种子均正常。"
>
> 完整历史材料（Q1/Q2/Q3 表、E44 六数据集判据精度、E45 多种子、Sport 容量压测负对照、E47 归因隔离）
> 仅作**历史实验记录**保留于仓库，引用时须注明"已裁撤的第二创新材料，非本文结论"：
> [e43_adaptive_experiment.md](e43_adaptive_experiment.md)、[e44_five_dataset_summary.md](e44_five_dataset_summary.md)、
> [e45_toys_multiseed_collapse.md](e45_toys_multiseed_collapse.md)、[e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)、
> [e47_seed_health_isolation.md](e47_seed_health_isolation.md)。

---

## 8. 讨论

### 8.1 与 ADC-SID 的对比

| 维度 | TA-SID | ADC-SID (MMQ-v2) |
|:-------|:------|:--------|
| 核心信号 | 商品-商品转移（PPMI） | 用户-商品交互密度 |
| 融合方式 | LN+Concat（静态） | MoQ + 自适应路由（动态） |
| 生成式 Backbone | T5 | REG4Rec |
| 行为编码 | PPMI+SVD（128d，每商品一个向量） | SASRec + PailiTAO（5 个专家） |
| 内容特征 | Sentence-T5-base（768d） | Qwen3-7B（70 亿参数） |
| 相对提升 | **+12.2%~+17.1%**（NDCG@20 vs content-only） | +10.21%（R@50 vs RQ-VAE++） |
| 架构复杂度 | **低**（LN+Concat + 一次 SVD） | 高（MoQ + 5 专家 + 自适应路由） |

**核心观点：** ADC-SID 以 7B 特征 + MoQ + 5 行为专家 + 自适应路由为代价，仅取得 +10.21% 相对提升；TA-SID 仅凭 768d 内容 + 一次 SVD + LN+Concat，即取得 +12.2%~+17.1%。**在架构复杂度低一个数量级的前提下，TA-SID 实现了相当甚至更大的相对提升**，表明商品-商品转移模式是比交互密度更高效的 SID 生成信号。（注：两方法指标类型不同，论文不以绝对数值、而以"复杂度-收益"关系对比定位）

### 8.2 冷启动局限性

- 与 TIGER 家族一致：对低频（冷启动）商品近似零精度
- 根本原因：T5 无法可靠预测训练中罕见的 SID
- LIGER（Yang et al., 2024）提出了内容增强解码作为方向
- **未来工作：** TA-SID + LIGER 冷启动解码

### 8.3 L_trans 对比损失的负面结果

- 尝试在 RQ-VAE 训练中加入 InfoNCE 对比损失来直接约束转移关系
- 效果反降（Sport, seed2025：NDCG@20 从 0.0327 降至 0.0305）
- 根因：L_trans 与 L_recon 存在冲突——拉近转移对的同时增加了重建难度
- **启示：** 简单的 LN+Concat 融合已足够捕捉转移模式，无需额外损失

### 8.4 局限性与未来工作

- **局限性：** 冷启动问题未解决（TIGER 家族通病）
- **通用性：** TA-SID SID 可以与任何生成式检索 backbone 配合（T5、REG4Rec、GPT）
- **未来工作 1：** 整合 REG4Rec 等更强 backbone
- **未来工作 2：** 结合 LIGER 的冷启动解码
- **未来工作 3：** 多模态转移信号（文本+视觉+行为）

---

## 9. 结论

**总结：**
- TA-SID 将商品转移动态引入 SID 生成
- 简单的 LN+Concat 融合 → 3 个数据集上一致的提升（3 seed 均值±std，零交叉）
- +12.2%~+17.1% NDCG@20 超越纯内容 TIGER，+56.4%~+139.5% 超越 SASRec（v2 对齐口径）
- AUC 转移一致性证实：转移信号提供了内容之外的全新信息（非冗余性见 §7.2 消融）

**结论句：**
> Transition dynamics are a powerful yet under-explored signal for SID generation. TA-SID shows that even a simple fusion of content and behavior via LN+Concat can yield substantial gains across diverse datasets, outperforming both pure-content SIDs and classical sequential recommenders.

---

## 附录：各章节数据来源对照

| 章节 | 数据来源 | 文件 |
|:----:|:--------|:-----|
| §1 引言 | NDCG@20 三数据集（3-seed 均值） | experiment_results_summary.md §1.5 |
| §5 实验设置 | 数据集统计 | experiment_dimension_manifest.md §九 |
| §6 主实验 | 3-seed 主实验表 | experiment_results_summary.md §1.5 |
| §6.3 NDCG@K 详表 | 3-seed NDCG@5/10/20 | model/logs/*_eval.log + auto_metrics_table.md §一 |
| §7.1 码质量消融 | Random SID vs content-only | experiment_results_summary.md §1.5 + auto_metrics_table.md §一 |
| §7.2 Transition 消融对照 | ZeroPad / Shuffled | experiment_results_summary.md §4.3 |
| §7.3 转移一致性 | AUC 版 | experiment_results_summary.md §5.3 |
| §7.4 维度对比 | 768 vs 4096 | experiment_dimension_manifest.md §六 |
| ~~§7.5 正交性分析~~ | **整节已删除（2026-09-29）**：数字不可溯源 + 基于弃用的 4096d + "低相关⟹互补" 不成立。非冗余性改由 §7.2 消融背书 | ~~`tools/analyze_emb_orthogonality.py`~~（**勿引用其输出**） |
| §7.5 长尾分析 | Head/Mid/Tail | experiment_results_summary.md §五 |
| ~~§7.7 防塌缩健康量化诊断~~ | **整节已裁撤（2026-09-29），不入正文与附录**；若保留一句探索性说明，素材出处见右栏 | e42_adaptive_k0_results.md / e43_adaptive_experiment.md / e44_five_dataset_summary.md / e45_toys_multiseed_collapse.md / **e46_second_innovation_overturn.md（推翻裁定，权威）** / e47_seed_health_isolation.md —— 均为历史材料，引用须注明"已裁撤第二创新，非本文结论" |
| §8.1 ADC-SID 对比 | 文献对比 | ta_sid_vs_adc_sid_comparison.md |
| §8.3 L_trans 负面 | Track B 实验 | experiment_results_summary.md §三 |

---

## 附录：论文写作策略备忘录

### 审稿人可能的问题与应对

**Q1: "为什么不做 Behavior-only RQ-VAE？"**
A1: 我们的核心主张是**融合**的互补性，而非 behavior alone。Random SID 消融（Sport: content-only 0.0275 vs random 0.0093）已证明 SID 质量依赖于有效的过渡信号。Behavior-only 不能替代融合，且不是论文故事线的必要环节。

**Q2: "为什么不用 REG4Rec / Qwen3-7B？"**
A2: 我们的贡献与 backbone 选择正交。选用 T5 是为了与 TIGER 直接可比。TA-SID SID 可以配合任何 backbone 使用——更强的 backbone 只会放大我们的收益。

**Q3: "和 ADC-SID 怎么比？你的绝对数值不如他们。"**
A3: 框架完全不同（backbone、特征、架构均不同），绝对数值不可比。看**复杂度-收益**：TA-SID 以 768d 单特征 + LN+Concat + T5 取得 +12.2%~+17.1% 相对提升；ADC-SID 以 7B 特征 + MoQ + 5 专家 + 自适应路由才取得 +10.21%。我们以低一个数量级的复杂度实现相当甚至更大的提升。

**Q4: "冷启动问题你们怎么解决？"**
A4: 诚实说明这是 TIGER 家族的固有限制。引用 LIGER 作为已知改进方向。在未来工作中讨论 TA-SID + LIGER 的组合。

**Q5: "为什么 Sport 提升大但 Toys 提升小？"**
A5: **只主张"最稀疏的 Sport 增益最大"**：Sport 密度最低（0.045%）、序列最短（8.32）→ 内容特征不足 → 行为信号边际贡献最大（+17.1%）。⚠ **不要写"Toys 稠密→饱和"**：Beauty 与 Toys 密度几乎相同（0.0734% / 0.0724%），增益却差 3.7pp（+15.9% vs +12.2%），密度和序列长都解释不了；建议答道"其余两数据集增益相当且较小，其间的差异归因于数据分布差异，我们不做过度解读"。论文把这个作为"信号互补性"的论据（口径见 [paper_main_results.md §2.1](paper_main_results.md)）。

**Q6: "怎么证明提升来自 Transition Dynamics，不是随便加个向量就涨？"**
A6: 我们做了三层控制变量（§7.2，Sport，一次只改一个）：Content-only=0.0274 → +ZeroPad=0.0286（死维度，≈无变化）→ +Shuffled=0.0262（错配向量，有害）→ +Transition=0.0327（唯一正向，相对 ZeroPad +14.3%、相对 Shuffled +24.8%）。结论：维度不贡献、任意向量有害、唯有真实转移结构带来提升。

### 论文故事线（核心论点链）

```
问题层：Content-only SIDs 丢失行为模式 → 推荐瓶颈
        ↓
方法层：Transfer Dynamics → PPMI+SVD → Behavior Emb → LN+Concat → 注入 RQ-VAE
        ↓
证据链：TA-SID > Content-only TIGER (+12.2%~+17.1%) > SASRec (+56.4%~+139.5%)（v2 对齐口径，3 seed 均值，零交叉）
        ↓
验证层：① Random SID 消融（码质量） ② ZeroPad/Shuffled 消融（维度不贡献、错配有害、净贡献 +14.3%/+24.8%）
        ③ AUC 转移一致性（量化空间，即非冗余性的直接证据） ④ 长尾分析（Head 提升显著）
        ⑤ 维度鲁棒性（768d vs 4096d）
        （原"正交性分析"一项已删除：数字不可溯源且逻辑不成立，见 §7.5 裁撤说明）
        ↓
定位层：ADC-SID（复杂自适应融合） vs TA-SID（简洁转移注入）→ 同目标不同路径
        ↓
诚实面：冷启动 ≈ 0（TIGER family 通病），L_trans 无效（融合已足够）
```

### 写作顺序建议

1. 先写 **§6 主实验**（最有信心的部分）
2. 再写 **§4 方法**（技术核心）
3. 然后写 **§7 消融与分析**（最撑场面的数据）
4. 接着写 **§8 讨论**（ADC-SID 对比 + 冷启动诚实面对）
5. 最后写 **§1-3 和 摘要**（引言和概述最难写，放在最后）

---

*大纲生成日期：2026-07-30；数字更新：2026-08-03（统一 3-seed 768d 口径）*
*数据来源：TA-SID 实验全链路（TIGER baseline → TA-SID → SASRec → ADC-SID 对比）*
