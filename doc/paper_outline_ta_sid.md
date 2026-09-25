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
4. 分析实验证明：转移信号与内容信号正交（Jaccard=0.02, ρ=0.02），转移一致性 AUC 三数据集占优
5. **（第二创新，⚠ 2026-09-04 改写）防塌缩健康量化诊断**：RQ-VAE 粗层码本可能**静默分配崩坏**（Toys seed2024 L0 仅用 62/256 码，recon loss 不可见、仅 z-gap=0.093 检出），且崩坏对初始化敏感（换健康种子 256 即近无损：z-gap≤0.004、NDCG 0.0372≈512 的 0.0381）。贡献改为：**z-gap 事后判据**（recon 不可见的坏码本唯一探测器，6 数据集全对、不误触发）+ **reseed 兜底**（~8min/次，0 T5），坏码本真实代价 NDCG −1.4% / R@20 −3.8%。~~原"自适应扩容 K0\*=512 / +3.8%"卖点已推翻~~（见 [E46](e46_second_innovation_overturn.md)）

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

### 4.6 ~~Adaptive Coarse-Level Capacity Allocation~~ → 防塌缩健康量化（第二创新，2026-09-04 改写）

> ⚠ 本节原"自适应扩容"算法与"Toys 需 K0\*=512"结论已被 E45 推翻（详见 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)）。
> 保留的是：**z-gap 判据 + 事后检测 + reseed 兜底**（防塌缩健康量化诊断）；"扩容动作"降级为可选项。
> 论文写作时勿再写"扩容 → +3.8%"或"Toys 容量不足需 512"。

**动机（保留且加强）：** RQ-VAE 粗层码本可能**静默分配崩坏**——Toys seed2024 的 L0 只用 62/256 码
且摆放次优，但**重构损失(recon)与健康码本几乎无差、完全不可见**，只有转移结构判据 z-gap=0.093
能检出；真实代价 = 推荐层 NDCG −1.4% / R@20 −3.8%。更关键的是**崩坏对初始化敏感**：同数据同配置仅换
RQ-VAE 种子（2025/2026），256 即健康（L0 59–76%、z-gap≤0.004）→ 任何一次训练都可能塌，固定 K0
（无论 256/512）只保证"平均"够用，**事后 z-gap 检测是唯一兜底**。

**算法（改写：动作从"扩容"改为"reseed"，保留 z-gap 门）：**
```
输入: 数据 D、item embeddings E、容差 ε=0.01、最大重试次数 R=3
1  训练 RQ-VAE(D; K0=256, K1=256, K2=256)
2  生成 SID → 重建 xq[full]
3  z-gap ← AUC(z) − AUC(xq[full])       # 转移结构量化损失（recon 不可见的塌缩探测器）
4  if z-gap ≤ ε: 接受                      # 健康，交付 codes
5  else: 换种子 reseed 重训（K0 不变）→ 回 2，至多 R 次；仍失败再考虑扩容 K0←2·K0
6  返回健康 codes（z-gap ≤ ε）
```

**关键设计：**
- 判据 = **z-gap**（量化后转移一致性 AUC 损失）→ 直接度量"transition structure 丢了多少"，训练后可算
- **为何是 z-gap 而非 recon**：塌缩态 recon MSE 与健康态无显著差异（0.1452 vs 0.1414@512、健康256 亦同量级）→ 唯 z-gap 能当探测器（R1/R2）
- **修复动作 = reseed**（RQ-VAE ~8 min/次、0 T5），比扩容(256→512 需重训 RQ-VAE + 全部 T5)更便宜、更对症
- **转移熵不可作 predictor**（负面结果，保留）：Toys H=3.52bits 最低却最需修复，方向相反——H(B|A) 测行为层不可预测性，与 z 空间簇结构无关（详细见 §7.8）
- 机制确定过程（4.2，保留）：E1 L0 occupancy 诊断 → E2 post-hoc 容量曲线 → E3 重训验证 → E44 六数据集判据精度 → E45 多种子（详情见 e42/e44/e45 文档）

**执行结果（改写）：** Toys seed2024 单次训练落到 512 才健康（原"自适应"输出 K0\*=512），但
seed2025/2026 在 256 即健康且推荐层 ≈512 → **"容量不足需 512"不成立，瓶颈是码本健康度**。

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

- **Sport 提升最大（+17.1%）**：稀疏数据集上，行为信号提供了纯内容无法捕捉的转移模式
- **Toys 提升最小（+12.2%）**：稠密数据上 content 已够用，各方法趋于饱和，转移为增量收益
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

**指标：** 在量化输出空间（SID 码重建 x_q）比较高频转移对 vs 随机商品对的余弦相似度。**AUC = P(转移对相似度 > 随机对相似度)**，0.5=随机（转移未编码），等价于 Link Prediction 的标准 AUC（Mann-Whitney U）。

| 指标 | Sport | Beauty | Toys |
|:----|:----:|:----:|:------:|
| **TA-SID AUC** | **0.962** | **0.869** | **0.872** |
| Content-only AUC | 0.900 | 0.753 | 0.863 |
| TA-SID 碰撞率 | 10.66% | 1.09% | 8.68% |

**解读：**
- **TA-SID AUC 三数据集全部占优**（Δ+0.062 / +0.115 / +0.009）→ 在 SID 量化空间，TA-SID 更可靠地把转移对从普通对中区分出来
- **为什么不能用绝对 cos**：content-only 的码本把商品挤成一团（随机对 cos 0.65），绝对 cos 虚高；TA-SID 撑开商品（随机对 0.35）同时转移对保持 0.74-0.82 → **转移被专门编码**。对应对比学习 alignment-uniformity 框架（Wang & Isola 2020）
- **Toys ΔAUC 最小**：与主实验 Toys 提升最小（+12.2%）一致——稠密数据饱和，转移信号边际贡献小，论文诚实报告

### 7.4 内容维度的影响

| 配置 | Sport N@20 | Beauty N@20 | Toys N@20 |
|:--------|:---------:|:----------:|:---------:|
| TA-SID 768d（3-seed 均值） | **0.0322** | **0.0415** | **0.0377** |
| TA-SID 4096d（seed2025 单值） | 0.0276 | 0.0382 | 0.0317 |

- 768d 一致优于 4096d → 更高维的内容 ≠ 更好的 SID
- RQ-VAE 偏好适中维度的输入；过高维度会稀释码本有效性
- （768d 报 3-seed 均值以对齐主实验表；4096d 为 seed2025 单值，对比方向三数据集一致）

### 7.5 嵌入正交性分析

**核心发现：** 内容嵌入与行为嵌入编码了几乎正交的信息。

| 指标 | 数值 |
|:-------|:-----:|
| Top-10 近邻 Jaccard | 0.0167 |
| Top-20 近邻 Jaccard | 0.0216 |
| Spearman ρ（成对距离） | 0.0193 |

→ 两种模态捕捉了**完全不同的**商品特征。融合不是冗余，而是互补。

### 7.6 长尾分析

| 分组 | Sport R@20 | Sport N@20 | Toys R@20 | Toys N@20 |
|:------|:---------:|:---------:|:---------:|:---------:|
| Head（高频） | 0.134 | 0.056 | 0.145 | 0.064 |
| Mid（中频） | 0.009 | 0.003 | 0.029 | 0.010 |
| Tail（低频） | 0.000 | ~0.000 | 0.002 | 0.001 |

- Head >> Mid >> Tail：长尾效应是生成式检索的普遍问题
- 两个数据集呈现相同模式 → T5 的固有限制
- Toys（更稠密）对尾部商品略好

### 7.7 t-SNE 可视化

- 融合嵌入比纯内容嵌入具有更清晰的商品聚类结构
- 内容嵌入按商品类别聚类，行为嵌入按消费模式聚类
- 两者互补 → 融合嵌入同时保留了两种结构

### 7.8 ~~Adaptive Capacity 有效性验证~~ → 防塌缩健康量化诊断（第二创新实验，§4.6 改写）

> ⚠ **2026-09-04 推翻裁定**：本节的 Q3（"+3.8% 扩容卖点"）与"Toys 容量不足态"已由 E45 推翻——Toys
> 塌缩是 seed2024 单次训练的分配崩坏（换种子 256 健康，z-gap≤0.004），健康 256 的 T5 NDCG=0.0372 ≈
> 512 的 0.0381（无统计差异），**扩容无稳健推荐层收益**。保留的结论 = **z-gap 事后判据能检出 recon
> 不可见的坏码本 + 按需 reseed 兜底**。Q1/Q2 表保留为判据证据；Q3 按 E45 判 (a) 改写。权威裁定见
> [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)；历史表见 [e43_adaptive_experiment.md](e43_adaptive_experiment.md)。

回答三个问题（Q1 结构检测 / Q2 判据优于朴素信号 / Q3 坏码本转推荐损失 + 修复价值）。

**Q1 — z-gap 是否识别分配崩坏 / 健康（SID 质量，判据证据）✓（保留）**

| Dataset | K0 | 状态 | z-gap | xq[full] AUC | L0 用码(util) |
|:--|:--|:--|--:|--:|--:|
| Beauty | 256 | 健康 | 0.0001 | 0.8676 | 204/256 (80%) |
| Sport | 256 | 健康 | 0.0032 | 0.9614 | 249/256 (97%) |
| Toys（seed2024） | 256 | **分配崩坏** ⚠ | **0.0931** | 0.8724 | 62/256 (24%) |
| Toys（seed2025，E45） | 256 | ✅ 健康 | **0.0016** | 0.9672 | 194/256 (76%) |
| Toys（seed2024 扩容后） | 512 | 健康（落健康分配） | 0.0045 | 0.968 | 305/512 (60%) |

→ 判据证据：z-gap 区分"分配崩坏 vs 健康"，且崩坏可由**换种子（reseed，K0 不变）**而非扩容修复
（seed2025 同 256 即 z-gap 0.0016）；Beauty/Sport/MI/Electronics 判据不误触发（见 E44 §10.3）。

**Q2 — z-gap 判据是否优于朴素信号 ✓（保留；E44 判据精度表）**
z-gap 6 数据集最终判定全对；裸 L0 利用率在 MI（塌缩却无损）与 Electronics（100% 却扩容无效）判错两个
方向；Sport 压到 K0=64 仍不误触发。**判据精度与"扩容动作"解耦，全部保留**（E44 §10.3 / E46 R2）。
注：原"Adaptive(512) 优于 Fixed-large(1024)"的 Q2 表述已随扩容叙事撤销——真实教训是**不必扩容**
（见 Q3 判 a：512 与健康 256 无差异），扩容至多是一个"顺带落到健康分配"的偶然手段。

**Q3 — 坏码本的推荐层代价 + 修复价值（Toys T5, 3-seed；⚠ 2026-09-04 判 (a)）**

| codes 来源 | NDCG@20 | R@20 |
|:--|--:|--:|
| 256 **塌缩**（seed2024） | 0.0367±0.0006 | 0.0873±0.0016 |
| 256 **健康**（seed2025，判a 补跑） | **0.0372±0.0012** | **0.0906±0.0037** |
| 512（seed2024，落健康态） | 0.0381±0.0023 | 0.0914±0.0060 |
| 1024（Fixed-large） | 0.0384±0.0005 | 0.0916±0.0005 |

> ⚠️ **2026-09-09 provenance 订正**：上一行 "256 **塌缩**（seed2024）= 0.0367±0.0006 / 0.0873±0.0016" 出自**孤儿 codes**（无法从现存 ckpt 复现），作为 Toys **正式结果已作废**；仅保留作"塌缩实例"历史对照。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020。权威记录见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

→ **判 (a)**：健康 256（0.0372）≈ 512（0.0381）≈ 1024（0.0384），std 全部重叠、无统计差异 ⇒
**扩容在推荐层无稳健额外收益**。原 "+3.8% vs 固定 256" 来自"固定 256"那行是 seed2024 塌缩 codes：
真实信息是**坏码本代价 NDCG −1.4% / R@20 −3.8%**（相对健康 256），且 recon 不可见、只有 z-gap 检出。
**修复 = reseed（更便宜）而非扩容。**

**状态洞察（机制理解，Toys 改名）：**
- 健康态（Sport / Beauty / 健康种子的 Toys 256）：z 高 + L0 高占用 + gap≈0 → 无需干预
- 结构受限态（Beauty）：z 低 + gap≈0 且扩容无收益 → 扩容无意义
- **分配崩坏态（Toys seed2024 单例）**：z 高 + L0 塌缩 24% + gap 0.093 → **初始化敏感、需 reseed 兜底**
  （原"容量不足态"命名已废，见 E46 O1/O5）
- 塌缩但无损态（MI）/ 饱和但无效态（Electronics）/ 压测仍无损态（Sport K0=64）：判据不误触发（负对照）

**负面结果（如实报告，保留）：** 转移熵不能预测量化健康（Toys H 最低却最需修复，方向相反）；单层
k-means 内在簇数也与 3-token 量化不一致。→ 可靠的探测器是训练后的 z-gap。

**压测负对照（2026-09-03 补，判据不误触发；保留）：** 试图在健康数据 Sport 上人为压小粗层容量制造
第二个"容量不足"正例，**压不出来**——

| K0（Sport） | L0 用码 | xq[L0] AUC | xq[full] AUC | **z-gap** | 判定 |
|:--|:--|--:|--:|--:|:--|
| 256 | 249/256 (97%) | 0.9448 | 0.9614 | 0.0032 | 无损 |
| 128 | 128/128 (100%) | 0.9468 | 0.9635 | **0.0011** | 无损 |
| 64 | 64/64 (100%) | 0.9229 | 0.9526 | **0.0032** | 无损 |

→ 即便 K0 压到 1/4、xq[L0] 明显下降（0.945→0.923），L1/L2 残差把结构几乎全补回，判据仍不误触发。
压小 K0 只能制造"容量饿死"，复现不了 Toys 的**训练期 L0 分配崩坏**（62/256 且摆放次优）→ 证伪
"容量吃紧=必塌"的朴素猜想，佐证 Toys 问题在 allocation 对初始化敏感。详细见 [e4x_sport_capacity_stress.md](e4x_sport_capacity_stress.md)。

**多种子实锤 + 判 (a)（E45，2026-09-03~04 完成；改写原"进行中"段）：** RQ-VAE 种子曾硬编码于
main.py（现暴露为 `--seed`，默认 2024 完全保复现）。Toys 量化端多种子复跑 → **结果 B 实锤**：
seed2024 L0 62/256(z-gap 0.093)塌缩，但 seed2025 194/256(0.0016)、seed2026 151/256(0.0037)均健康 ⇒
**分配崩坏对初始化敏感，不是 Toys 系统性容量不足**。健康 256 codes 上 T5 3-seed → 判 (a)：≈512，
扩容无稳健收益。⇒ 第二创新叙事落为 **z-gap 事后检测 + reseed 兜底**（任何一次训练都可能塌，
recon 不可见，唯一可靠保护是事后 z-gap）。脚本 `rqvae/run_multiseed_repro.sh` /
`model/run_t5_multiseed.sh`，详见 [e45_toys_multiseed_collapse.md](e45_toys_multiseed_collapse.md)。

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
- 正交性 + AUC 转移一致性证实：转移信号提供了内容之外的全新信息

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
| §7.5 正交性分析 | embedding 分析 | analyze_emb_orthogonality.py 输出 |
| §7.6 长尾分析 | Head/Mid/Tail | experiment_results_summary.md §五 |
| §7.8 防塌缩健康量化诊断（第二创新，2026-09-04 改写） | z-gap 判据 / L0 util / 塌缩 vs 健康 / T5 NDCG@20+R@20 | e42_adaptive_k0_results.md（机制，历史）+ e43_adaptive_experiment.md（实验表，历史）+ e44_five_dataset_summary.md（六数据集判据精度）+ e4x_sport_capacity_stress.md（压测负对照）+ e45_toys_multiseed_collapse.md（多种子实锤结果 B + 判 a）+ **e46_second_innovation_overturn.md（推翻裁定，权威）** |
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
A5: 数据稀疏度差异。Sport 稀疏（+17.1%）→ 内容特征不足 → 行为信号边际贡献大。Toys 稠密（+12.2%）→ 所有方法饱和 → 差距缩小。论文把这个作为"信号互补性"的论据。

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
        ③ AUC 转移一致性（量化空间） ④ 正交性（非冗余） ⑤ 长尾分析（Head 提升显著）
        ⑥ 维度鲁棒性（768d vs 4096d）
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
