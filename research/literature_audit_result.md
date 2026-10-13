# Literature Audit Result — Semantic ID × Behavior Structure

> **本文件是审计结果**，不改动、不覆盖协议原文 `research/literature_audit.md`。
> 审计日期：2026-10-08　｜　执行：Literature Auditor（对抗性立场）
> 被审对象：`doc/behavior_structure_study_plan.md`（v0.2）所定义的 RQ1–RQ6 研究问题链。

## 0. 审计元信息与证据规则

**问题链**：Behavioral structure → continuous item representation → RQ-VAE/RVQ → discrete Semantic ID
→ behavioral structure preservation/distortion → generative recommendation。

**证据标记**（全文遵守协议 §7）：

- `[Paper evidence]` — 已打开 PDF 全文（`pdftotext -layout`）逐字核对。
- `[Code evidence]` — 官方仓库代码位置；仅证明代码行为，不自动等同论文贡献。
- `[Auditor inference]` — 审计员推断，**非**论文陈述。
- `[abstract-level / 未核实]` — 付费墙或抓取被拦，仅有摘要/搜索级信息，**不得作为核心证据**。

**判定口径的两个关键纪律**（对应协议 §6 与 §8）：

1. 「论文使用 behavior-aware SID」≠「论文研究了 behavior structure preservation」。必须看它是否把**行为/协同结构本身**作为被测量对象。
2. 「做了 layer-wise / capacity / prefix analysis」≠「该分析的对象是行为结构」。必须指出**每层分析的是什么结构**（重建损失 / 码本几何 / 语义邻域 / 行为邻域）。

**检索覆盖**：12 类关键词矩阵（协议 §4）× 2022–2026（重点 2024–2026）；WebSearch + arXiv 全文 PDF + Semantic Scholar Graph API + 官方 GitHub。
精读全文约 **40+ 篇**（分五个并行审计流），另有 ~20 篇仅达检索级。**无法打开的关键条目已在 §9 逐条列出**。

---

# 1. Executive Summary

**1）当前问题是否已有直接先例？**
**没有**任何单一已发表工作完整覆盖「行为信号 → 表示 → 量化 → SID → 行为结构保持/失真分析 → 生成式推荐」全链，并以行为/协同结构为**显式分析对象**。**Within the searched literature, no direct precedent was identified** 覆盖该完整闭环（详见 §5）。**但是**，链条的每一段都已被单独占据，且有两篇已站在闭环边缘：**GrIS**（CIKM'26，逐 SID 前缀层协同一致性间隙 δ^CF + Hit/NDCG）与 **TopoTok**（RecSys'26，逐量化层 top-20 邻域拓扑重叠 63%→30%→27% + 性能）。

**2）哪些 RQ 已被覆盖？**
**RQ1（系统性结构损失）已被 Level A 覆盖**：TopoTok 直接给出「连续 vs 量化 top-k 邻域重叠、逐 RQ-VAE 层」——即 RQ1 的度量本体；SIDInspector/SIDScope 的 D3 给出**行为（共现）**前缀召回；DRQ 给出量化后 item-to-item 检索保留率（RQ-VAE@20 = 0.5561）；"Understanding Semantic IDs" 给出 SID 邻域仅恢复编码器 10-NN 的 32.2%。
**RQ2（发生在哪个阶段）已被覆盖**：TopoTok 逐层、GrIS 逐前缀层 δ^CF、RecLoop 逐层码熵、MoC 逐层信息量、DIGER/DRQ 逐层码本塌缩。
**RQ4（行为相似性 → SID prefix / code-sharing）已被覆盖**：SIDInspector/SIDScope D3/D5、GrIS δ^CF、ReSID Table 9、Latte 的 Kendall(tree, CF)。

**3）哪些 RQ 只有部分覆盖？**
**RQ3（不同结构类型可保持性）**：SIDInspector/SIDScope 的 D4 分 head/mid/tail、TopoTok 分 inter-group/intra-group/inter-item、HypRQ/VarLenRec/AsymRec/FineSID/H²Rec 分流行度——**但**在已检索文献内，**未识别到**以一个受控协议在同一物品全集上比较多种行为结构类型可保持性的工作。
**RQ5（codebook capacity 的 trade-off）**：InforID（容量预算 vs NDCG）、DIGER（K∈{128,256,512} vs 优化稳定性）、FlexCode/VarLenRec（按流行度在协同码本与语义码本间分配容量）、DRQ（协同监督反而降低深层有效容量）——各占一角，**未识别到**把「有限容量」作自变量、把「行为结构保持」作因变量的受控研究。
**RQ6（结构保持 ↔ 性能）**：**相关性已被覆盖**（SIDInspector D3↔候选召回 Spearman 0.976，自述非因果；SIDScope 前缀对齐↔候选曝光）；**因果/干预证据部分覆盖**（Hourglass 层交换干预 → 性能；TopoTok 方法同时抬升 overlap 与性能，但无剂量-响应扫描）。

**4）哪些 RQ 目前仍有明显空缺？**
- RQ3 的**受控跨类型比较**（同量化器、同物品全集、多行为结构类型并列）——**最薄的一块地**。
- RQ5 的**「容量 × 行为结构保持」交互项**（作为受控自变量）。
- RQ6 的**单调剂量-响应**（刻意调节结构保持程度、其余固定、观察性能）。

**5）最危险的 novelty overlap 是什么？**
🔴 **TopoTok**（RQ1/RQ2 的度量本体几乎正面撞车，且含 SASRec 协同教师配置）与 **SIDInspector/SIDScope**（RQ1 的行为变体 + RQ4 的 prefix-behavior 对齐，D3 度量与我们高度同构）。次级危险：**GrIS**（RQ2/RQ4 的逐层协同结构间隙）、**"Understanding Semantic IDs"**（RQ1 的语义变体 + 代际损失）、**ReSOT**（"Relational Structure Distortion"，未核实）。

**6）是否建议继续进入实验阶段？**
**不建议按 v0.2 原样 Continue。** RQ1/RQ2/RQ4 的原始表述已被直接先例覆盖，必须**重定义**。判定为 **B. Narrow**（详见 §8）：保留 RQ3 的受控跨类型比较作为主轴（当前最薄），RQ5/RQ6 改写为「容量/因果的受控交互」，并**删除**任何「首次发现量化损失结构」类主张。

---

# 2. Literature Landscape

按协议 §2 分类。每条附**最危险代表**与占据判定。

### (a) Semantic ID / 生成式推荐（骨架文献）
TIGER、LETTER、ETEGRec、OneRec、GRID/Practitioner's Handbook、OneSearch、SaviorRec……为全部 RQ 的**背景**（Level D），不构成问题重合。

### (b) behavior-aware / collaborative-aware SID（方法轴，已被占满）
FORGE（i2i co-occurrence 共现统计进量化）、COSETTE/MARIUS（共现统计引导码学习）、QuaSID（Hamming 引导协同双塔对比）、SimCIT（ALS 协同多视图）、DAS（三路 i2i/u2u）、GrIS（协同图 + RQ-GAE）、Restoring Collaborative Signals（二阶 co-view HPF 分解）。
→ **判定**：把「行为/协同信号注入 SID」作为方法贡献的这条轴**已被占满**（与内部 `doc/prior_art_novelty_audit.md` §0 一致）。**这些工作多数不做结构保持/失真分析。**

### (c) collaborative-enhanced tokenization（同 (b)，注重训 tokenizer）
COSETTE、GRID、QuaSID、SimCIT。Level B/C。

### (d) quantization analysis（RQ2 所在轴）
TopoTok（逐层拓扑）、DRQ（逐层有效容量/塌缩）、DIGER（逐层码使用 + K sweep）、MoC（逐层信息量）、RecLoop（逐层码熵）、Hourglass（逐层熵/Gini/std + 层交换）、GrIS（逐前缀层 δ^CF/δ^Sem + 各层 cardinality/Gini）。
→ **判定**：**layer-wise 分析已被大量占据**；关键是「分析的是什么结构」——几乎全是**重建/码本几何/语义**，**行为排序逐层扫描**最薄。

### (e) structure preservation / distortion（RQ1 所在轴）
TopoTok（**最强**：连续 vs 量化 top-20 邻域重叠、逐层）、"Understanding Semantic IDs"（SID 邻域仅恢复 32.2% 编码器 10-NN）、SIDInspector/SIDScope（D3 共现前缀召回）、DRQ（检索保留率）、ReSOT（Relational Structure Distortion，未核实）、Latte（forced transitivity，Kendall(tree, CF) = −0.7170）、What Makes a Good SID（语义邻域保持跨设计）。
→ **判定**：**RQ1 作为「是否发生结构损失」已被覆盖**；作为「行为结构」的变体被 SIDInspector/SIDScope 部分覆盖。

### (f) generative recommendation（下游，全 RQ 的落点）
以上几乎所有方法论文都报 Hit/NDCG/Recall，**说明 (d)(e) 的分析已被连到性能**。

**Landscape 总结**：这是一个**高度拥挤**的领域。2026 年（模拟时间线）出现了一波**诊断/资源类**工作（SIDInspector、SIDScope、FORGE、Reproducibility Study、Auditing Recovery Test），它们把「SID 质量诊断」系统化，**直接压缩了"做一个 SID 诊断研究"的空间**。

---

# 3. Key Paper Audit

| Paper | Year | Main idea | Behavior | Quantization | Structure preservation | Layer-wise | Capacity | Prefix | Recommendation | Overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| **TopoTok** (RecSys, 2607.18600) | 2026 | 多级蒸馏保拓扑的 tokenization | 语义为主；ETEGRec/SASRec 配置下为协同 | RQ-VAE 3×256 | **是**：连续 vs 量化 top-20 邻域重叠 | **是**：L1/L2/L3 = 63/30.3/27.6% | 部分（utilization 作层路由） | 部分 | 是（+9.42% R@5） | 🔴 A（RQ1/RQ2） |
| **SIDInspector** (CIKM, 2606.10375) | 2026 | mapping-first 诊断资源 | **是**（D3 共现前缀召回） | 覆盖 RQ-VAE/RK-Means | **是**：D3 行为前缀对齐 | 部分（D3 多为 level-1） | 是（D1 利用率/D4 尾部容量） | **是**（D5 trie） | 受限（自述非因果） | 🔴 A（RQ1 行为变体/RQ4） |
| **SIDScope** (TOIS 投稿, 2608.18779) | 2026 | SIDInspector 扩展 | **是**（行为前缀） | 覆盖 | **是**：D3(k) 加权前缀召回 | 部分 | 是（D4 head/mid/tail） | 是 | 受限（候选曝光，非端到端） | 🔴 A（RQ1/RQ4 行为变体） |
| **GrIS** (CIKM, 2610.01533) | 2026 | 图引导 SID 平衡协同/语义 | **是**（SASRec 协同图） | RQ-GAE/RecDMoN | **是**：逐前缀层 δ^CF | **是**：L1–L3 δ^CF（Sports TIGER 0.184/0.244/0.299） | 是（逐层 cardinality/Gini） | **是**（δ^CF 即前缀一致性） | 是（H@5/10, N@5/10） | 🟠 B（近 A，RQ2/RQ4） |
| **Understanding SIDs** (2607.24995) | 2026 | SID→item selection 分析 | 否（编码器语义空间） | 8 种 SID 构造 | **是**：SID 邻域仅恢复 32.2% 10-NN；末位仅保留 29.9% 目标 | **是**（末位 token） | 部分 | **是** | 是（ISD +31.2% NDCG@10） | 🔴 A（RQ1 语义变体/RQ4） |
| **ReSOT** (KDD, 10.1145/3770855.3817834) | 2026 | OT 对齐 item-item 关系 | 部分（关系结构） | RVQ | **是**（"Relational Structure Distortion"）— **未核实** | 未核实 | 部分（碰撞） | 部分 | 是 | 🟠 A（未核实） |
| **DRQ** (2606.01844) | 2026 | 解耦几何重建与分布匹配 | 部分（CL 注入行为） | RQ-VAE | **是**：item-to-item 检索保留率 RQ-VAE@20=0.5561 | **是**：逐层有效容量 K_eff 455/2146/2345 | **是**：逐层 K_eff/利用率/Gini | 部分 | 受限（单工业数据集） | 🟠 B（近 A，RQ1/RQ2/RQ5） |
| **Latte** (2605.06331) | 2026 | SID 树度量的受迫传递性分析 | 否 | 树/量化 | **是**：Kendall(tree dist, CF sim) = −0.7170 (TIGER) | 部分 | 否 | 是 | 部分 | 🟠 B（RQ1/RQ4） |
| **Reproducibility Study** (SIGIR-AP, 2609.24430) | 2026 | SID 设计可复现性 | 部分（协同正则） | 覆盖 | **是**：local semantic preservation | 部分 | 是（L1 归一化熵） | 部分 | 是 | 🟠 B（RQ1/RQ5/RQ6） |
| **RecLoop** (TOIS, 2606.17707) | 2026 | 生成式推荐信息茧房 | 部分（SID vs CID 协同 ID） | SID/CID | 否（码空间多样性，非 item 结构） | **是**：逐层码熵（L0 53.6%/L1 40.4%/L2 13.4%） | 是（码集中度） | 是 | 是 | 🟡 C（RQ2） |
| **MoC** (2410.09560) | 2024 | 可扩展语义表示 | 否 | RQ-VAE | 部分（逐层信息量 NMI） | **是**（逐层信息/单 ID 增益） | 部分 | 是 | 是 | 🟡 C（RQ2） |
| **DIGER** (SIGIR, 2601.19711) | 2026 | 可微 SID | 否 | RQ-VAE | 部分（逐层码使用/塌缩） | **是** | **是**：K∈{128,256,512} | 是 | 是 | 🟠 B（RQ2/RQ5） |
| **InforID** (2608.09685) | 2026 | 自适应语义容量分配 | 否 | PQ/RQ | 部分（子空间重建损失） | 部分 | **是**：预算 B 64→352 sweep | 部分 | 是 | 🟠 B（RQ5） |
| **VarLenRec/PIBA** (2605.17779) | 2026 | 变长 tokenization | **部分**（协同 vs 内容按流行度分配长度） | RQ | 部分（信息预算 L*∝p^(−α/γ)） | 部分（逐层 γ/l 信息，**假设非实测**） | **是**（长度=容量） | 部分 | 是 | 🟠 B（RQ5/RQ3） |
| **FlexCode** (Roblox, 2511.20673) | 2025 | 双码本（协同+语义） | **是**（CF 码本） | 双码本 | 否 | 否 | **是**（按流行度分配 CF/语义容量） | 是 | 是 | 🟠 B（RQ5/RQ3） |
| **QuaSID** (KDD, 2603.00632) | 2026 | 按碰撞严重度加权的 SID 学习 | **是**（协同双塔对比） | RQ-VAE | 否（碰撞严重度/Hamming 代理） | 否 | 部分 | 部分 | 是 | 🟡 C（RQ1 弱） |
| **COSETTE/MARIUS** (2508.14910) | 2025 | 协同语义 tokenization | **是**（共现对比） | RQ-VAE | 否（仅碰撞 + R@10） | 否 | 否 | 否 | 是 | 🟡 C |
| **FORGE** (KDD, 2509.20904) | 2026 | 工业 SID 构造 + 基准 | **是**（i2i co-occurrence 进 InfoNCE） | RQ-VAE | 否（自承代理指标与检索不相关） | 否 | 部分（Gini/utilization） | 否 | 是（含在线 A/B） | 🟡 C（RQ1/RQ5 弱） |
| **Hourglass** (EMNLP Ind., 2409.xxxx) | 2024 | 逐层量化诊断 + 层交换干预 | 否 | RQ-VAE | 部分（逐层熵/Gini/std） | **是** | **是** | 否 | 是（**层交换干预→性能**） | 🟠 B（RQ2/RQ6 因果） |
| **Guided SID** (Meta, 2609.22227) | 2026 | 引导粗层使细层可学 | 否（属性/类别） | RQ-VAE | 部分（逐层 prefix recall，对象是属性） | **是** | 否 | 是 | 是 | 🟡 C（RQ2/RQ4 非行为） |
| **FineSID** (AAAI'27, 2609.36670) | 2026 | 可扩展 SID 学习 | 否（内容；命名"collaborative"实为 Lipschitz 界） | RQ-VAE | **理论**：成对距离保持 ≤ε | 否 | 是（尾部保证） | 部分 | 是 | 🟡 C（RQ5 理论） |
| **AST / Not All Branches** (RecSys, 3831756) | 2026 | 自适应分支 SID 树 | 否 | 树 | **部分**（前缀 mask 余弦 0.758/0.874/0.920）— **未核实** | 是 | 否 | 是 | 是 | 🟡 C（RQ4，未核实） |
| **ReSID** (2602.02338) | 2026 | 信息论重构 tokenization | **是**（Louvain 共现社区 vs 类别） | RQ | 部分（Table 9 目标 vs 历史码重叠） | 否 | 否 | 是 | 是 | 🟡 C（RQ3/RQ4） |
| **Restoring Collab. Signals** (2607.27682) | 2026 | 个性化自然语言恢复协同信号 | **是**（二阶 co-view 分解） | 用既有 SID（不改 tokenizer） | **部分**（"collaboration loses"、"coarse SID 碰撞"，定性） | 否 | 部分 | 否 | 是（hit@K） | 🟠 B（RQ1/RQ3，定性） |
| **Kuaishou Static→Dynamic** (2608.21012) | 2026 | 单层大码本替代多层小码本 | 弱（hash 协同消歧码） | 大码本 | 否（码本质量诊断，自承"limited evidence"） | **是**（逐层利用率/条件利用） | **是** | 部分 | 是 | 🟡 C（RQ2/RQ5） |
| **What Makes a Good SID** 见 Reproducibility Study | — | — | — | — | — | — | — | — | — | — |
| **Model-scaling View** (KDD, 2509.25522) | 2026 | SID GR 的 scaling law | 弱（CF 仅外部注入） | SID | 否（SI 容量瓶颈） | 否 | **是**（离散化丢弃语义信息） | 否 | 是 | 🟡 C（RQ5） |
| 其余（SimCIT/DAS/ATLAS/H²Rec/HypRQ/AsymRec/EAGER/PLUM/DECOR/…） | 2024–26 | 行为注入或长尾方法 | 是/否 | 是 | 否（多为方法） | 否 | 部分 | 否 | 是 | 🟢 C/D |

> 表中「Overlap」列：🔴 = Level A 直接重合；🟠 = Level B 机制重合；🟡 = Level C 部分重合；🟢 = Level D 背景。

---

# 4. RQ-by-RQ Audit

## RQ1 — 连续表示 → 量化 → SID 过程中是否发生系统性行为结构损失？

- **已有研究**：TopoTok（连续 vs 量化 top-20 邻域重叠，逐层，`[Paper evidence]` Fig 1 / §1 / §4.3.1 / Table 2）；SIDInspector/SIDScope D3（共现前缀召回，`[Paper evidence]` §3.2–4.1）；DRQ Table 4（item-to-item 检索保留率 RQ-VAE@20=0.5561，`[Paper evidence]` §5.5）；"Understanding Semantic IDs"（SID 邻域恢复 32.2% 编码器 10-NN，`[Paper evidence]` Abstract）；Latte（Kendall(tree, CF)=−0.7170，`[Paper evidence]`）；ReSOT（Relational Structure Distortion，`[abstract-level，未核实]`）。
- **最近似工作**：**TopoTok**（度量本体同构）+ **SIDInspector/SIDScope**（对象换成行为共现）。
- **已解决程度**：**基本解决（Level A）**。「是否发生系统性结构损失」不再是新问题。
- **尚未解决**：把「行为/协同**排序**结构」作为唯一被测量对象、给出量化前后**受控**失真度量——但 SIDInspector/SIDScope 已覆盖其行为的共享前缀版本，缝隙很窄。
- **重合风险**：🔴 **High**。
- **结论**：**RQ1 原表述不可作为贡献；必须降级为「引用 TopoTok/SIDInspector 的既有观察，我们复现并扩展其数据集覆盖面」。**

## RQ2 — 结构损失主要发生在哪个量化 stage？

- **已有研究**：TopoTok 逐层（L1 63%→L2 30.3%→L3 27.6%，`[Paper evidence]` Table 2）；GrIS 逐前缀层 δ^CF（`[Paper evidence]` Table 8）；RecLoop 逐层码熵（L0 53.6%/L1 40.4%/L2 13.4%，`[Paper evidence]` Fig 7）；MoC 逐层信息量（`[Paper evidence]` §5.4）；DIGER/DRQ 逐层码本塌缩（`[Paper evidence]` §6.2 / §5.3）；Hourglass 逐层熵/Gini/std + 层交换（`[Paper evidence]`）。
- **最近似工作**：**TopoTok**（逐层拓扑）+ **GrIS**（逐前缀层协同间隙）。
- **已解决程度**：**已覆盖（Level A/B）**。
- **尚未解决**：以**行为排序指标**在**多数据集**上做 coarse-to-fine 的「哪一层断」定位诊断（TopoTok 是单方法论文、metric 为邻域重叠；GrIS 只到 L1–L3 前缀余弦间隙、且是方法论文）。
- **重合风险**：🔴 **High**。
- **结论**：**RQ2 必须改写为「跨数据集的行为排序逐层定位诊断」，并显式区别于 TopoTok（拓扑邻域）与 GrIS（前缀余弦间隙）。**

## RQ3 — 不同类型的行为结构是否有不同可保持性？

- **已有研究**：SIDInspector/SIDScope D4（head/mid/tail 分层的 unique-SID 比，`[Paper evidence]` §3.2）；TopoTok inter-group/intra-group/inter-item 分层（`[Paper evidence]` §4.5）；HypRQ-VAE（top-20%/bottom-80%）、VarLenRec、AsymRec、FineSID（尾部保证）、H²Rec（head/tail 权衡）——均按**流行度/长尾**分层（多为方法动机）。
- **最近似工作**：**SIDInspector/SIDScope D4**（频率结构作为独立坐标）。
- **已解决程度**：**部分覆盖（Level C）**。
- **尚未解决**：**未识别到**单个受控研究，在**同一量化器、同一物品全集**下，并列比较**多种行为结构类型**（强/弱关联、behavioral-NN、local co-occurrence、frequency）的可保持性。SIDScope 明确声明不做跨数据集因果比较。
- **重合风险**：🟠 **Medium**。
- **结论**：**这是当前最薄的一块地，可作为主轴**——但须框定为「测量/分类贡献，至多部分新」，并引用 SIDInspector/SIDScope 为最接近先例。

## RQ4 — 行为相似性是否映射到 SID prefix / code-sharing？

- **已有研究**：SIDInspector/SIDScope D3/D5（前缀 co-occurrence 对齐 + trie，`[Paper evidence]` §3.2/§4.1）；GrIS δ^CF（逐前缀层协同一致性，`[Paper evidence]` Table 8）；ReSID Table 9（目标 vs 历史码重叠，`[Paper evidence]`）；Latte（树距离 vs CF 相似 Kendall，`[Paper evidence]`）；AST 前缀 mask 余弦（`[abstract-level，未核实]`）；"Understanding Semantic IDs"（SID 邻域恢复）。
- **最近似工作**：**SIDInspector/SIDScope**（行为前缀保持，直接命中）+ **GrIS**。
- **已解决程度**：**已覆盖（Level A，行为变体）**。
- **尚未解决**：几乎无实质空白。
- **重合风险**：🔴 **High**。
- **结论**：**RQ4 不可作为贡献；降级为引用 + 复现。**

## RQ5 — 有限 codebook capacity 是否影响行为结构保持，形成语义/行为 trade-off？

- **已有研究**：InforID（预算 B 64→352，重建损失 vs NDCG，`[Paper evidence]` Fig 3/Table 3）；DIGER（K∈{128,256,512}，"larger codebooks lead to inferior results"，`[Paper evidence]` §6.2）；FlexCode/VarLenRec（按流行度在协同码本与语义码本间分配容量，`[Paper evidence]` §2.3/§3；Eq 2/4）；DRQ（逐层 K_eff；`[Paper evidence]` §5.3：协同监督使深层 K_eff 从 3365.96 掉到 930→509，即「协同重塑不自动改善符号稳健性」）；Kuaishou（逐层条件利用率，`[Paper evidence]` §1）；FORGE（Gini/utilization，`[abstract-level]`）；Reproducibility Study（L1 归一化熵 vs NDCG@10 Pearson r=−0.02，`[Paper evidence]`）。
- **最近似工作**：**VarLenRec**（显式语义↔协同 trade-off）+ **FlexCode**（双码本容量分配）+ **DRQ**（协同监督反而降深层容量）。
- **已解决程度**：**部分覆盖（Level B）**——各要素被多篇分占，trade-off 的**交互项**未被单独占据。
- **尚未解决**：**未识别到**把「有限容量」作独立自变量、把「行为结构保持」作因变量的受控研究。**注意**：Reproducibility Study 的「码本熵与 NDCG 不相关」与我方 E49 结论（codebook health 不预测 NDCG）**同向**——这会**削弱**「容量→行为保持→性能」链条的先验合理性。
- **重合风险**：🟠 **Medium**（可被审稿人用多篇拼装成「已覆盖」）。
- **结论**：**RQ5 改写为「容量 × 行为结构保持」的交互控制，须明说区别于 VarLenRec/FlexCode/DIGER。**

## RQ6 — 结构保持是否与生成式推荐性能有系统关系？

- **已有研究**：**相关性**——SIDInspector D3↔候选召回 Spearman 0.976（Musical）/0.657（All_Beauty，NDCG）（`[Paper evidence]`，**自述 D3 是候选曝露诊断、非排序指标、非因果**）；SIDScope 前缀对齐↔候选曝光（`[Paper evidence]`）；GrIS δ^CF 与 H@K 并列（`[Paper evidence]`）；Reproducibility Study 熵 vs NDCG r=−0.02（**零相关**，`[Paper evidence]`）。**干预/因果**——Hourglass 层交换干预 → 性能变化（`[Paper evidence]`）；TopoTok 方法同时抬升 overlap 与性能（非受控剂量-响应）。
- **最近似工作**：**Hourglass**（干预证据）+ **SIDInspector**（相关性）。
- **已解决程度**：**相关性已覆盖；因果/干预部分覆盖（Level B）**。
- **尚未解决**：**单调剂量-响应**——刻意调节行为结构保持程度、其余固定、观察性能单调变化。**未识别到**此设计。
- **重合风险**：🟠 **Medium**。
- **结论**：**RQ6 改写为「结构保持程度 → 性能的受控剂量-响应」，并须披露 Reproducibility Study 的零相关反例。**

---

# 5. Direct Precedent Audit

**问题**：是否存在一篇工作已覆盖 Behavior → representation → quantization → SID → structure preservation → recommendation 全链？

**回答**：**Within the searched literature, no direct precedent was identified** 完整覆盖该 6 阶段链、且以行为/协同结构为显式分析对象并连到生成性能的单一工作。

文献的占据模式是**三向分裂**（`[Auditor inference]`，基于 5 个并行审计流的全文核对）：

1. **协同信号入量化**（行为→表示→SID→性能，缺失真分析）：COSETTE/MARIUS、FORGE、QuaSID、SimCIT、DAS。
2. **「量化导致结构失真/保持」的显式分析**（缺行为对象）：TopoTok、Understanding Semantic IDs、What Makes a Good SID、DRQ、ReSOT——**对象几乎全是语义/内容结构**。
3. **协同结构与 SID 的保持关系**（缺完整闭环）：**GrIS**（δ^CF 逐前缀协同间隙 + H@K，但为**静态间隙**、方法论文）与 **Restoring Collaborative Signals**（明确断言协同结构丢失，但为诊断+推理期修复，无量化失真闭环）。

**最接近的单一先例 = GrIS**（协同结构逐前缀层 + 性能），其次 **TopoTok**（若把「结构」放宽为任意嵌入拓扑，TopoTok 在 SASRec 配置下即直接作用于协同嵌入）。

**审计员对抗性提示**（`[Auditor inference]`）：可区分的缝隙只剩一条——**在注入行为信号的前提下，以「量化前 vs 量化后」的协同结构失真为受控对照，并把它作为与生成性能的剂量-响应链来度量**——须明确区别于 GrIS（静态间隙、非受控）与 TopoTok（语义拓扑、非行为对象、非受控剂量）。

---

# 6. Novelty Risk Matrix

| RQ | Direct precedent | Mechanistic overlap | Partial overlap | Novelty risk | Continue? |
|---|---|---|---|---|---|
| **RQ1** | TopoTok（度量本体）；SIDInspector/SIDScope D3（行为变体）；"Understanding SIDs" | DRQ（检索保留率）；Latte | ReSOT（未核实） | 🔴 **High** | ❌ 不可作贡献，降级引用/复现 |
| **RQ2** | TopoTok（逐层）；GrIS（逐前缀层 δ^CF） | RecLoop/MoC/DIGER/DRQ/Hourglass（逐层，非行为） | — | 🔴 **High** | ⚠️ 需改写为跨数据集行为排序定位 |
| **RQ3** | — | SIDInspector/SIDScope D4（频率分层） | HypRQ/VarLenRec/AsymRec/FineSID/H²Rec（长尾） | 🟠 **Medium** | ✅ **主轴（最薄）**，框为测量/分类贡献 |
| **RQ4** | SIDInspector/SIDScope D3/D5；GrIS δ^CF | AST（未核实）；ReSID Table 9；Latte | "Understanding SIDs" | 🔴 **High** | ❌ 不可作贡献，降级引用 |
| **RQ5** | — | VarLenRec；FlexCode；DRQ；DIGER；InforID | Kuaishou；FORGE；Reproducibility Study | 🟠 **Medium** | ⚠️ 改写为「容量×行为保持」交互 |
| **RQ6** | — | Hourglass（干预）；SIDInspector（相关性） | SIDScope；GrIS；Reproducibility Study（零相关反例） | 🟠 **Medium** | ⚠️ 改写为受控剂量-响应 |

**风险分级**：🔴 High（审稿人可一句话否决）/ 🟠 Medium（可用邻近工作拼装质疑）/ 🟢 Low。

---

# 7. Reviewer Attack Test

假设 SIGIR/WWW/KDD/CIKM 审稿人。列出 ≥10 条质疑并逐条判定。

| # | 质疑 | 文献是否已解决该质疑？ | 判定 |
|---|---|---|---|
| A1 | "This problem has already been studied by TopoTok." | **是**——TopoTok 直接做逐层连续 vs 量化邻域重叠并连性能。 | 🔴 **成立**，须正面引用并区分（行为对象 + 受控）。 |
| A2 | "SIDInspector/SIDScope 已经测量了行为前缀对齐（D3）。" | **是**——D3 即共现前缀召回，且 D4 已分 head/tail。 | 🔴 **成立**，须引用并说明差异（我们做跨类型受控比较，非单 artifact 描述）。 |
| A3 | "The proposed analysis is merely a diagnostic of existing RQ-VAE behavior." | **部分**——TopoTok/DRQ/SIDInspector 均为诊断类；这恰是本领域 2026 的拥挤带。 | 🟠 **成立**，「做一个诊断」本身不是贡献，须靠**受控设计 + 跨数据集**作答。 |
| A4 | "Behavior-aware tokenization already preserves collaborative structure." | **否**——FORGE/COSETTE/GRIS 等证明注入行为**不自动**保持结构；DRQ 甚至显示协同监督降低深层有效容量。 | 🟢 **可用作反证**（对我们有利）。 |
| A5 | "The claimed structural loss is an artifact of the chosen metric." | **未完全**——TopoTok 用单尺度聚合 metric；SIDInspector 自述 D3 非排序指标；ReSOT 用 GW/Jaccard。 | 🟠 **开放**，须多指标交叉（overlap + Jaccard/RBO + 排序相关）。 |
| A6 | "Behavior representation and evaluation criterion come from the same co-occurrence source." | **未解决（本领域通病）**——COSETTE/DAS/FORGE 均用 co-occurrence 构造又用其评估。 | 🟠 **开放且高危**，须显式披露循环性并加独立判据。 |
| A7 | "The observed effect is caused by popularity, not behavioral structure." | **部分**——SIDInspector D4、HypRQ/VarLenRec 均显示流行度分层效应显著。 | 🟠 **成立**，须把 popularity 作协变量控制/分层。 |
| A8 | "Codebook capacity analysis is standard quantization analysis." | **部分**——DIGER/DRQ/InforID/Kuaishou 已做容量-利用率-塌缩分析。 | 🟠 **成立**，纯容量分析不新，须靠「容量×行为保持」交互区分。 |
| A9 | "The correlation with recommendation performance is not causal." | **部分**——Hourglass 已给层交换干预；SIDInspector 明确自述非因果。 | 🟠 **成立**，须补受控剂量-响应（否则只剩相关性）。 |
| A10 | "Codebook health/entropy does not predict NDCG anyway." | **是（对我们不利）**——Reproducibility Study：归一化熵 vs NDCG r=−0.02；与我方 E49 同向。 | 🔴 **成立**，直接削弱「保持→性能」链条先验，**必须正面回应**。 |
| A11 | "Layer-wise RQ-VAE analysis already exists (TopoTok, GrIS, MoC, RecLoop)." | **是**。 | 🔴 **成立**，RQ2 原表述不可作贡献。 |
| A12 | "Why is behavioral structure the right structure? Semantic structure works in TIGER." | **开放**——TopoTok 主叙事即语义拓扑。 | 🟠 **开放**，须论证行为结构对推荐任务的必要性。 |
| A13 | "Shared-prefix ≠ behavioral similarity; code-sharing is an artifact of codebook size." | **部分**——SIDInspector 的 category-prefix 对照（D3 0.447 vs 学习导出 0.055–0.154）已说明前缀对齐可被类别而非行为驱动。 | 🟠 **成立**，须加类别/随机前缀对照（我方 E49 ZeroPad/Shuffled 已部分具备）。 |

**小结**：A1/A2/A11/A10 是**致命级**（须重定义 RQ）；A3/A5/A6/A7/A9/A13 是**须补实验设计**级；A4 反而**对我方有利**。

---

# 8. Final Adjudication

## 结论：**B. Narrow**

**判定理由**（依据协议 §8 与「宁可判定不新」原则）：

1. **不能 Continue**：RQ1（系统性结构损失）、RQ2（逐层定位）、RQ4（行为相似性→prefix）的**原始表述**均已被 Level A 直接先例覆盖（TopoTok / SIDInspector / SIDScope / GrIS / Understanding SIDs）。若按 v0.2 原样进入实验，审稿人可一击否决。

2. **不必 Abandon**：整条**受控比较链**（同量化器、同物品全集、跨数据类型、跨数据集、含剂量-响应）在已检索文献内**未见**单一占据者；RQ3 的受控跨类型比较是最薄的可用地；GrIS/TopoTok 各自都差一块（GrIS 差「量化前后受控失真框架」，TopoTok 差「行为对象 + 受控剂量」）。

3. **必须做的收窄（Narrow 的具体含义）**：
   - **删除**所有「首次发现量化损失行为结构」类主张（A1/A11 致命）。
   - **RQ1/RQ2/RQ4 降级**为「引用既有先例（TopoTok/SIDInspector/GrIS）+ 我们在 3 数据集复现并扩展」——作为**证据基础**而非**贡献**。
   - **主轴收窄到 RQ3**：单一受控协议下、同一物品全集、并列比较**多种行为结构类型**的可保持性（含「某些类型可能无差异」这一诚实可能）。
   - **RQ5 收窄**为「容量 × 行为结构保持」的**交互控制**（自变量=容量，因变量=行为保持），显式区别于 VarLenRec/FlexCode/DIGER。
   - **RQ6 收窄**为**受控单调剂量-响应**（区别于相关性与 Hourglass 的层交换）。
   - **必须正面回应 A6（共现源循环性）、A7（流行度混淆）、A10（熵-NGCG 零相关反例）**——这三条决定收窄后的设计是否站得住。

4. **本阶段禁止**：不新增 RQ、不设计新方法、不重新引入已停止的 Adaptive Transition-Preserving Quantization（协议 §9 第 12–14 条）。

---

# 9. 未核实项与检索缺口（诚实清单）

以下条目**未取得全文**或**仅有搜索级信息**，不得作为核心证据，引用前须补审：

1. **ReSOT**（KDD'26, 10.1145/3770855.3817834）— ACM DL Cloudflare 403；Semantic Scholar 取得摘要（"Relational Structure Distortion"、GW-OT、Jaccard@10），**正文与代码仅部分核实**（GitHub grw-zju/ReSOT 的 `evaluate.py` 无 Jaccard 函数）。**Level A 潜在，最高优先补审。**
2. **Codebook-Based Semantic IDs in GR**（RecSys, 10.1145/3773078.3831806）— 付费墙；仅摘要 + Semantic Scholar abstract。**框架性风险最高**（其"four fractures"直指「SID 应保持什么」）。
3. **Not All Branches Are Equal / AST**（RecSys, 10.1145/3773078.3831756）— 付费墙；0.758/0.874/0.920 为搜索级。**未核实。**
4. **QuaSID**（KDD, 10.1145/3770855.3818486）— ACM 拦截；检索级。
5. **HG-Rec / HypRQ-VAE**（ICML poster 65614 / OpenReview ALJsIAmO54）— 仅搜索级。
6. **TextRay**（10.1145/3773078.3831887）— 逐层重建表为搜索级。
7. **DiscRec**（Liu et al. 2025，被 2607.27682 引为"content–collaborative exclusivity"论据）— **未获取全文，arXiv 编号未确认**。
8. **MMGRec**（2404.16555）— 已下载但本审计未逐段复核 (iii) 缺口。
9. **Temporal Structure Preservation Through Discrete VQ**（Zenodo）、**SemTop-VQ**（IEEE）— 抓取被拦，仅搜索级。
10. **FORGE**（2509.20904）的 Embedding HitRate/Gini 度量细节 — 本审计为检索级（内部 `doc/prior_art_novelty_audit.md` §11 曾全文精读，可交叉引用）。
11. 多个 2607.x/2608.x/2609.x/2610.x 为 2026 模拟时间线 ID；**已打开的以 `[Paper evidence]` 标记者经 PDF 原文核对**，仅 ACM 条目元数据来自搜索，须复核。

**检索覆盖声明**：本审计覆盖 12 类关键词矩阵、2022–2026（重点 2024–2026）、arXiv + Semantic Scholar API + 官方 GitHub。**「未识别到」严格限于已检索范围**，不等于全局空白（协议 §7 Rule 7）。

---

## 附：审计流程与证据来源

- 协议：`research/literature_audit.md`（未改动）。
- 审计流（5 并行 + 本审计员复核）：
  - 结构保持/失真（VQ/RQ-VAE）：全文 ~26 篇 + 6 篇一般域。
  - 逐层 + 容量分析：19 次检索 + 16 篇全文。
  - SID 前缀-行为映射（RQ4）：独立语料 `/tmp/audit/*/p.txt`。
  - 结构→性能链（RQ6）。
  - 2026 对抗性检索：27 篇语料 `/tmp/audit_sid/*.txt`。
  - 全链先例（Direct Precedent）：6 阶段 × 20 篇逐段核对。
- 本审计员独立原文核验关键条目：TopoTok、SIDInspector、SIDScope、GrIS、Understanding SIDs、DRQ、Latte、Hourglass、Reproducibility Study（均 `pdftotext -layout` 逐字核对）。
