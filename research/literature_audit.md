# Semantic ID × Behavior Structure — Literature Audit Protocol

## 0. 你的角色

你现在不是论文写作助手，也不是研究方向设计助手。

你的角色是：

> **严格的研究问题文献审计员（Literature Auditor）**

目标不是证明我的研究方向成立，而是主动寻找能够否定、削弱或覆盖该研究问题的已有工作。

必须采取“审稿人 / advisor / novelty auditor”的立场。

---

# 1. 当前研究问题背景

我们正在研究生成式推荐中的 Semantic ID。

当前关注的问题链为：

Behavioral structure
→ continuous item representation
→ RQ-VAE / residual vector quantization
→ discrete Semantic ID
→ behavioral structure preservation / distortion
→ generative recommendation

当前候选研究问题：

### RQ1

行为结构从连续 item representation 到最终 Semantic ID 的过程中，是否发生系统性的结构损失？

### RQ2

如果存在结构损失，它主要发生在哪一个 RQ-VAE / RVQ quantization stage？

### RQ3

不同类型的行为结构是否具有不同的可保持性？例如：

* 强 / 弱 item-item behavioral association
* behavioral nearest neighbors
* local co-occurrence structure
* item frequency / popularity-related structure

### RQ4

行为相似性是否会映射到 Semantic ID 的 prefix/code-sharing structure？

### RQ5

有限 codebook capacity 是否会影响 behavioral structure preservation，并形成 semantic structure 与 behavioral structure 之间的 trade-off？

### RQ6

behavioral structure preservation 是否与最终 generative recommendation performance 有系统关系？

---

# 2. 最重要的总问题：RQ0

首先审计：

> 是否已经存在工作完整研究了以下链条：

Behavior / collaborative signal
→ item representation
→ vector quantization / RQ-VAE
→ Semantic ID
→ preservation / distortion of behavioral or collaborative structure
→ generative recommendation

如果存在高度相似工作，必须优先报告。

不能因为论文没有使用“behavior structure loss”这个术语，就判断它与本问题无关。

必须检查：

* 方法是否实质上相同
* 研究对象是否相同
* 分析指标是否实质相同
* 机制解释是否相同
* 是否已经做了 layer-wise / capacity / neighborhood / prefix analysis
* 是否已经把结构保持与推荐效果联系起来

---

# 3. 文献检索范围

优先覆盖：

1. Semantic ID / generative recommendation
2. RQ-VAE / residual vector quantization
3. behavior-aware / collaborative-aware Semantic ID
4. collaborative-enhanced tokenization
5. item-item behavioral / transition / co-occurrence structure
6. discrete item representation
7. hierarchical / prefix-based Semantic ID
8. neighborhood preservation
9. similarity / topology / graph structure preservation
10. quantization distortion / information loss
11. codebook capacity / quantization capacity
12. discrete representation learning for recommendation

时间范围：

* 优先覆盖 2022–2026
* 特别检查 2024–2026
* 对关键早期工作可向前追溯

必须同时检查：

* 顶会论文
* arXiv
* conference proceedings
* journal papers
* official project pages
* official GitHub / supplementary materials（仅用于验证实现）

---

# 4. 检索策略

禁止只使用一个关键词。

必须建立 query matrix。

至少覆盖以下组合：

### Semantic ID × behavior

* "Semantic ID" + behavior
* "Semantic ID" + behavioral
* "Semantic ID" + collaborative
* "semantic identifier" + collaborative
* "item tokenization" + collaborative
* "generative recommendation" + collaborative tokenization

### Semantic ID × structure

* "Semantic ID" + neighborhood
* "Semantic ID" + similarity
* "Semantic ID" + structure preservation
* "Semantic ID" + structural preservation
* "Semantic ID" + topology
* "Semantic ID" + locality
* "Semantic ID" + prefix

### Quantization × behavior

* "RQ-VAE" + collaborative
* "RQ-VAE" + behavior
* "residual vector quantization" + collaborative
* "vector quantization" + recommendation
* "quantization" + behavioral structure

### Quantization × preservation

* "vector quantization" + structure preservation
* "residual quantization" + structure preservation
* "quantization" + neighborhood preservation
* "quantization" + similarity preservation
* "quantization" + topology preservation
* "quantization" + information loss

### Codebook × structure

* "codebook capacity" + structure
* "codebook size" + representation preservation
* "codebook" + neighborhood
* "codebook" + collaborative
* "discrete representation" + neighborhood preservation

---

# 5. 每篇论文必须完成“证据级审计”

对于每篇核心论文，必须记录：

| 字段                   | 要求                                         |
| -------------------- | ------------------------------------------ |
| Paper                | 完整标题                                       |
| Authors              | 作者                                         |
| Venue                | 会议/期刊/arXiv                                |
| Year                 | 年份                                         |
| URL                  | 官方论文链接                                     |
| Code                 | 官方代码（如有）                                   |
| Main problem         | 作者真正解决的问题                                  |
| Input representation | Semantic ID 的输入是什么                         |
| Behavior signal      | 是否使用行为/协同信息                                |
| Quantization         | 是否使用 RQ-VAE / VQ / RVQ                     |
| SID construction     | 如何构造 SID                                   |
| Structure analysis   | 是否分析结构保持                                   |
| Layer-wise analysis  | 是否分析不同量化层                                  |
| Capacity analysis    | 是否分析 codebook capacity                     |
| Prefix analysis      | 是否分析 prefix/code sharing                   |
| Recommendation       | 是否验证生成式推荐                                  |
| Key evidence         | 原文具体证据                                     |
| Evidence location    | section / figure / table / equation / page |
| Relevance            | 与当前 RQ 的相关程度                               |
| Overlap              | 与当前问题的重合程度                                 |

---

# 6. 必须区分四种相关性

不要简单写“相关 / 不相关”。

使用：

### Level A — Direct overlap

论文直接研究与当前 RQ 基本相同的问题。

### Level B — Mechanistic overlap

研究机制高度相似，即使术语不同，也可能覆盖当前问题。

### Level C — Partial overlap

只覆盖当前问题的一部分。

### Level D — Background only

仅提供背景，不构成问题重合。

特别注意：

> “论文使用 behavior-aware SID”

并不自动意味着：

> “论文研究了 behavior structure preservation”。

必须根据原文证据判断。

---

# 7. 证据可靠性规则

这是本审计最重要的部分。

### Rule 1

任何“论文已经做过 X”的结论，都必须能够定位到原论文证据。

至少提供：

* section
* figure/table/equation
* page（如果可获得）

### Rule 2

搜索引擎 snippet 不得作为最终证据。

搜索结果只能用于发现论文。

### Rule 3

摘要只能支持摘要中明确表达的内容。

不能根据摘要推断论文进行了未明确描述的实验。

### Rule 4

官方代码只能证明代码实现了什么。

不能把代码功能自动写成论文贡献。

### Rule 5

如果无法确认，必须写：

> “Evidence insufficient / unable to verify.”

不能猜测。

### Rule 6

区分：

* Paper explicitly states
* Paper experimentally demonstrates
* Code indicates
* Auditor inference

不得混淆。

### Rule 7

禁止因为“没有搜到”而写：

> “没有任何研究做过 X。”

只能写：

> “Within the searched literature and available evidence, no direct work was identified that explicitly studies X.”

---

# 8. 对每一个 RQ 做独立审计

最终必须分别回答：

## RQ1

是否已有工作直接测量：

continuous representation
→ quantized representation
→ Semantic ID

过程中 behavioral / collaborative structure 的损失？

重点寻找：

* AUC
* Recall@K
* Jaccard
* neighborhood overlap
* similarity correlation
* topology preservation
* graph structure distortion
* quantization distortion

---

## RQ2

是否已有：

* RVQ layer-wise analysis
* residual-wise analysis
* codebook-level analysis
* coarse-to-fine structural preservation analysis

如果已有，必须说明：

> 它分析的结构是什么？

不能仅因为它做了 layer-wise quantization analysis，就认为它解决了 behavioral structure loss。

---

## RQ3

是否已有工作区分：

* strong vs weak associations
* local vs global structure
* frequent vs infrequent items
* neighborhood structure
* co-occurrence structure

并分析它们在 quantization 中的不同保持程度？

---

## RQ4

重点审计：

> behavioral similarity 是否映射到 SID prefix similarity / code sharing？

寻找：

* shared prefix
* hierarchical code
* prefix locality
* token-level locality
* code-level neighborhood
* semantic locality
* discrete neighborhood

如果已有研究只证明“semantic similarity → SID similarity”，不要自动认为它解决了“behavior similarity → SID similarity”。

---

## RQ5

重点审计：

> codebook capacity 是否造成 behavioral structure preservation 的变化或 trade-off？

检查：

* K sweep
* codebook size
* quantization capacity
* utilization
* collision
* distortion
* information bottleneck
* semantic/behavior trade-off

如果已有工作只研究 reconstruction loss / semantic quality，而没有 behavioral structure，不要写成已经解决。

---

## RQ6

重点审计：

> structure preservation 是否与 generative recommendation performance 有系统关系？

检查是否已有：

* structure preservation → Recall/NDCG
* quantization quality → recommendation
* code locality → recommendation
* neighborhood preservation → recommendation
* correlation analysis
* intervention / ablation
* causal-style evidence

特别区分：

### Correlation

结构保持越好，推荐越好。

### Causal/intervention evidence

主动改变结构保持程度，并观察推荐变化。

二者不能混淆。

---

# 9. 必须重点检查最近工作

重点审计 2024–2026 年：

* Semantic ID
* behavior-aware Semantic ID
* collaborative-enhanced tokenization
* generative recommendation
* RQ-VAE / RVQ-based item tokenization
* discrete item representation

对于每篇高度相关论文必须回答：

> 如果审稿人看到我们的 RQ，他会不会说：

> “This has already been studied in [Paper X].”

如果答案可能是“是”，必须明确标红并解释重合点。

---

# 10. 最终输出结构

最终生成：

`research/literature_audit.md`

结构如下：

# 1. Executive Summary

用不超过 1000 字回答：

1. 当前问题是否已有直接先例？
2. 哪些 RQ 已经被覆盖？
3. 哪些 RQ 只有部分覆盖？
4. 哪些 RQ 目前仍有明显空缺？
5. 最危险的 novelty overlap 是什么？
6. 是否建议继续进入实验阶段？

---

# 2. Literature Landscape

按照：

* Semantic ID
* behavior-aware SID
* collaborative tokenization
* quantization analysis
* structure preservation
* generative recommendation

分类总结。

---

# 3. Key Paper Audit

使用表格：

| Paper | Year | Main idea | Behavior | Quantization | Structure preservation | Layer-wise | Capacity | Prefix | Recommendation | Overlap |
| ----- | ---: | --------- | -------- | ------------ | ---------------------- | ---------- | -------- | ------ | -------------- | ------- |

---

# 4. RQ-by-RQ Audit

分别输出：

## RQ1

* 已有研究
* 最近似工作
* 已解决程度
* 尚未解决
* 重合风险
* 结论

RQ2–RQ6 同样。

---

# 5. Direct Precedent Audit

单独回答：

> 是否存在一篇工作已经覆盖：

Behavior
→ representation
→ quantization
→ Semantic ID
→ structure preservation
→ recommendation

如果存在，详细说明。

如果没有，也必须说明：

> “No direct precedent was identified within the searched literature.”

而不是写：

> “No one has done this.”

---

# 6. Novelty Risk Matrix

| RQ  | Direct precedent | Mechanistic overlap | Partial overlap | Novelty risk | Continue? |
| --- | ---------------- | ------------------- | --------------- | ------------ | --------- |
| RQ1 |                  |                     |                 |              |           |
| RQ2 |                  |                     |                 |              |           |
| RQ3 |                  |                     |                 |              |           |
| RQ4 |                  |                     |                 |              |           |
| RQ5 |                  |                     |                 |              |           |
| RQ6 |                  |                     |                 |              |           |

风险分级：

* 🔴 High
* 🟠 Medium
* 🟢 Low

---

# 7. Reviewer Attack Test

假设你是 SIGIR / WWW / KDD / CIKM 审稿人。

请主动提出至少 10 个可能的质疑，例如：

* “This problem has already been studied by ...”
* “The proposed analysis is merely a diagnostic of existing RQ-VAE behavior.”
* “Behavior-aware tokenization already preserves collaborative structure.”
* “The claimed structural loss is an artifact of the chosen metric.”
* “The behavior representation and evaluation criterion are derived from the same co-occurrence source.”
* “The observed effect is caused by popularity rather than behavioral structure.”
* “Codebook capacity analysis is standard quantization analysis.”
* “The correlation with recommendation performance is not causal.”

然后逐条判断：

> Existing literature already addresses this criticism?
> Or does it remain an open issue?

---

# 8. Final Adjudication

最后只允许给出以下三种结论之一：

### A. Continue

现有文献没有直接覆盖核心问题，可以进入最小诊断实验。

### B. Narrow

问题有研究基础，但需要缩小/重新定义研究问题。

### C. Abandon

已有工作已经直接覆盖核心问题，不建议继续。

禁止在 literature audit 阶段自行提出新的“创新点”或新的方法。

---

# 9. 严格禁止事项

禁止：

1. 为了证明 novelty 而刻意忽略相似论文。
2. 只搜索 Semantic ID，不搜索 quantization / representation preservation 文献。
3. 只看摘要。
4. 只依赖搜索引擎 snippet。
5. 把论文中的 inference 当成实验事实。
6. 把代码实现当成论文贡献。
7. 把“behavior-aware”自动等同于“behavior structure preservation”。
8. 把“semantic locality”自动等同于“behavior locality”。
9. 把 correlation 写成 causality。
10. 把“没有找到”写成“没有人做过”。
11. 在没有证据的情况下判断 novelty。
12. 在本阶段设计新的方法。
13. 在本阶段扩展新的 RQ。
14. 将已经明确停止的 Adaptive Transition-Preserving Quantization 重新作为第二创新方向。

---

# 10. 审计原则

核心原则只有一句：

> **宁可判定一个 RQ 不新，也不要在证据不足时判定它新。**

本审计的目标不是帮助论文“找到创新点”，而是尽可能提前发现：

> **这个问题为什么可能不值得做。**
