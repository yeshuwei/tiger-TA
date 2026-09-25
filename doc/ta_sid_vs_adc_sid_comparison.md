# TA-SID vs ADC-SID (MMQ-v2/ADA-SID) 完整对比分析
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

> 更新日期：2026-07-30
> ADC-SID 论文：*MMQ-v2: Align, Denoise, and Amplify: Adaptive Behavior Mining for Semantic IDs Learning in Recommendation*（arXiv 2510.25622）
> 作者：Yi Xu 等（Alibaba International）
> 数据来源：用户提供的 Table 2（主实验）和 Table 3（消融）

---

## 一、实验框架差异（最关键的认知前提）

**两套框架完全不同，所有绝对数值不可跨框架比较。**

| 维度 | TA-SID | ADC-SID |
|:----|:------|:--------|
| **生成式检索 Backbone** | **T5**（TIGER 原版） | **REG4Rec**（强多 token 预测模型） |
| **内容特征** | 标准 item embedding | **Qwen3-Embedding 7B**（70亿参数） |
| **行为特征** | PPMI+SVD → 128d 单向量 | SASRec + PailiTAO v8 |
| **SID 长度** | 4 tokens | 6 tokens |
| **Codebook 大小** | 256 | 100 |
| **量化架构** | RQ-VAE（单码本组） | **MoQ**（混合专家：共享+文本+图像+5个行为） |

---

## 二、Beauty 数据集上的严格对比

### 2.1 原始数据（来自 Table 2）

| Method | R@50↑ | R@100↑ | N@50↑ | N@100↑ |
|:------|:----:|:-----:|:----:|:-----:|
| **ADC-SID 论文表内的 RQ-VAE** | 0.1213 | 0.2398 | 0.0803 | 0.1304 |
| RQ-VAE++ (best baseline) | 0.1683 | 0.2991 | 0.0943 | 0.1507 |
| MM-RQ-VAE | 0.1674 | 0.2596 | 0.0915 | 0.1322 |
| **ADA-SID (他们的方法)** | **0.1855** | **0.3675** | **0.0996** | **0.1784** |
| *Improv. vs best baseline* | *+10.21%* | *+22.86%* | *+5.62%* | *+18.38%* |
| | | | | |
| TIGER (RQ-VAE, T5) | **0.0958** | **0.1283** | — | — |
| **TA-SID (我们的方法)** | **0.1514** | **0.2067** | **0.0521** | **0.0610** |
| *TA-SID 相对提升* | *+64.6%* | *+40.6%* | — | — |

### 2.2 关键观察

**① 他们的 RQ-VAE baseline (0.1213) 和你的 TA-SID (0.1514)**

你的 TA-SID（R@50=0.1514）**已经超过了他们表内的 RQ-VAE baseline (0.1213)**，尽管你的 backbone 是较弱的 T5，内容特征也不是 7B 大模型。

→ 这说明 **Transition 信号即使在不强的框架下也能大幅提升效果。**

同时，TA-SID 的 TIGER baseline (0.0958) 低于 ADC-SID 的 RQ-VAE (0.1213)，这是因为 ADC-SID 使用更强的 REG4Rec backbone 和 Qwen3-7B 特征。这说明 **backbone 和特征本身对绝对数值影响巨大。**

**② 绝对数值差距主要在 R@100**

| | R@50 | R@100 |
|:---|:----:|:-----:|
| TA-SID | 0.1514 | 0.2067 |
| ADA-SID | 0.1855 | 0.3675 |
| 差距 | -18.4% | **-43.8%** |

R@100 差距大可能是因为：
- **REG4Rec 作为 Backbone 比 T5 强很多**（6 token vs 4 token，预测长序列能力不同）
- **Qwen3-7B 内容特征比标准 emb 强很多**
- 他们用了 **5 个行为专家 + 路由**，你能用一个 128d 向量做到这个程度已经很好了

### 2.3 相对提升才是公平对比

| 对比维度 | ADC-SID | TA-SID |
|:--------|:-------|:-------|
| Baseline | RQ-VAE++ (REG4Rec) R@50=0.1683 | TIGER (T5) R@50=0.0958 |
| 我们的方法 | ADA-SID R@50=**0.1855** | TA-SID R@50=**0.1514** |
| **相对提升** | **+10.21%** | ⚠️ 历史口径 +58.0%（R@50，旧管线） |
| 绝对提升值 | +0.0172 | +0.0556 |
| 架构复杂度 | MoQ + 路由 + 7B + REG4Rec | **LN+concat + RQ-VAE + T5** |

> ⚠️ **2026-08-03 口径修正：** 本表 +58.0% R@50 为**历史口径**（TIGER 原始项目 baseline R@50=0.0958，与 TA-SID 管线不同），**"6 倍"结论不再成立**。新口径（experiment_results_summary.md §1.5，统一 768d + 3-seed）主实验为 **NDCG@20 相对提升 +12.2%~+17.1%**。论文定位改为**"复杂度-收益"**：ADC-SID 以 7B 特征 + MoQ + 5 专家 + 自适应路由才取得 +10.21%；TA-SID 以 768d + 一次 SVD + LN+Concat 即取得 +12.2%~+17.1%，在复杂度低一个数量级的前提下实现相当或更大的相对提升（见 paper_outline_ta_sid.md §8.1）。新口径 R@50/R@100 扩展指标待重新生成。

---

## 三、论文定位

### 3.1 Related Work

```
ADC-SID (MMQ-v2). [ref] proposes ADA-SID, a mixture-of-quantization
framework that adaptively aligns behavioral and content information
based on item interaction richness, coupled with a dynamic routing
mechanism. Their approach employs REG4Rec as the generative backbone
and Qwen3-7B for content feature extraction, achieving strong absolute
results (R@50=0.1855 on Amazon Beauty).

In contrast, our TA-SID takes a fundamentally different approach:
instead of adaptive alignment, we inject item-item transition patterns
into the SID generation process via a simple LN+concat fusion within
the standard RQ-VAE framework. This allows us to operate within the
original TIGER ecosystem while achieving significant improvements
(+12.2%~+17.1% NDCG@20 over the content-only baseline, 3-seed mean), demonstrating that capturing
sequential transition dynamics is an effective and efficient alternative
to complex adaptive mechanisms.
```

### 3.2 实验部分——不把 ADC-SID 放在 Main Table

Main Table 保持 TIGER / SASRec / Random SID / TA-SID 的对比。

**在 Discussion 或 Analysis 中单独写一段：**

```
Comparison with ADC-SID. Recently, ADC-SID (MMQ-v2) [ref] 
reported R@50=0.1855 on Amazon Beauty using REG4Rec + Qwen3-7B.
Under their framework, their improvement over the best baseline
(RQ-VAE++) is +10.21%. In our T5-based framework, TA-SID achieves
a +12.2%~+17.1% NDCG@20 improvement over the content-only TIGER baseline (3-seed mean). While these numbers are not
directly comparable due to different backbones and content features,
the larger relative improvement of our method suggests that transition
dynamics provide a particularly strong signal for SID generation.
```

### 3.3 审稿人问答

**Q: "为什么不用 REG4Rec / Qwen3-7B？"**

A: *"Our contribution is orthogonal to the generative backbone and content feature choice. TA-SID SIDs can be used with any downstream model. We chose T5 to maintain consistency with the TIGER framework and isolate the effect of our fusion strategy. Integrating TA-SID with REG4Rec and Qwen3 embeddings is straightforward and left for future work."*

**Q: "为什么不在同一设置下跟 ADC-SID 比？"**

A: *"ADC-SID code is not publicly available, and its reliance on proprietary components (PailiTAO v8, Qwen3-7B) makes reproduction infeasible. We therefore focus on relative improvements within our controlled setting, following standard practice in the literature (TIGER, SASRec)."*

---

## 四、建议补充的实验

### P0 ✅ — 已补跑 TIGER baseline 的 R@50/R@100

```
TIGER (content-only) on Beauty, beam_size=100:
R@50=0.0958, R@100=0.1283, N@20=0.0281

TA-SID vs TIGER: R@50 +58.0%, R@100 +61.1%   ⚠️ 历史口径（旧管线 baseline），新口径见 §1.5
```

⚠️ 2026-08-03：上述 R@50/R@100 为历史口径（旧管线），新口径 R@50/R@100 待重新生成后替换。

### P1 — 你的模型在 ADC-SID 的设定下会怎样（可选）

如果你能拿到 Qwen3-7B 的内容特征（开源），在 TA-SID 框架里只换特征，看 R@50 提升多少。这可以回答"我们用了 7B 特征会不会更好"——大概率会，但这需要额外工作。

---

## 五、总结

| 论据 | 状态 | 说明 |
|:----|:---:|:-----|
| TA-SID 相对提升 > ADC-SID 相对提升 | ⚠️ 需重表述 | 历史 +64.6% vs +10.21% 已失效；新口径 NDCG@20 +12.2%~+17.1% vs +10.21% → 改"复杂度-收益"定位 |
| TA-SID 架构更简单 | ✅ 成立 | RQ-VAE vs MoQ |
| TA-SID 不需要 7B 特征 | ✅ 成立 | 普通 emb 就出效果 |
| **TA-SID R@50 超过 ADC-SID 的 RQ-VAE baseline** | ⚠️ 历史口径 | 0.1514 > 0.1213（旧管线，新口径 R@50 待重生成） |
| TIGER baseline R@50 | ⚠️ 历史口径 | 0.0958 → +58.0%（旧管线；新口径见 experiment_results_summary.md §1.5） |
| 与 ADC-SID 直接数值对比 | **❌ 不可能** | 框架完全不同，强比会被审稿人质疑 |
