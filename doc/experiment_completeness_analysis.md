# 实验完成度评估与后续计划
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

## 一、核心贡献完成情况（方案 §3）

### Contribution 1：行为信号引入 RQ-VAE 量化
| 要求 | 状态 | 证据 |
|:----|:---:|:-----|
| Behavior emb 融入 fused embedding | ✅ | LN(content·768d) + LN(behavior·128d) → concat → 896d |
| 三数据集验证 | ✅ | Sport/Toys/Beauty 全部完成 |

### Contribution 2：Transition-enhanced item representation
| 要求 | 状态 | 证据 |
|:----|:---:|:-----|
| Item transition graph | ✅ | `build_transition_graph.py` PPMI+SVD→behavior_emb |
| Content+behavior 融合 | ✅ | `fuse_embeddings.py` → fused_item_emb.parquet |

### Contribution 3：Plug-in 替换提升
| 要求 | 状态 | 证据 |
|:----|:---:|:-----|
| 不改 T5 只换 SID → 提升 | ✅ | 三数据集 +18%~+58% NDCG@20 |
| SASRec 基线对比 | ✅ | +25%~+99% |
| Random SID 消融 | ✅ | 同 pipeline 对比 |

**三个核心贡献全部完成。** 不需要 L_trans。

---

## 二、语义 ID 质量指标（方案 §7.4）

| 指标 | 状态 | 说明 |
|:----|:---:|:-----|
| Codebook utilization | ✅ 完成 | L1: 79.7%~97.3%, L2/L3: 100% |
| Collision rate | ✅ 完成 | 三数据集全部 0% |
| Transition consistency | ✅ 完成 | Beauty 0.894, Sport 0.830, Toys 0.860 |
| **Invalid ID rate** | ❌ **未做** | 需修改 eval 函数，~0.5h 工作量 |

---

## 三、消融实验（方案 §8）

| 消融变体 | 状态 | 必要性 |
|:---------|:---:|:------:|
| Content only (TIGER) | ✅ 三数据集 | **核心基线** |
| Content + Behavior (TA-SID) | ✅ 三数据集 | 主模型 |
| Random SID | ✅ 三数据集 | 码质量消融 |
| **Content + Random emb** | **✅ Sport ✅** | **Transition vs 噪声——已补，结果干净** |
| Behavior only | ❌ 未跑 | 已不重要（Transition vs Random 消融更具说服力） |
| Content+Behavior+L_trans | ❌ 放弃 | 已证无效 |

---

## 四、基线对比（方案 §7.2 + 三篇论文对照）

### 你已经有的
| 基线 | 你的数据 | TIGER 原文 | ADC-SID | RPG |
|:----|:-------:|:---------:|:-------:|:---:|
| SASRec | ✅ 三数据集 | ✅ | ❌ | ✅ |
| GRU4Rec/Caser | ❌ | ✅ | ❌ | ❌ |
| BERT4Rec | ❌ | ✅ | ❌ | ❌ |
| S3-Rec | ❌ | ✅ | ❌ | ❌ |
| LSH SID | ❌ | ✅ | ❌ | ❌ |
| Random SID | ✅ | ✅ | ❌ | ✅(OPQ→Random) |
| RQ-VAE SID (TIGER) | ✅ | ✅ | ✅ | ✅ |
| **ADC-SID** | ❌ | ❌ | ✅ | ❌ |

### 关键发现

**ADC-SID** 是你最相关的竞品，它也做内容+行为融合，但用的是**自适应融合**（每个 item 动态决定内容 vs 行为的权重）。你的方案是固定的 LN+Concat。ADC-SID 在 Beauty R@50 上 +10.21%。

**RPG** 在 Beauty NDCG@10=0.0464，远高于你的 0.0326。但 RPG 用了 64 token 的长语义 ID + 图约束解码 + OPQ 并行量化，方法完全不同，不是直接竞争。

---

## 五、分析实验对照

| 实验 | 你的状态 | TIGER 有？ | ADC-SID 有？ | RPG 有？ |
|:----|:-------:|:---------:|:-----------:|:-------:|
| Longtail (Head/Mid/Tail) | ✅ 完成(Sport+Toys) | ❌ | ✅ | ❌ |
| Cold-start | ✅ 完成(Sport+Toys) | ✅ | ✅(New items) | ❌ |
| t-SNE 可视化 | ✅ 完成(9张图) | ❌ | ❌ | ❌ |
| Diversity (temperature) | ❌ 未跑 | ✅ | ❌ | ❌ |
| SID 质量指标 | ✅ 完成 | ❌ | ❌ | ❌ |

---

## 六、后续实验建议（按优先级排列）

### ① ADC-SID 对比定位（已完成为文档，见 `ta_sid_vs_adc_sid_comparison.md`）

### ② 补充 R@50/R@100 评估（~0.5h，重要性：高）
重新跑 T5 evaluation（beam_size=100, topk=[50,100]），获得与 ADC-SID 直接可比的数值。已有完整 Beauty checkpoint，无需重新训练。

### ③ 内容 vs 行为 embedding 正交性分析（~0.5h，重要性：高）
利用已有 embedding 数据（item_emb.parquet + behavior_emb.parquet），证明 transition 信号编码了内容之外的独立信息。

### ④ Invalid ID Rate（~0.5h，重要性：低）
✅ 已完成（三数据集 ≈ 0%），仅用于质量报告。

### ② Behavior-only 消融（~4h，重要性：高）
只保留 behavior emb（128d）做 RQ-VAE 输入，可以清晰回答"行为语义单独能不能生成有用的 SID？"
- 必要性：消融表里缺这一列，审稿人可能会问
- 工作量：一个新的 RQ-VAE 训练（~2h）+ T5 训练（~2h），可选 1 个数据集（Sport）

### ③ LSH SID 基线（~2h，重要性：低）
TIGER 论文里 LSH < RQ-VAE < TA-SID 的层次可被完整复现。但既然 TA-SID 已经超过 RQ-VAE，LSH 的结果是隐含的，**不一定需要做**。

### ④ Diversity 实验（~2h，重要性：低）
调整 T5 生成温度（0.5/1.0/1.5/2.0），看 Entropy@K 变化。TIGER 论文有这个实验，但你的论文故事线里不是必需的。

---

## 七、论文写作前必须做的非实验工作

| 任务 | 重要性 | 说明 |
|:----|:-----:|:-----|
| 读 ADC-SID 论文 | **极高** | 最相关竞品，需要定位你的创新差异 |
| 读 LIGER 论文 | **高** | 支撑冷启动讨论（TIGER family 冷启动=0 的引用来源） |
| 读 RPG 论文 | 中 | 了解 generative retrieval 前沿，但方法差异大 |
| 写冷启动讨论 | 高 | 诚实说明 TIGER family 的固有限制，引用 LIGER |

---

## 八、总结

| 类别 | 完成度 | 还需要做 |
|:----|:-----:|:---------|
| 核心贡献（3 个） | **100%** | 无 |
| SID 质量指标 | **100%** | ✅ 全部完成 |
| 消融实验 | **90%** | ✅ Content+Random emb 已完成（Sport），Transition 净贡献明确 |
| 基线对比 | **80%** | ✅ ADC-SID 文献对比已完成 |
| 分析实验 | **85%** | ✅ emb 正交性分析已完成 |
| 文献综述 | **30%** | ADC-SID 已读 ✅，LIGER/RPG 未读 |

**最重要的事（按顺序）：**
1. ✅ ADC-SID 论文对比完成（见 `ta_sid_vs_adc_sid_comparison.md`）
2. ✅ R@50/R@100 评估完成
3. ✅ 内容 vs 行为 embedding 正交性分析完成
4. ✅ **Transition vs Random 噪声消融完成**——证明 Transition 净贡献 +16.4%
5. 读 LIGER（冷启动引用）
