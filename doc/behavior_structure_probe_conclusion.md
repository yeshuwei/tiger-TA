# Behavior Structure Preservation vs Recommendation Performance — 结论

**日期**：2026-09-23　**产出脚本**：`tools/behavior_structure_probe.py` + `tools/plot_behavior_auc_vs_ndcg.py`
**原始结果**：[behavior_structure_probe.md](behavior_structure_probe.md)（自动生成）· `doc/behavior_structure_probe_results.csv`（长表）
**图**：`doc/figures/behavior_auc_vs_ndcg_full.png` / `..._l0.png`（另有 `_all_degmatch` 口径两份）

---

## 0. 一句话结论

> **SID 确实保留了行为结构，且其保留程度与下游 NDCG@20 排序高度一致（数据集内 Spearman ρ = 0.94，两数据集一致；数据集内标准化后合并 r = 0.935, p < 0.0001）。但该指标有一层高达 0.88–0.91 的「content 地板」，且绝对值跨数据集不可比 —— 故它可作**同一数据集内的 SID 排序指标**，不可作跨数据集的绝对质量阈值。**

---

## 1. 回答三个问题

### Q1. 行为结构是否被 SID 保留？

**是，但需分层看。** 6 个 variant 的 `xq[full]` Behavior AUC 全部显著高于 0.5（Sport 0.876–0.965 / Toys 0.813–0.965，`topk`×`degmatch` 口径），说明量化后的 SID 空间普遍保留了「哪些商品经常一起出现」这一结构。

但关键分层事实是：**完全没有行为信息的 Content-only 就拿到了 0.908（Sport）/ 0.875（Toys）**，即行为结构判据的绝大部分可由 content 相似度单独满足（content 相似的商品本来就更可能被同一用户依次点击）。真正由行为向量**增量**贡献的部分是 0.06–0.09。

### Q2. 不同 SID 方法的行为保持能力是否与推荐效果一致？

**一致。** 两个数据集、两种正样本口径、两个空间（`xq[full]`/`xq[L0]`）下，排序**恒为**

```
TA-SID > Raw-CoOcc > ItemCF > {Content-only, ZeroPad} > Shuffled
```

与 NDCG@20 排序相比，**每个数据集恰好只有 1 处倒置**，且该倒置对在全部口径下稳定复用：

| 数据集 | Behavior AUC 排序 | NDCG@20 排序 | 倒置处 |
|:--|:--|:--|:--|
| Sport | TA-SID > Raw > ItemCF > **Content > ZeroPad** > Shuffled | TA-SID > Raw > ItemCF > **ZeroPad > Content** > Shuffled | Content↔ZeroPad（NDCG 差 0.0005，在 3-seed std 内） |
| Toys | TA-SID > Raw > **ItemCF** > Content > ZeroPad > Shuffled | TA-SID > **ItemCF > Raw** > Content > ZeroPad > Shuffled | Raw↔ItemCF（NDCG 差 0.0015） |

注意 Sport 的那处倒置是 **Content vs ZeroPad**，而二者在行为上**本就应当等价**（ZeroPad = content + 128 维零向量，行为信息为零）—— 它们的 Behavior AUC 差 0.0011、NDCG 差 0.0005，都在噪声内。**换言之，唯一的"不一致"出现在两个按理应当打平的点之间，不构成对排序一致性的实质反例。**

> ⚠️ 该对相邻点的先后会随口径翻转：`doc/e49grid_diag_summary.md` 在本格报"Sport 6/6 同序"，
> 即因其负样本池未剔除 E 内边而把 ZeroPad 排到了 Content 之上。分歧的成因与结论见 §6.1，
> **对外不要主张 6/6 或 4/6**。

### Q3. Behavior AUC 是否可以作为 SID 质量评价指标？

**可以，但仅限同一数据集内的相对排序。** 支持与限制并列：

**支持**
- 数据集内 Spearman ρ = 0.9429（Sport 与 Toys **完全一致**），p = 0.0048；
- Pearson r = 0.98 / 0.89（`xq[full]`）、0.99 / 0.96（`xq[L0]`）；
- 数据集内 z-score 后合并 12 点，Pearson r = **0.9354，p < 0.0001**（消掉数据集间整体偏移后仍然成立）；
- **剂量–响应自洽**：加对齐行为（TA-SID）→ 最高 AUC + 最高 NDCG；加错配行为（Shuffled）→ 最低 AUC + 最低 NDCG（且**低于** Content-only）；加零行为（ZeroPad）≈ Content-only（两项都打平）。三档干预给出一致的单调关系。

**限制（不可忽略）**
1. **Content 地板高**：0.875–0.908 的 AUC 与行为信息无关。指标主要度量的是「embedding 在共现图上是否平滑」，而非「行为结构是否被编码」。用它给 SID 打分时，必须同时报出 Content-only 基线，否则无法区分"行为带来的增益"与"content 本身就有的相似度"。
2. **跨数据集绝对值不可比**：Sport 的 AUC 区间 0.876–0.965、Toys 0.813–0.965，同分位点差异达 0.06；直接 Pooled 的 Pearson 只有 0.48（带偏移），必须数据集内标准化后才恢复到 0.94。**故该指标不能设跨数据集的绝对阈值。**
3. **定义优势**：判据图 E 由 `window=3` 相邻对构成，而这正是 TA-SID / Raw-CoOcc 构造行为向量所用的同一张图 —— 这两个 variant 相对 Content-only / ZeroPad 有天然对齐红利。ItemCF 用的是**另一张图**（整篮共现）却仍排第 3，是该隐忧的一点缓解，但不消除。
4. **样本量极小**：每个数据集 n = 6 个点。ρ = 0.94 已是 n=6 下除完美外的次高值，但 6 点的相关性检验检出力很弱，**不宜单独作为论文的核心论证**，应与消融表的逐变体对照并用。

---

## 2. 主表（`xq[full]`，`topk` 正样本 × `degmatch` 负样本）

| 数据集 | Variant | Behavior AUC | NDCG@20 | Hamming AUC | L0-match AUC |
|:--|:--|--:|--:|--:|--:|
| Sport | **TA-SID** | **0.9647** | **0.0322** | **0.7498** | 0.7450 |
| Sport | Raw-CoOcc | 0.9469 | 0.0305 | 0.6881 | 0.6688 |
| Sport | ItemCF | 0.9397 | 0.0295 | 0.7374 | 0.7179 |
| Sport | Content-only | 0.9080 | 0.0275 | 0.6856 | 0.6849 |
| Sport | ZeroPad | 0.9069 | 0.0280 | 0.6590 | 0.6558 |
| Sport | Shuffled | 0.8763 | 0.0264 | 0.6253 | 0.6283 |
| Toys | **TA-SID** | **0.9651** | **0.0377** | **0.8416** | **0.8370** |
| Toys | Raw-CoOcc | 0.9543 | 0.0340 | 0.7655 | 0.7620 |
| Toys | ItemCF | 0.9455 | 0.0355 | 0.7796 | 0.7735 |
| Toys | Content-only | 0.8751 | 0.0336 | 0.7172 | 0.7135 |
| Toys | ZeroPad | 0.8716 | 0.0334 | 0.6957 | 0.6910 |
| Toys | Shuffled | 0.8127 | 0.0284 | 0.6225 | 0.6215 |
| Beauty | TA-SID | 0.9560 | 0.0415 | 0.8847 | 0.8655 |
| Beauty | Content-only | 0.8365 | 0.0358 | 0.6584 | 0.6435 |

> **Beauty 缺 4/6 变体**（无 Raw-CoOcc / ItemCF / ZeroPad / Shuffled 的 ckpt 与 codes），
> 仅 2 点，**不做相关分析**，仅作方向性描述：TA-SID 在 AUC 与 NDCG 上双双高于 Content-only，
> 与 Sport/Toys 同向。

## 3. 相关分析（Behavior AUC vs NDCG@20）

| 空间 | 正样本口径 | 指标 | 范围 | n | Pearson | Spearman | p (Spearman) |
|:--|:--|:--|:--|--:|--:|--:|--:|
| xq[full] | topk | cosine sim | Sport | 6 | 0.978 | **0.943** | 0.005 |
| xq[full] | topk | cosine sim | Toys | 6 | 0.893 | **0.943** | 0.005 |
| xq[full] | topk | cosine sim | Pooled (z-scored) | 12 | **0.935** | — | <0.0001 |
| xq[full] | all | cosine sim | Sport | 6 | 0.965 | 0.943 | 0.005 |
| xq[full] | all | cosine sim | Toys | 6 | 0.906 | 0.943 | 0.005 |
| xq[full] | all | cosine sim | Pooled (z-scored) | 12 | **0.935** | — | <0.0001 |
| xq[L0] | topk | cosine sim | Sport | 6 | 0.990 | 0.943 | 0.005 |
| xq[L0] | topk | cosine sim | Toys | 6 | 0.960 | 0.886 | 0.019 |
| xq[full] | topk | Hamming | Toys | 6 | 0.962 | **1.000** | <0.001 |
| xq[full] | topk | Hamming | Sport | 6 | 0.834 | 0.886 | 0.019 |
| xq[full] | topk | Pooled (z-scored) | 12 | 0.898 | — | <0.0001 |

> **不写 Pooled 未标准化值**：Sport 与 Toys 的 AUC/NDCG 各有整体量纲差，直接合并会被
> offset 主导（Pearson 仅 0.48）。上表 Pooled 一律指「数据集内 z-score 后合并」。

## 4. 稳健性：两轴口径 × 两个空间

- **正样本口径**：`topk`（频次最高 1000 条边，`count≥6`）与 `all`（从全部 E 均匀采样 2×10⁴ 条，
  含 `count=2` 的长尾弱边）给出**同一排序**，仅绝对值整体平移（如 TA-SID Sport `xq[full]`：
  0.9647 → 0.7718）。ρ 在两种口径下皆为 0.9429。
- **空间**：`xq[full]`（3 层码本向量之和）与 `xq[L0]`（仅第 1 层）**排序一致**（两数据集皆然），
  但 `xq[L0]` 的 AUC 系统性更低，差距最大 0.0497（Toys Shuffled：0.7630 vs 0.8127）——
  即后两层码本确实进一步承载了行为结构。
  `z`（量化前 latent，容量天花板）与 `xq[full]` 几乎重合（14 格中最大差 **0.0063**），
  说明**行为结构在量化中基本无损**，与本项目 `doc/e49grid/` 的既有结论一致。
- **离散 token**：Hamming（前 3 个 token 差异位数）在 Toys 上 ρ = 1.000（完美），Sport 0.886；
  `all` 口径下弱边主导时 Hamming 判别力退化到 ~0.53–0.59。离散指标对正样本强弱更敏感。

## 5. 负采样控制的有效性（本轮的**新增**方法要点）

`degmatch` 负采样按 ⌊log₂deg⌋ 分桶、候选取自与正样本端点同桶，等比均匀随机负采样显著地
堵住了流行度捷径：

| 数据集 | 正样本口径 | 正样本端点 degree 均值 | degmatch 负样本 | uniform 负样本 |
|:--|:--|--:|--:|--:|
| Sport | topk | 179.7 | 162.0 (Δ=−17.7) | 20.2 (**Δ=−159.5**) |
| Sport | all | 65.6 | 63.0 (Δ=−2.6) | 19.8 (Δ=−45.8) |
| Toys | topk | 59.0 | 55.7 (Δ=−3.2) | 16.6 (Δ=−42.3) |

均匀负样本的端点流行度只有正样本的 1/9（Sport topk），即分类器原本可凭"热门商品彼此都相似"
拿分。这与 `tools/behavior_space_diagnostic.py` 对 `cons_auc` 系统性定义优势的告警同源。
**即便如此，两种负采样口径给出的变体排序一致**（ρ 同为 0.9429），说明结论不是被该捷径撑起来的。

## 6. 产物与复现

```bash
# 主实验（CPU，分钟级；只读，不训练）
python3 tools/behavior_structure_probe.py --datasets Sport,Toys,Beauty \
  --spaces "xq[L0],xq[full],z" --n_pos 20000 --n_neg 20000 --seed 2024

# 图
python3 tools/plot_behavior_auc_vs_ndcg.py --pos_regime topk --neg_mode degmatch
python3 tools/plot_behavior_auc_vs_ndcg.py --pos_regime all  --neg_mode degmatch
```

| 产物 | 说明 |
|:--|:--|
| `doc/behavior_structure_probe.md` | 自动生成原始结果（含全空间 × 全口径） |
| `doc/behavior_structure_probe_results.csv` | 长表：dataset × variant × space × pos_regime |
| `doc/figures/behavior_auc_vs_ndcg_full.{png,pdf}` | Full SID AUC vs NDCG（主图） |
| `doc/figures/behavior_auc_vs_ndcg_l0.{png,pdf}` | L0 SID AUC vs NDCG |
| `doc/figures/behavior_auc_vs_ndcg_{full,l0}_all_degmatch.*` | `all` 正样本口径 |

**复用的既有模块（未重复实现）**：`tools/sid_consistency_contrast.py`（ckpt/码本加载）、
`tools/sid_per_layer_diagnostic.py`（`rebuild_xq_partial`/`normalize_rows`/`encode_z`）、
`tools/e49_summary.py`（NDCG 日志解析）、`rqvae/build_transition_graph.py`（E 图构建）。

**与 `doc/e49grid/` 的关系**：e49grid 是本实验的 `pos=topk × neg=uniform` 格的**前身**；本轮在此之上
新增了 ①度受控负采样 ②`all` 正样本口径 ③Hamming/L0-match 离散指标 ④与 NDCG@20 的相关检验。

两处 `topk × neg=uniform` 数值**可相互对照但并非同口径**，实测 mean|Δ| = 0.013、max|Δ| = 0.032
（本轮系统性偏高，12 格中 11 格为正）。差异来源已定位，**不是 seed 噪声**：

| 环节 | `doc/e49grid/`（`sid_capacity_diagnostic.py`） | 本轮（`behavior_structure_probe.py`） |
|:--|:--|:--|
| 负样本是否排除 E 内边 | **否**（全库随机抽，含 E 内边与自环） | **是**（排除自环与 E 内边） |
| 负样本量 | 5000 | 20000 |
| 正样本来源 | `history`（不含 target），单向计数 | `history+[target]`，window 双向计数 |

其中「负样本是否排除 E 内边」是系统偏高方向的主因：把真实的边留在负样本集里会抬高负样本的
余弦均值，从而压低 AUC。**本轮的负样本集在定义上更严格**（负样本应当是真正的非边），
故以本轮为准；e49grid 仅作趋势对照。

### 6.1 与 `doc/e49grid_diag_summary.md` 的 Sport 排序分歧（已对账）

该报告记 **Sport 6/6 同序**、本轮记 **Sport 4/6**。分歧点**唯一且已定位**：
只出在 **Content-only ↔ ZeroPad** 这一对相邻点的先后上，其余 4 个位置两管线完全一致
（两者之间的 Spearman ρ = 0.943，即恰好一次相邻互换）。

| 口径 | Sport `xq[full]` 排序（尾段） | 与 NDCG 同序 |
|:--|:--|:--:|
| e49grid（负样本**含** E 内边，n_neg=5000） | ItemCF > **ZeroPad 0.9021 > Content 0.8999** > Shuffled | 6/6 |
| 本轮 `degmatch`（负样本**排除** E 内边与自环） | ItemCF > **Content 0.9080 > ZeroPad 0.9069** > Shuffled | 4/6 |
| 本轮 `uniform`（同样排除 E 内边） | ItemCF > **Content 0.9105 > ZeroPad 0.9076** > Shuffled | 4/6 |

三点必须记清：

1. **不是 `degmatch` 造成的**。本轮自己的 `uniform` 负样本同样给出 Content > ZeroPad
   （0.9105 vs 0.9076），也是 4/6。真正翻转该项的是 **e49grid 未把 E 内边剔出负样本池** ——
   保留真实边会抬高负样本相似度均值，且抬高幅度随变体不同（对 ZeroPad 抬得比对 Content 少），
   于是把这 0.002 量级的差距翻了向。
2. **差距在噪声地板内**，两个方向都是：e49grid 差 0.0022、本轮差 0.0011；而二者 NDCG 只差
   0.0005（ZeroPad 0.0280 vs Content 0.0275，3-seed std 之下）。**无论哪一边都不构成
   "诊断指标错判"的证据。**
3. **被翻转的是两个按构造就应当等价的变体**。ZeroPad = Content 的 768d 内容向量 ⊕ 128d 零向量，
   行为信息恒为 0 —— 它的 AUC/NDCG 与 Content-only 打平**正是本实验的预期行为**
   （见 §1 Q3 的"剂量–响应自洽"第 3 条）。在一个预期打平、实测差 0.001 的对上发生先后翻转，
   属噪声，**不应计入"同序率"**。

> 因此对外表述建议统一为：**"Sport 与 Toys 上，行为诊断与下游 NDCG 在 6 变体中各有 1 处相邻倒置，
> 且两处倒置都发生在噪声量级（≤0.0015 NDCG）的相邻点之间；两端的 TA-SID（恒第一）与
> Shuffled（恒最后）跨数据集稳定。"** 不主张 6/6 也不主张 4/6 —— 单点数据支撑不起这个区别。
