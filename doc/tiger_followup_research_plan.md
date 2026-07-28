# TIGER 后续研究方案：面向二区论文的稳健创新路线

本文档基于 TIGER 论文与 `tiger-main` 代码，整理一个可行性较高、创新强度适中、适合继续写论文的研究方案。目标不是做一个风险很大的全新框架，而是在 TIGER 的核心瓶颈处做稳定改进：**第一阶段的 Semantic ID 生成，也就是 item tokenizer 设计**。

推荐选题暂定为：

> **Transition-Aware Semantic IDs for Generative Sequential Recommendation**

中文可以写作：

> **面向生成式序列推荐的序列转移感知语义 ID 方法**

也可以简称为 **TA-SID**，即 **Transition-Aware Semantic ID**。

---

## 1. 总体判断

我不建议直接沿着“把 TIGER 的 Transformer 换得更大”这个方向做。原因是：

1. TIGER 的第二阶段本质是 next token prediction，直接换更大的 encoder-decoder，容易被审稿人认为只是模型规模变化。
2. 如果没有更多数据或更强算力，堆模型不一定稳定提升。
3. TIGER 真正有研究价值的地方在第一阶段：如何把 item 变成适合生成式推荐的离散 token。
4. 近年的生成式推荐工作也普遍把 **item tokenization / semantic identifier / semantic code** 当作关键问题。

因此，更稳的研究切入点是：

> 不改或少改第二阶段生成模型，重点改第一阶段 Semantic ID 生成方式，让 Semantic ID 不只表达商品内容语义，还表达用户行为序列中的推荐关系。

原始 TIGER 的主线是：

$$
\text{item content}
\rightarrow
\text{content embedding}
\rightarrow
\text{RQ-VAE}
\rightarrow
\text{Semantic ID}
\rightarrow
\text{T5 next-token prediction}
$$

本文建议改成：

$$
\text{item content}
+
\text{item transition behavior}
\rightarrow
\text{recommendation-aware item embedding}
\rightarrow
\text{RQ-VAE}
\rightarrow
\text{Transition-aware Semantic ID}
\rightarrow
\text{T5 next-token prediction}
$$

直观理解是：TIGER 原来给商品编码时主要看“这个商品是什么”；我们的改进让它同时看“这个商品在用户序列中通常和哪些商品一起出现、前后接着什么商品”。

---

## 2. 研究问题

### 2.1 TIGER 的关键假设

TIGER 把推荐任务从“检索物品向量”改成“生成物品的 Semantic ID”。对一个用户历史序列：

$$
S_u = (i_1, i_2, \dots, i_t),
$$

传统序列推荐通常预测下一个 item：

$$
\hat{i}_{t+1}
= 
\arg\max_{i \in \mathcal{I}}
P(i \mid i_1, i_2, \dots, i_t).
$$

TIGER 先把每个 item $i$ 映射成一个 Semantic ID：

$$
\operatorname{SID}(i)
=
(c_{i,1}, c_{i,2}, \dots, c_{i,m}),
$$

然后模型不再直接预测 item id，而是生成下一个 item 对应的 code 序列：

$$
P(\operatorname{SID}(i_{t+1}) \mid S_u)
=
\prod_{l=1}^{m}
P(c_{i_{t+1},l}
\mid
S_u, c_{i_{t+1},<l}).
$$

所以，TIGER 的效果很依赖第一阶段生成的 Semantic ID 是否好。

### 2.2 原始 TIGER 的潜在不足

TIGER 第一阶段主要依赖 item content features，例如：

- title
- brand
- category
- price
- description

这些内容特征被 Sentence-T5 等文本编码器转换为连续向量：

$$
x_i = f_{\text{content}}(\text{title}_i, \text{brand}_i, \text{category}_i, \text{price}_i).
$$

然后 RQ-VAE 对 $x_i$ 进行量化，得到 Semantic ID。

这个设计很合理，但它有一个问题：

> 内容相似不一定等于推荐关系相似。

例如：

- 两个口红颜色很相似，内容 embedding 很接近，但用户买完第一个不一定会马上买第二个。
- 洗发水和护发素内容类别不同，但用户行为序列中经常连续出现。
- 手机和手机壳文本差异大，但序列转移关系强。
- 新手用户买完入门商品后，可能更容易转向配件、耗材或进阶商品。

因此，推荐任务中的“相似性”至少包含两种：

$$
\text{item similarity}
=
\text{content similarity}
+
\text{behavioral transition similarity}.
$$

原始 TIGER 更偏向前者，而我们的方案补充后者。

---

## 3. 核心创新点

本文可以提出一个简单但清晰的观点：

> 生成式推荐中的 Semantic ID 不应该只服务于内容重构，也应该服务于下一个物品预测。为此，本文提出序列转移感知的 Semantic ID 生成方法，使 item tokenizer 同时编码内容语义和用户行为转移关系。

可以包装成三个贡献：

1. **Recommendation-aware item tokenizer**  
   提出一种面向推荐目标的 Semantic ID 生成方法，把用户序列中的 item transition signal 引入 RQ-VAE 量化阶段。

2. **Transition-enhanced item representation**  
   从用户交互序列中构建 item-item transition graph，学习 item behavior embedding，并与内容 embedding 融合。

3. **Stable plug-in improvement for TIGER**  
   不改变 TIGER 的第二阶段生成模型，只替换第一阶段 Semantic ID，即可提升 Recall、NDCG、冷启动和长尾推荐效果。

---

## 4. 从数据角度理解方法

### 4.1 原始数据

原始数据是用户按时间排序的交互序列：

```text
user_0: [9839, 11863, 11752, 9449, 10076, 11155]
user_1: [3309, 4572, 9079, 4136, 10955, 58]
user_2: [4386, 6362, 4208, 5665, 453, 8876, 8649]
```

在 TIGER 中，这些数据会被变成 next-item prediction 样本：

| 历史序列 | 目标物品 |
|---|---:|
| `[9839]` | `11863` |
| `[9839, 11863]` | `11752` |
| `[9839, 11863, 11752]` | `9449` |
| `[9839, 11863, 11752, 9449]` | `10076` |

原始 TIGER 只在第二阶段使用这些序列数据。

我们的关键改进是：

> 在第一阶段生成 Semantic ID 时，也利用这些序列数据。

### 4.2 构造 item transition graph

从用户序列中统计相邻 item 的转移关系。

如果用户序列为：

$$
(i_1, i_2, i_3, \dots, i_t),
$$

则可以得到转移边：

$$
(i_1 \rightarrow i_2),
(i_2 \rightarrow i_3),
\dots,
(i_{t-1} \rightarrow i_t).
$$

统计所有用户后，得到 item-item transition matrix：

$$
A_{ij}
=
\#\{(u,t): i_t^u=i, i_{t+1}^u=j\}.
$$

其中 $A_{ij}$ 表示 item $i$ 后面接着出现 item $j$ 的次数。

为了减少热门物品的影响，可以做归一化：

$$
P_{ij}
=
\frac{A_{ij}}{\sum_{j'} A_{ij'}}.
$$

也可以使用 PPMI 矩阵：

$$
\operatorname{PPMI}(i,j)
=
\max
\left(
\log
\frac{P(i,j)}{P(i)P(j)},
0
\right).
$$

这样，原始用户序列就多生成了一份数据：

```text
item_i -> frequently transitioned items
9839   -> 11863, 9449, ...
11863  -> 11752, 10076, ...
11752  -> 9449, ...
```

它描述的不是商品内容，而是行为转移结构。

### 4.3 学习 item behavior embedding

用 item transition graph 学一个行为向量：

$$
g_i \in \mathbb{R}^{d_g}.
$$

这个向量表示 item $i$ 在用户行为序列中的位置和功能。

最稳的实现有三种，建议优先从第一种开始：

| 方法 | 优点 | 风险 |
|---|---|---|
| PPMI + Truncated SVD | 实现简单、稳定、可解释 | 表达能力有限 |
| Node2Vec / DeepWalk | 适合图结构，实验容易讲 | 需要调随机游走参数 |
| LightGCN item embedding | 推荐味更强 | 实现和训练复杂度更高 |

为了二区论文的稳健性，推荐第一版使用：

$$
G = \operatorname{SVD}(\operatorname{PPMI}(A)).
$$

最终每个 item 得到一个行为 embedding：

$$
g_i = G_i.
$$

### 4.4 融合内容语义和行为转移

原始 TIGER 的输入是内容 embedding：

$$
x_i \in \mathbb{R}^{d_x}.
$$

我们的方案额外引入行为 embedding：

$$
g_i \in \mathbb{R}^{d_g}.
$$

然后融合：

$$
h_i
=
\operatorname{MLP}
\left(
[x_i; g_i; p_i]
\right),
$$

其中：

- $x_i$ 是内容向量；
- $g_i$ 是行为转移向量；
- $p_i$ 是可选的 item 统计特征，例如 popularity、category id、price bucket。

如果想让方法更简单，可以先不用 $p_i$：

$$
h_i
=
\operatorname{MLP}
([x_i; g_i]).
$$

最终，$h_i$ 就是 recommendation-aware item embedding。

---

## 5. RQ-VAE 量化阶段

### 5.1 原始 RQ-VAE

RQ-VAE 先把输入向量编码为 latent representation：

$$
z_i = E_\theta(h_i).
$$

然后进行多层 residual quantization。

第 $l$ 层 codebook 为：

$$
\mathcal{C}_l
=
\{e_{l,1}, e_{l,2}, \dots, e_{l,K}\}.
$$

初始残差为：

$$
r_{i,1}=z_i.
$$

第 $l$ 层选择最近的 codeword：

$$
c_{i,l}
=
\arg\min_{k}
\|r_{i,l} - e_{l,k}\|_2^2.
$$

然后更新残差：

$$
r_{i,l+1}
=
r_{i,l} - e_{l,c_{i,l}}.
$$

经过 $m$ 层后，得到：

$$
\operatorname{SID}(i)
=
(c_{i,1}, c_{i,2}, \dots, c_{i,m}).
$$

量化后的向量为：

$$
\hat{z}_i
=
\sum_{l=1}^{m}
e_{l,c_{i,l}}.
$$

### 5.2 改进后的训练目标

总损失可以写成：

$$
\mathcal{L}
=
\mathcal{L}_{\text{recon}}
+
\lambda_1 \mathcal{L}_{\text{trans}}
+
\lambda_2 \mathcal{L}_{\text{balance}}.
$$

第一项是重构损失：

$$
\mathcal{L}_{\text{recon}}
=
\|h_i - \hat{h}_i\|_2^2.
$$

第二项是转移感知损失。它的目标是：如果 item $i$ 经常转移到 item $j$，那么它们的量化表示应该更接近。

可以使用对比学习形式：

$$
\mathcal{L}_{\text{trans}}
=
-
\sum_{(i,j)\in \mathcal{E}}
\log
\frac{
\exp(\operatorname{sim}(\hat{z}_i,\hat{z}_j)/\tau)
}{
\sum_{n\in \mathcal{N}_i}
\exp(\operatorname{sim}(\hat{z}_i,\hat{z}_n)/\tau)
}.
$$

其中：

- $(i,j)$ 是 item transition graph 中的正样本边；
- $\mathcal{N}_i$ 是负样本集合；
- $\tau$ 是 temperature；
- $\operatorname{sim}(\cdot,\cdot)$ 可以使用 cosine similarity。

第三项是 codebook balance loss，用来防止 codebook collapse：

$$
\mathcal{L}_{\text{balance}}
=
\sum_{l=1}^{m}
\operatorname{KL}(q_l(c)\|U(c)).
$$

其中：

- $q_l(c)$ 是第 $l$ 层 codebook 的使用分布；
- $U(c)$ 是均匀分布。

这个损失鼓励不同 codeword 都被充分使用。

### 5.3 最低风险版本

为了先跑通实验，第一版不一定要加复杂损失。可以先做一个最小可行版本：

$$
h_i = \operatorname{MLP}([x_i; g_i])
$$

然后仍然使用原始 RQ-VAE 损失：

$$
\mathcal{L}
=
\mathcal{L}_{\text{recon}}
+
\mathcal{L}_{\text{rqvae}}.
$$

这样创新点仍然成立：

> 原始 TIGER 的 RQ-VAE 输入只包含内容语义；本文的 RQ-VAE 输入融合了序列转移行为。

如果这个版本已经有效，再加入 $\mathcal{L}_{\text{trans}}$ 和 $\mathcal{L}_{\text{balance}}$ 作为增强版。

---

## 6. 第二阶段保持 TIGER 不变

第二阶段仍然是 TIGER 的生成式推荐模型。

每个 item 的 Semantic ID 例如：

```text
9839  -> [225, 198, 146, 160]
11863 -> [8, 29, 66, 64]
11752 -> [23, 181, 190, 86]
```

加入 codebook offset 后：

$$
(225,198,146,160)
\rightarrow
(225,454,658,928).
$$

用户历史序列：

```text
[9839, 11863, 11752]
```

会被展开为 Semantic ID token 序列：

```text
[225,454,658,928, 8,285,578,832, 23,437,702,854]
```

模型目标是生成下一个 item 的 Semantic ID：

$$
P(c_{t+1,1}, c_{t+1,2}, c_{t+1,3}, c_{t+1,4}
\mid
\text{history SID tokens}).
$$

这一阶段不需要大改。这样论文的实验逻辑非常干净：

> 同一个生成模型，不同的 item tokenizer。性能变化主要来自 Semantic ID 质量。

---

## 7. 实验设计

### 7.1 数据集

优先使用 TIGER 原文中的三个 Amazon Reviews 子集：

| Dataset | 特点 |
|---|---|
| Beauty | 商品语义强，适合验证 content + behavior |
| Sports and Outdoors | 类目复杂，序列转移较明显 |
| Toys and Games | 物品数量适中，适合做补充验证 |

如果时间足够，可以增加一个 MovieLens 或 Yelp 数据集，但不建议一开始就加。先复现 TIGER 原数据集更稳。

### 7.2 Baselines

推荐至少包含：

| 方法 | 作用 |
|---|---|
| Random ID | 证明语义 ID 有意义 |
| LSH SID | 证明学习式量化优于随机投影 |
| TIGER RQ-VAE SID | 最关键 baseline |
| TA-SID, ours | 本文方法 |
| SASRec / BERT4Rec | 传统序列推荐 baseline |
| P5 | 生成式推荐 baseline，可选 |

最重要的比较是：

$$
\text{TIGER RQ-VAE SID}
\quad
\text{vs.}
\quad
\text{TA-SID}
$$

因为这样能直接说明：只改第一阶段 Semantic ID，就能提升推荐。

### 7.3 主指标

使用 TIGER 原文一致的指标：

$$
\operatorname{Recall@K}
=
\frac{|\operatorname{TopK}(u) \cap \{i^*_u\}|}{1}
$$

因为 leave-one-out 设置中每个用户只有一个测试目标，所以 Recall@K 等价于目标 item 是否出现在 top-K 中。

NDCG@K 为：

$$
\operatorname{NDCG@K}
=
\frac{1}{\log_2(r_u + 1)},
$$

其中 $r_u$ 是真实目标 item 在推荐列表中的排名。如果目标 item 不在 top-K 内，则 NDCG@K 为 0。

建议报告：

- Recall@5
- Recall@10
- NDCG@5
- NDCG@10

### 7.4 Semantic ID 质量指标

为了让论文更像一篇完整研究，而不只是刷推荐指标，建议增加以下分析指标。

**1. Codebook utilization**

$$
\operatorname{Utilization}_l
=
\frac{|\{c_{i,l}: i\in\mathcal{I}\}|}{K}.
$$

这个指标表示第 $l$ 层 codebook 有多少比例被使用。

**2. Collision rate**

$$
\operatorname{CollisionRate}
=
1
-
\frac{
|\{\operatorname{SID}(i): i\in\mathcal{I}\}|
}{
|\mathcal{I}|
}.
$$

这个指标表示有多少 item 被映射到相同 Semantic ID。

**3. Invalid ID rate**

生成模型可能输出不存在的 Semantic ID：

$$
\operatorname{InvalidRate@K}
=
\frac{
\#\text{invalid generated IDs in top-K}
}{
\#\text{generated IDs in top-K}
}.
$$

如果 TA-SID 让生成空间更贴近真实 item 分布，Invalid ID rate 可能降低。

**4. Transition consistency**

衡量 Semantic ID 是否更符合用户行为转移关系：

$$
\operatorname{TC}
=
\frac{1}{|\mathcal{E}|}
\sum_{(i,j)\in\mathcal{E}}
\operatorname{sim}(\hat{z}_i,\hat{z}_j).
$$

其中 $\mathcal{E}$ 是 item transition graph 中的高频边。

### 7.5 冷启动实验

TIGER 原文强调 Semantic ID 有冷启动能力。我们的方案也应该保留这一部分。

实验设置：

1. 从 test target items 中移除 5%、10%、20% 的 item。
2. 这些 item 不参与第二阶段生成模型训练。
3. 但它们仍然有 content embedding。
4. 使用 TA-SID 为这些 unseen items 生成 Semantic ID。
5. 测试模型能否召回这些 unseen items 或其相近物品。

指标：

- Cold Recall@10
- Cold NDCG@10
- unseen item hit rate

预期结论：

> TA-SID 同时利用内容语义和行为结构，因此在冷启动或半冷启动场景下比原始 TIGER 更稳。

### 7.6 长尾实验

把 item 按训练集中出现次数分成三组：

| 组别 | 定义 |
|---|---|
| Head | 交互次数 top 20% |
| Mid | 交互次数中间 60% |
| Tail | 交互次数 bottom 20% |

分别报告 Recall@10 和 NDCG@10。

这一部分很重要，因为 TA-SID 的卖点之一是：

> 行为转移关系可以帮助长尾 item 借助相邻 item 或相似 transition pattern 获得更好的离散表示。

---

## 8. 消融实验

建议设计以下消融：

| Variant | 说明 |
|---|---|
| Content only | 原始 TIGER RQ-VAE |
| Behavior only | 只用 $g_i$，不用 $x_i$ |
| Content + Behavior | 本文主模型 |
| Content + Behavior + transition loss | 加 $\mathcal{L}_{trans}$ |
| Content + Behavior + balance loss | 加 $\mathcal{L}_{balance}$ |
| Full model | 所有模块 |

如果结果理想，应该看到：

$$
\text{Content + Behavior}
>
\text{Content only}
>
\text{Random ID}.
$$

更强版本可能是：

$$
\text{Full model}
>
\text{Content + Behavior}
>
\text{Content only}.
$$

---

## 9. 代码实现路线

### 9.1 目标

尽量复用 `tiger-main` 的第二阶段，只生成新的 Semantic ID 文件。

原始文件类似：

```text
tiger-main/data/Beauty/index_rqvae.json
```

其格式为：

```json
{
  "9839": [225, 198, 146, 160],
  "11863": [8, 29, 66, 64]
}
```

我们的目标是生成：

```text
tiger-main/data/Beauty/index_tasid.json
```

格式保持完全一致：

```json
{
  "9839": [34, 201, 77, 0],
  "11863": [34, 205, 12, 0]
}
```

这样第二阶段 dataloader 几乎不用改，只需要把使用的 index 文件从 `index_rqvae.json` 换成 `index_tasid.json`。

### 9.2 实现步骤

**Step 1: 构造 transition matrix**

从：

```text
inter.json
```

读取用户序列，统计相邻 item pair：

$$
A_{ij} \leftarrow A_{ij} + 1.
$$

**Step 2: 学习 behavior embedding**

第一版建议：

```text
A -> PPMI -> TruncatedSVD -> g_i
```

得到：

```text
item_id -> behavior embedding
```

**Step 3: 读取 content embedding**

复用 TIGER 或代码中已有的 item content embedding。

得到：

```text
item_id -> content embedding
```

**Step 4: 融合 embedding**

拼接后用 MLP 投影：

$$
h_i = \operatorname{MLP}([x_i; g_i]).
$$

**Step 5: 训练 RQ-VAE**

输入从 $x_i$ 改成 $h_i$。

输出仍然是：

```text
item_id -> [c1, c2, c3, c4]
```

**Step 6: 训练 TIGER 第二阶段**

复用原训练流程，只替换 Semantic ID 文件。

**Step 7: 评估**

使用 Recall@K、NDCG@K、codebook utilization、collision rate、invalid ID rate、cold-start、long-tail 等指标。

---

## 10. 预期论文结构

可以按下面结构写：

1. **Introduction**
   - 介绍生成式推荐和 TIGER。
   - 指出 Semantic ID 是关键。
   - 指出内容语义和推荐行为目标存在 gap。
   - 提出 TA-SID。

2. **Related Work**
   - Sequential recommendation。
   - Generative recommendation。
   - Semantic ID / item tokenization。
   - Vector quantization in recommendation。

3. **Preliminaries**
   - Sequential recommendation task。
   - TIGER generative retrieval。
   - RQ-VAE Semantic ID。

4. **Method**
   - Item transition graph construction。
   - Behavior embedding learning。
   - Content-behavior fusion。
   - Transition-aware RQ-VAE。
   - Generative recommendation with TA-SID。

5. **Experiments**
   - Datasets。
   - Baselines。
   - Metrics。
   - Main results。
   - Ablation。
   - Cold-start。
   - Long-tail。
   - Semantic ID quality analysis。

6. **Conclusion**
   - TA-SID 是一个可插拔的 item tokenizer。
   - 它提升了生成式推荐中 Semantic ID 与推荐目标的一致性。

---

## 11. 预期结果与论文卖点

论文中最理想的结果不是所有指标大幅超过 TIGER，而是形成稳定证据链：

1. TA-SID 在 Recall@10 / NDCG@10 上稳定优于原始 TIGER。
2. TA-SID 在 long-tail items 上提升更明显。
3. TA-SID 的 invalid ID rate 更低或相近。
4. TA-SID 的 codebook utilization 更高。
5. 可视化显示 TA-SID 的 code 不仅按内容聚类，也符合序列转移结构。

可以在论文中强调：

> 在不增加第二阶段生成模型复杂度的情况下，改进第一阶段 Semantic ID 即可带来稳定收益。这说明生成式推荐的关键不仅是更强的 decoder，也是更推荐友好的 item tokenizer。

---

## 12. 风险与备用方案

### 风险 1：主指标提升不明显

可能原因：

- transition embedding 太弱；
- fusion 方式太简单；
- TIGER 第二阶段训练波动大；
- Amazon 数据序列较短，transition signal 不够强。

备用方案：

1. 使用 multi-hop transition，不只看相邻 item：

$$
A_{ij}
=
\sum_{\Delta=1}^{L}
\alpha^\Delta
\#\{i_{t}=i, i_{t+\Delta}=j\}.
$$

2. 加入 category-level transition。
3. 加入 $\mathcal{L}_{trans}$ 对比学习损失。
4. 在 long-tail 和 cold-start 场景中突出提升。

### 风险 2：behavior embedding 被热门 item 支配

备用方案：

1. 使用 PPMI 降低 popularity bias。
2. 对 transition matrix 做 row normalization。
3. 对 item popularity 加 log smoothing：

$$
w_{ij}
=
\frac{A_{ij}}{\log(1+\operatorname{pop}(j))}.
$$

### 风险 3：审稿人认为创新不够

备用包装：

1. 强调问题不是简单融合 embedding，而是 **recommendation-aware item tokenization**。
2. 增加 Semantic ID 质量分析，而不只报告 Recall。
3. 展示该方法是 plug-in，可以兼容 TIGER、P5 类生成式推荐模型。
4. 加入 cold-start 和 long-tail，体现实用价值。

---

## 13. 时间规划

### 第 1-2 周：复现与数据理解

- 跑通 `tiger-main`。
- 确认 `inter.json`、`index_rqvae.json`、dataloader 和 metric。
- 得到原始 TIGER 的 Beauty 结果。

### 第 3-4 周：TA-SID 最小版本

- 构建 item transition matrix。
- 实现 PPMI + SVD behavior embedding。
- 融合 content embedding 和 behavior embedding。
- 训练新的 RQ-VAE。
- 生成 `index_tasid.json`。

### 第 5-6 周：主实验

- 在 Beauty 上跑完整对比。
- 如果有效，再跑 Sports and Outdoors、Toys and Games。
- 记录 Recall@5、Recall@10、NDCG@5、NDCG@10。

### 第 7-8 周：分析实验

- codebook utilization。
- collision rate。
- invalid ID rate。
- long-tail。
- cold-start。
- ablation。

### 第 9-10 周：论文初稿

- 写 Introduction、Method、Experiments。
- 画 pipeline 图。
- 画 Semantic ID 可视化。
- 整理 related work。

---

## 14. 最终推荐路线

最推荐的路线是：

1. **不要大改 TIGER 第二阶段。**
2. **专注第一阶段 Semantic ID。**
3. **先做 content embedding + transition embedding 融合。**
4. **用相同 TIGER 模型比较不同 Semantic ID。**
5. **用长尾、冷启动、invalid ID、codebook utilization 支撑论文完整性。**

论文的核心句可以这样写：

> Existing generative recommender systems rely heavily on semantic item identifiers, but most identifiers are generated from item content alone. We argue that item identifiers for recommendation should be aware of both content semantics and behavioral transitions. To this end, we propose a transition-aware semantic identifier framework that incorporates item transition signals into residual quantization, producing recommendation-aware discrete item tokens for generative sequential recommendation.

中文表达为：

> 现有生成式推荐方法高度依赖物品语义 ID，但这些 ID 往往主要由内容语义生成。本文认为，推荐系统中的物品 ID 不应只描述“物品是什么”，还应描述“物品在用户行为序列中如何转移”。因此，本文提出序列转移感知的语义 ID 生成方法，将 item transition signal 引入残差量化过程，为生成式序列推荐构建更符合推荐目标的离散物品 token。

---

## 15. 参考文献与相关方向

- TIGER: Recommender Systems with Generative Retrieval  
  https://arxiv.org/abs/2305.05065

- LETTER: A Lightweight yet Effective framework for item Tokenization in generative recommendation  
  https://arxiv.org/abs/2405.07314

- MMGRec: Multimodal Generative Recommendation with Transformer Model  
  https://arxiv.org/abs/2404.16555

- SETRec: Enhancing Generative Recommendation by Semantic Tokenization and Efficient Transformer  
  https://arxiv.org/abs/2502.10833

- ReSID: Learning Recommendation-aware Semantic IDs for Generative Retrieval  
  https://arxiv.org/abs/2602.02338

