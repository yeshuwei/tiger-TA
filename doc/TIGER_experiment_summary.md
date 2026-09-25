# TIGER论文实验设计与实验结果总结
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

论文：Recommender Systems with Generative Retrieval (TIGER)

## 1. 实验总体目标

论文提出 Semantic ID + Generative Retrieval 的推荐框架。

实验整体目标：

1. 验证TIGER整体推荐性能是否优于传统推荐模型。
2. 验证Semantic ID是否是性能提升的关键因素。
3. 验证生成式推荐是否具备传统推荐系统不具备的能力，例如冷启动和多样性控制。

整体实验逻辑：

提出方法
↓
验证效果提升（主实验）
↓
验证创新模块有效性（消融实验）
↓
验证额外能力（冷启动、多样性）

---

# 实验1：主实验（Main Experiment）

## 实验目的

验证：

> TIGER生成式推荐框架是否能够超过传统Sequential Recommendation方法。

核心问题：

- Generative Retrieval是否适用于推荐任务？
- Semantic ID是否能够提升推荐准确率？

---

## 实验任务

任务：

Sequential Recommendation（序列推荐）

输入：

用户历史商品序列

例如：

商品A → 商品B → 商品C

预测：

下一次交互商品D。

---

## 数据集

Amazon Product Reviews三个公开数据集：

|数据集|领域|用户数|商品数|
|-|-|-|-|
|Beauty|美妆|22363|12101|
|Sports and Outdoors|运动|35598|18357|
|Toys and Games|玩具|19412|11924|

数据处理：

- 根据时间排序用户行为
- 使用leave-one-out策略
- 最后一个商品作为测试集

---

## Baseline

对比模型：

### RNN模型

- GRU4Rec

### CNN模型

- Caser

### Transformer模型

- SASRec
- BERT4Rec

### 自监督推荐

- S3-Rec

### LLM推荐

- P5

---

## 评价指标

### Recall@K

衡量：

真实商品是否出现在Top-K推荐列表中。

---

### NDCG@K

衡量：

正确商品排序位置。

越靠前，得分越高。

---

## 实验结果

TIGER在三个Amazon数据集上均超过baseline。

示例：

Beauty数据：

|方法|Recall@5|
|-|-|
|S3-Rec|0.0387|
|TIGER|0.0454|

提升约17%。

NDCG@5提升约29%。

---

## 实验结论

证明：

1. Generative Retrieval可以用于推荐任务。
2. Semantic ID表示方式有效。
3. Transformer生成推荐结果优于传统检索式推荐。

---

# 实验2：Semantic ID消融实验

## 实验目的

验证：

> TIGER性能提升是否来自Semantic ID。

核心问题：

如果替换Semantic ID生成方式，效果是否下降？

---

## 实验设计

固定Transformer推荐模型。

改变商品ID表示方式：

---

## 方法1：Random ID

随机编号。

特点：

没有任何商品语义。

---

## 方法2：LSH Semantic ID

利用哈希方法生成编码。

具有部分相似性。

---

## 方法3：RQ-VAE Semantic ID

TIGER提出方法。

通过：

商品embedding

↓

RQ-VAE量化

↓

Semantic ID

---

## 指标

- Recall@K
- NDCG@K

---

## 实验结果

Beauty数据：

|方法|Recall@5|
|-|-|
|Random ID|0.0296|
|LSH ID|0.0379|
|RQ-VAE Semantic ID|0.0454|

---

## 实验结论

证明：

性能提升不是来自Transformer本身。

真正原因：

Semantic ID提供了商品之间的语义结构。

相似商品共享token，使模型能够学习商品之间关系。

---

# 实验3：冷启动实验（Cold-start）

## 实验目的

验证：

> TIGER是否可以推荐没有历史交互的新商品。

解决传统推荐系统冷启动问题。

---

## 实验设计

模拟新商品：

删除测试集中5%的商品。

这些商品：

- 训练阶段没有出现
- 没有历史交互记录

---

## 对比方法

### TIGER

利用：

商品内容

↓

Semantic ID

↓

生成推荐

---

### Semantic-KNN

传统相似搜索方法。

---

## 指标

主要：

Recall@K

衡量新商品是否能够被推荐。

---

## 实验结果

TIGER在不同Recall@K指标下均超过Semantic-KNN。

---

## 实验结论

证明：

Semantic ID能够利用商品内容信息。

即使商品没有历史行为，也可以生成对应表示并完成推荐。

---

# 实验4：推荐多样性实验

## 实验目的

验证：

> TIGER是否可以控制推荐结果的多样性。

解决传统推荐系统容易产生热门偏置的问题。

---

## 实验方法

调整Transformer生成参数：

temperature。

---

## 参数影响

低temperature：

- 生成更加确定
- 推荐更加集中


高temperature：

- 探索更多可能
- 推荐更加多样

---

## 指标

Entropy@K

衡量Top-K推荐结果的信息丰富程度。

Entropy越高：

推荐越多样。

---

## 实验结果

随着temperature提高：

Entropy@K增加。

说明：

生成式模型可以通过控制生成策略调整推荐多样性。

---

# 实验总结表

|实验|目的|指标|结果|
|-|-|-|-|
|主实验|验证整体推荐性能|Recall@K、NDCG@K|TIGER超过传统推荐模型|
|Semantic ID消融|验证Semantic ID有效性|Recall@K、NDCG@K|RQ-VAE Semantic ID最好|
|冷启动实验|验证新商品推荐能力|Recall@K|优于Semantic-KNN|
|多样性实验|验证生成控制能力|Entropy@K|temperature提高，多样性增强|

---

# 对论文实验设计的启发

TIGER实验结构：

1. 整体性能验证
2. 核心模块消融
3. 特殊能力验证

对应自己的研究：


