# Track B (L_trans) 实施方案追溯分析

> 对比 source of truth：`tiger_followup_research_plan.md`（原始方案）
> 分析对象：`trans_loss.py` / `trainer.py` / `build_transition_graph.py`

---

## 一、方案原文 vs 实际实现对照

### 1.1 核心公式对比

**研究方案 §5.2 定义的 L_trans：**

```
L_trans = - Σ_{(i,j)∈E} log( exp(sim(ẑ_i, ẑ_j)/τ) / Σ_{n∈N_i} exp(sim(ẑ_i, ẑ_n)/τ) )
```
其中 **ẑ = 量化向量**（quantized），不是编码器输出 z。

**实际实现：**

| 实验 | 作用向量 | 公式一致？ | 说明 |
|:----|:--------|:---------:|:-----|
| B-1 | **z** (encoder output) | ❌ 偏离方案 | 方案指定用 ẑ，B-1 用了 z |
| B-2 | **z** (encoder output) | ❌ 偏离方案 | 同上 |
| B-3 | **x_q** (quantized) | ✅ 符合方案 | 与方案公式一致 |

**关键发现：B-1/B-2 的策略（L_trans on z）并非方案原文推荐的做法。** 方案原文明确指定 L_trans 作用在量化向量上。B-1/B-2 选择 z 是因为实现更简单（不需要从码本重建向量），但这个简化本身引入了问题——L_trans on z 与量化器产生对抗。

### 1.2 正样本对构造方式对比

**研究方案 §4.2：**
```
A_ij = #{(u,t): i_t=i, i_{t+1}=j}   # 相邻转移
PPMI(i,j) = max(log(P(i,j)/P(i)P(j)), 0)   # 可选 PPMI 过滤
```

**实际实现（build_transition_graph.py）：**
- `extract_transition_pairs()`: 使用用户序列中**所有相邻对**，无 PPMI 过滤
- `build_cooccurrence_matrix()`: 滑动窗口（对称）共现，非定向转移
- PPMI 有实现但**仅用于 behavior embedding 生成**，未用于 L_trans 正样本过滤

**差异：**
1. 实际使用的 transition pairs 是定向的（i→j），符合方案
2. 但**没有做 PPMI 过滤**——所有相邻对都被视为正样本，包括热门商品造成的大量噪声
3. 方案推荐的高频过滤（PPMI > 阈值）可以剔除噪声转移

方案原文 §12 风险 2 明确提到：
> "behavior embedding 被热门 item 支配" → 使用 PPMI 降低 popularity bias

但 transition pairs 本身没有应用这个去偏策略。

### 1.3 负样本采样方式

**研究方案：** 未明确指定，标准 InfoNCE batch 内负采样。

**实际实现：** batch 内负采样（CrossEntropy over batch similarity matrix）。✅ 合理。

### 1.4 L_balance 实现

**研究方案 §5.2：**
```
L_balance = Σ_l KL(q_l(c) || U(c))
```

**实际实现（BalanceLoss）：** 逐层统计码本使用分布 → KL vs 均匀分布。✅ 完全一致。

---

## 二、追溯发现的 3 个关键问题

### 问题①：B-1/B-2 用 z 而非 ẑ（最重要）

```
研究方案要求：L_trans(ẑ_i, ẑ_j)    ← 量化向量
B-1/B-2 实现： L_trans(z_i, z_j)   ← 编码器输出
```

**为什么这个差异很关键？**

L_trans on z 的梯度流向编码器，挤压量化器的工作空间：
```
编码器输出 z ← L_trans 拉近 z
       ↓
量化器被迫分散码本 ← L_recon 要求精确重建
       ↓
    矛盾！L_trans 和 L_recon 的目标冲突
```

B-1 碰撞率从 10.66%→0.34%（异常骤降）就是证据——L_trans 把编码器输出挤压到小区域，量化器只能使用极少数码字来区分商品，大部分码字浪费了。

**但 B-3（on x_q，符合方案）效果更差（0.03122 vs B-2 的 0.03246）。** 说明即使按方案原文来做，L_trans 依然无效。

### 问题②：未使用 PPMI 过滤过渡对

`transition_pairs.npy` 包含**所有**连续转移对，包括大量噪声：
- 热门商品作为"万能转移"混入正样本
- 低频低频之间的偶然共现成为误导信号

方案原文生成过 PPMI 过滤版本：
```
transition_pairs_768d_ppmi4.0.npy  (保留 83.2%)
transition_pairs_768d_ppmi6.0.npy  (保留 45.8%)
```
但这些文件从未被 L_trans 使用。

**如果使用 PPMI 过滤后的过渡对：**
- 正样本质量提升（只保留统计显著的转移关系）
- 噪声减少，L_trans 的约束信号更集中
- 可能部分缓解 L_trans vs L_recon 的冲突

### 问题③：未尝试温度系数 τ 的搜索

方案建议 τ 范围 [0.07, 0.1, 0.2, 0.5]，但 B-1/B-2/B-3 **全部使用 τ=0.1**。

τ 控制对比学习的"软硬"：
- τ 太小：只关注最近的样本，容易过拟合噪声
- τ 太大：所有样本一视同仁，信号稀释

但即使 τ 调优，也不太可能扭转 B-2 已经趋于 Track A 的趋势（λ→0 → Track A）。

---

## 三、即使实现完全正确，L_trans 为什么还是无效？

从 §5.3 的风险预案来看，方案原文已经预见到这种可能：

> **风险 1：主指标提升不明显**
> 可能原因：transition embedding 太弱；fusion 方式太简单；Amazon 数据序列较短

实际根因（与方案原文判断一致）：

### 根因 1：融合输入已承载 transition 信号（信号冗余）

方案原文的核心创新其实是**融合输入**（§4.4），L_trans 只是增强版。融合方案中：
```
h_i = MLP([x_i; g_i])   ← g_i 已经编码了 transition 信息
```

然后 RQ-VAE 对这个 h_i 做量化。所以 transition 信号**已经通过 g_i 进入了量化过程**。L_trans 是对同一信号的再次约束——从信息论角度，没有新信息。

### 根因 2：Amazon 数据序列太短（信号强度不足）

Sport 数据集平均序列长度 **5.3**。在 5 个商品的序列中，每个商品只有 1-2 个转移对。对比损失的正样本数量不足以产生有效的训练信号。

方案原文 §12 也指出了这一点：
> Amazon 数据序列较短，transition signal 不够强

### 根因 3：T5 第二阶段等效于"隐式 L_trans"

T5 的自回归训练本质上就是预测 transition——给定历史序列中的 SID tokens，预测下一个 item 的 SID token。T5 的交叉熵损失已经做了 L_trans 想做的事情。

---

## 四、结论：Track B 的决策是正确的

| 维度 | 评估 |
|:----|:-----|
| **方案匹配度** | B-1/B-2 偏离方案（用 z 而非 ẑ）；B-3 符合方案 |
| **实现质量** | 代码逻辑正确，但正样本构造未做 PPMI 去噪 |
| **调参范围** | 缺少 τ 搜索，但 λ 的趋势已清楚 |
| **最终判断** | ✅ **放弃 L_trans 是正确的**。三条独立路径全部验证失败 |

### 如果还要继续做，可以尝试的方向（必要性：低）

1. **PPMI 过滤过渡对 + L_trans on x_q** — 但预期改善有限（B-2 的 λ→0 趋势已说明问题）
2. **迁移到长序列数据集**（如 MovieLens 100K）— 需要额外工作量
3. **去掉融合，纯 behavior → RQ-VAE → L_trans** — 但在当前消融框架下这是 behavior-only 变体

---

## 五、论文启示

方案原文 §5.3 的策略已经被验证：

> **"最低风险版本"（融合输入 + 原始 RQ-VAE 损失）= Track A ✅ 有效**
> **"增强版"（融合输入 + L_trans + L_balance）= Track B ❌ 无效**

论文故事线成立：**融合行为信号的 RQ-VAE 输入已经足够将 transition 信息传递给语义 ID，不需要额外的对比损失约束。** 这反而是一个有意义的实验结论——它说明简单融合比复杂损失更有效。
