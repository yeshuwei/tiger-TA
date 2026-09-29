# 有向转移结构诊断实验方案（Stage 2）

## 0. 实验定位

**研究候选：** Transition-role-aware Semantic ID / 有向转移结构进入 Semantic ID tokenizer。

**当前阶段目标：** 不提出新模型，不证明创新点成立，只验证一个可证伪的问题：

> 在现有 TA-SID 的连续表示到 RQ-VAE 离散表示的过程中，是否存在可观测、非 popularity 的 item transition-role 信息损失？

只有诊断实验通过，才进入第二创新的方法设计。

### 当前候选的边界

- 不声称“首次提出有向 transition”。
- 不声称“directional Semantic ID”已经成立。
- 当前更准确的概念：**transition-role information / 出入转移结构角色**。
- 不把 item→item 的有向关系误写成 pair-level directional token。
- 现阶段不重新设计 tokenizer。

---

# 1. 总体研究问题

定义用户行为序列：

\[
S_u=[i_1,i_2,\ldots,i_T]
\]

构建相邻 item 的有向转移：

\[
i_t\rightarrow i_{t+1}
\]

研究三个问题：

### RQ1：数据中是否存在稳定的非对称 item transition？

\[
i\rightarrow j \neq j\rightarrow i
\]

### RQ2：连续表示是否包含这种 transition-role 信息，而 RQ-VAE 离散化是否造成明显损失？

\[
z_i \rightarrow x_i^q
\]

比较：

\[
AUC(z)\quad vs.\quad AUC(x_q)
\]

并定义：

\[
\Delta_{\mathrm{role}}
=
AUC(z)-AUC(x_q)
\]

### RQ3：这种方向性是否只是 popularity 的代理？

通过 popularity correlation、对称化和随机方向控制进行验证。

---

# 2. 实验范围与固定约束

## 2.1 数据集

只使用：

- Amazon Sport
- Amazon Toys

不新增数据集，除非现有结果无法完成诊断。

## 2.2 数据来源

只使用训练集行为序列构建 transition graph。

**禁止使用验证集/测试集信息构造 transition matrix。**

## 2.3 transition 定义

第一版严格使用**相邻 item transition**：

\[
C_{ij}
=
\#\{i_t=i,\ i_{t+1}=j\}
\]

即：

\[
i_t\rightarrow i_{t+1}
\]

不使用当前 TA-SID 的 \(w=3\) 对称窗口作为本实验的方向定义。

原因：当前实验研究的是 next-item direction，需要一个明确的有向关系。

## 2.4 不允许的操作

本阶段暂时禁止：

- 新颖性继续大范围检索
- 重新设计第二创新
- 重新修改 RQ-VAE 主体
- 重新训练完整 T5
- 直接以推荐指标作为第一判断依据
- 仅凭 probe AUC 宣布创新成立

---

# 3. 实验总流程

```text
Stage 1A
数据层方向性诊断
        ↓
是否存在稳定非对称结构？
        ↓ Yes
Stage 1B
连续表示 vs 量化表示
        ↓
是否存在明显 transition-role information loss？
        ↓ Yes
Stage 1C
Popularity / Symmetric / Random controls
        ↓
方向信息是否具有独立价值？
        ↓ Yes
Stage 2
进入第二创新方法设计
```

任何一个关键 Gate 不通过，都停止继续开发该方向。

---

# 4. Stage 1A：原始数据方向性诊断

## 4.1 构建有向 transition matrix

对每个用户序列：

\[
[i_1,i_2,\ldots,i_T]
\]

统计：

\[
C_{ij}
=
\sum_u\sum_t
\mathbf{1}(i_t=i,i_{t+1}=j)
\]

得到：

\[
C\in\mathbb{R}^{|\mathcal I|\times|\mathcal I|}
\]

只保留：

\[
C_{ij}>0
\]

的有向边。

---

## 4.2 条件转移概率

定义：

\[
P_{\mathrm{out}}(j|i)
=
\frac{C_{ij}}
{\sum_k C_{ik}}
\]

同时定义反向概率：

\[
P_{\mathrm{rev}}(i|j)
=
\frac{C_{ji}}
{\sum_k C_{jk}}
\]

注意：这里比较的是实际方向上的条件概率，不应简单把两个矩阵做转置后当成概率相同。

---

## 4.3 Pair-level asymmetry

定义：

\[
A_{ij}
=
\log
\frac{P(j|i)+\epsilon}
{P(i|j)+\epsilon}
\]

其中：

\[
\epsilon
\]

取固定的小常数，例如 \(10^{-8}\)。

解释：

- \(A_{ij}>0\)：\(i\rightarrow j\) 相对更强；
- \(A_{ij}<0\)：\(j\rightarrow i\) 相对更强；
- \(A_{ij}\approx0\)：近似对称。

### 必须报告

每个数据集报告：

1. 有向边数量；
2. 同时存在 \(i\rightarrow j\) 和 \(j\rightarrow i\) 的 reciprocal pairs 数量；
3. \(|A_{ij}|\) 的均值；
4. \(|A_{ij}|\) 的中位数；
5. \(|A_{ij}|>\{0.5,1.0\}\) 的比例；
6. \(A_{ij}\) 的分布图。

### 注意

只在 reciprocal pair 上计算方向差异。

如果 \(C_{ij}>0\) 但 \(C_{ji}=0\)，这是单向边，但不能直接用普通的 log ratio 把零概率解释成无限强方向性；使用平滑项后单独统计即可。

---

# 5. Stage 1A-B：Popularity 混淆诊断

## 5.1 Item popularity

定义：

\[
pop_i
=
\#\{\text{item }i\text{在训练序列中出现}\}
\]

同时：

\[
d_i^{out}
=
\sum_j C_{ij}
\]

\[
d_i^{in}
=
\sum_j C_{ji}
\]

计算 Spearman：

\[
\rho(d^{out},pop)
\]

\[
\rho(d^{in},pop)
\]

\[
\rho(d^{out},d^{in})
\]

---

## 5.2 Transition-role asymmetry

为了避免把“热门程度”误认为方向结构，对每个 item 定义：

\[
P_{out,i}(j)
=
\frac{C_{ij}}{\sum_k C_{ik}}
\]

\[
P_{in,i}(j)
=
\frac{C_{ji}}{\sum_k C_{ki}}
\]

定义：

\[
R_i
=
\left\|
P_{out,i}-P_{in,i}
\right\|_1
\]

然后分析：

\[
\rho(R,pop)
\]

如果 \(R_i\) 明显非零且与 popularity 的相关性有限，说明 transition-role 信息不完全等价于热门程度。

### 结果输出

| Dataset | Out-pop ρ | In-pop ρ | Out-In ρ | Mean R | Median R | R-pop ρ |
|---|---:|---:|---:|---:|---:|---:|
| Sport | | | | | | |
| Toys | | | | | | |

---

# 6. Gate 1：数据层停止规则

### No-Go A

如果：

- reciprocal pair 基本对称；
- \(|A_{ij}|\) 极小；
- item-level \(R_i\) 接近 0；

则：

> 数据本身没有足够的方向性结构。

**停止该方向。**

### No-Go B

如果存在明显 asymmetry，但：

\[
\rho(R,pop)
\]

极高，且控制 popularity 后方向性基本消失，则：

> 所谓 direction 很可能只是 popularity proxy。

**停止或重新定义研究问题，不进入 tokenizer。**

### Pass A

只有当：

1. 数据中存在稳定 asymmetry；
2. transition-role 信息不完全由 popularity 解释；

才进入 Stage 1B。

---

# 7. Stage 1B：连续表示与量化表示的结构保持诊断

## 7.1 使用已有 TA-SID

不重新训练模型。

提取：

### 连续 latent

\[
z_i
\]

即 RQ-VAE encoder 输出的连续 latent。

### Quantized latent

\[
x_i^q
\]

即 RQ-VAE 完成 RVQ 后的量化 latent / reconstruction representation。

**必须确认二者来自同一次 TA-SID/RQ-VAE 实验。**

---

# 8. Stage 1B 的核心任务：Transition-role Recovery

目标：

> 判断 item representation 是否包含足以恢复 transition-role / directional relation 的信息。

---

## 8.1 Pair 数据构造

只使用训练集 transition graph 中的 reciprocal pairs：

\[
C_{ij}>0,\quad C_{ji}>0
\]

对于每个 pair \((i,j)\)，计算：

\[
A_{ij}
=
\log
\frac{P(j|i)+\epsilon}
{P(i|j)+\epsilon}
\]

定义标签：

\[
y_{ij}
=
\begin{cases}
1,&A_{ij}>0\\
0,&A_{ij}<0
\end{cases}
\]

为了避免近似对称样本造成噪声，主实验建议设置一个最小 margin：

\[
|A_{ij}|\ge \delta
\]

\(\delta\) 作为预注册参数，例如 0.5。

同时保留：

\[
0<|A_{ij}|<\delta
\]

作为 sensitivity analysis，而不是混入主实验。

---

# 9. 数据划分

不能让 probe 记住固定 item pair。

采用：

## Edge-disjoint split

train / validation / test 中：

\[
(i,j)
\]

pair 完全不重复。

同时保证每个 item 在训练阶段具有足够 transition statistics。

### 推荐

- Train：70%
- Validation：10%
- Test：20%

按照 pair 分层抽样，使正负方向标签比例基本一致。

---

# 10. 防止 item identity leakage

至少加入一个控制：

### ID baseline

使用 item ID / one-hot 或等价的 ID-based representation 训练同样 probe。

如果 ID baseline 非常强而：

\[
z/x_q
\]

很弱，则说明 probe 很可能依赖 item identity。

此时需要重新设计 probe。

### 更稳妥的最终实验

如果计算资源允许，进一步做 item-disjoint test：

- test 中的 item 不出现在 train；
- 仅使用其 learned representation；
- probe 不能记忆 item ID。

这作为 robustness experiment。

---

# 11. Probe 设计

对于任意 representation \(r_i\)，构造：

\[
h_{ij}
=
[r_i;r_j;r_i-r_j]
\]

其中：

\[
r\in\{z,x_q\}
\]

训练相同结构的轻量 MLP：

\[
f(h_{ij})\rightarrow \hat y_{ij}
\]

建议：

- 2-layer MLP；
- hidden dimension 固定；
- 相同 optimizer；
- 相同 learning rate；
- 相同 epoch / early stopping；
- 相同 train/validation/test split；
- 3 random seeds。

不能为了让 z 或 xq 更强而分别调不同模型。

---

# 12. 核心指标

报告：

\[
AUC_z
\]

\[
AUC_q
\]

以及：

\[
\boxed{
\Delta_{\mathrm{role}}
=
AUC_z-AUC_q
}
\]

同时报告 3 seeds：

\[
mean\pm std
\]

结果表：

| Dataset | AUC(z) | AUC(xq) | Δrole |
|---|---:|---:|---:|
| Sport | | | |
| Toys | | | |

---

# 13. 必须增加一个更直接的方向性指标

为了避免 probe 完全依赖分类器能力，增加一个 reversal consistency analysis。

对于测试 pair：

\[
(i,j)
\]

同时输入：

\[
(i,j)
\]

和：

\[
(j,i)
\]

定义：

\[
D_{ij}
=
s(i,j)-s(j,i)
\]

判断：

\[
sign(D_{ij})
\]

是否与：

\[
sign(A_{ij})
\]

一致。

报告：

\[
Direction\ Accuracy
\]

这样同时有：

- AUC；
- direction accuracy。

---

# 14. Gate 2：量化损失停止规则

### No-Go C

如果：

\[
AUC_z\approx0.5
\]

说明连续 latent 本身没有可恢复的 transition-role 信息。

**停止。**

### No-Go D

如果：

\[
AUC_z>AUC_q
\]

但差异非常小、不同 seed 不稳定，说明：

> RQ-VAE 没有稳定的结构损失。

**停止。**

### Pass B

需要同时观察到：

\[
AUC_z>0.5
\]

且：

\[
AUC_z>AUC_q
\]

并且 gap 在不同 seed / 两个数据集上具有一致趋势。

此时才进入 Stage 1C。

**不预先规定“0.05 就算成功”，先看效应大小、稳定性和数据集一致性。**

---

# 15. Stage 1C：方向是否真的比对称结构有价值

这一阶段解决：

> “你是不是只是增加了一路 behavior feature？”

---

## 15.1 三种 transition condition

### Real Direction

真实：

\[
C_{ij}
\]

与：

\[
C_{ji}
\]

保持原始方向。

### Symmetric

构造：

\[
C^{sym}_{ij}
=
C_{ij}+C_{ji}
\]

完全消除方向。

### Random Direction

保持 pair 的连接关系和边总量，但随机交换：

\[
i\rightarrow j
\]

与：

\[
j\rightarrow i
\]

的方向。

---

# 16. 对照实验的目的

不是马上比较推荐指标。

首先比较：

\[
Transition\ Role\ Recovery
\]

即：

| Condition | AUC / Direction Accuracy |
|---|---:|
| Real Direction | |
| Symmetric | |
| Random Direction | |

理想证据：

\[
Real > Symmetric
\]

且：

\[
Real > Random
\]

如果：

\[
Real\approx Symmetric
\]

说明 direction 没有额外信息。

如果：

\[
Real\approx Random
\]

说明方向可能主要是噪声。

---

# 17. 最终 Go / No-Go 决策矩阵

| Gate | 必须观察到 | 失败后 |
|---|---|---|
| 1A 数据方向性 | 稳定 asymmetry | 放弃 |
| 1A popularity control | 非纯 popularity | 放弃/重定义 |
| 1B 连续表示 | \(AUC(z)>0.5\) | 放弃 |
| 1B 量化损失 | \(AUC(z)>AUC(x_q)\) 且稳定 | 放弃 |
| 1C 对照 | Real > Symmetric / Random | 放弃 |
| 全部通过 | 方向结构有必要进入 tokenizer | 进入第二创新设计 |

---

# 18. 成功后的第二阶段才做什么

只有全部 Gate 通过，才允许进入：

> **Transition-role-aware Tokenization**

届时再讨论：

\[
M_{out},M_{in}
\]

如何进入：

\[
e_i^{transition}
\]

以及如何与：

\[
e_i^{content}
\]

融合后进行 RQ-VAE。

还需要重新设计：

- direction-preserving objective；
- symmetric vs directional ablation；
- popularity-controlled ablation；
- T5 recommendation experiment；
- 与现有 TA-SID 的比较。

**这些现在全部不做。**

---

# 19. 实验最终应形成的证据链

如果成功，最终应该得到：

```text
① 原始行为序列存在 item transition asymmetry
                    ↓
② asymmetry 不是简单 popularity proxy
                    ↓
③ TA-SID 连续 latent 能编码 transition-role
                    ↓
④ RQ-VAE quantization 后该信息下降
                    ↓
⑤ 对称化 / 随机方向不能复现该结构
                    ↓
⑥ 因此存在 tokenizer-level transition-role
   preservation problem
                    ↓
⑦ 才设计第二创新进行修复
```

如果其中任意一个箭头断掉：

```text
停止 B
```

而不是继续换名字、换结构、换实验。

---

# 20. 当前执行清单

### Phase 0：准备
- [ ] 确认 Sport/Toys 训练序列路径
- [ ] 确认 TA-SID 的 \(z_i\) 保存位置
- [ ] 确认 \(x_i^q\) 保存位置
- [ ] 确认 item ID 映射一致
- [ ] 确认没有使用 validation/test 构建 transition

### Phase 1A：数据
- [ ] 构建 adjacent directed transition matrix
- [ ] 统计 reciprocal pairs
- [ ] 计算 \(A_{ij}\)
- [ ] 绘制 asymmetry distribution
- [ ] 计算 out/in degree
- [ ] 计算 popularity correlation
- [ ] 计算 \(R_i\)

### Phase 1B：表示
- [ ] 构造 reciprocal pair dataset
- [ ] edge-disjoint train/val/test
- [ ] 训练 z probe
- [ ] 训练 xq probe
- [ ] 3 seeds
- [ ] AUC
- [ ] direction accuracy
- [ ] \(\Delta_{\mathrm{role}}\)

### Phase 1C：控制
- [ ] Real Direction
- [ ] Symmetric
- [ ] Random Direction
- [ ] 比较 direction recovery

### Final Gate
- [ ] Go / No-Go
- [ ] **只有 Go 才进入第二创新设计**

---

## 当前研究纪律

**这份方案确定后，在 Phase 1 完成前禁止再进行大规模文献检索和新方案发散。**

实验过程中如果发现某个指标不好，先判断它对应哪个 Gate；**不能为了“救结果”临时增加新的指标或修改判据。**

只有实验结果出来后，才能决定：

> **继续 / 修改诊断 / 放弃。**
