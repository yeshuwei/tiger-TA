# 4.2 Adaptive K0 Selection — 机制确定与三数据集实证

> ⚠ **2026-09-04 补注**：本文档"机制定为 gap-based **扩容**闭环 / 三数据集 K0\*={256,256,512}"的
> 方法主张已被 E45 推翻（Toys 塌缩是 seed2024 单例，reseed 而非扩容才是正确动作；判 a：健康 256 ≈ 512）。
> 机制侧有价值的部分（转移熵 predictor 失败、z-gap 作为判据）保留。权威裁定见 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)。

> 生成: 2026-08-13 | 数据: 768d 模型(Beauty/Sport/Toys) + 4.1 Toys K0 sweep
> 结论先行:**转移熵不能预测容量需求(方向相反),单层 k-means 内在簇数也与 3-token 量化行为不一致;
> 可靠的自适应信号是训练后的 z-gap → 机制定为 gap-based 反馈闭环,4.2 无需任何新 RQ-VAE 训练。**
>
> ⚠ 协议注记（2026-08-21）：本表为 e43 时代 **5000 随机对**口径。E44 统一为 3000 随机对后，
> 规范值为 Beauty z-gap=0.0001 / Sport **0.0032**（本表 0.0024 废弃）/ Toys **0.0931**（本表 0.0933 废弃）；
> z/xq AUC 在 4 位小数略有抽样差异（如 Toys z 0.9652→0.9656）。以 E44 六数据集表为准。

## 1. 三个数据集在 K0=256 的完整画像

| Dataset | items | H_trans (bits) | z AUC | xq[full]@256 | **z-gap@256** | L0 用码@256 | K0* | z-gap(K0*) |
|---|---|---|---|---|---|---|---|---|
| Beauty | 12101 | 3.7746 | 0.8688 | 0.8687 | **0.0001** | 204/256 (80%) | **256** | 0.0001 |
| Sport | 18357 | 3.8780 | 0.9645 | 0.9621 | **0.0024** | 249/256 (97%) | **256** | 0.0024 |
| Toys | 11924 | 3.5161 | 0.9652 | 0.8719 | **0.0933** | 62/256 (24%) ⚠ | **512** | 0.0045 |

脚本: `tools/transition_entropy.py` | 诊断: [sid_capacity_diagnostic.md](sid_capacity_diagnostic.md) + 4.1 sweep

## 2. 关键发现:三个数据集真的需要不同容量(adaptive 前提成立)

- **Beauty/Sport 在 256 上就是近无损的**(gap ≤ 0.003),且 L0 利用率健康(80%/97%) → **K0\* = 256**。
- **Toys 是唯一容量不足的数据集**(gap=0.093, L0 塌缩 62/256) → 扩容到 512 修复(gap→0.0045)。
- 结论:**固定 K0=256 牺牲 Toys 的 transition structure;固定 K0=1024 对 Beauty/Sport 是无意义扩容**。
  Adaptive 的价值 = 给每个数据集恰好需要的容量,不为 Beauty/Sport 浪费、不放任 Toys 丢失。

## 3. 两个候选预测器都失败(重要负面结果)

### 3.1 转移熵(4.2 原假设)❌ — 方向相反

| Dataset | H(B|A) weighted (bits) | perplexity | 容量需求 |
|---|---|---|---|
| Beauty | 3.7746 | 13.69 | 256(低) |
| Sport | 3.8780 | 14.70 | 256(低) |
| **Toys** | **3.5161**(最低) | **11.44** | **512(高)** |

**Toys 转移熵最低,却最需要扩容** — 与"复杂度高→容量大"假设完全相反。
原因: H(B|A) 度量行为层转移分布的*不可预测性*(多样性);而 K0 需求由 *z 空间的簇结构与 encoder allocation* 决定
(E1/E2 已证明 Toys 的问题是 codebook allocation 而非行为层多样性)。
→ **transition entropy 不能作为 adaptive predictor,论文应如实报告这个反例。**

### 3.2 单层 k-means 内在簇数 ❌ — 与 3-token 量化不一致

`tools/intrinsic_capacity.py`: 对 z 做单层 k-means(k=64..1024)扫 transition AUC:
- Beauty: AUC 对 k 几乎不敏感(0.844→0.854),与 z-gap 小一致(扩容无收益)。
- Sport/Toys: 缓慢上升,到 1024 仍未饱和(0.955/0.948 < η·z)。

单层 k-means 曲线不能直接当 L0 需求:真实 RQ-VAE 的 L1/L2 残差量化额外提供 256² 组合容量,
所以 3-token 量化在 L0=512 就达到 0.968,而单层要 8000+ 码。→ 不能作为免训练预测器。

## 4. 机制定论: gap-based 反馈闭环(取代 entropy)

> **Adaptive coarse-level capacity allocation(操作化定义):**
> 1. 训练 RQ-VAE(默认 K0=256)。
> 2. 测 z-gap@K0 = AUC(z) − AUC(xq[full])。
> 3. gap ≤ ε(取 0.01)→ 容量充足,保持当前 K0。
> 4. gap > ε → 按 2× 扩容,重训,回步骤 2,直至 lossless。
> 5. **K0\* = 最小满足 z-gap ≤ ε 的粗层容量。**

### 为什么这是"自适应"而非"grid search"
- **有准则**:终止条件由 lossless 定义(z-gap≤ε),非"挑指标最好的"。
- **按需触发**:仅当 gap 超阈值才扩容,Beauty/Sport 零额外训练。
- **结果因数据而异**:三数据集自动落在 {256, 256, 512},不是统一最优超参。
- **成本**:每数据集 0–1 次额外 RQ-VAE 训练(Toys 已验证 512 达标,无需再扫)。

### 顺带的三态洞察(可写进论文)
- **容量充足态**(Sport): z 高 + L0 利用 97% + gap≈0 → 256 恰好够。
- **结构受限态**(Beauty): z 低(0.87)+ gap≈0 但扩容无收益(AUC 对 k 不敏感)→ 扩容无意义。
- **容量不足态**(Toys): z 高 + L0 塌缩 24% + gap 大 + 扩容有收益 → 必须扩容。
- 统一 K0 无法同时满足三态;自适应机制的正确行为 = 仅在不足态扩容至 lossless。

## 5. 对 4.3 的影响(最小 T5 集合)

4.2 **零新增 RQ-VAE 训练**(Beauty/Sport 用现有 256 模型,Toys@512 来自 4.1)。

4.3 对比表(T5 重训,3-seed):
| Dataset | Fixed-small K0=256 | **Ours Adaptive K0\*** | Fixed-large K0=1024 |
|---|---|---|---|
| Beauty | 现有 baseline T5 ✅ | =256(复用) | — |
| Sport | 现有 baseline T5 ✅ | =256(复用) | — |
| Toys | 现有 baseline T5 ✅ | **512(新训)** | 1024(新训) |

**新增 T5 只有 2 个配置:Toys@512(adaptive 主菜) + Toys@1024(fixed-large 对照)**
(Beauty/Sport 的 fixed-large 臂用 quantizer 指标 gap/util/recon 与 256 无差异论证,免训。)

## 文件
- `tools/transition_entropy.py` — 转移熵计算(history 连续对,加权/均匀,含 --with-target 变体)
- `tools/intrinsic_capacity.py` — z 空间单层 k-means 容量扫描
- 4.1 曲线: [doc/figures/e41_k0_sweep_curve.png](figures/e41_k0_sweep_curve.png)
