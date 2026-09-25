# E47 Seed 敏感度隔离筛查（Q1/Q2/Q3，2026-09-04）

> **一句话结论：** 灾难性 L0 塌缩在 9 个 数据集×seed 组合里仅出现 1 个（**Toys×seed2024**），
> 且 **Toys 塌缩是 TA-SID 行为融合特有的**——同一 seed2024、同一 896d 维度，把行为通道换成
> 128 个零（ZeroPad）就健康（L0 163 vs 62）。**诱因是行为内容改变量化难度，不是维度、不是
> RQ-VAE 普遍问题。** 这修正了 E46 的"任何训练都可能塌"的过度泛化 → 塌缩是
> **"行为融合 × Toys"** 的交互事件。
>
> 回答用户三个问题：Q1(Beauty/Sport 多种子波动？)→ 有轻微 z-gap 波动但**从不 L0 塌缩**；
> Q2(content-only 也塌吗？)→ **不塌**，fusion 特有；Q3(z-gap↔NDCG 跨数据相关？)→ 需 NDCG 锚点（可选 T5 臂）。

---

## 1. 目的与实验（全部量化端 gate，无 T5）

为回答三个问题而设，全链路复用 E45 gate（A 训练 RQ-VAE → B codegen → C z-gap 诊断，3000ep/batch1024/256³/896d-或-768d，仅 seed/输入不同）：

| 实验 | 输入 | seeds | 成本 | 目的 |
|---|---|---|---|---|
| S0 | content-only 768d | {2024}（用既有 ckpt，零训练） | 0 | Q2 第一问：content-only 是否也塌 |
| S1 | TA-SID fused 896d | Beauty/Sport × {2025,2026} | ~35min | Q1：B/S 是否跨 seed 波动/塌缩 |
| S2a | content-only 768d | Toys × {2025,2026} | ~15min | Q2：content-only 是否跨 seed 稳定 |
| S2b | ZeroPad 896d（content+128零） | Toys × {2024,2025} | ~15min | 维度-隔离：896d 死维 vs 行为内容 |

脚本：`rqvae/run_multiseed_repro.sh`（TA-SID）、`rqvae/run_contentonly_multiseed.sh`、`rqvae/run_zeropad_multiseed.sh`。
逐 seed 诊断报告：`doc/e4x_{ds}_seed{S}.md`、`doc/e4x_toys_contentonly_seed{S}.md`、`doc/e4x_toys_zeropad_seed{S}.md`。

---

## 2. 结果

### 2.1 Toys 隔离矩阵（★ 决定性对照）

> ⚠️ **2026-09-09 provenance 订正**：本表 0.0367 / L0=62 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**；本处仅作"塌缩实例"历史对照保留。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020（健康 codes，L0=207，z-gap 0.005）。权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

| Toys 输入 | 维度 | seed2024 | seed2025 | seed2026 | L0 塌缩率 | z-gap 全谱 |
|---|---|---|---|---|---|---|
| **TA-SID**（content+行为） | 896 | **62/256 ⚠ 塌缩** | 194/256 | 151/256 | **1/3** | +0.093 / +0.002 / +0.004 |
| **ZeroPad**（content+0） | 896 | 163/256 ✅ | 185/256 ✅ | — | 0/2 | ≈0（xq 略高 z）|
| **Content-only**（content） | 768 | 148/256 ✅ | 169/256 ✅ | 150/256 ✅ | 0/3 | ≈0（xq 略高 z）|

**ZeroPad@seed2024（163）与 TA-SID@seed2024（62）：同一 seed、同一 896d 架构超参，唯一差别 = 128 维是"真实行为"还是"0"。**
→ 塌缩不是维度效应（ZeroPad 证明 896d 死维不会塌），**是行为信号本身**把 Toys 的量化推进了易塌区。

### 2.2 Beauty / Sport 跨 seed（Q1）

| 数据集 | seed2024（既有） | seed2025 | seed2026 | 判定 |
|---|---|---|---|---|
| **Beauty** | L0 204, gap **0.0001** | L0 203, gap **0.0128 ⚠>ε** | L0 206, gap 0.0078 | L0 从不塌；z-gap 有波动，@2025 边缘超标 |
| **Sport** | L0 249, gap 0.0032 | L0 235, gap −0.0001 | L0 236, gap 0.0040 | 全健康、z-gap 稳定 |

Beauty@2025 的 z-gap 0.0128 是**第三个 ">ε 但无塌缩"** 观测（连同 Electronics 饱和型 0.0116）——细看是
encoder 层 z AUC 从 0.868(seed2024) 掉到 0.856(seed2025)，非码本分配崩坏。L0 利用正常(80%)、k-means 扫描确认容量非瓶颈。

### 2.3 z-gap 语义细化（对 E46 R5/判据叙事的重要修正）

- ε=0.01 更准确的角色是**"重建保真度报警线"**，不是"塌缩检测线"。>ε 的集合含三种形态：
  - **灾难塌缩**：Toys@2024，L0 24%、gap **0.093 ≫ ε** → 已知 NDCG −1.4%（R@20 −3.8%）
  - **饱和无塌缩**：Electronics，L0 100%、gap 0.0116
  - **健康分配微损**：Beauty@2025，L0 80%、gap 0.0128
- ⇒ **灾难性代价只出现在 gap ≫ ε（~0.09 级）的塌缩**；0.01–0.05 灰色区代价未知（可能轻微或可忽略）。
  0.01–0.05 区是否掉 NDCG = Q3 的开放问题（见 §4，可选 Beauty@2025 T5 臂）。

---

## 3. 对三个问题的判定

### Q1：Beauty/Sport 跑多种子了吗？有波动吗、影响结果吗？
- **跑前状态**：此前所有 RQ-VAE 是 seed2024 单点；现 Beauty/Sport 补了 seed{2025,2026}。
- **答案**：灾难性 L0 塌缩**不是普适现象**——9 个 数据集×seed 里仅 Toys@2024 一个。Beauty/Sport L0 跨 seed 从不塌（80%/92%+）。
  Beauty 的 **z-gap 跨 seed 有轻微波动**（0.0001→0.0128→0.0078，@2025 越过 ε 但幅度远小于塌缩级）；
  Sport 完全稳定。是否影响 NDCG = 仅 Beauty@2025 一个存疑点（见 §4 可选臂）。

### Q2：seed 问题是 TA-SID 特有，还是 content-only 也塌？（★ 本实验最有信息量）
- **答案**：**TA-SID（行为融合）特有**。content-only 768d 三 seed（148/169/150）与 ZeroPad 896d（163/185）
  全部健康、z-gap≈0，从不塌缩。
- 归因：**行为内容**（不是 896 维度）改变量化难度。ZeroPad@2024 对照一锤定音。
- 对叙事影响：E46 的"任何数据集、任何一次训练都可能塌"**过度泛化** → 修正为
  **"TA-SID 融合输入在 Toys 上呈 1/3 塌缩风险；content-only/ZeroPad 在该数据稳定；Beauty/Sport 融合后也稳定"**。
  塌缩是 行为融合 × 数据难度的交互事件，不是 RQ-VAE 的普遍属性。

### Q3：z-gap 与 NDCG 跨数据集相关吗？
- 跨数据集**绝对相关不成立**（NDCG 由数据难度主导，E46 O4 已述）。正确形态 = 阈值泛化。
- 现有 NDCG 锚点：Beauty@2024(gap 0.0001)→0.0415、Sport@2024(0.0032)→0.0322、
  Toys@2024 塌缩(0.093)→0.0367、Toys@2025 健康(0.0016)→0.0372。**惩罚侧只有 Toys 一个实例**。
- 新增灰色区观测（Beauty@2025 gap 0.0128）**缺 NDCG**——这是唯一能同时测"阈值泛化到第二数据集"
  和"灰色区代价"的锚点，但代价是一次 T5（见 §4，用户未定）。

---

## 4. 可选 Q3 臂（用户决策，未跑）

**Beauty@2025 codes（gap 0.0128 边缘超标）→ T5 重训（tag 如 `bh2025`）对照 Beauty@2024 的 NDCG 0.0415。**
两种结局：
- NDCG ≈ 0.0415（无惩罚）→ ε=0.01 报警线对"健康分配微损"不产生下游代价 → 阈值语义=保守，判据精度叙事干净；
- NDCG 掉 → 灰色区有真实代价，ε 阈值需下调或改为分级，判据叙事需补"风险分级"。
成本 ~40min/seed（T5 单 seed 探针，若显著再补 3-seed）。**暂不跑，等用户指示。**

---

## 5. 文件与资产

- 脚本：`rqvae/run_multiseed_repro.sh`（TA-SID 多种子）/ `rqvae/run_contentonly_multiseed.sh` / `rqvae/run_zeropad_multiseed.sh`
- 输入：`data/Toys/fused_item_emb_zeropad.parquet`（新建，contentonly+128 零）；Beauty 用 `fused_item_emb.parquet`（896d，旧命名，symlink 为 `fused_item_emb_768d.parquet`）
- 诊断：`doc/e4x_beauty_seed{2025,2026}.md`、`doc/e4x_sport_seed{2025,2026}.md`、`doc/e4x_toys_contentonly_seed{2024,2025,2026}.md`、`doc/e4x_toys_zeropad_seed{2024,2025}.md`
- 模型：`rqvae/ckpt/{Beauty,Sport}_TA_SID_seed{2025,2026}/`、`Toys_Content768d_seed{2025,2026}/`、`Toys_Zeropad_seed{2024,2025}/`
- 关联：[[session-2026-09-04-seed-health-screening]]（待建） | 权威裁定 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)（D1 状态已更新）
