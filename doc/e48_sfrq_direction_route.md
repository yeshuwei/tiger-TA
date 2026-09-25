# E48 SF-RQ（Semantic-First Residual Quantization）方向设计与可执行路线（2026-09-09）

> **状态：已裁定停投（2026-09-09）。** 用户裁定：**SF-RQ 不再跑 Step 1-3，D2 正式撤第二创新**（见
> [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md) D2）。本档保留为方向评估落档；
> Step-0（sinkhorn 探针，gate0 PASS-with-caveat）产物**并入论文稳健性叙事**（"塌缩刻画"），详见下。
> 停投理由：provenance 订正已使主表用健康 codes（"修复坏码本"动机消失）；塌缩真实成本 ≈0.0002 NDCG
> 且仅 1/3 概率；z-gap 判据 + reseed 兜底两个零成本手段已覆盖审稿人关心的稳健性 → 继续跑 SF-RQ 无投稿边际收益。
>
> 本文档回答三件事：① 这个方向的目的与意义（含诚实定位，**它是对第一创新的"加固/保险"，不是"效果加强"**）；
> ② 方法机制与可证伪预测；③ 分步可执行路线（每步带判据/成本/产物）。
> 属于 E46 D2（创新 2 方向）候选之一的评估落档。**执行与否已由用户裁定（停投）。**
>
> 关联：裁定 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)（D2 未定）、
> 实锤 [e47_seed_health_isolation.md](e47_seed_health_isolation.md)（塌缩=行为融合×Toys×seed2024）、
> 推翻 [e45_toys_multiseed_collapse.md](e45_toys_multiseed_collapse.md)（健康 256≈512）。
>
> ⚠️ **2026-09-09 provenance 订正（影响本档部分表述）**：主表 Toys codes（L0=62，NDCG 0.0367）经审计为**孤儿**
> ——无法从任何现存 ckpt 复现；幸存 seed2024 best_collision 实际健康（L0=207）。Toys 正式主结果已订正为
> **NDCG@20 = 0.0377±0.0007 / R@20 = 0.0911±0.0020**。本档中凡以 0.0367/62 作"塌缩实例"的历史对照仍可读，
> 但**勿作 Toys 正式结果引用**。权威记录见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。
> 该订正**不削弱**本方向的保险动机（E47：TA-SID 融合在 Toys 上仍以 seed/epoch 敏感方式出现塌缩、recon 静默、仅 z-gap 可察），
> 但"修复 Toys 主表坏码本"已非痛点（主表已用健康 codes）；Step 1 的 Repair 演示目标改为**现存 best_loss（L0=15）塌缩模型**，见下。

---

## 1. 一句话定位

**SF-RQ 不是让第一创新"更聪明"，是保证第一创新的结果"稳定不靠运气"**：
第一创新（行为融合输入 → RQ-VAE）在难数据（Toys）上有 1/3 概率**静默塌缩** L0 码本
（重构损失不可见，只有 z-gap 揪得出），SF-RQ 通过"在学到语义的 z 上显式构造 L0 分区并冻结"，
把这个"种子彩票"变成确定性事件。**作为支持性贡献 + 塌缩刻画是扎实的；作为独立的第二算法卖点不够。**

---

## 2. 问题陈述：为什么需要它

### 2.1 第一创新带来了什么副作用（E47 因果链，实锤）

| 环节 | 事实 | 证据 |
|---|---|---|
| ① 融合输入改变量化几何 | 同一 seed2024、同一 896d：ZeroPad(行为→0) 健康 163/256，TA-SID(真实行为) 塌缩 62/256 | E47 隔离矩阵 |
| ② 塌缩有数据集×seed 交互 | 9 个 数据集×seed 组合仅 1 个塌缩：**TA-SID×Toys×seed2024**（1/3） | E47 |
| ③ 塌缩是**静默的** | 塌缩码本重构损失与健康几乎无差（L1/L2 残差补偿），只有 z-gap(0.093≫ε) 检测得出 | E45/E47 |
| ④ 标准 RQ-VAE 无防护 | L0 用 argmin 无熵均衡；无码本复活/EMA；`kmeans_init` 在**随机 encoder 的首批 z** 上懒初始化（`vq.py` `init_emb` 首次 forward 触发），挡不住漂移 | 源码阅读 |
| ⑤ 下游真实代价 | 塌缩 NDCG 0.0367 vs 健康 0.0372（−1.4%），R@20 −3.8%（相对） | E45 判 (a) 表 |

### 2.2 为什么这对第一创新是"切肤之痛"而不是别人的问题

塌缩**只在第一创新的融合输入上发生**（content-only/ZeroPad 全 seed 健康，E47）。
所以第一创新每在更难的数据集/新数据上落地一次，就有静默出坏码本的风险——
**审稿人复现翻车、使用者换数据翻车**，而常规手段（看重构损失）完全发现不了。

### 2.3 一句话问题定义

> 第一创新的行为融合输入在难数据上以 1/3 概率触发**重构不可见**的粗层码本塌缩；
> 现有 RQ-VAE 训练无法阻止、现有指标无法察觉。需要一个**让语义分区确定存活**的量化器设计。

---

## 3. 方向的真实价值（诚实版，防过度宣称）

| 维度 | 判断 | 说明 |
|---|---|---|
| **期望推荐精度增益** | ≈ **0.0002 NDCG** | 塌缩代价 −1.4% 但仅 1/3 概率 → E[修复收益]=1/3×0.0005≈0.0002。**不可作为 accuracy 卖点** |
| **真正卖点** | **确定性 / 可复现性 / 稳健性** | 每次都是 0.0372，而非 1/3 概率掉 0.0367；对部署与审稿人复现是保险 |
| **新颖性** | 中偏弱 | freeze/两阶段/k-means 热启动均为 VQ 文献熟工具；**独特处** = RQ 残差掩盖塌缩（recon 失效）+ z-gap 探测器 + 与融合输入强绑定 |
| **审稿风险** | "修自家 bug" | 塌缩或可归因于自家量化器配置（argmin L0 / 无 EMA/复活 / 随机 z 上 kmeans）。需用 C0-C3 对照自证是"方法设计"而非"bug 修复" |
| **定位建议** | **支持性贡献** | 与第一创新的关系 = 保险/加固；论文中以"方法稳健性 + 塌缩刻画"一节呈现，**不占独立第二创新位** |
| **边界（不做什么）** | 不声称 > 健康基线 | 健康 256 ≈ 512 无统计差异，没有精度空间给"比健康更好"（E45 判 a 教训） |

---

## 4. 方法机制（SF-RQ）

### 4.1 核心思想

把 RQ-VAE 里"未被任何目标约束、训练中自由漂移"的粗层码本 L0，改为
**在学到语义的 z 上显式构造、训练期锁死**的分区：

1. **暖机（Stage A）**：按现役标准训至 z 有意义（encoder 收敛）。
2. **切分+锁定（A→B）**：在全量数据的当前 z 上跑 k-means(256) → 写入 L0 码本 →
   **冻结 L0**（`rq.vq_layers[0].embedding.weight.requires_grad_(False)`）。
3. **细调（Stage B）**：继续训 encoder + L1/L2 残差，让残差在冻结粗分区上补细节。

### 4.2 为什么机制上成立（关键推断）

- 塌缩模型里 **z 本身健康**（z AUC 0.966，E45）——塌的只是 L0 码本分配（62 码），不是 encoder 输出。
- 在健康 z 上 k-means(256) 的分区天然**转移对齐**（fresh kmeans-256 的 xq[L0] AUC 0.945 ≫ 塌缩的 0.854）。
- commit loss 把 z 钉在冻结码本邻近 → L0 想再塌也塌不掉；
- 残差层 L1/L2 本就"补细节"，换 L0 后细调重新适应即可，重构不损。

### 4.3 两个变体 + 兜底梯

| 变体 | 做法 | 适用 |
|---|---|---|
| **Repair（优先）** | 用现有已训完 ckpt 当 Stage A 产物 → 换 L0 → 冻结 → 细调 | **Stage A 零重训**、对现役管线触碰最小；可演示"修复真实塌缩模型" |
| **Prevent** | 暖机 T0 epochs 中途切分冻结，续训至 3000 | 避免 Stage A 养成坏残差习惯；总 epoch 对齐 |

**兜底三件套**（从干净到放松）：Repair 硬冻结 → 锚定正则（λ‖L0−c‖²）→ 条件门（仅 z-gap 超阈值时启用）。
健康数据上若冻结侵蚀收益 → 退条件门，SF-RQ 变"按需修复"。

---

## 5. 可证伪预测与失败预案

| # | 预测 | 若不符 → 预案 |
|---|---|---|
| **H1** | Toys 塌缩率 1/3 → **0/3**，z-gap 全 ≤ε | 用码多但 z-gap 仍高 → k-means 分区不够转移对齐 → 改行为加权切分，或降级"修复到健康区间"叙事；z 已被 Stage A 污染 → 移后切分/加复活 |
| **H2** | 细调后 recon 不劣化、NDCG ≥ 0.0372 地板 | 掉到 0.036x → "确定性地板 vs 幸运 seed 天花板"的真实权衡，如实报告；或退锚定版 |
| **H3** | Beauty/Sport（健康数据）冻结后 NDCG/AUC 不侵蚀 | 侵蚀 → 退条件门，SF-RQ 变按需修复（叙事更普适、纯度降） |

**防 E43 死因**：一切结论建立在**同 seed 对照（frozen vs unfrozen）**上，禁止从单 seed/单数据集导出"方法级"主张。

---

## 6. 与第一创新的隔离保证

**改什么（仅 RQ-VAE 训练规程）**：加切分冻结阶段。
**不改什么**：融合输入构造（同一 parquet）、RQ-VAE 架构（256³/e32/encoder-decoder）、T5 全链路。

**隔离对照梯**：

| 对照 | 内容 | 排除的混淆 |
|---|---|---|
| **C0 回归** | 改代码后先用现役参数重训 seed2025 → 复现健康(194/256) | 改代码破坏复现 |
| **C1 主隔离** | 同 seed2024：加冻结 vs 不加，唯一差异=冻结 | seed 运气/噪声 |
| **C2 机制归因** | 冻结后立即测 z-gap（细调前/后各一次） | 结构来自新分区而非细调 |
| **C3 创新1不蚀** | Beauty/Sport 冻结版 NDCG & 转移 AUC 守现役数字 | 用创新 1 收益为方向 1 买单 |

**叙事纪律**：创新 1 主表（+9~17% over content-only）用现役健康 run，方向 1 用独立受控对照（frozen vs unfrozen 同 seed）+ 展示 frozen-TA-SID 仍大幅赢 content-only → 两贡献各自可证、不互相挪用。

---

## 7. 可执行路线（分步 + 判据）

> 原则：**先量化端、后 T5；每步有预注册判据，不事后挑结论**。
> Step 0/1 本质是"机制诊断"（回答：塌缩可修复？锁定零代价？），不是方法承诺。

### Step 0 — sinkhorn-L0 机制探针（最廉价，零代码）

> ✅ **已执行（2026-09-09）**：gate0 = L0 62→256、z-gap 0.0070≤ε（形式 PASS），但 sinkhorn 强制均匀以
> z −0.025（0.9656→0.9407）为代价 → 触发路线预案"需结构式分配、SF-RQ 必要性上升、仍前进"。
> 完整结果见 [e48_step0_sinkhorn_gate0.md](e48_step0_sinkhorn_gate0.md)。

- **内容**：`--sk_epsilons 0.003 0 0.003`（L0 开 sinkhorn 均衡用量），Toys 融合输入 seed2024 重训。
- **目的**：验证"塌缩 = 用量分配崩坏、可修复"；同时给 C0 提供现役参照。
- **输入（provenance 已核验，2026-09-09）**：必须用 **`data/Toys/fused_item_emb_768d.parquet`（896d）**——E45/E47 的塌缩锚点 62/256 出自 896d 的 `Toys_TA_SID_768d`（诊断默认 CONFIG），**不是** 4224d 的 `Toys_TA_SID`（=4096d robustness track 副本，L0=110/z-gap≈0，是另一现象）。二者同名易混，勿用错。
- **成本**：~8 min（3000ep 一跑）；**产物**：ckpt + 诊断报告。
- **命令**：
  ```bash
  python rqvae/main.py --data_path data/Toys/fused_item_emb_768d.parquet \
    --ckpt_dir rqvae/ckpt/Toys_TA_SID_skl0_seed2024 \
    --num_emb_list 256 256 256 --sk_epsilons 0.003 0 0.003 --seed 2024
  # 然后 codegen + sid_capacity_diagnostic（复用现有 gate，参照 run_zeropad_multiseed.sh 的 B/C 段）
  ```
- **判据（gate 0）**：L0 用码 62→**≥130** 且 z-gap **≤0.01** → 机制确认、方向可押；
  若强制均匀伤重构（xq[full] AUC 掉）→ 说明需结构式分配，SF-RQ 必要性上升，仍前进。

### Step 1 — Repair 验证（正实验，最便宜）

> ⚠️ **provenance 订正后的 Repair 目标**：legacy L0=62 那个 ckpt 已不存在（孤儿，见顶部横幅）。
> 现存可用的塌缩样本只有幸存 `best_loss_model.pth`（**L0=15**，896d legacy seed2024）——用它演示
> "把真实塌缩模型修复到健康区间"。健康 `best_collision`（L0=207）现已是主表正式 codes，**不是** Step 1 目标。

- **内容**：新建 `rqvae/sf_rq_repair.py`（~120–160 行），**复用现存塌缩 ckpt**（`Toys_TA_SID_768d` 的 `best_loss_model.pth`，896d，legacy seed2024，L0=15，零 Stage A 重训）：load `state_dict` → 全量数据过 encoder 取 z → `layers.kmeans`(256) 写回 L0 → 冻结 → trainer 续训 ~500ep。
- **判据（gate 1）**：修复后 L0 ≥130、z-gap ≤0.01、recon(xq[full] AUC) 不劣于健康基线(~0.965)。
- **成本**：写脚本 ~1–2h + 跑 ~10 min；**产物**：repaired ckpt + codes + 诊断报告。

### Step 2 — 健康模型回归（C0/C3 量化端）

- **内容**：同一 repair 流程过 Toys seed2025/2026（应本健康）+ Beauty seed2024 + Sport seed2024。
- **判据（gate 2）**：修复后健康模型仍健康（C3 量化端：z-gap、L0、recon 不劣化）。
  若 Repair 弄坏健康模型 → 退"条件门"设计。

### Step 3 — T5 端判定（只在 gate 0–2 全过后才烧）

- **内容**：repaired Toys codes → T5 3-seed（tag `sfrq`）对照；若 gate2 过，补 Beauty/Sport 现役守住验证。
- **判据（gate 3）**：Toys NDCG ≥ **0.0372** 地板（且 std 不劣）；Beauty ≈0.0415、Sport ≈0.0322（±噪声内）。
- **成本**：T5 ~30–50 min/seed；Toys 3-seed ≈2h。

### 决策汇总

| 步 | 内容 | 成本 | 通过判据 | 不通过 → |
|---|---|---|---|---|
| 0 | sinkhorn-L0 探针 | 8 min | L0≥130 & z-gap≤ε | 仍可前进（结构式必要性↑） |
| 1 | repair 塌缩 ckpt | ~1.5h | L0≥130 & z-gap≤ε & recon 不劣 | F4 预案（污染 z/移切分/复活） |
| 2 | 健康模型回归 | ~30 min | 健康模型不劣化 | 退条件门 |
| 3 | T5 3-seed | ~2h | NDCG≥0.0372 地板 | 如实报告确定性 vs 天花板权衡 |

### ⛔ 停投裁定（2026-09-09，用户拍板）

**本路线到 Step-0 为止，Step 1-3 不执行；D2 撤第二创新，聚焦第一创新与论文。**

- **已产出的、并入论文稳健性叙事**（"塌缩刻画"一节）：
  1. E45/E47 塌缩实锤（行为融合×Toys 的 seed/epoch 敏感、recon 静默、仅 z-gap 可察）；
  2. z-gap 判据 + reseed 兜底 = 稳健性做法（零成本，已够）；
  3. **Step-0 sinkhorn 负结果** = "为什么不用均匀化/扩容量"的诚实对照：强制均匀伤 encoder z
     （同 seed 0.9656→0.9407），证明塌缩修复须"只动 L0 分配、不动 encoder 动力学"→ 支持 reseed 而非
     sinkhorn/扩容作为手段。
- **不再投入**：repair 脚本（Step 1）、健康模型回归（Step 2）、T5 sfrq 对照（Step 3）全部取消。
- 保留本档 + [e48_step0_sinkhorn_gate0.md](e48_step0_sinkhorn_gate0.md) 为评估/探针落档，不删除。

---

## 8. 工程清单

**新增**
- `rqvae/sf_rq_repair.py`：load ckpt → z → kmeans → 换 L0 → 冻结 → 续训（核心，唯一实质新代码）。
- （可选）`rqvae/run_sfrq_repair.sh`：批处理多 seed。

**改动**
- 现役管线（main.py / trainer.py / 数据 / 下游）**零改动**（Step 0 纯 CLI 参数；repair 独立脚本）。
- 冻结经 `requires_grad_(False)` 实现；优化器建在 `model.parameters()` 上，torch 自动跳过 grad=None 参数，**无需改 optimizer**。

**复用**
- `rqvae/datasets.py`（数据加载）、`rqvae/models/layers.py`（kmeans）、`rqvae/trainer.py`（fit）、
  `rqvae/generate_sid_codes_param.py`（codegen）、`tools/sid_capacity_diagnostic.py`（gate）、T5 runner。

**命名（依 [naming_convention.md](naming_convention.md)）**
- 修复模型 `rqvae/ckpt/{DS}_SFRQ_seed{S}/`；codes `data/{DS}/{DS}_sfrq_seed{S}_sid.npy`；
  诊断 `doc/e4x_{ds}_sfrq_seed{S}.md`；Step 0 用 tag `skl0`。

---

## 9. 关键代码事实（设计依据，已核实）

- `vq.py` `init_emb` 懒触发在首次训练 forward → L0 k-means 落在**随机 encoder** 的首批 z 上（为何 kmeans_init 挡不住塌缩）。
- ckpt 为 dict：`{"state_dict": ..., "optimizer": ..., "best_collision_rate": ...}`（trainer.py save），load 可直接复用。
- `sk_epsilons` 已是 `main.py` CLI 参数（Step 0 零代码）。
- 残差量化 `rq.py`：`x_q` 逐层累加，L1/L2 天然补细节（换 L0 后细调可适应）。
- Track B 死代码（L_trans/L_balance，λ=0）与新方向无关，勿混淆。

---

## 相关
[[session-2026-09-04-seed-health-screening]]（E47 记忆）| [e47_seed_health_isolation.md](e47_seed_health_isolation.md) | [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md) | [e45_toys_multiseed_collapse.md](e45_toys_multiseed_collapse.md)
