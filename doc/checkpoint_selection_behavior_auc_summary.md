# RQ-VAE Checkpoint Selection 对行为结构保持的影响 —— 实验结果（2026-09-28）

> **实验设计**：[exp/RQ-VAE checkpoint selection实验设计.md](exp/RQ-VAE%20checkpoint%20selection实验设计.md)
> **产出脚本**：[tools/checkpoint_selection_probe.py](../tools/checkpoint_selection_probe.py)
> **产出数据**：[checkpoint_selection_behavior_auc.csv](checkpoint_selection_behavior_auc.csv)（9 行）、
> [checkpoint_selection_code_structure.csv](checkpoint_selection_code_structure.csv)（18 行）、
> [checkpoint_selection_toys_provenance.csv](checkpoint_selection_toys_provenance.csv)、
> [checkpoint_selection_toys_provenance_matrix.csv](checkpoint_selection_toys_provenance_matrix.csv)
> **运行日志**：[checkpoint_selection_probe.log](checkpoint_selection_probe.log)（主实验）、
> [checkpoint_selection_toys_provenance.log](checkpoint_selection_toys_provenance.log)（provenance）
> **复现**：
> ```bash
> python tools/checkpoint_selection_probe.py --stage both --datasets Beauty,Sport,Toys \
>     --seeds 2024 2025 2026 --device cuda
> ```
> 本实验未重新训练任何 RQ-VAE；未改模型结构 / 数据 / pair sampling / AUC 定义 /
> seed（全部复用 `doc/g2_stage_probe_results.csv` 的 9 个 Dataset×Seed 单元）。

---

## 0. 结论速览（先读这一节）

**本实验的设计意图——"只改 checkpoint selection、其余全不变"——在现有 checkpoint 上
无法成立为一个干净的单变量对照。** 原因见 §5.1：全部 9 个单元里
`best_loss_model.pth` 都停在 **epoch 49**（`eval_step=50` 的第一个评估点）、
`best_collision_rate=inf`、保存的 `best_loss` 字段与 `best_collision` 那份**逐位相同**，
且重建损失高 ~40%、L0 使用率跌至 15–79/256。它**不是一个被另一种准则选中的充分训练模型**，
而是"最早的一个评估快照"。

因此对设计 §1 的核心问题，答案必须分两层给出：

| 读法 | 答案 |
|---|---|
| **按 §3 公式字面读**（`gap_loss = AUC(z_loss) − AUC(C_loss)`，两臂各自的 z） | **情况 C（数据集间不一致）**：Beauty 的 delta_gap 为负（−0.0070），Sport/Toys 为正（+0.0082 / +0.0133）。但该读法被 §5.1 的混淆污染，**不应据此下结论**。 |
| **同 z 口径**（固定 `z_collision`，只比量化步 `z → C`） | **9/9 单元一致为正**，Δ = **+0.0138 ~ +0.1163**（全库 +0.0715 ± 0.0408）。即：**给定同一个 z，best_loss 的量化损失是 best_collision 的 10–130 倍**。 |

**一句话**：在"checkpoint 已经存在"的前提下，collision-based selection 对 **z→C 这一步**
确实在做实事（同 z 口径全 9 格一致、幅度大）；但**不能**由此推出
"collision-based selection 是行为结构保持的必要条件"——因为对照臂不是一个公平的
替代准则，而是一个欠训练的早期快照。本实验**不足以**支持或否定 §6 情况 A 的因果表述。

---

## 1. 口径复核（与 g2 对齐的自检）

B/C 空间的 AUC 与 g2 已发布值逐格对齐，确认采样路径 / z 编码 / C 重建与 g2 逐位复用：

| dataset | seed | g2 `space=B` | 本实验 `auc_z` | Δ | g2 `space=C` | 本实验 `auc_best_collision` | Δ | L0 distinct (g2/本) | n_pos |
|:--|:--:|--:|--:|--:|--:|--:|--:|:--:|--:|
| Beauty | 2024 | 0.962335 | 0.962335 | +1.1e-16 | 0.956025 | 0.956025 | 0 | 204 / 204 | 1000 |
| Beauty | 2025 | 0.960089 | 0.960089 | −1.1e-16 | 0.955619 | 0.955619 | 0 | 203 / 203 | 1000 |
| Beauty | 2026 | 0.970789 | 0.970789 | +1.1e-16 | 0.962542 | 0.962542 | 0 | 206 / 206 | 1000 |
| Sport | 2024 | 0.965484 | 0.965484 | −7.4e-08 | 0.964747 | 0.964747 | −1.3e-07 | 249 / 249 | 998 |
| Sport | 2025 | 0.957976 | 0.957976 | −7.2e-08 | 0.957659 | 0.957659 | +2.0e-07 | 235 / 235 | 998 |
| Sport | 2026 | 0.962762 | 0.962762 | −2.0e-07 | 0.962582 | 0.962582 | −4.8e-07 | 236 / 236 | 998 |
| Toys | 2024 | 0.969677 | 0.969677 | +1.1e-16 | 0.965060 | 0.965060 | 0 | 207 / 207 | 1000 |
| Toys | 2025 | 0.972214 | 0.972214 | 0 | 0.970498 | 0.970498 | 0 | 194 / 194 | 1000 |
| Toys | 2026 | 0.976117 | 0.976117 | +1.1e-16 | 0.968805 | 0.968805 | 0 | 151 / 151 | 1000 |

Sport 的 ~1e-7 残差来自 float32 编码器的非确定性，非口径差异。主指标为
`topk` 正样本 × `degmatch` 负样本（与 g2 §主指标一致）。

---

## 2. 逐 Dataset × Seed 结果

主指标 AUC（`topk`×`degmatch`）。`delta_gap = gap_loss − gap_collision`（§3 原公式）。
`delta_gap_common_z` 为**追加列**，见 §5.2 的定义与理由。

| dataset | seed | `auc_z` | `auc_best_collision` | `gap_best_collision` | `auc_z_best_loss` | `auc_best_loss` | `gap_best_loss` | `delta_gap` | `delta_gap_common_z` |
|:--|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Beauty | 2024 | 0.962335 | 0.956025 | +0.006310 | 0.853975 | 0.855773 | −0.001798 | **−0.008108** | +0.100252 |
| Beauty | 2025 | 0.960089 | 0.955619 | +0.004470 | 0.840873 | 0.845473 | −0.004600 | **−0.009070** | +0.110146 |
| Beauty | 2026 | 0.970789 | 0.962542 | +0.008247 | 0.850775 | 0.846272 | +0.004503 | **−0.003744** | +0.116270 |
| Sport | 2024 | 0.965484 | 0.964747 | +0.000737 | 0.947378 | 0.937946 | +0.009432 | **+0.008695** | +0.026801 |
| Sport | 2025 | 0.957976 | 0.957659 | +0.000317 | 0.949801 | 0.943894 | +0.005907 | **+0.005590** | +0.013765 |
| Sport | 2026 | 0.962762 | 0.962582 | +0.000180 | 0.949698 | 0.939327 | +0.010371 | **+0.010192** | +0.023256 |
| Toys | 2024 | 0.969677 | 0.965060 | +0.004617 | 0.891417 | 0.858828 | +0.032589 | **+0.027972** | +0.106232 |
| Toys | 2025 | 0.972214 | 0.970498 | +0.001716 | 0.904103 | 0.890195 | +0.013908 | **+0.012192** | +0.080303 |
| Toys | 2026 | 0.976117 | 0.968805 | +0.007312 | 0.909484 | 0.902560 | +0.006924 | **−0.000388** | +0.066245 |

### 2.1 mean ± std（3 seed）

| dataset | `auc_z` | `auc_best_collision` | `gap_best_collision` | `auc_best_loss` | `gap_best_loss` | `delta_gap` | `delta_gap_common_z` |
|:--|:--|:--|:--|:--|:--|:--|:--|
| Beauty | 0.964404 ± 0.005642 | 0.958062 ± 0.003885 | +0.006342 ± 0.001889 | 0.849173 ± 0.005730 | −0.000632 ± 0.004662 | **−0.006974 ± 0.002838** | +0.108889 ± 0.008083 |
| Sport | 0.962074 ± 0.003801 | 0.961663 ± 0.003632 | +0.000411 ± 0.000290 | 0.940389 ± 0.003113 | +0.008570 ± 0.002354 | **+0.008159 ± 0.002347** | +0.021274 ± 0.006740 |
| Toys | 0.972669 ± 0.003244 | 0.968121 ± 0.002783 | +0.004548 ± 0.002799 | 0.883861 ± 0.022544 | +0.017807 ± 0.013269 | **+0.013259 ± 0.014210** | +0.084260 ± 0.020285 |
| **全部 9 格** | 0.966383 ± 0.006120 | 0.962615 ± 0.005338 | +0.003767 ± 0.003132 | 0.891141 ± 0.041564 | +0.008582 ± 0.010704 | +0.004815 ± 0.011701 | **+0.071474 ± 0.040766** |

**要点**：

1. **`gap_best_collision` 极小且稳定**：三个数据集均 < 0.007，Sport 甚至只有 4e-4。
   这与既有"当前较小的量化行为结构损失"的观察一致。
2. **`auc_z_best_loss` 自身就塌了**：0.891 / 0.849 / 0.949，比 counterpart 低
   0.011（Sport）~ 0.115（Beauty）。**这是 delta_gap 出现负号的直接原因**（§5.1）。
3. `delta_gap` 的**符号在数据集间不一致**（Beauty 全负、Sport 全正、Toys 2 正 1 负），
   且 std 与均值同量级（0.0117 vs 0.0048）→ 按 §3 字面读只能落到 §6 情况 C。
4. `delta_gap_common_z` **9/9 为正**，最小值 +0.0138（Sport 2025），最大 +0.1163（Beauty 2026）。

---

## 3. 离散码结构对比（两个 checkpoint）

`compute_utilization` / `compute_collision_rate` 复用自
[tools/sid_quality.py](../tools/sid_quality.py)。utilization = 被至少一个 item 使用的码数 / 256。
`full_sid_collision_rate` 按 codes 文件全部 4 列计（含碰撞消解序号）。

| dataset | seed | ckpt | epoch | L0 distinct | L1 distinct | L2 distinct | L0 util | L1 util | L2 util | prefix-3 collision | full SID collision |
|:--|:--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Beauty | 2024 | best_collision | 1849 | 204 | 256 | 256 | 0.797 | **1.000** | **1.000** | 0.000496 | 0.000000 |
| Beauty | 2024 | best_loss | **49** | 44 | 171 | 255 | 0.172 | 0.668 | 0.980 | 0.019337 | 0.000000 |
| Beauty | 2025 | best_collision | 1949 | 203 | 256 | 256 | 0.793 | **1.000** | **1.000** | 0.000165 | 0.000000 |
| Beauty | 2025 | best_loss | **49** | 55 | 164 | 256 | 0.215 | 0.641 | 1.000 | 0.034956 | 0.000000 |
| Beauty | 2026 | best_collision | 2399 | 206 | 256 | 256 | 0.805 | **1.000** | **1.000** | 0.000331 | 0.000000 |
| Beauty | 2026 | best_loss | **49** | 56 | 169 | 256 | 0.219 | 0.660 | 0.977 | 0.015288 | 0.000000 |
| Sport | 2024 | best_collision | 2899 | 249 | 256 | 256 | 0.973 | **1.000** | **1.000** | 0.000000 | 0.000000 |
| Sport | 2024 | best_loss | **49** | 60 | 200 | 256 | 0.234 | 0.781 | 1.000 | 0.028599 | 0.000000 |
| Sport | 2025 | best_collision | 2549 | 235 | 256 | 256 | 0.918 | **1.000** | **1.000** | 0.000000 | 0.000000 |
| Sport | 2025 | best_loss | **49** | 79 | 226 | 256 | 0.309 | 0.883 | 1.000 | 0.010296 | 0.000000 |
| Sport | 2026 | best_collision | 2499 | 236 | 256 | 256 | 0.922 | **1.000** | **1.000** | 0.000000 | 0.000000 |
| Sport | 2026 | best_loss | **49** | 74 | 239 | 256 | 0.289 | 0.934 | 1.000 | 0.013782 | 0.000000 |
| Toys | 2024 | best_collision | 2049 | 207 | 256 | 256 | 0.809 | **1.000** | **1.000** | 0.000755 | 0.000000 |
| Toys | 2024 | best_loss | **49** | **15** | 127 | 256 | **0.059** | 0.496 | 0.934 | 0.050570 | 0.000000 |
| Toys | 2025 | best_collision | 1449 | 194 | 256 | 256 | 0.758 | **1.000** | **1.000** | 0.000587 | 0.000000 |
| Toys | 2025 | best_loss | **49** | 58 | 192 | 256 | 0.227 | 0.750 | 0.996 | 0.009728 | 0.000000 |
| Toys | 2026 | best_collision | 999 | 151 | 256 | 256 | 0.590 | **1.000** | **1.000** | 0.001174 | 0.000000 |
| Toys | 2026 | best_loss | **49** | 77 | 224 | 256 | 0.301 | 0.875 | 0.996 | 0.004864 | 0.000000 |

**要点**：

1. **`best_collision` 的 L1 utilization 恒为 1.000，best_loss 从不达到**（0.496–0.934）。
   L0 utilization 差距最大：best_collision 0.590–0.973，best_loss 0.059–0.309。
   （L2 utilization 两臂都接近满：best_collision 全 1.000，best_loss 0.934–1.000，
   区分度弱，不作主判据。）
2. **`full_sid_collision_rate` 18/18 全部为 0.000000。** 这是本实验一个值得单独记住的读数：
   `best_collision` 被选中去最小化的那个量，**在导出的 SID 上已经被 codegen 的碰撞消解循环
   抹平**（`generate_sid_codes_param.py` 第 4 列就是消解序号）。两份 checkpoint 导出的码在
   "有无重复 SID"这一层**没有区别**；它们的区别保留在 **prefix-3 碰撞率**（best_collision
   0.0000–0.0012 vs best_loss 0.0049–0.0506）与 **码本利用率**上。→ checkpoint selection
   真正带进来的是**健康的码本 / 编码器**，而不是"零碰撞"这个序性质本身。
3. **Toys 2024 `best_loss` 的 L0 util = 15/256。** 这正是
   [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md) §1
   提到的"best_loss 塌缩 L0=15"的那份快照——本实验确认它**就是 ep-49 的 best_loss 检查点**。
4. prefix-1 碰撞率对所有 ckpt 都在 0.98 以上（256 个 L0 码装不下 12101–18357 个 item），
   无区分度，故不列为主判据。

---

## 4. RQ-VAE 损失对比（eval 全量，**trainer 同口径**：逐 batch `reduction='mean'` 后求和）

> **⚠ 2026-09-28 订正（本节数字整体替换）**
> 初版本节用的是 `F.mse_loss(..., reduction='sum') / N`，即**逐元素均值的 768 倍**
> （768 = 输入维 `in_dim`，不是 `e_dim=32`；`sum` 累加的是 B×768 个元素，
> 却只除以 item 数 N），而 quant 列是逐 item 均值（~1 倍）。
> **两列量纲不一致**，导致 `total` 列被 768 倍膨胀的 recon 主导，
> 初版据此得出的「best_loss 的 total loss 反而更大」是**错误的**。
> 现改为与 `trainer._train_epoch` 的 `total_recon_loss` / `total_quant` **完全同口径**，
> 数值可与训练日志逐行对照（如 Toys 2024 实测 total=2.8132 vs 日志 2.8269；
> Beauty 2026 实测 2.7309 vs ckpt 字段 2.736835，均在轮内波动范围内）。
> 结论反转：**best_loss(ep49) 的总损失在 9/9 单元里确实更小**，checkpoint 选择逻辑是对的。

| dataset | seed | recon: best_collision → best_loss | quant: → | total: → | 谁的总损失更小 |
|:--|:--:|:--|:--|:--|:--|
| Beauty | 2024 | 1.7717 → 2.5315 (+42.9%) | 3.6017 → 0.2586 | 5.3734 → **2.7901** | ✅ best_loss |
| Beauty | 2025 | 1.7680 → 2.5447 (+43.9%) | 3.5349 → 0.3053 | 5.3029 → **2.8499** | ✅ best_loss |
| Beauty | 2026 | 1.7641 → 2.5606 (+45.2%) | 4.0082 → 0.1703 | 5.7723 → **2.7309** | ✅ best_loss |
| Sport | 2024 | 2.6617 → 3.6742 (+38.0%) | 6.6665 → 0.6730 | 9.3281 → **4.3471** | ✅ best_loss |
| Sport | 2025 | 2.7183 → 3.6793 (+35.4%) | 6.4502 → 0.5848 | 9.1685 → **4.2641** | ✅ best_loss |
| Sport | 2026 | 2.6955 → 3.6926 (+37.0%) | 6.6650 → 0.5223 | 9.3605 → **4.2149** | ✅ best_loss |
| Toys | 2024 | 1.7277 → 2.5397 (+47.0%) | 3.4624 → 0.2735 | 5.1901 → **2.8132** | ✅ best_loss |
| Toys | 2025 | 1.7720 → 2.4775 (+39.8%) | 3.2174 → 0.2561 | 4.9894 → **2.7336** | ✅ best_loss |
| Toys | 2026 | 1.8065 → 2.4400 (+35.1%) | 3.1286 → 0.2330 | 4.9351 → **2.6730** | ✅ best_loss |

**要点**：

1. `best_loss` 的**重建损失全面劣于** `best_collision`（**+35.1% ~ +47.0%**）——
   这一条初版就是对的，量纲修正后比例不变。
2. 但 `best_loss` 的**总损失反而更小**（9/9），因为其 **quant 项低一个数量级**
   （0.17–0.67 vs 3.13–6.67）。`train_loss = recon + quant_loss_weight × quant`
   由 quant 项支配 → **ep49 确实是这条训练轨迹上的总损失最小值**，
   `best_loss_model.pth` 的选取**忠实地执行了既定判据**（详见 §5.1 的机理）。
3. 因此，"best_loss 检查点更差"的正确表述是：**它在下游关心的重建质量上更差
   （recon 差 35–47%），但它在训练目标本身（recon+commitment）上更好**。
   训练目标与重建质量/下游行为**不对齐**——这才是 best_loss 不适合当对照臂的根本原因。

---

## 5. delta_gap 分析

### 5.1 关键发现：`best_loss_model.pth` 不是一个可用的对照臂

9/9 单元的 checkpoint 元数据完全同构：

| 属性 | `best_collision_model.pth` | `best_loss_model.pth` |
|:--|:--|:--|
| epoch | 999 – 2899（各单元不同） | **49（全部 9 个单元）** |
| `best_collision_rate` 字段 | 0.0059 – 0.0118 | **inf** |
| `best_loss` 字段 | 2.68 – 4.35 | **与左侧逐位相同** |
| 重建损失（trainer 口径） | 1.73 – 2.72 | 2.44 – 3.69 |
| 总损失（trainer 口径） | 4.94 – 9.36 | **2.67 – 4.35（更小）** |
| L0 utilization | 0.59 – 0.97 | 0.06 – 0.31 |

**机理（2026-09-28 定位，非代码 bug）**：`train_loss = recon + quant_loss_weight × quant`，
而这条训练轨迹上两项**反向演化**——重建损失单调下降（11.9 → 1.73），
**quant（承诺/码本）项单调上升（0.24 → 3.92，×16）**，于是总损失在 **ep42–79 触底后
持续上升**（Toys：ep49=2.8269 → ep2049=5.2210）。评估受
`(epoch_idx+1) % eval_step == 0` 约束（`eval_step=50`），**最早的评估点就是 epoch 49**，
所以总损失的最小值恰好在首次评估处被捕获，此后**没有任何评估点能再刷新它**。
`fit()` 的比较与保存逻辑经逐行复刻验证**完全正确**（9/9 单元 ep49 的总损失确实是
全体评估点中的最小值，见 §4）。→ 详见 [trainer_best_loss_root_cause.md](trainer_best_loss_root_cause.md)。

因此 best_loss 检查点的准确画像是：**在训练目标（recon+commitment）上更优，
但在重建质量上更差、码本更塌缩的早期快照**——它执行了既定判据，只是该判据
与下游关心的东西**不对齐**。

**这正是 `delta_gap` 出现负号的机制**：`gap_loss = AUC(z_loss) − AUC(C_loss)` 的基线
是一个**自身已经崩坏的 z**。一个已经很差的 z 再被量化，几乎无结构可失
（Beauty：z 从 0.9644 掉到 0.8485，但 C 只从 0.8492 掉到 0.8485），
于是 `gap_loss` 数值上**变小**。**这不是"量化更无害"，而是基线在移动。**
把两个 arm 的 gap 直接相减，混合了"编码器选点"与"量化步"两种效应，无法归因。

### 5.2 两种口径并列（不做替换，只做标注）

设计 §3 只给出一列 `auc_z`，但**两个检查点的编码器权重不同 ⇒ z 本身不同**，
"同一个 AUC(z)"在数学上不存在。为不篡改设计字段，本实验：

`checkpoint_selection_behavior_auc.csv` 的列布局 = **前 8 列严格按设计 §5 要求的字段与顺序**
（`dataset, seed, auc_z, auc_best_collision, gap_best_collision, auc_best_loss, gap_best_loss, delta_gap`），
其后 23 列为审计与上下文列（provenance 归属、epoch/元数据推断所需的 tag 与 run_dir、
其余 AUC 轴、各类损失）。具体做法：

- 保留设计要求的 8 列，其中 **`auc_z` = best_collision 编码器的 z**
  （与 g2 已发布的 B 空间读数一致，便于审计，见 §1）；
- **追加** `auc_z_best_loss`（best_loss 编码器的 z）、
  `gap_best_loss_common_z` = `AUC(z_collision) − AUC(C_loss)`、
  `delta_gap_common_z` = `gap_best_loss_common_z − gap_best_collision`，
  用于**隔离量化步本身的损失**（固定同一 z）。

| dataset | `delta_gap`（§3 原式，两臂各自 z） | `delta_gap_common_z`（固定 z，仅量化步） | 比值 |
|:--|--:|--:|--:|
| Beauty | **−0.006974 ± 0.002838** | **+0.108889 ± 0.008083** | 符号相反 |
| Sport | +0.008159 ± 0.002347 | +0.021274 ± 0.006740 | 2.6× |
| Toys | +0.013259 ± 0.014210 | +0.084260 ± 0.020285 | 6.4× |
| 全部 | +0.004815 ± 0.011701 | +0.071474 ± 0.040766 | 14.8× |

**同 z 口径下 9/9 为正、无一例外**，且其幅度是 §3 原式 delta_gap 的 2.6–14.8 倍。

若进一步直接问"z→C 这一步谁更省"，即按 `gap_best_loss_common_z / gap_best_collision` 计，
**best_loss 的量化损失是 best_collision 的 10.1–130.2 倍**
（逐格：Beauty 16.9× / 25.6× / 15.1×，Sport 37.4× / 44.4× / 130.2×，
Toys 24.0× / 47.8× / 10.1×；数据集均值 Beauty 19.2× / Sport 70.7× / Toys 27.3×）。
即两者在量化步上并非"差不多"，而是**相差一到两个数量级**。

### 5.3 落到设计 §6 的哪一档

| 情况 | 判据 | 本实验结果 |
|:--|:--|:--|
| **A** `best_loss` 的 gap 明显更大 | → 结论受 checkpoint selection 支配 | 仅在**同 z 口径**下成立（9/9），§3 原式下不成立 |
| **B** 两者 gap 都较小 | → 结论不主要依赖 collision selection | 否：best_loss 臂的 z 已崩，谈不上"都较小" |
| **C** 数据集间结论不同 | → 分别报告 | **按 §3 原式字面读，落此档**（Beauty 负 / Sport·Toys 正） |

**建议的表述**：按 §6 的措辞纪律，本实验应报为
**情况 C（按数据集分别报告）**，并同时声明 **§3 原式存在已识别的混淆（§5.1），
故 C 的"数据集差异"不宜解释为"机制在不同数据集上不同"**；
在控制了 z 之后（§5.2），全部 9 格指向同一方向。

---

## 6. Toys 数据版本同步检查（设计 §7）

检查方式：对每个 variant 的 codes 文件，用**全部现存** Toys checkpoint
（13 个训练目录 × best_collision / best_loss = 26 个候选）重建码本、按 RQ 真实前向
重放残差（`sf_rq.rederive_codes_sequential`，`use_sk=False` 对齐 codegen 主 pass），
与 codes 文件 L0 列做行级一致率。**判据只有 L0**（L0=100% 蕴含 L1=100%，L1/L2 不是独立证据；
见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md) §1 订正块）。
完整交叉矩阵见 [checkpoint_selection_toys_provenance_matrix.csv](checkpoint_selection_toys_provenance_matrix.csv)。

| variant | codes 文件 | L0 distinct | 判定 | 最佳匹配 ckpt | L0 一致率 |
|:--|:--|--:|:--|:--|--:|
| Content-only | `Toys_content768d_baseline_sid.npy` | 148 | ✅ OK | `Toys_Content768d_Baseline/Jul-31-2026_21-37-33` best_collision | **100.00%** |
| Content-only | `Toys_content768d_seed2025_sid.npy` | 169 | ✅ OK | `Toys_Content768d_seed2025/Sep-04-2026_17-07-25` best_collision | **100.00%** |
| Content-only | `Toys_content768d_seed2026_sid.npy` | 150 | ✅ OK | `Toys_Content768d_seed2026/Sep-04-2026_17-14-16` best_collision | **100.00%** |
| ZeroPad | `Toys_zeropad_seed2024_sid.npy` | 163 | ✅ OK | `Toys_Zeropad_seed2024/Sep-04-2026_17-21-10` best_collision | **100.00%** |
| ZeroPad | `Toys_zeropad_seed2025_sid.npy` | 185 | ✅ OK | `Toys_Zeropad_seed2025/Sep-04-2026_17-28-11` best_collision | **100.00%** |
| Shuffled | `Toys_shuffled_768d_sid.npy` | 243 | ✅ OK | `Toys_Shuffled_768d/Sep-23-2026_08-37-17` best_collision | **100.00%** |
| ItemCF | `Toys_itemcf_768d_sid.npy` | 161 | ✅ OK | `Toys_ItemCF_768d/Sep-23-2026_00-45-14` best_collision | **100.00%** |
| Raw-CoOcc | `Toys_rawcooc_768d_sid.npy` | 222 | ✅ OK | `Toys_RawCoOcc_768d/Sep-22-2026_22-36-51` best_collision | **100.00%** |
| **TA-SID（健康）** | `Toys_ta_sid_768d_seed2024_sid.npy` | **207** | ✅ OK | `Toys_TA_SID_768d/Jul-28-2026_23-32-18` best_collision | **100.00%** |
| TA-SID（孤儿·已作废） | `Toys_ta_sid_768d_sid.npy` | 62 | ❌ **ORPHAN** | 最佳仅 `Toys_TA_SID_768d` best_collision | 29.26% |
| **Random** | `Toys_random_sid.npy` | 122 | ❌ **ORPHAN** | 最佳 `Toys_TA_SID_seed2025` best_loss | **0.99%** |

**结论**：

1. **E49 Toys 主表用到的 6 个 variant（Content-only×3 seed、ZeroPad×2、Shuffled、
   ItemCF、Raw-CoOcc、TA-SID）全部 100.00% 配对到现存的 best_collision 检查点。**
   没有任何一个仍是孤儿。E49 表里**只有** TA-SID 一行曾受孤儿 codes 影响，
   且已在 [e48](e48_toys_main_codes_provenance_fix.md) 中订正为 0.0377，本检查**确认该订正到位**。
2. 旧孤儿 `Toys_ta_sid_768d_sid.npy`（L0=62）复现出文档记载的 **29.26%**，裁定不变：作废。
3. **新发现（超出设计 §7 清单，但属同一问题）**：`Toys_random_sid.npy`（L0=122）
   也是**孤儿**——最佳一致率 0.99%，且 `rqvae/ckpt/` 下**根本不存在 `Toys_Random*` 目录**
   （只有 `Sport_Random_Ablation`；[experiment_dimension_manifest.md](experiment_dimension_manifest.md) §78/93
   声称的 `rqvae/ckpt/Toys_Random/` 已失效）。其 ckpt 来自
   [ta_sid_track_a_plan.md](ta_sid_track_a_plan.md) 的第 111–118 行（`ckpt/Toys_Random`），
   该目录现已不在。
   **该 variant 不在 E49 结果表内**（E49 只有 Sport Random 辅助行 0.0281），
   但 [paper_audit_critique.md](paper_audit_critique.md) 第 90 行引用了
   "Random SID Toys = 0.0308"。**该数字与 0.0367 同类，建立在不可复现的孤儿 codes 上，
   不应继续引用**（除非能重新导出并注明）。
4. 全部 26 个候选里，**best_loss 从未达到 90% 一致率**（最高 24.84%，Toys Content-only
   2024）；best_loss 列普遍落在 5–25%——与 §3 的 L0 塌缩互相印证：ep-49 快照的
   L0 分配与任何已发布 codes 都对不上。

### 6.1 TA-SID 相对 Raw-CoOcc 的提升（按 §7 要求重算）

用 E49 订正后的健康数字（[e49_results_corrected.md](e49_results_corrected.md)）：

| 数据集 | 变体 | NDCG@20 | Recall@20 | vs Raw-CoOcc ΔN@20 | ΔR@20 |
|:--|:--|--:|--:|--:|--:|
| **Toys** | TA-SID（健康 codes，207） | 0.0377 ± 0.0007 | 0.0911 | **+10.88%** | **+13.59%** |
| Toys | Raw-CoOcc | 0.0340 ± 0.0015 | 0.0802 | — | — |
| **Sport** | TA-SID | 0.0322 ± 0.0005 | 0.0769 | **+5.57%** | **+5.20%** |
| Sport | Raw-CoOcc | 0.0305 ± 0.0006 | 0.0731 | — | — |

参照：旧孤儿数字 0.0367 相对 Raw-CoOcc 0.0340 只能算出 +7.9%，**订正后提升为 +10.9%**——
健康 codes 使 TA-SID 相对 Raw-CoOcc（即"用同样的行为共现信息但不做语义融合"）
的优势**变大了**。Toys 侧其余近邻：ItemCF 0.0355、Content-only 0.0336、ZeroPad 0.0334、
Shuffled 0.0284。TA-SID 在 Toys 上仍居首，且领先第二名 ItemCF 的幅度为
(0.0377−0.0355)/0.0355 = **+6.2%**。

---

## 7. 局限与不应推出的结论

按设计 §8（不提前假设结论 / 不自动提新方法），以下边界必须显式写明：

1. **不能**声称"collision-based selection 是行为结构保持的重要来源"。
   对照臂 `best_loss_model.pth` 是 ep-49 欠训练快照（§5.1），
   两者之间变化的**不是选择准则，而是训练量**——本实验无法把这两者分开。
2. **不能**声称"codebook capacity 是/不是瓶颈"（§6 两种情况都禁此项）。
   256 码下 prefix-1 碰撞率恒 >0.98，本实验未触及 capacity 维度。
3. **不能**把 `delta_gap` 的负号读成"量化反而提升结构保持"。
   Beauty 的负值来自基线 z 已崩（§5.1），不是量化的功劳。
4. **不能**把 §5.2 的同 z 结论外推为"用不同准则训练的模型也会如此"。
   它只说明：**在现存这两份 checkpoint 之间**，量化步的损失差是 10–130 倍。
5. `full_sid_collision_rate` 全 0（§3.2）说明 **checkpoint selection 的目标量已被
   codegen 后处理抹平**；凡以"碰撞率"为理由解释下游差异的表述，都应改口为
   "码本健康度 / 利用率"，或直接说明是 prefix-3 层级的差异。

---

## 8. 下一步应据此判断什么

1. **要回答设计 §1 的原问题，必须先拿到一个公平的对照检查点。** 现存的
   `best_loss_model.pth` 不具备对照资格。**但"另存一份总损失更低的检查点"这条路径
   已被证伪**（§5.1）：这条训练轨迹上总损失的最小值就在 ep42–79，之后再无改善，
   所以不存在"总损失确实下降的更晚评估点"可另存。
   可行的替代判据（本实验不执行，仅记录）：
   ① 按**固定 epoch 对齐**比较（如统一取 ep500/1000/2000），或
   ② 用**重建损失**而非总损失作选择准则（recon 单调改善，见 §4），或
   ③ 直接改用"下游行为"选择。
   其中 ②③ 都会改变实验定义，需先确认。在此之前，§6 的 A/B/C 三档无法被干净判定。
2. **`best_loss` 字段自 epoch 49 起从未刷新这一现象已被解释，且不是代码 bug**
   （2026-09-28 定位）：`train_loss = recon + quant`，其中 **recon 单调下降
   （11.9→1.73）而 quant 项单调上升（0.24→3.92，×16）**，故总损失在 ep42–79 触底后
   持续上升；`trainer.fit` 的比较与保存逻辑经逐行复刻验证**完全正确**，
   9/9 单元 ep49 的总损失确实是全体评估点中的最小值（§4）。详见
   [trainer_best_loss_root_cause.md](trainer_best_loss_root_cause.md)。
3. **Toys 主表已可放心使用**：§6 的 6 个 variant 全部 100% 可复现。
   需要处理的是 **`Toys_random_sid.npy` 孤儿**及其在
   [paper_audit_critique.md](paper_audit_critique.md) 中的 0.0308 引用。
4. TA-SID vs Raw-CoOcc 的提升按健康数字为 Toys **+10.9% / +13.6%**、
   Sport **+5.6% / +5.2%**，可直接替换文档中基于孤儿 codes 的 +7.9%。
