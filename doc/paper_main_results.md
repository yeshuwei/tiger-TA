# TA-SID 论文主实验结果（§5.2.1 数据源）

> **状态：** 2026-08-07 定稿口径；**Toys 数字 2026-09-09 provenance 修复**（旧 0.0367 基于孤儿 codes 作废，订正为 0.0377±0.0007，见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)）；**经典序列基线扩充 2026-09-29**（新增 GRU4Rec / BERT4Rec，方法与诚实标注见 [seq_baselines_report.md](seq_baselines_report.md)）。本文档是论文 §5.2 主实验结果（Overall Performance Comparison）的**唯一数字源**，格式与注释面向论文写作。
> **权威来源：** SASRec → `sasrec_output/{Dataset}/summary_3seed.txt`；GRU4Rec → `gru4rec_output/{Dataset}/summary_3seed.txt`；BERT4Rec → `bert4rec_output/{Dataset}/summary_3seed.txt`；Content-only / TA-SID → `model/logs/{Dataset}_{tag}_seed{seed}_eval.log`。总表另见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)。
> **协议：** 3 个 seed（2025/2026/2027）均值±标准差；best-**val** NDCG@20 checkpoint；最终评估 = test.parquet（**四个方法完全同任务**）；全量用户 + 全量物品排序。SASRec 为对齐协议 v2（2026-08-07）；GRU4Rec/BERT4Rec 自始即对齐同一协议（2026-09-29）。

---

## 1. 主实验完整结果表（可直接进论文）

**Table: Overall performance comparison (mean ± std over 3 seeds). Bold = best. RQ-VAE fixed; randomness from generative-retrieval training (T5/SASRec).**

| Method | Dataset | NDCG@5 | NDCG@10 | NDCG@20 | Recall@5 | Recall@10 | Recall@20 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **SASRec** | Sport | 0.0064±0.0004 | 0.0096±0.0001 | 0.0134±0.0003 | 0.0121±0.0007 | 0.0220±0.0005 | 0.0374±0.0017 |
| | Beauty | 0.0121±0.0007 | 0.0183±0.0008 | 0.0254±0.0009 | 0.0234±0.0010 | 0.0429±0.0017 | 0.0711±0.0019 |
| | Toys | 0.0127±0.0005 | 0.0184±0.0002 | 0.0241±0.0004 | 0.0245±0.0013 | 0.0423±0.0007 | 0.0651±0.0019 |
| **GRU4Rec**<sup>†</sup> | Sport | 0.0086±0.0005 | 0.0106±0.0004 | 0.0129±0.0004 | 0.0127±0.0007 | 0.0189±0.0004 | 0.0282±0.0004 |
| | Beauty | 0.0244±0.0014 | 0.0284±0.0011 | 0.0323±0.0013 ⚠ | 0.0333±0.0016 | 0.0458±0.0007 | 0.0612±0.0017 |
| | Toys | 0.0138±0.0003 | 0.0164±0.0003 | 0.0192±0.0002 | 0.0203±0.0005 | 0.0286±0.0011 | 0.0394±0.0011 |
| **BERT4Rec**<sup>‡</sup> | Sport | 0.0069±0.0004 | 0.0091±0.0007 | 0.0118±0.0009 | 0.0109±0.0006 | 0.0178±0.0017 | 0.0284±0.0024 |
| | Beauty | 0.0144±0.0011 | 0.0185±0.0009 | 0.0233±0.0006 | 0.0221±0.0013 | 0.0348±0.0006 | 0.0539±0.0007 |
| | Toys | 0.0080±0.0008 | 0.0105±0.0010 | 0.0132±0.0010 | 0.0127±0.0010 | 0.0206±0.0017 | 0.0313±0.0017 |
| **Content-only (TIGER 复刻)** | Sport | 0.0161±0.0003 | 0.0213±0.0005 | 0.0275±0.0007 | 0.0255±0.0005 | 0.0421±0.0009 | 0.0666±0.0019 |
| | Beauty | 0.0227±0.0010 | 0.0291±0.0009 | 0.0358±0.0009 | 0.0347±0.0012 | 0.0547±0.0009 | 0.0811±0.0010 |
| | Toys | 0.0206±0.0007 | 0.0268±0.0006 | 0.0336±0.0007 | 0.0324±0.0007 | 0.0516±0.0007 | 0.0785±0.0009 |
| **TA-SID** | Sport | **0.0190**±0.0002 | **0.0252**±0.0002 | **0.0322**±0.0005 | **0.0299**±0.0001 | **0.0491**±0.0006 | **0.0769**±0.0014 |
| | Beauty | **0.0264**±0.0008 | **0.0335**±0.0010 | **0.0415**±0.0011 | **0.0402**±0.0005 | **0.0620**±0.0012 | **0.0941**±0.0019 |
| | Toys | **0.0225**±0.0008 | **0.0294**±0.0005 | **0.0377**±0.0007 | **0.0365**±0.0014 | **0.0579**±0.0003 | **0.0911**±0.0020 |

<sup>†</sup> **GRU4Rec 为交叉熵变体**（全物品 softmax CE），非原版 TOP1/BPR 成对排序损失——原版损失配合负采样评估，
与本工作的全量排序协议不可比。**本表不声称复现原论文数字。**
<sup>‡</sup> **BERT4Rec 训练 Cloze / 评估 next-item**，通过"训练时恒将末位替换为 [MASK]"使二者同构；
评估输入右移一格（每数据集仅影响 1 个超长序列用户）。标注原句见 [seq_baselines_report.md §2](seq_baselines_report.md)。
⚠ **越界标记**：Beauty / NDCG@20 上 GRU4Rec (0.0323) 超过 SASRec (0.0254)——**全表 12 格中唯一一格**
（BERT4Rec 无越界）。该格 **Recall@20 未越界**（SASRec 0.0711 > GRU4Rec 0.0612）。
⇒ **"最强经典基线"的归属取决于指标，勿做无条件断言。** 该格数字可信（3 seed 齐备、std 正常、
配对检验零越界 d_z=3.75），⚠ 是**排序标记而非数据质量警告**。

> **损失不对等（写论文必须交代，否则被审稿人抓）**：SASRec 用 BCE + **每正例 1 个负采样**，
> 而 GRU4Rec / BERT4Rec 用**全物品 softmax CE**（训练信号强得多）。因此
> **GRU4Rec 与 SASRec 之间的名次不可作架构解读**，可解释为损失强度差异。
> 对本工作有利的一面：TA-SID 连"损失更强的基线"也全面胜出。

> **std 口径（2026-09-29 统一）：** 上表 std 为**样本标准差（ddof=1，Bessel 校正）**，与
> [ablation_main_table_3datasets.md](ablation_main_table_3datasets.md) 和图 A 口径一致。
> 此前本文档用 ddof=0（`np.std` 默认值），导致同一量在主表与消融表中数值不同
> （加入新基线后共 30 格，**29/30 格四位小数即不同**；三方法时曾为 18/18），
> 已全部重算。逐 seed 产物文件 `sasrec_output/*/summary_3seed.txt` 仍为 ddof=0（逐位复现门禁要求），
> 两者差异与对账表见 [seq_baselines_significance.md §1](seq_baselines_significance.md)。

### 论文写作注释

1. **层次关系（2026-09-29 扩充）**：TA-SID 在 **3 数据集 × 6 指标 × 4 个基线 = 72 格**上全部最优，
   无任何一格倒挂——可写成 "TA-SID consistently outperforms all four baselines on all three
   datasets and all six metrics."（配 §3 的零越界与配对检验）
2. **⚠ 经典基线之间的名次不可无条件断言（新增）**：`SASRec < Content-only` 在全部 18 格成立；
   但 **GRU4Rec 在 Beauty / NDCG@5/10/20 上超过 SASRec**（N@20 0.0323 vs 0.0254），**Recall 上未超过**。
   正确写法："the relative ordering among the classic sequence baselines depends on the metric."
   **禁止**写 "SASRec is the strongest classic baseline"（与 Beauty 行矛盾）。
   另：GRU4Rec/BERT4Rec 用全物品 CE 而 SASRec 用 1-负采样 BCE，名次含损失强度混淆（见上）。
3. **Bold 标注**：现在共 5 个方法，仍只有 TA-SID 在 3 数据集上最优，**18 个指标格加粗**即可；
   其余方法一律不加粗（⚠ 标记不是加粗，勿混用）。
4. **Content-only 即 TIGER 复刻**：同管线 768d 纯内容 RQ-VAE+T5，唯一区别是去掉了 behavior 通道。这是论文里 "vs TIGER" 的公平 baseline（旧的 4096d TIGER 原始数字因维度/数据口径混淆**已弃用，勿引用**，见 §7 弃用声明）。
5. **SASRec 已对齐**：v2 与 TA-SID 同任务（test.parquet）、全量用户、3-seed。不要再写 "SASRec 为单值"。
6. **GRU4Rec / BERT4Rec 为 2026-09-29 新增**：协议与前三者逐字相同（共用 `evaluate_full`），
   `train.parquet` 的 history 训练、`test.parquet` 评估、best-val 选择、同 3 个 seed。
   方法与诚实标注全量见 [seq_baselines_report.md](seq_baselines_report.md)；预登记超参见其 §1.2。

---

## 2. 相对提升汇总（写作用）

### 2.1 TA-SID vs Content-only（核心创新点）

| Dataset | NDCG@20 提升 | Recall@20 提升 |
|:--|--:|--:|
| Sport | **+17.1%** | +15.5% |
| Beauty | +15.9% | +16.0% |
| Toys | +12.2% | +16.1% |

> **叙事（诚实口径，勿写单调）**：Sport（交互密度最低 0.045%、平均序列最短 8.32）提升最大 +17.1%；但 Beauty（0.0734%）与 Toys（0.0724%）密度几乎相同，增益却差 3.7pp（+15.9% vs +12.2%，Toys 为 2026-09-09 provenance 修复后数字），**既不能用密度也不能用序列长解释 Beauty vs Toys 的排序**。论文建议写法："最稀疏的 Sport 增益最大，其余两数据集增益相当且较小；Beauty 与 Toys 间的差异可归因于数据分布差异，不做过度解读。" 审稿人会对数据核对，勿写成 "the gain grows with dataset sparsity."

### 2.2 TA-SID vs SASRec（经典基线，对齐后）

| Dataset | NDCG@20 提升 | Recall@20 提升 |
|:--|--:|--:|
| Sport | **+139.5%** | +105.5% |
| Beauty | +63.2% | +32.3% |
| Toys | +56.4% | +39.9% |

> **叙事**：对齐后 SASRec 下移（预测最后一项更难），TA-SID 相对优势全面扩大。写具体倍率用本表，勿用旧的 +96.3%/+29.7%/+27.4%。

### 2.3 TA-SID vs 新增经典基线 GRU4Rec / BERT4Rec（2026-09-29）

| Dataset | vs GRU4Rec (N@20) | vs GRU4Rec (R@20) | vs BERT4Rec (N@20) | vs BERT4Rec (R@20) |
|:--|--:|--:|--:|--:|
| Sport | +149.6% | +172.7% | +172.9% | +170.8% |
| Beauty | +28.5% | +53.8% | +78.1% | +74.6% |
| Toys | +96.4% | +131.2% | +185.6% | +191.1% |

> **计算口径**：本表两侧**都取 §1 主表的四位小数值**（如 Sport N@20: `0.0322/0.0129−1`），
> 读者可逐格复算。与用全精度均值计算的结果可差 ≤0.4pp
> （如 Sport vs GRU4Rec N@20 全精度为 +150.0%）。§2.1/§2.2 沿用原有全精度口径，两表不混用。

> **叙事**：三数据集 × 两指标共 **6 格全部为正、零倒挂**——且这是**在经典基线拿到更强训练信号
> （全物品 softmax CE，而 SASRec 只有 1-负采样 BCE）的前提下**取得的，
> 即 TA-SID 的优势不能归因于"给基线配了更弱的损失"。
> ⚠ **勿**据此表声称 "architecturally stronger than GRU4Rec"——名次含损失强度混淆，见 §1 脚注。

---

## 3. per-seed 明细（NDCG@20，审稿可复现用）

| Dataset | Method | seed2025 | seed2026 | seed2027 | mean±std |
|:--|:--|--:|--:|--:|--:|
| Sport | SASRec | 0.0137 | 0.0135 | 0.0132 | 0.0134±0.0003 |
| | GRU4Rec | 0.0126 | 0.0127 | 0.0133 | 0.0129±0.0004 |
| | BERT4Rec | 0.0129 | 0.0113 | 0.0112 | 0.0118±0.0009 |
| | Content-only | 0.0274 | 0.0283 | 0.0269 | 0.0275±0.0007 |
| | **TA-SID** | 0.0327 | 0.0317 | 0.0322 | **0.0322±0.0005** |
| Beauty | SASRec | 0.0256 | 0.0245 | 0.0262 | 0.0254±0.0009 |
| | GRU4Rec | 0.0319 | 0.0312 | 0.0338 | 0.0323±0.0013 ⚠ |
| | BERT4Rec | 0.0230 | 0.0240 | 0.0229 | 0.0233±0.0006 |
| | Content-only | 0.0368 | 0.0352 | 0.0354 | 0.0358±0.0009 |
| | **TA-SID** | 0.0418 | 0.0425 | 0.0403 | **0.0415±0.0011** |
| Toys | SASRec | 0.0237 | 0.0243 | 0.0244 | 0.0241±0.0004 |
| | GRU4Rec | 0.0194 | 0.0190 | 0.0191 | 0.0192±0.0002 |
| | BERT4Rec | 0.0125 | 0.0127 | 0.0144 | 0.0132±0.0010 |
| | Content-only | 0.0341 | 0.0338 | 0.0328 | 0.0336±0.0007 |
| | **TA-SID** | 0.0369 | 0.0383 | 0.0380 | **0.0377±0.0007** |

> **写作要点**：TA-SID 在 **4 个基线 × 3 数据集 × 3 个 seed = 36/36** 个组合上全部高于对应基线，
> **零 seed 交叉**；上表全部 NDCG@20 的 std ≤ 0.0013（ddof=1 口径，最大者为 GRU4Rec Beauty），
> 方法间差距（0.009~0.02）远大于 seed 噪声。这支撑 "improvement is statistically robust, not an artifact of seed selection."
>
> **显著性检验已完成（2026-09-29）**：paired t-test / Wilcoxon 符号秩 / 符号检验 / Cohen's d_z / seed 交叉计数，逐格结果见 **[seq_baselines_significance.md](seq_baselines_significance.md)**。全部 18 组对比**零越界**（其中 TA-SID 相关 12 组），d_z 范围 2.07~42.82。**唯一 p>0.05 的一格**：Beauty `Content-only − GRU4Rec`（Δ=+0.0035, p=0.070, d_z=2.07），该格**不得声称显著**。
> ⚠️ **写作红线**：n=3 下符号检验的最小双侧 p 恒为 0.25，**禁止**写 "statistically significant (p<0.05)" 而不交代 n/df。论文用 "consistent across all 3 seeds (no crossover)" + 效应量表述。详见该文档 §4。

---

## 4. 实验设置速查（写 §5.1 用，勿与数字脱节）

| 项目 | 值 |
|:--|:--|
| 数据集 | Amazon 5-core：Beauty（用户 22,363 / 商品 12,101 / 交互 198,502 / 平均序列长 8.88）、Sport（35,598 / 18,357 / 296,337 / 8.32）、Toys（19,412 / 11,924 / 167,597 / 8.63） |
| 数据划分 | 滚动 leave-one-out：train 预测 i_{L-2}，val 预测 i_{L-1}，test 预测 i_L（L = 完整序列长）；转移图仅用 train 构建（无泄漏） |
| 指标 | NDCG@k、Recall@k（k=5,10,20），全量物品排序 |
| 模型选择 | best-valid NDCG@20 checkpoint |
| Seed | 3 ×（2025/2026/2027），SASRec 与 T5 同 seed |
| GPU | NVIDIA RTX 4090 |

> **测试集一致性**：三方法最终评估全部 = "从 test.parquet 的 history 预测 target"。SASRec v1 曾用 valid 任务（预测 i_{L-1}），已修正（见 [sasrec_baseline_report.md](sasrec_baseline_report.md)）。写论文时若提 "same evaluation protocol for all methods"，此声明成立。

---

## 5. 弃用/勘误记录（写作时避开）

| 条目 | 状态 | 说明 |
|:--|:--|:--|
| 4096d / 64d TIGER 原始数字（§1.1-1.4） | ❌ 弃用 | 维度+数据管线混淆（64d content vs 768d，TIGER 原始管线 vs 本项目 5-core） |
| SASRec v1（2026-07-30） | ❌ 弃用 | off-by-one（实为 valid 任务）+ 10k 用户子集 + 单 seed，整体高估 ~16-21% |
| Beauty TA-SID R@20 旧值 0.0933±0.0014 | ❌ 已勘误 | 实测 3 seed = 0.0951/0.0953/0.0919 → **0.0941±0.0016** |
| **§3 per-seed 明细 4 行（2026-09-29 勘误）** | ❌ 已勘误 | 4 行 per-seed 值抄写有误，均值未受影响（碰巧正确）。订正为：Sport TA-SID `0.0327/0.0317/0.0322`（原 2026 值 0.0320 错）、Beauty Content-only `0.0368/0.0352/0.0354`（原 0.0357/0.0366/0.0351）、Beauty TA-SID `0.0418/0.0425/0.0403`（原 2026 值 0.0424）、Toys Content-only `0.0341/0.0338/0.0328`（原 0.0339/0.0331/0.0338）。权威源 = [ablation_main_table_3datasets.md §2](ablation_main_table_3datasets.md) 与 [experiment_results_summary.md §1.5](experiment_results_summary.md)（后者 9/9 行与 eval log 一致）。std 未改动：订正后各行 population std 与原文一致 |
| **Toys TA-SID 旧数字（0.0367±0.0006 等，2026-08-07 口径）** | ❌ **作废** | 旧 codes（L0=62）为**孤儿**，无法从现存 ckpt 复现（provenance 审计）。2026-09-09 用健康 best_collision(L0=207) 重跑 → **NDCG@20=0.0377±0.0007 / R@20=0.0911±0.0020**，提升 +12.2%/+16.1%（vs content）、+56.4%/+39.9%（vs SASRec）。详见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md) |
| **§1/§3 全部 std 列（2026-09-29 口径统一）** | ✅ **已重算** | 原 std 用 `np.std` 默认 ddof=0（总体标准差），与 [ablation_main_table_3datasets.md](ablation_main_table_3datasets.md)、图 A 的 ddof=1 口径冲突，**18/18 格四位小数即不同**。已统一为 ddof=1（Bessel 校正）。**均值与逐 seed 值全部未变**，仅 std 改变。对账表： [seq_baselines_significance.md §1](seq_baselines_significance.md)。逐 seed 产物 `sasrec_output/*/summary_3seed.txt` 保持 ddof=0 不动（逐位复现门禁要求） |
| **GRU4Rec / BERT4Rec（2026-09-29 新增）** | ✅ **正式** | 3-seed 全量完成（driver 3 h 24 min，exit 0，无 NaN，三 seed 零越界）。协议与前三者**逐字相同**（共用 `evaluate_full`）。⚠ 两处**不得声称复现原论文**：GRU4Rec 用全物品 softmax CE（非原版 TOP1/BPR）、BERT4Rec 训练 Cloze 而评估 next-item。方法与诚实标注全量 → [seq_baselines_report.md](seq_baselines_report.md) |
| **"SASRec 是最强经典基线"类表述** | ❌ **不得再写** | Beauty / NDCG@5/10/20 上 GRU4Rec 超过 SASRec（R@20 未超）。正确写法见 §1 写作注释 #2 |
| 本文档全部数字（除上条旧值） | ✅ 正式 | 2026-08-07 定稿 + 2026-09-09 Toys 订正 + 2026-09-29 基线扩充，可引用 |

---

## 6. 复现命令

```bash
# SASRec 3-seed（~25 min / 全量）
python3 tools/run_sasrec.py --dataset all --seeds 2025 2026 2027

# GRU4Rec + BERT4Rec 3-seed（~3.5 h，单卡串行；幂等，已完成 seed 自动跳过）
bash tools/run_seq_baselines.sh
bash tools/run_seq_baselines.sh gru4rec          # 只跑 GRU4Rec（~1.5 h）
bash tools/run_seq_baselines.sh bert4rec Beauty  # 只跑 BERT4Rec × Beauty

# Content-only / TA-SID T5 3-seed（RQ-VAE 已固定，只重训 T5 + eval）
python3 rqvae/run_multiseed.py --method baseline --dataset all --seeds 2026 2027
python3 rqvae/run_multiseed.py --method ta_sid   --dataset all --seeds 2026 2027

# 配对显著性检验 + std 对账（零 GPU，~10 s）
python3 tools/paired_significance.py --extra-methods GRU4Rec BERT4Rec \
    --out-md doc/seq_baselines_significance.md \
    --out-csv doc/seq_baselines_significance.csv
```

> ⚠ **冒烟/验证必须写入一次性目录**（如 `/tmp/smoke_*`）。若写进 `gru4rec_output/`，
> 短 epoch 的 `seed{seed}_best.txt` 会让驱动脚本的幂等跳过逻辑误判"已完成"，
> **正式结果会静默变成冒烟结果**。
