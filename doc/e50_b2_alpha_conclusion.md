# E50 b2 α 标定结论 + step 10 实施方案（待裁）

> 📌 **2026-10-07 状态更新**：**step 10 已跑完**（2026-09-30 16:31，9/9 路，码本门禁全 PASS）。
> 本文件 §4（实施方案）与 §5（请你裁定）**已过期，仅作历史记录**；
> **正式结果与裁决见 [`doc/e50_b2_full_conclusion.md`](e50_b2_full_conclusion.md)。**
> §3 的内部冲突**已按本文件建议裁定**（RQ-VAE 固定 seed=2024）。

> **本文件性质**：α 标定（预注册 `doc/exp/ab_decision_and_design.md` §3.5 第 8 步）的**结果与订正记录**，
> 以及 step 10 的实施方案供裁。**未启动 step 10**（撰写时）。
> 日期：2026-09-30 ｜ 分支 `experiments-2026-09-25` ｜ 代码已提交 `756656e`（tag `post-e50-b2-impl`）

---

## 1. α 标定：结果

**方法**（严格按 §3.5，一字未改）：Sport × T5-seed2025 × α ∈ {0.002, 0.02, 0.2}，
RQ-VAE 用 `main.py` 默认 seed=2024，输入 `data/Sport/item_emb_768d_contentonly.parquet`（768d），
对齐目标 `data/Sport/behavior_emb.parquet`（128d，**非** `_ppmi`）。
判据 = **下游 val NDCG@20**（禁用潜空间诊断）。驱动 `rqvae/run_b2_alpha_calib.sh`，
产物 `rqvae/logs/e50_b2_alpha_summary.tsv` + `e50_b2_alpha_driver.log`。

**参照**：b4（Content-only 768d，同协议同 seed）= val **0.032264**。

| α | RQ-VAE L0 unique | 门禁 | val NDCG@20 | test NDCG@20 | 裁决 |
|:--|--:|:--:|--:|--:|:--|
| 0.002 | 182/256 | ✅ | 0.033961 | 0.0294 | ≥ b4 |
| **0.02 ★** | 250/256 | ✅ | **0.036009** | **0.0311** | ≥ b4 |
| 0.2 | 254/256 | ✅ | 0.033319 | 0.0290 | ≥ b4 |
| b4（α=0） | 151/256 | — | 0.032264 | 0.0274 | — |

**⇒ α\* = 0.02**（预注册规则：val 最高者）。峰在区间**内部**（α=0.2 回落）⇒ 该 sweep 不是边界假象。
α\* 恰好等于 LETTER (CIKM'24) 的推荐值，属**独立复现而非调参所得**。

### 1.1 同协议全族对照（Sport × T5-seed2025）

全部行已逐行验配（训练日志 `save_path` ↔ eval log `Loading model from`）：

| arm | 注入位置 | L0 unique | val NDCG@20 | test NDCG@20 |
|:--|:--|--:|--:|--:|
| **b0** TA-SID（896d） | **输入级** | 249 | 0.037393 | **0.0327** |
| RawCoOcc | 输入级 | — | 0.035147 | 0.0311 |
| **b2 α=0.02 ★** | **损失级** | 250 | **0.036009** | **0.0311** |
| ItemCF | 输入级 | — | 0.034784 | 0.0298 |
| b2 α=0.002 | 损失级 | 182 | 0.033961 | 0.0294 |
| b2 α=0.2 | 损失级 | 254 | 0.033319 | 0.0290 |
| b6 ZeroPad | 安慰剂 | — | 0.032609 | 0.0286 |
| b4 Content-only | 无 | 151 | 0.032264 | 0.0274 |
| Shuffled | 输入级（错配） | — | 0.031695 | 0.0262 |

### 1.2 结论（对 §3.3 判据）

1. **三个 α 全部 > b4** ⇒ **损失级注入确实有效**（相对无行为信号的纯内容基线）。
2. **b2（0.0311）< b0（0.0327）** ⇒ 落在预注册的 **`b2 ≪ b0`** 分支：
   **输入级更优**，按 §3.3 应「同上报更简者更优，并把 b2 作为负面结果诚实报告」。
   ⚠️ 但注意阈值定义是 `≈` = `|Δ| < 3-seed std`（量级 ≤0.0011）。**单 seed** 下 Δ=0.0016，
   尚**不能**断定是 `≪` 还是 `≈` —— **这正是需要 step 10 的 3 seed 的理由**。
3. **Δ(b0−b4)=+0.0053，Δ(b2−b4)=+0.0037** ⇒ b2 追回 b0 增益的 **~70%**。
   这是本方案最有价值的量化表述：**损失级不是零，但系统性弱于输入级。**
4. val 与 test 在本族内**同序**，val 可作代理（比值 0.83–0.89；Toys TA-SID 0.79–0.83）。

---

## 2. 本次运行中订正的两个**自身错误**（留档，勿重演）

### 2.1 eval log 列数陷阱 → 真值 0.0294，不是 0.0327

`tools/eval_trained_model.py` 的 NDCG 行**列数 = `len(--topk_list)`**。旧 lane 用 `5,10,20`（3 列），
2026-09 后统一用 `5,10,20,50,100`（**5 列**）。驱动脚本用 `awk '{print $NF}'` 取 @20，
在 5 列下取到的是 **@100**。⇒ α=0.002 的 test 一度被报成 `0.0294/0.0327/0.0327`，
真值 `0.0172/0.0231/**0.0294**`。**已订正 TSV；α=0.02/0.2 的行改由 eval log 表头重算。**

> 影响面：**val（决策量）不受影响**——它来自 T5 训练日志的 `'NDCG@20': x` 正则，故 α\* 未被污染。
> 教训已入 memory `eval-log-ndcg-column-trap`。

### 2.2 val/test 配错日志 → 一个被撤回的假发现

初版用 `tiger_ta_sid_sport.log`（`save_path=tiger_ta_sid_sport.pth`）取 b0 的 val=0.032240，
但 b0 的 test 来自 `Sport_ta_sid_768d_seed2025_eval.log`，它加载的是
**`tiger_ta_sid_sport_768d.pth`** —— **另一次训练**。
据此曾得出「b0 的 val/test 比值 1.01 离群 ⇒ val 与 test 跨变体不同序」的结论，**该结论已作废**。

正确配对后 b0 val = **0.037393**，比值 0.874，落回全族 0.83–0.89 内。
旁证：Sport TA-SID seed2026/2027 的 val 为 0.036948 / 0.036949，三 seed 量级一致。

> **纪律**：任何 val/test 对照必须先用 `save_path` ↔ `Loading model from` 逐行验配。
> 这是 `ablation_main_table_3datasets.md` §5.1「命名陷阱」的**同型**错误。
> Legacy 命名极乱：Sport TA-SID seed2025 用 `tiger_ta_sid_sport_768d`，seed2026/2027 用
> `tiger_ta_sid_768d_sport_seed*`；且 2025 的 eval log 是 **3 列**旧格式。

---

## 3. ⚠️ 预注册文件的一处内部冲突（**仲裁前不得开跑**）

`ab_decision_and_design.md` §3.5「其余纪律」写了：

> - **三 seed 同 seed 同步**（RQ-VAE 与 T5 用同一 seed）。

**这与三处矛盾**：

| 出处 | 说的是什么 |
|:--|:--|
| §3.1 固定项 | `seed 2025/2026/2027`（**只约束 T5**） |
| 主表实际协议（`run_behavior_ablation.sh` 头部） | 「一个变体 = **一次 RQ-VAE（默认 seed=2024）** → 一个 codes.npy → T5 用 2025/2026/2027 各训一次」 |
| 本次 α 标定（已执行） | RQ-VAE seed **2024** + T5 seed 2025 |

**后果**：若照 §3.5 字面执行（RQ-VAE 也随 T5 seed 变），b2 会有 3 份 codes/数据集，
而 **b0/b4/b6 每个数据集只有 1 份 codes** ⇒ **与基线不可比**，整张对照表作废。
§4.2-E「3 seed × 3 数据集 = 9 次 RQ-VAE」沿用了同一错误假设。

**建议裁定**：以**主表协议**为准 —— **RQ-VAE 固定默认 seed=2024（每数据集一份 codes），T5 三个 seed 2025/2026/2027**。
据此 §3.5 该行应改为「**T5 三 seed；RQ-VAE 固定 seed=2024 以与 b0/b4/b6 可比**」，
§4.2-E 的成本行同步改为 3 次 RQ-VAE（不是 9 次）。

> 本项**不是判据变更**，是**修正一处自相矛盾的表述**，且方向是**让实验与既有基线可比**。
> 但按预注册纪律，改预注册文件需你确认。

---

## 4. step 10 实施方案（待裁后启动）

### 4.1 配置（钉死）

| 项 | 值 |
|:--|:--|
| α\* | **0.02**（冻结，不再调） |
| 输入 | `data/{D}/item_emb_768d_contentonly.parquet`（768d） |
| 对齐目标 | `data/{D}/behavior_emb.parquet`（128d） |
| RQ-VAE | `--num_emb_list 256 256 256`，其余全默认，**seed=2024** |
| 新参数 | `--align_weight 0.02 --align_data_path … --align_tau 0.1`（τ 固定不扫） |
| T5 | `model/main.py`，`--seed 2025/2026/2027`，rest 与主表逐字一致（200 ep / early_stop 10 / beam 30） |
| 评测 | `tools/eval_trained_model.py --beam_size 30 --topk_list 5,10,20,50,100` |

### 4.2 产物（幂等，绝不覆盖既有文件）

- codes：`data/{D}/{D}_b2align0.02_sid.npy`
- RQ-VAE ckpt：`rqvae/ckpt/{D}_B2_align0.02/`
- T5 ckpt / log：`model/ckpt/tiger_b2align0.02_{d}_seed{S}.pth`、`model/logs/tiger_b2align0.02_{d}_seed{S}.log`
- eval：`model/logs/{D}_b2align0.02_seed{S}_eval.log`
- 汇总：`rqvae/logs/e50_b2_summary.tsv`（**test 列按 eval log 表头取，不用 `$NF`**）

**Sport × seed2025 已存在**（本次标定产物），驱动直接跳过 ⇒ 实跑 **8 路 + 2 次 RQ-VAE**。

### 4.3 门禁与判据（沿用 §4.2-E / §3.3）

1. 每个数据集 RQ-VAE 训完先查 **L0 unique ≥ b4 的 90%**，不达标记塌缩、跳过该数据集的 T5。
   （Sport α=0.02 实测 250 > 151，预期无风险。Toys 需注意——Toys 是历史塌缩数据集。）
2. 出结论时按 **§4.2 问题 D 的两个增量**写法：`b0−b4`（输入级，配 b6 安慰剂）与 `b2−b4`（损失级）。
   **不得**直接用 `b0 vs b2` 作单变量结论（二者同时差输入与损失两个因素）。
3. 配对显著性用 `tools/paired_significance.py`，std 用 ddof=1。

### 4.4 成本（实测口径，见 §3.6 订正块）

| 数据集 | RQ-VAE | T5 × 3 seed | eval × 3 | 小计 |
|:--|--:|--:|--:|--:|
| Sport | 9 min | 241 min | 12 min | ~262 min（**已跑 1 lane，实需 ~182 min**） |
| Toys | 9 min | 146 min | 12 min | ~167 min |
| Beauty | 9 min | 197 min | 12 min | ~218 min |
| **合计** | | | | **~11.7 h**（Sport 已扣 1 lane ⇒ 实需 ~10.7 h） |

---

## 5. 请你裁定

1. **§3 的内部冲突**：是否按建议改 §3.5 那一行为「RQ-VAE 固定 seed=2024」？（不改则 b2 与基线不可比）
2. **是否启动 step 10**（~10.7–11.7 h）。
3. **范围**：3 数据集全域（预注册），还是先做 Sport+Toys？

**若你倾向不跑**：现有单 seed 结果已足以支撑「损失级有效但弱于输入级（~70%）」的**方向性**结论，
但**无法**走 §3.3 的阈值判据（`≈` 需 3-seed std），论文里只能写成"单 seed 初步观察"。

---

## 附：本次新增/改动文件

| 文件 | 性质 |
|:--|:--|
| `rqvae/main.py`、`rqvae/trainer.py`、`rqvae/models/rqvae.py` | b2 实现（已提交 756656e，含回归门禁） |
| `rqvae/run_b2_alpha_calib.sh` | α 标定驱动（幂等 + 码本门禁前置） |
| `rqvae/logs/e50_b2_alpha_summary.tsv`、`e50_b2_alpha_driver.log` | α 标定产物 |
| `doc/exp/ab_decision_and_design.md` §3.5/§3.6 | 追加实测成本订正块（判据未动）；**§3 冲突待裁** |
| `data/Sport/Sport_b2align0.{002,02,2}_sid.npy` | α 三个 codes |
