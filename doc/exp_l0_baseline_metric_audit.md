# EXP-L0 × Clean TIGER 指标口径审计（只读）

审计日期：2026-10-19

审计对象：`integration/artifacts/exp_l0/runs/20261018T202823Z`（5 臂）与 Toys Clean TIGER 基线

约束：只读；未修改任何代码/配置/产物，未重跑训练；仅新增一次只读 GPU 复算（不落盘）与本文档。

---

## 结论摘要

1. **基线 `NDCG@20 = 0.0341` / `Recall@20 = 0.0795` 是 test 集指标**，来自
   [tiger_content768d_baseline_toys.log](model/logs/tiger_content768d_baseline_toys.log) 第 76 epoch 的
   `Test Dataset:` 行（单 seed 2025），并由
   [Toys_content768d_baseline_seed2025_eval.log](model/logs/Toys_content768d_baseline_seed2025_eval.log) 独立复算。
2. **EXP-L0 `run.json` 的 `result.best_ndcg@20` 是 valid 集指标**（模型选择指标），不是 test。
3. 因此**「EXP-L0 vs Clean TIGER」此前采用的比较（arm 的 valid 对 baseline 的 test）口径错配**。
   据此得到的 `lc1e-4 +18.6%` **必须撤回**。
4. 同口径复核（本次只读复算，见 §4）：

   | 臂 | valid N@20 | vs baseline valid | test N@20 | vs baseline test |
   |---|---|---|---|---|
   | baseline | 0.039808 | — | 0.034084 | — |
   | lc1e-4 | 0.040444 | **+0.000636 (+1.60%)** | 0.033515 | **−0.000569 (−1.67%)** |
   | lc1e-3 | 0.039014 | −0.000794 | 0.032033 | −0.002051 |
   | lc1e-2 | 0.037674 | −0.002134 | 0.031311 | −0.002773 |
   | le0.1 | 0.039269 | −0.000539 | 0.031531 | −0.002553 |
   | le1.0 | 0.033796 | −0.006012 | 0.028671 | −0.005413 |

5. **没有任何一臂在 test 上超过基线**；`lc1e-4` 的 test 差（−0.00057）小于基线自身 3-seed 的极差
   （0.0328–0.0341，跨度 0.0013），只能判为「无差异」。
6. 训练协议、数据划分、评价函数、候选生成、选择规则**同构**（§3）；差异集中在
   **T5 初始化种子、DataLoader worker 数、CSG 臂重训 tokenizer/SID**，见 §3.2。

---

## §1 基线指标来源（provenance）

### 1.1 一级证据：逐 epoch 训练日志（本次新发现）

[tiger_content768d_baseline_toys.log](model/logs/tiger_content768d_baseline_toys.log) 是 `model/main.py`
的 `--log_path` 日志，完整记录了该次训练每个 epoch 的 valid/test 指标。关键块（epoch 76，1-based）：

```
2026-07-31 22:26:13 - Validation Dataset: {'NDCG@20': 0.03980784884796207, ...}
2026-07-31 22:26:26 - Best NDCG@20: 0.03980784884796207
2026-07-31 22:26:26 - Test Dataset: {'Recall@20': 0.07945402289552642, ...}
2026-07-31 22:26:26 - Test Dataset: {'NDCG@20': 0.03408397088203524, ...}
2026-07-31 22:26:26 - Best model saved to model/ckpt/tiger_content768d_baseline_toys.pth
...
2026-07-31 22:30:35 - Early stopping triggered.
```

- **valid**（模型选择用）：N@20 = `0.03980784884796207`，R@20 = `0.09360632170083488`
- **test**（论文报告用）：N@20 = `0.03408397088203524`，R@20 = `0.07945402289552642`
- 最佳 epoch = **76**（1-based），总共训练 86 epoch，`early_stop=10` 触发。
- **单 seed 2025，单次运行，不是 3-seed 均值。**

**这直接回答了「0.0341/0.0795 是 valid 还是 test」：是 test。** 此前该 valid 数字从未被记录进任何汇总表。

### 1.2 二级证据：独立复算

[Toys_content768d_baseline_seed2025_eval.log](model/logs/Toys_content768d_baseline_seed2025_eval.log)
（mtime 2026-07-31 22:30:51）由 [tools/eval_trained_model.py](tools/eval_trained_model.py) 生成：

- `Test samples: 19412`
- `NDCG@5/10/20 = 0.0210 / 0.0272 / 0.0341`
- `Recall@5/10/20 = 0.0330 / 0.0524 / 0.0795`

与 §1.1 的 test 行四位小数完全一致，且只评估 **test.parquet**。

### 1.3 配置

日志首行记录的配置（`model/main.py` argparse 全量）：

| 项 | 值 |
|---|---|
| dataset_path | `data/Toys` |
| code_path | `data/Toys/Toys_content768d_baseline_sid.npy` |
| save_path | `model/ckpt/tiger_content768d_baseline_toys.pth` |
| seed | 2025 |
| batch_size / infer_size | 256 / 96 |
| num_epochs / early_stop | 200 / 10 |
| lr | 1e-4 |
| max_len / beam_size / topk_list | 20 / 30 / [5,10,20] |
| 模型结构 | 4+4 层, d_model 128, d_ff 1024, 6 heads, d_kv 64, dropout 0.1, vocab 1025 |

tokenizer 配置（`rqvae/ckpt/Toys_Content768d_Baseline/Jul-31-2026_21-37-33/best_collision_model.pth` 的 args）：
`epochs 3000, batch 1024, lr 1e-3, num_emb [256,256,256], e_dim 32, kmeans_init True, lambda_trans 0, lambda_balance 0, seed None`。
**注意 `seed: None`** —— 基线 RQ-VAE 的 KMeans 初始化未固定随机种子。

### 1.4 checkpoint 与 SID

- 日志写的 `model/ckpt/tiger_content768d_baseline_toys.pth` 现已不存在；同 mtime（2026-07-31 22:26）
  的文件是 [tiger_content768d_baseline_toys_seed2025.pth](model/ckpt/tiger_content768d_baseline_toys_seed2025.pth)
  （18,408,707 B），符合 [doc/naming_mapping.md](doc/naming_mapping.md) 的重命名规则。
- **重命名这一步没有哈希级证据**，只有 mtime + 命名约定。但 §1.2 的复算以该 ckpt 精确复现了 4 位小数，
  等价于反证「这个 ckpt 就是当时评测的那个」。
- SID 文件 `data/Toys/Toys_content768d_baseline_sid.npy`：
  sha256 `bb2693a6…`，manifest 记录 `ckpt_sha256 8069f998…`，`.npy` mtime 2026-07-31 21:44（早于 T5 训练）。
  **评测日志本身不记录 SID 路径**；SID 归属由「训练日志 config + 训练早于 .npy mtime + 复算复现」三方交叉确认。

### 1.5 3-seed 上下文（用于判断噪声尺度）

| seed | test N@20 | test R@20 |
|---|---|---|
| 2025 | 0.0341 | 0.0795 |
| 2026 | 0.0338 | 0.0784 |
| 2027 | 0.0328 | 0.0777 |

单 seed 极差 0.0013（N@20）。**任何小于该量级的 arm−baseline 差都不能当作效应。**

---

## §2 EXP-L0 的评价流程（代码明确证实）

### 2.1 字段映射

`run.json` → `result`：

| 字段 | 实际含义 | 产生位置 |
|---|---|---|
| `best_ndcg@20` | **valid** NDCG@20 的最大值 | [tiger_ext/fixed_train.py:710](tiger_ext/fixed_train.py#L710)（`evaluate(self.tiger, valid_loader, ...)`）→ 720 |
| `best_epoch` / `best_ckpt` | 取得上述 valid 最大值的 epoch / `best.pth` | 717-726 |
| `epochs[i].valid_ndcgs/valid_recalls` | 该 epoch 的 **valid** 指标 | 705-707 |
| `epochs[i].test_ndcgs/test_recalls` | **仅在该 epoch 改善时**才评估并记录 | 722-733 |
| `n_test_evals` | test 实际评估次数（= `n_best_saves`，本次 5 臂均为 46/47/41/46/50，两两相等） | 725 |
| `num_epochs_executed` / `early_stopped` | 实跑轮数 / 是否早停 | 745-747, 811 |

`run.json` **不存储**「最佳 epoch 的 test 指标」为顶层标量；需要从
`result.epochs[best_epoch].test_ndcgs` 取（本次审计即如此取数）。`run_summary.json` 若只抄
`best_ndcg@20`，拿到的必然是 valid。

### 2.2 选择规则

[tiger_ext/fixed_train.py:705-741](tiger_ext/fixed_train.py#L705-L741)：

```python
improved = ndcg20 > best_ndcg          # ndcg20 来自 valid
if improved:
    ... evaluate(test_loader) ...      # 同时评估 test
    torch.save(self.tiger.state_dict(), best_ckpt)
```

- 严格大于、按 epoch 顺序、首次达到即更新；`best_ndcg` 初始 0.0。
- 早停 patience 10，非改善则 `counter += 1`，改善则清零。
- **test 指标不参与任何选择、不参与早停、不参与保存决策。** 代码中没有任何 `test` 影响控制流的路径。

### 2.3 test 是否每 epoch 评估

否。**只在 valid 改善的 epoch 评估**（上表 5 臂的 test 次数 41–50 次，而非 77–124 次）。这既省时，
也意味着 test 序列被重复测量了 41–50 次；但由于它从不进入选择，**不构成 test 集调参泄漏**。
（严格说，test 上的多次观测在事后分析者手里存在「挑最好一次」的风险，本次分析未使用该自由度。）

### 2.4 五臂是否共享同一套评估

代码层面共享：同一个 `_build_loader`（[fixed_train.py:588](tiger_ext/fixed_train.py#L588)），
同一个 `tiger_main.evaluate`，`beam_size=30`、`infer_size=96`、`max_length=5`（4 SID token + EOS）、
`topk_list=[5,10,20]`、`GenRecDataset(mode="evaluation", max_len=20)`；valid/test parquet 是同一对文件
（sha256 `f77a6dc7…` / `cf696aa1…`）。`run.json.config` 中 5 臂的 `t5_config`、`batch_size`、`lr`、
`num_epochs`、`early_stop`、`seed`、`num_workers` 全部逐字相同，**仅 `sid_path`/`tokenizer_ckpt`/`lambda_*` 不同**。

### 2.5 summary 的来源

`integration/artifacts/.../run_summary.json` 由 [integration/summarize_l0_run.py](integration/summarize_l0_run.py)
从 `run.json` 逐字段抄录，不重算。因此 summary 中的「最佳验证集指标」= valid，「测试集指标」需显式指明
取哪个 epoch；本次审计直接回到 `run.json` 取 `epochs[best_epoch].test_*`。

---

## §3 可比性逐项核对

### 3.1 代码/数据上相同（可以认为同口径）

| 项 | 基线 | EXP-L0 | 判定 |
|---|---|---|---|
| split 文件 | `data/Toys/{train,valid,test}.parquet` | 同 | 相同（同一份文件） |
| split 定义 | leave-one-out：valid 目标 = `seq[-2]`，test 目标 = `seq[-1]`（[rqvae/prepare_data.py:104-140](rqvae/prepare_data.py#L104-L140)） | 同 | 相同 |
| 评价函数 | `model.main.evaluate` | 同（`mods["tiger_main"].evaluate`） | 相同 |
| beam / infer / max_len / topk | 30 / 96 / 20 / [5,10,20] | 同 | 相同 |
| 指标计算 | `calculate_pos_index` 精确 token 序列匹配，按 batch 平均再对 batch 平均 | 同 | 相同 |
| 选择规则 | valid NDCG@20 严格改善；test 仅改善时评估；early stop 10 | 同 | 相同 |
| T5 结构 | 4+4, 128, 1024, 6 heads, vocab 1025 | 同 | 相同（`best.pth` 92 个张量键/形状逐一相同） |
| 物品表 | `item_emb_768d_contentonly.parquet` | 同（sha `6cc8dfbc…`） | 相同 |
| optimizer | Adam, lr 1e-4 | 同 | 相同 |

### 3.2 不相同（影响「逐位可复现」，但不必然影响结论）

1. **T5 初始化种子（最重要）**
   - 基线：[model/main.py:260](model/main.py#L260) 先 `model = TIGER(config)`，第 [265 行](model/main.py#L265)
     才 `set_seed(2025)` ⇒ **权重初始化使用的是进程默认 RNG 状态，未固定**。
   - EXP-L0：[tiger_ext/fixed_train.py:423](tiger_ext/fixed_train.py#L423) 在构造 RQVAE/TIGER/adapters
     **之前**调用 `set_seed(2025)` ⇒ 初始化被固定。
   - 影响：基线 ckpt 不可由任何 seed 复现；arm 与 baseline 的差异里包含一次未知初始化抽样。
   - 量级约束：基线 3-seed 极差 0.0013，`lc1e-4` 与基线的 test 差 0.00057 小于该极差。
   - 代码有明确注释承认这一点（[fixed_train.py:680-684](tiger_ext/fixed_train.py#L680-L684)），
     并在 loop 前再次 `set_seed` 以对齐 shuffle 顺序。

2. **DataLoader worker 数**：基线 `num_workers=4`（main.py 默认），EXP-L0 冻结为 `1`
   （`run.json.config.num_workers=1`）。eval loader 用 `shuffle=False`，顺序不变；train loader 的
   epoch 间排列由 loop 前的 `set_seed` 决定，两者调用点一致，**未逐位验证**是否完全相同的 shuffle 序列。

3. **adapters**：EXP-L0 无论 λ 是否为 0 都会构造 `SIAAdapter(128)`（4,128 参数）与
   `PSAAdapter(128,768)`（99,072 参数），并放进同一个 Adam 的可训练参数集
   （[fixed_train.py:439-452](tiger_ext/fixed_train.py#L439-L452)）。但：
   - λ=0 时 `self.sia/self.psa = None`，`train(..., **mounts)` 不会调用 adapter ⇒ 无梯度 ⇒ Adam 跳过；
   - `best.pth` 只存 `self.tiger.state_dict()`，**不含 adapter**，键集/形状与基线完全一致（92 keys）；
   - 因此 λ=0 时 **T5 的计算图与更新规则与 Clean TIGER 等价**，但**初始化与 worker 不同 ⇒ 不是逐位等价**。

4. **CSG 臂的 tokenizer/SID 是重训的**（见 §3.3），ETE 臂用基线 tokenizer/SID。
   ⇒ `lc1e-*` 与 `le*` 不是「同一模型加不同正则」，它们在 SID 空间上就不同，不能互相当作对方的对照。

5. **valid/test 不是同一预测任务**：valid 目标是 `seq[-2]`（history 少一项），test 目标是 `seq[-1]`。
   这是 **valid 系统性高于 test 的结构性原因**，不是 bug：
   基线 valid 0.039808 → test 0.034084（−0.005724，−14.4%）；各臂差值 0.0051–0.0077，量级一致。
   **任何 valid 与 test 的混用都会凭空制造 14%–20% 的「提升」。**

### 3.3 CSG 臂的 tokenizer 是否经过重新训练

**是，三臂各自从零重训。** 证据（`run.json`）：

| 臂 | tokenizer ckpt sha | SID sha | SID 唯一行 / dup rate |
|---|---|---|---|
| baseline | `8069f998…`（未固定 KMeans seed，`seed=None`） | `bb2693a6…` | 11924 / 0.000419 |
| lc1e-4 | `c12ba03e…`（run-local `ckpt/Oct-19-2026_04-28-24/`） | `28a28468…` | 11924 / 0.000587 |
| lc1e-3 | `e98192bc…`（`…/05-45-22/`） | `f4cda480…` | — |
| lc1e-2 | `cb600450…`（`…/07-03-40/`） | `fd0544bd…` | — |

tokenizer 日志（`lc1e-4/logs/tokenizer.log`）显示 `--epochs 3000 --batch-size 1024 --seed 2024
--lambda-csg 1e-4`，且 `seeding seed=2024 … kmeans_random_state_calls=[...]` ⇒ KMeans 被显式定种。
ETE 两臂 `sid_path` 与 `tokenizer_ckpt` **逐字等于基线**（`bb2693a6…` / `8069f998…`）。

---

## §4 只读复核测量

为避免依赖「数值接近」做推断，本次用一次性脚本（`/tmp/audit_eval_valid.py`，不写任何文件）
走**历史评测路径**（`sys.path` 指向 `model/`，`TIGER` + `model.main.evaluate` +
`GenRecDataset(mode="evaluation")`，batch 96、beam 30、max_len 20）：

| 对象 | test N@20 | test R@20 | valid N@20 | valid R@20 |
|---|---|---|---|---|
| baseline seed2025 ckpt + baseline SID | **0.03408397088203524** | **0.07945402289552642** | **0.03980784884796207** | **0.09360632170083488** |
| lc1e-4 `best.pth` + lc1e-4 SID | **0.03351538827450111** | **0.07950533683333784** | **0.04044356019112277** | **0.09545361272660383** |

交叉校验：
- baseline 四个数字与 §1.1 训练日志、§1.2 eval 日志**逐位相同**；
- lc1e-4 四个数字与 `run.json` 的 `epochs[96]` **逐位相同**。

⇒ 两套代码（`main.py` 训练循环 / `fixed_train.py`）产出的指标**在同一测量尺度上**，
差异不是评测实现造成的。

---

## §5 对 Stage-0 选点规则的影响

原规则：「ΔNDCG@20 ≥ −0.001 视为不劣，取满足条件的最大 λ；无满足者记录『未找到正向工作点』」。
基线端数字是 **test**。因此规则必须在 test 口径下执行：

| 读法 | CSG 合格臂 | ETE 合格臂 | 结论 |
|---|---|---|---|
| **test 对 test（正确口径）** | 仅 `lc1e-4`（−0.00057） | 无（le0.1 为 −0.00255） | λc = 1e-4；λe 回退锚点 0.1，并记录「未找到正向工作点」 |
| valid 对 valid（若改用 valid 比较） | `lc1e-4`(+0.00064)、`lc1e-3`(−0.00079)、`le0.1`(−0.00054) | `le0.1`(−0.00054) | 最大合格 λc = 1e-3，λe = 0.1 |

两种读法在 **λe = 0.1** 上一致。λc 的分歧由 `lc1e-3` 是否被码本塌缩判据排除决定
（`lc1e-3` 的 L0 distinct = 118/148 ≈ 80%，若按参照的 90% 阈值则排除）⇒ 排除后两读法都落在 **λc = 1e-4**。
**但必须明确：这里的「合格」只表示「未劣化超过 0.001」，不表示「有提升」。**

---

## §6 已确认事实 vs 仅凭现有文件无法确认

### 代码/文件明确证实（可直接引用）

1. 0.0341/0.0795 是 **test**、seed 2025、单次运行；valid 对应值为 0.039808/0.093606。
2. `run.json.best_ndcg@20` 是 valid；test 指标需从 `epochs[best_epoch].test_*` 取。
3. 模型选择只用 valid；test 不参与任何控制流。
4. 五臂的 split、eval code、beam、topk、选择规则、early stop 完全一致。
5. 三臂 CSG 各自重训 tokenizer（seed 2024）并生成自己的 SID；两臂 ETE 使用基线 tokenizer + 基线 SID。
6. 所有 `best.pth` 与基线 ckpt 的 T5 键集/形状一致，不含 adapter。
7. 基线 T5 初始化在代码顺序上未定种；EXP-L0 在构造前定种。
8. valid 与 test 是不同目标位置的两个任务，系统性 gap ≈ 0.006。

### 仅凭现有文件无法确认

1. **基线启动时外部是否设置过全局随机状态**（`torch.manual_seed` / `PYTHONHASHSEED` / 启动脚本）。
   日志无法证明「绝对没有任何外部定种」，只能说 `main.py` 自身没有。
2. **ckpt 重命名的哈希级证据**：`..._toys.pth` → `..._toys_seed2025.pth` 只有 mtime 与命名约定；
   复算精确复现可作强旁证，但不是 sha256 记录。
3. **基线评测时使用的 SID 路径**：eval 日志不记录 `--code_path`；由训练日志 config + 复现结果反推。
4. **train loader 的 shuffle 序列是否与基线逐位相同**：worker 数 4 vs 1，未逐位验证。
5. **`lc1e-3` 在 L0 的码本塌缩是否应按 90% 阈值判负**：这是应用既有判据，不是本次审计新增的事实。
6. **多 seed**：EXP-L0 每个臂只有 seed 2025 一次运行；任何「臂间差异」都未做配对显著性检验。

---

## §7 对后续分析（CSG / ETE 各解决什么问题）的直接含义

- **不能说** CSG 或 ETE 带来了推荐效果提升。同口径下最好的 `lc1e-4` 也只是「test −0.00057、
  valid +0.00064」，均小于基线自身 seed 极差。
- **可以说**两个模块的机制确实生效了：CSG 臂的 tokenizer loss 曲线、码本 distinct 计数、SID
  重复率（0.000587 vs 基线 0.000419）都发生了变化；ETE 臂 `mounts.sia/psa=true` 且 adapter 参与前向。
  但这属于**机制证据**，不是效果证据。
- 若要把「各解决什么问题」写成结论，需要的是**同一 seed、同一初始化口径下的对照臂**（例如
  λ=0 的 fixed_train 臂），而不是直接对历史基线。当前**没有** λ=0 的 fixed_train 臂。
- 因此 Stage-0 的诚实结论是：**在 test 口径下未找到正向工作点；`λc=1e-4` 为不劣化锚点，
  `λe=0.1` 为回退锚点。** 这与原计划的「标定工作点」目标一致，但**不支持任何增益叙事**。

---

## 附：本次审计未做的事

- 未修改 `model/`、`rqvae/`、`tiger_ext/`、`integration/` 下任何源码或配置。
- 未运行任何训练；只做了一次不落盘的 GPU 前向评测（约 2 分钟）。
- 未启动 EXP-B1。
- 用户提问的最后一条在「CSG 实验使用的 tokenizer 是否经过重新」处被截断；按语义补全为
  「是否经过重新训练」并在 §3.3 回答。若原意是「是否重新生成 SID」或「是否重新初始化码本」，
  答复分别是：SID 由各臂 tokenizer 重新生成（§3.3 表），码本由 RQ-VAE 从头训练而非复用基线。
