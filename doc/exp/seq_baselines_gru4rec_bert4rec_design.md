# 新增经典序列基线 GRU4Rec + BERT4Rec：方案与执行记录

**日期**：2026-09-29
**目标**：回应"只挑了一个经典序列基线（SASRec）"的审稿风险；一并完成配对显著性检验。

---

## 0. 执行摘要（含两处对原方案的修正）

| 项 | 状态 |
|:--|:--|
| 通用层抽取 `tools/seq_baselines_common.py` | ✅ 完成 |
| `tools/run_sasrec.py` 重构为 import 通用层 | ✅ 完成 |
| **门禁方案修正**（原"diff 存档"不可实现，见 §2） | ✅ 已替换为 Gate A + Gate B |
| **Gate A** 存档 ckpt 重评估 | ✅ **PASSED**（差 ~1e-7 = 打印舍入） |
| **Gate B** 四路重构等价（含 best.pth md5 相同） | ✅ **PASSED**（见 §2.4.1） |
| **std 口径统一**（ddof=0 vs ddof=1 冲突，见 §3） | ✅ 已全量重算并统一为 ddof=1 |
| `tools/run_gru4rec.py` + 自检 8/8 | ✅ 完成 |
| `tools/run_bert4rec.py` + 自检 15/15 | ✅ 完成 |
| `tools/paired_significance.py` + 报告 | ✅ 完成（6/6 对比零越界） |
| 阶段 5 步 1：CPU 自检 | ✅ GRU4Rec 11/11、BERT4Rec 15/15 |
| 阶段 5 步 2：冒烟（Toys/seed2025/20ep） | ✅ 双通过（见 §0.1） |
| 三基线 3-seed 全量训练 | ⏳ 运行中（driver task `bfb8152m5`） |

### 0.1 阶段 5 步 1–2 实测

**CPU 自检**（`--selftest --device cpu`）：GRU4Rec **11/11**、BERT4Rec **15/15**。
其中 BERT4Rec 的合成确定性测试（`x_{t+1}=(x_t+1) mod V`）末位恢复 **acc=1.000**（随机 0.0200），
打乱对照 **acc=0.010**（≈随机）—— 证明掩码/右移实现正确且指标确实反映训练。

**冒烟**（Toys / seed 2025 / 20 epoch，写入 `/tmp/smoke_*` 一次性目录）：

| 方法 | CE 首 → 末 | val N@20 | test N@20 | 判定 |
|:--|:--|--:|--:|:--|
| GRU4Rec | 11.224 → 7.700 | 0.0140 | 0.0112 | ✅ 单调下降，远超随机 |
| BERT4Rec | 19.195 → 8.712 | 0.0085 | 0.0081 | ✅ 单调下降 |

Toys 词表 V=19413 ⇒ 均匀预测基线 `ln V = 9.87`。两方法都从**高于** `ln V` 起步
（随机初始化下 logits 本身有方差，`CE = logsumexp − logit_target` 会超过均匀值），
并在 20 epoch 内降到其下 —— 收敛行为健康。

> **与方案预估的一处出入（记录）**：方案 §阶段 5 步 2 预估 GRU4Rec 首 epoch CE ≈ 9.39。
> 实测 **11.224**。原因是该预估假设"初始即均匀分布"，忽略了 GRU 初始隐状态非零、
> `out_emb` 为 `N(0,1)` 默认初始化，二者相乘使 logits 有可观方差。
> **判据的实质（loss 降到 `ln V` 以下）成立**，仅预估数字有误，非实现问题。

> **另一处修正**：GRU4Rec 自检 T2/T3 原用 margin=30 制造"完美预测"。
> float32 下 `30·ε ≈ 3.6e-6 ≫ 4e^-30`，`CE = log(1+4e^-30)` 被 `e^30` 完全吞掉 → **恒为精确 0.0**，
> 使 T3（"扰动 pad 列 loss 不变"）退化为恒真断言，检测不出 `-inf` 掩码失效。
> 已改为 margin=10（CE≈1.8e-4，可表示）+ 容差 1e-6（float32 抵消误差本身约 6e-8），
> 并新增 **T3b 阳性对照**（扰动样本实际读取的**有效**列必须改变 loss；因 `hs` 是 one-hot，
> 扰动 `w_1[1]` 是无效扰动，须扰动 `w_1[3]`）。

> **时间估计修正**：方案预估 GRU4Rec ~35-60 min、BERT4Rec ~50-90 min（全套）。
> 冒烟实测外推为 GRU4Rec ~20 min/(seed·数据集)、BERT4Rec ~30 min；
> 全量首跑实测（Sport，早停于 epoch ~140）约 **7-10 min/seed** ⇒ 9+9 次训练约 **3-5 小时**
> （单卡串行）。方案的数字低估，以实测为准。

### 0.2 R1 冻结超参（预登记记录，看 test 数字之前冻结）

`seed{seed}_config.txt` 只记录了**显式 argv**（`--dataset/--seeds/--output_dir`）+ git HEAD。
生效的超参一部分来自 `parser.set_defaults`（见下），**未体现在该文件里**。
为满足 R1"超参在看 test 前冻结且可审计"，此处显式固化全部生效值：

**git HEAD（记录于每个 config.txt）**：`9bcac4a6b8a50d4c42d140addb8865a77f7f5ec1`

| 超参 | SASRec | GRU4Rec | BERT4Rec | 来源 |
|:--|:--|:--|:--|:--|
| `hidden_units` | 50 | **100** | **64** | common 默认 50；后两者 `set_defaults` 覆盖 |
| `num_layers` / `num_blocks` | 2 | 1（GRU 层数，硬编码） | **2** | BERT4Rec `--num_blocks` |
| `num_heads` | 2 | — | **2** | BERT4Rec |
| `dropout_rate` | 0.2 | 0.2 | 0.2 | common |
| `mask_prob` | — | — | **0.2** | BERT4Rec Cloze |
| `batch_size` | 128 | 128 | 128 | common |
| `lr` | 0.001 | 0.001 | 0.001 | common |
| `maxlen` | 200 | 200 | 200 | common |
| `num_epochs` | 200 | 200 | 200 | common |
| `eval_interval` | 20 | 20 | 20 | common |
| `patience` | 5 | 5 | 5 | common（单位=评估次数） |
| `ce_chunk` | — | **2048** | **2048** | 分块 CE 行数上限 |
| optimizer | Adam, betas=(0.9,0.98) | Adam 默认 | Adam 默认 | 各自论文口径 |

> **不做的调参**：三者一律不做逐方法调参（R1）。SASRec 的 `betas=(0.9,0.98)` 是其原论文设置；
> GRU4Rec/BERT4Rec 用 Adam 默认 betas，同为其各自论文口径 —— 这是**结构差异而非不公平**。

---

## 1. 重构设计

通用层 `tools/seq_baselines_common.py` 承载**协议性**代码（三基线必须逐字相同）：

| 组件 | 内容 |
|:--|:--|
| `load_tiger_data` / `build_dataset` | 数据切分与适配（原 L54-115 / L487-495） |
| `evaluate_full` | 全量用户 × 全量物品排序，`eval_batch=512`（原 L277-330） |
| `seed_everything` | `torch/np/random` 三行，**不含任何 cudnn 开关** |
| `xavier_init_zeros` | Xavier + 指定行置零（原 L346-354，顺序不可改） |
| `run_seed` | 训练骨架：eval 循环 / best-val ckpt / 早停（原 L337-429，控制流一字不改） |
| `write_summary_3seed` | 产物格式（原 L507-529） |
| `run_datasets` / `add_common_args` / `resolve_device` | CLI 与编排（原 L469-531） |

方法专属代码留在各自脚本：SASRec 的 `WarpSampler`（负采样）+ 因果 mask + BCE + `betas=(0.9,0.98)`；
GRU4Rec 的 GRU + 分块全物品 CE；BERT4Rec 的双向 encoder + Cloze。

**接口约定**：
- `model.predict(user_ids, log_seqs, item_indices) -> (B, len(item_indices))`
- `trainer_factory(model, dataset, args, seed) -> trainer`，需实现 `train_epoch(epoch)` / `close()`
- `model_factory` 必须完成"建模型 + 初始化"，因为 **WarpSampler 必须在模型初始化之后创建**（RNG 顺序）

---

## 2. ⚠ 门禁方案修正：原"逐位 diff 存档"在构造上不可能通过

### 2.1 原方案

> 重跑 `Sport` seed 2025/2026 → 与 `sasrec_output/Sport/seed*_epochs.txt` 逐位 diff，全空即通过。

### 2.2 实测：不通过，但**不是重构的锅**

同一个 epoch 20、同一个 seed 2025、**同一份未修改的 HEAD 原始代码**，今天跑两次：

| 来源 | val N@20 | test N@20 |
|:--|--:|--:|
| 存档（2026-08-07） | 0.014310 | 0.012929 |
| 原版代码 今日（全量 200ep） | 0.014282 | 0.013172 |
| 原版代码 今日（20ep 探针） | 0.014725 | 0.013220 |
| 重构代码 今日 | 0.014116 | 0.012592 |

**原版代码今天两次互不相同**（0.014725 / 0.014282）⇒ 训练路径本就跨运行不可复现。

### 2.3 根因

`WarpSampler` 默认 `n_workers=3`（cuda），三个子进程各以 `base_seed+i` 播种、各自 shuffle 用户，
结果 `put()` 进同一个 `multiprocessing.Queue`。**队列的 FIFO 顺序取决于 OS 调度**，因此
batch 到达主进程的顺序跨运行不同 ⇒ 逐 batch SGD 轨迹不同 ⇒ 权重不同 ⇒ 指标不同。

原 docstring 里"WarpSampler 按 seed 播种（worker = base_seed + i），保证同 seed 可复现"
**只对"每个 worker 产生的 batch 内容"成立，对"整体训练轨迹"是过度声明**。此条已修正。

### 2.4 替换后的门禁（两条，均可实现）

**Gate A — 产物保真（保护已发表数字的那一条）**
用重构后的代码加载**存档的** `sasrec_output/Sport/seed2025_best.pth` 重新评估：

```
重构代码重评估  val  N@20=0.014655  R@20=0.041266
重构代码重评估  test N@20=0.013695  R@20=0.039328
存档 epochs.txt 最佳行(epoch 100)  val 0.014655      test 0.013695
→ 差 4.7e-07 / 1.1e-07（= 打印舍入）
```
⇒ 评估路径、模型类、权重加载、数据集构造**全部逐位正确**。

脚本：`/tmp/eval_ckpt_check.py`（隔离实验，一次性）；审计证据留在
`sasrec_output_verify/`（重构后跑出的产物，**绝不写 `sasrec_output/`**）。

**Gate B — 训练路径等价（同 session、同条件 head-to-head）**
把采样器串行化（`n_workers=1`）后，batches 按严格顺序产生，训练即逐位可复现。于是：

```bash
# 原版（HEAD）+ 唯一改动 n_workers=1
sed 's/n_workers=1 if args.device == "cpu" else 3,/n_workers=1,/' \
    <(git show HEAD:tools/run_sasrec.py) > /tmp/orig_nw1_run_sasrec.py

python3 /tmp/orig_nw1_run_sasrec.py --dataset Sport --seeds 2025 \
        --num_epochs 40 --eval_interval 20 --output_dir /tmp/gB_orig_a
python3 tools/run_sasrec.py         --dataset Sport --seeds 2025 \
        --num_epochs 40 --eval_interval 20 --n_workers 1 --output_dir /tmp/gB_new_a
diff /tmp/gB_orig_a/Sport/seed2025_epochs.txt \
     /tmp/gB_new_a/Sport/seed2025_epochs.txt      # 必须全空
```

**通过判据**：`orig_a ≡ orig_b ≡ new_a ≡ new_b` 四路逐位相同（`epochs.txt` 6 位小数，一位 bit 差异必显形）。

新增 `--n_workers`（默认 `None` = 原行为 cpu1/cuda3）**零语义变化**，仅用于让门禁可实现。

#### 2.4.1 Gate B 实测结果：✅ PASSED（2026-09-29）

四路各 40 epoch × 2 次 eval（epoch 20 / 40），Sport / seed 2025：

| 运行 | 代码来源 | 采样 | epoch 20 (val/test N@20) | epoch 40 (val/test N@20) |
|:--|:--|:--|:--|:--|
| `gB_orig_a` | HEAD 原版 | `n_workers=1` | 0.013487 / 0.012446 | 0.013134 / 0.012354 |
| `gB_new_a`  | 重构版 | `--n_workers 1` | 0.013487 / 0.012446 | 0.013134 / 0.012354 |
| `gB_orig_b` | HEAD 原版 | `n_workers=1` | 0.013487 / 0.012446 | 0.013134 / 0.012354 |
| `gB_new_b`  | 重构版 | `--n_workers 1` | 0.013487 / 0.012446 | 0.013134 / 0.012354 |

```
diff orig_a epochs.txt new_a epochs.txt   → 空
diff orig_a epochs.txt orig_b epochs.txt  → 空   （n_workers=1 训练路径确定）
diff orig_a epochs.txt new_b epochs.txt   → 空
md5sum 四路 epochs.txt  → f5ac1cd7ab113344aafbb47c4c35e08d ×4
md5sum 四路 seed2025_best.pth → 39a533bec5a13a0876101c01d4205358 ×4
```

**判定强度**：不只是打印精度相同 —— `best.pth`（完整 float32 权重）的 md5 也四路一致。
- `orig_a ≡ orig_b` ⇒ `n_workers=1` 下**训练轨迹逐位确定**（反证 §2.3 根因正确：不确定性确实只来自
  `multiprocessing.Queue` 的到达顺序）。
- `orig_a ≡ new_a ≡ new_b` ⇒ 重构是**纯语义等价变换**，无任何数值副作用。

复现命令（~4 min/路，CPU 训练）：

```bash
sed 's/n_workers=1 if args.device == "cpu" else 3,/n_workers=1,/' \
    <(git show HEAD:tools/run_sasrec.py) > /tmp/orig_nw1_run_sasrec.py
python3 /tmp/orig_nw1_run_sasrec.py --dataset Sport --seeds 2025 \
        --num_epochs 40 --eval_interval 20 --output_dir /tmp/gB_orig_a
python3 tools/run_sasrec.py --dataset Sport --seeds 2025 \
        --num_epochs 40 --eval_interval 20 --n_workers 1 --output_dir /tmp/gB_new_a
# orig_b / new_b 同前两条，仅换 output_dir
```

**结论**：阶段 0 门禁**通过**，无需降级为"复制粘贴复用"。已发表的 `sasrec_output/` 数字未受任何影响
（本次全部写入 `/tmp/gB_*`，从未触碰权威目录）。

### 2.5 对论文的影响（诚实边界）

- 已发表的 3-seed 数字**仍然有效**：每个 seed 是一次合法实现，且方法间差距（0.009~0.02）比
  run-to-run 噪声（同 epoch 跨运行 ~0.0006）大一个数量级以上。
- 但"同 seed 可逐位复现"**不能声称**。论文应写 "run-to-run variation from asynchronous
  mini-batch sampling is of the same order as between-seed variation"，或
  在可复现性声明中限定为"给定 checkpoint 可逐位复现下游评估"。

---

## 3. ⚠ std 口径统一（ddof=0 vs ddof=1）

### 3.1 事实

| 出处 | ddof |
|:--|:--|
| `sasrec_output/*/summary_3seed.txt`（`np.std` 默认） | **0**（总体标准差） |
| `doc/paper_main_results.md` §1/§3（原） | **0** |
| `doc/experiment_results_summary.md` §1.5（原） | **0** |
| `doc/ablation_main_table_3datasets.md` | **1**（样本标准差） |
| 图 A（`figA_claims_verification.md`） | **1** |

逐格反算确认（`doc/seq_baselines_significance.md §1`）：**18/18 格四位小数即不同**。

### 3.2 处置

**统一为 ddof=1**（Bessel 校正；与消融表、图 A 一致；也是领域惯例）。

- 已重算并写入：`doc/paper_main_results.md` §1（18 个 std）+ §3（9 个 std）；
  `doc/experiment_results_summary.md` §1.5（6 个）+ 3-seed 明细（6 个）。
- **均值与逐 seed 值全部未变**，仅 std 变化。
- 逐 seed 产物文件保持 ddof=0 不动（门禁与历史一致性），差异记录在
  `doc/seq_baselines_significance.md §1`，论文 §5.1 加一句 "standard deviation uses
  Bessel's correction (n−1)"。
- 生成命令：`python3 tools/paired_significance.py --emit-corrected-table`

### 3.3 连带修正

`doc/paper_main_results.md` §3 原句"所有方法 std ≤ 0.0016"→ 改为
"§3 全部 NDCG@20 的 std ≤ 0.0011（ddof=1 口径）"。

---

## 4. 三基线的协议对齐

| 项 | SASRec | GRU4Rec | BERT4Rec |
|:--|:--|:--|:--|
| 训练数据 | 仅 `train.parquet` 的 history | 同 | 同 |
| 不使用 | 任何 split 的 target | 同 | 同 |
| 评估 | `evaluate_full`（共用） | 同 | 同 |
| 最终测试任务 | 从 test.history 预测 target | 同 | 同 |
| 模型选择 | best-val NDCG@20 | 同 | 同 |
| Seed | 2025/2026/2027 | 同 | 同 |

**刻意偏离（论文必须标注，原句见 `doc/seq_baselines_report.md`）**：
- GRU4Rec 用**全物品 softmax CE**，非原版 TOP1/BPR（负采样损失与本工作全量排序协议不可比）
- BERT4Rec 训练 Cloze / 评估 next-item，通过"训练时恒将末位替换为 [MASK]"使二者同构；
  评估输入右移一格（位置索引一致，每数据集仅影响 1 个超长序列用户）

---

## 5. 产物布局

```
{sasrec,gru4rec,bert4rec}_output/{Sport,Toys,Beauty}/
├── seed{seed}_epochs.txt    5 列 header 三方法逐字相同
├── seed{seed}_best.pth / _best.txt
├── seed{seed}_config.txt    CLI + git HEAD（幂等判据 / 预登记超参冻结）
├── seed{seed}_loss.tsv      仅 GRU4Rec/BERT4Rec：epoch/loss 收敛自检
└── summary_3seed.txt
```

脚本层断言：`out_dir` 以 `sasrec_output` 结尾且方法名不是 SASRec → 直接退出
（`seq_baselines_common.guard_output_dir`）。

---

## 6. 预登记处理规则（跑之前定死）

**R1 数字优先**：如实报告，禁止调参调到低于 Content-only、禁止换 loss、禁止挑 seed 子集。
超参在看 test 数字**之前**冻结并记入 `seed{seed}_config.txt`。

**R2 一次性全局排查**（非逐格调参）：①协议核验 ②泄漏核验 ③评估路径核验。

**R3 三种结局**：
- **A（预期）** 经典基线 < SASRec < Content-only < TA-SID → 主表新增 2 行
- **B** 经典基线 > SASRec 但仍 < Content-only → 叙事不变，但删掉"SASRec 是最强经典基线"类表述
- **C** 某经典基线 > Content-only（任一格）→ 收紧为"优于全部经典 ID 基线与内容基线中的最优者"，
  被超过格标 "comparable"，**不得**写 "ours 更强"

**R4 禁止的补救**：换 seed、换评估口径、把新基线挪到附录。
