# 经典序列基线报告：GRU4Rec + BERT4Rec（对齐 SASRec/TA-SID 协议）

> **状态（2026-09-29）：** 方法与门禁部分**已定稿**；结果表**待全量跑完填入**
> （driver task `bfb8152m5`，预计 4-5 h）。方案与执行记录见
> [exp/seq_baselines_gru4rec_bert4rec_design.md](exp/seq_baselines_gru4rec_bert4rec_design.md)。
> 本文件是这两个基线的**唯一数字源与诚实标注源**。

> 实验人：Hanpee　｜　生成日期：2026-09-29

---

## 0. 为什么加这两个基线

论文原有 **SASRec** 一个经典序列基线。审稿风险直接：「只挑了一个基线」。
GRU4Rec（Hidasi et al., ICLR 2016）与 BERT4Rec（Sun et al., CIKM 2019）是 TIGER 原论文
使用的基线，且至今仍是全领域共同参照系（2025-2026 的 LSIG / DiGR / VaLiDRec 等仍以二者
作为序列基线）。选择依据是**共同参照系地位**，不是"最新"。

**硬约束（已满足）**：本工作**未改动**论文已发表的任何 SASRec / TA-SID / 消融数字。
`tools/run_sasrec.py` 经重构后与 HEAD 原版**逐位等价**（§3），SASRec 产物目录
`sasrec_output/` 全程未被写入。

---

## 1. 方法

### 1.1 协议对齐（三基线 + TA-SID 严格同口径）

| 项 | SASRec | GRU4Rec | BERT4Rec |
|:--|:--|:--|:--|
| 训练数据 | 仅 `train.parquet` 的 **history** | 同 | 同 |
| 不使用的数据 | 任何 split 的 `target` | 同 | 同 |
| 模型选择 | best-val NDCG@20 | 同 | 同 |
| 评估实现 | `evaluate_full`（共用） | 同 | 同 |
| 评估人口 | 全量用户（`sorted(users)`） | 同 | 同 |
| 排序范围 | 全量物品（`arange(1, itemnum+1)`） | 同 | 同 |
| 最终测试任务 | 从 `test.history` 预测 `test.target` | 同 | 同 |
| Seed | 2025 / 2026 / 2027 | 同 | 同 |

「全量用户 × 全量物品」是硬性要求：不做负采样、不排除已交互项。三者与 TA-SID T5
评估的是**同一个人口、同一个任务**，因此可直接比较。

### 1.2 模型与超参

| 项 | GRU4Rec | BERT4Rec |
|:--|:--|:--|
| 结构 | 1 层 GRU(100→100) + 独立输出 embedding | 2 层双向 Transformer，H=64，2 头 |
| 损失 | 全物品 softmax CE（见 §2.1） | Cloze CE（见 §2.2） |
| Dropout | 0.2（embedding） | 0.2 |
| 优化器 | Adam (lr=1e-3) | Adam (lr=1e-3) |
| maxlen / batch | 200 / 128 | 200 / 128 |
| 训练 | ≤200 epoch，val N@20 早停（patience=5 次评估） | 同 |

超参在**看到 test 数字之前**冻结。⚠️ 注意 `seed{seed}_config.txt` 只记录了**显式 argv**
（`--dataset/--seeds/--output_dir`）+ git HEAD（`9bcac4a6b8a50d4c42d140addb8865a77f7f5ec1`），
**不含从 `parser.set_defaults` 生效的超参**。全部生效值的显式固化表见
[exp/seq_baselines_gru4rec_bert4rec_design.md §0.2](exp/seq_baselines_gru4rec_bert4rec_design.md)。
复现时以该表 + 上述 git HEAD 为准，不要只看 `config.txt`。

**不做的调参**：三者一律不做逐方法调参。SASRec 用其原论文的 `betas=(0.9,0.98)`，
GRU4Rec/BERT4Rec 用 Adam 默认 betas —— 同为其各自论文口径，属结构差异而非不公平。

---

## 2. 必须随论文发表的诚实标注（可直接粘贴）

> **GRU4Rec (cross-entropy variant).** 我们采用全物品 softmax 交叉熵变体，而非
> Hidasi et al. (ICLR 2016) 原版的 session-parallel mini-batch + TOP1/BPR 成对排序损失。
> 原因：(i) 原版损失配合"100 负采样 + Recall@20"评估口径，与本工作三方法统一的
> **全量物品排序**协议不可比；(ii) CE 变体的监督信号（逐位置 next-item 预测）与本文
> SASRec 基线严格同构。模型结构（100 维 GRU、embedding dropout 0.2、输入/输出 embedding
> 不共享）按原论文设置，未在测试集上做任何调参。另：原论文面向匿名 session，本工作数据是
> 用户级序列（Amazon 5-core），不做 session 切分，与另两个基线一致。
> **本表不声称复现原论文数字。**

> **BERT4Rec 的目标不一致性.** 训练目标是 Cloze（掩码项复原），评估是 next-item 排序；
> 我们通过"训练时恒将序列末位替换为 [MASK]"使二者结构同构。评估输入相对训练输入右移一格
> （保持位置索引一致，每数据集仅影响 1 个超长序列用户，已核实）。双向编码使被 mask 位置
> 可见左右上下文，与自回归基线在信息可得性上不同——但评估时输入不含目标项，**不构成泄漏**。

> **训练数据口径（三基线一致）.** 三个序列基线均只使用 `train.parquet` 的 history 做
> next-item 自监督，均不使用任何 split 的 target；最终评估对三者都是"从 `test.parquet`
> 的 history 预测 target"。

> **可复现性边界（重要，勿过度声称）.** 见 §5。简言之：给定 checkpoint 的下游评估**逐位可
> 复现**；但"同一 seed 端到端逐位复现"**不成立于默认配置**，原因是 SASRec 的 `WarpSampler`
> 多进程采样的 batch 到达顺序由 OS 调度决定（§4.2）。

---

## 3. 重构门禁：SASRec 逐位等价（保证已发表数字未受影响）

为了让三个基线共用同一份协议代码，`run_sasrec.py` 的通用部分被抽到
`tools/seq_baselines_common.py`。这是一次**纯语义等价重构**，用两条门禁证明：

### Gate A — 产物保真

用重构后的代码加载**存档的** `sasrec_output/Sport/seed2025_best.pth` 重新评估：

| 来源 | val N@20 | test N@20 |
|:--|--:|--:|
| 重构代码重评估 | 0.014655 | 0.013695 |
| 存档 `epochs.txt` 最佳行 (epoch 100) | 0.014655 | 0.013695 |

差 4.7e-07 / 1.1e-07 = 打印舍入 ⇒ **评估路径、模型类、权重加载、数据集构造全部逐位正确**。

### Gate B — 训练路径等价（四路逐位）

把采样器串行化（`--n_workers 1`）后四路对照，Sport / seed 2025 / 40 epoch：

| 运行 | 代码来源 | epoch 20 | epoch 40 |
|:--|:--|:--|:--|
| `orig_a` / `orig_b` | HEAD 原版 | 0.013487 / 0.012446 | 0.013134 / 0.012354 |
| `new_a` / `new_b` | 重构版 | 0.013487 / 0.012446 | 0.013134 / 0.012354 |

```
四路 epochs.txt  diff 全空，md5 = f5ac1cd7ab113344aafbb47c4c35e08d
四路 best.pth    md5 = 39a533bec5a13a0876101c01d4205358   ← 完整 float32 权重逐 bit 相同
```

**结论**：`orig_a ≡ orig_b ≡ new_a ≡ new_b`，重构无任何数值副作用。已发表数字未受影响。

> `--n_workers` 默认 `None` = 原行为（cpu 1 / cuda 3），**零语义变化**；仅为让门禁可实现而新增。

---

## 4. std 口径统一与可复现性边界

### 4.1 std：ddof=0 → ddof=1（已统一）

| 出处 | 原 ddof |
|:--|:--|
| `sasrec_output/*/summary_3seed.txt`（`np.std` 默认） | 0 |
| `doc/paper_main_results.md` §1/§3、`doc/experiment_results_summary.md` §1.5 | 0 |
| `doc/ablation_main_table_3datasets.md`、图 A | **1** |

逐格反算确认 **18/18 格四位小数即不同**。已统一为 **ddof=1**（Bessel 校正；与消融表、
图 A 一致；也是领域惯例）。**均值与逐 seed 值全部未变，仅 std 变化。**
论文 §5.1 需加一句 "standard deviation uses Bessel's correction (n−1)"。

逐 seed 产物文件保持 ddof=0 不动（历史一致性），差异记录在
[seq_baselines_significance.md §1](seq_baselines_significance.md)。

### 4.2 可复现性边界（诚实声明，勿写成"可逐位复现"）

**发现过程**：同一个 epoch、同一个 seed、**同一份未修改的 HEAD 代码**，两次运行得到
0.014725 与 0.014282。原版代码**今天两次互不相同** ⇒ 训练路径跨运行本就不可复现。

**根因**：`WarpSampler` 默认 `n_workers=3`（cuda），三个子进程各以 `base_seed+i` 播种、
各自 shuffle 用户，结果 `put()` 进同一个 `multiprocessing.Queue`。**队列的 FIFO 顺序取决于
OS 调度**，因此 batch 到达主进程的顺序跨运行不同 ⇒ 逐 batch SGD 轨迹不同 ⇒ 权重不同。

原 docstring 的"按 seed 播种保证同 seed 可复现"**只对"每个 worker 产生的 batch 内容"成立，
对"整体训练轨迹"是过度声明**（已修正）。

**论文可用的措辞**：run-to-run variation from asynchronous mini-batch sampling is of the
same order as between-seed variation。可复现性声明限定为"**给定 checkpoint 的下游评估
逐位可复现**"（Gate A 已证）。

**注意**：`rqvae/main.py` 设置了 `cudnn.deterministic/benchmark`，但 `run_sasrec.py`
**从未设置**。重构时"顺手"加上会改变 kernel 选择并使 Gate B 必然失败。新脚本同样不加。

---

## 5. 结果

> ⏳ 待全量跑完填入。产物：`gru4rec_output/{Sport,Toys,Beauty}/summary_3seed.txt`、
> `bert4rec_output/{...}/summary_3seed.txt`。

---

## 6. 命名陷阱与复现命令

### 6.1 命名陷阱

- **禁止**用 `--output_dir sasrec_output*` 跑非 SASRec 方法。脚本层已有断言
  （`seq_baselines_common.guard_output_dir`），但**不要依赖它**：它会 `exit`，
  在驱动脚本里表现为静默失败。手工跑时自查。
- **冒烟测试必须写入一次性目录**（如 `/tmp/smoke_*`）。若写进 `gru4rec_output/`，
  20-epoch 的 `seed{seed}_best.txt` 会让驱动脚本的幂等跳过逻辑误判"已完成"，
  **正式结果会变成冒烟结果且无任何报错**。
- 三方法的 `seed{seed}_epochs.txt` 是 **5 列同名 header**，聚合时靠目录区分方法，
  不靠文件名。

### 6.2 复现命令

```bash
conda activate tiger_t5

# 自检（CPU，秒级）
python3 tools/run_gru4rec.py  --selftest --device cpu
python3 tools/run_bert4rec.py --selftest --device cpu

# 全量（单卡串行，~4-5 h；幂等，已完成的 seed 自动跳过）
bash tools/run_seq_baselines.sh
bash tools/run_seq_baselines.sh gru4rec          # 只跑 GRU4Rec
bash tools/run_seq_baselines.sh bert4rec Beauty  # 只跑 BERT4Rec × Beauty

# 聚合 + 配对显著性（零 GPU）
python3 tools/paired_significance.py --extra-methods GRU4Rec BERT4Rec \
    --out-md doc/seq_baselines_significance.md \
    --out-csv doc/seq_baselines_significance.csv

# 阶段 0 门禁复现（~4 min/路）
sed 's/n_workers=1 if args.device == "cpu" else 3,/n_workers=1,/' \
    <(git show HEAD:tools/run_sasrec.py) > /tmp/orig_nw1_run_sasrec.py
python3 /tmp/orig_nw1_run_sasrec.py --dataset Sport --seeds 2025 \
        --num_epochs 40 --eval_interval 20 --output_dir /tmp/gB_orig_a
python3 tools/run_sasrec.py --dataset Sport --seeds 2025 \
        --num_epochs 40 --eval_interval 20 --n_workers 1 --output_dir /tmp/gB_new_a
```

---

## 7. 预登记处理规则（跑之前定死，避免事后合理化）

- **R1 数字优先**：如实报告，禁止调参调到低于 Content-only、禁止换 loss、禁止挑 seed 子集。
- **R2 只做一次性全局排查**（协议 / 泄漏 / 评估路径），不逐格调参。
- **R3 三种结局**：A（预期）经典基线 < SASRec < Content-only < TA-SID → 主表新增 2 行；
  B 经典基线 > SASRec 但仍 < Content-only → 删掉"SASRec 是最强经典基线"类表述；
  C 某经典基线 > Content-only → 收紧为"优于全部经典 ID 基线与内容基线中的最优者"，
  被超过格标 "comparable"，**不得**写 "ours 更强"。
- **R4 禁止的补救**：换 seed、换评估口径、把新基线挪到附录。
