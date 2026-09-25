# SASRec 基线实验报告 v2（对齐协议）

> **状态（2026-08-07）：** v2 重跑完成，修正了 v1 的三个问题，数字为**正式口径**：
> ① off-by-one（v1 的 "test" 实为 valid 任务，现改为 test.parquet）；② 全量用户（去掉 10,000 子集）；③ 3-seed mean±std。
> 主实验总表见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)。
> 本文档是 SASRec 基线的完整方法与结果存档。

> 生成日期：2026-08-07（v2）；v1 生成于 2026-07-30
> 实验人：Hanpee

---

## 方法（v2 修正）

### 评估协议对齐

TIGER 数据为滚动划分（L = 完整序列长度）：

| split | history | target | TA-SID T5 用途 |
|:-----|:-------|:-------|:---------------|
| train | [i₁..i_{L-3}] | i_{L-2} | 训练（滑窗） |
| valid | [i₁..i_{L-2}] | i_{L-1} | 模型选择（val） |
| test | [i₁..i_{L-1}] | i_L | **最终评估（test）** |

SASRec v2 的评估与 TA-SID 严格同任务：
- **val**（选 checkpoint）：从 valid.history 预测 valid.target —— 与 T5 验证集一致
- **test**（报告）：从 test.history 预测 test.target —— 与 T5 最终测试集一致

v1 的错误：其 "test" 从 train.history+train.target 预测 valid.target，实际是 **valid 任务**（预测 i_{L-1} 而非 i_L），导致 SASRec 数字整体虚高（对齐后 Sport/Toys/Beauty 的 N@20 分别下降 -18%/-16%/-21%）。

### 实验设置

| 项目 | 值 |
|:----|:---|
| 实现 | 基于 pmixer/SASRec.pytorch 移植，适配 TIGER 数据格式 |
| 模型结构 | 2层 Transformer → 50d hidden → BCE loss |
| 优化器 | Adam (lr=0.001, betas=(0.9, 0.98)) |
| 训练 | 最大 200 epoch，val NDCG@20 早停（patience=5 次评估），best-val checkpoint |
| 评估 | **全量物品排序** × **全量用户**（与 TA-SID 同人口） |
| 随机性 | 3 个 seed（2025/2026/2027，与 T5 一致），采样器按 seed 播种可复现 |
| GPU | NVIDIA RTX 4090（3 数据集 × 3 seed 总耗时 ~25 min） |

---

## 结果总表（best-val checkpoint，3-seed mean±std）

| Method | Sport N@20 | Sport R@20 | Beauty N@20 | Beauty R@20 | Toys N@20 | Toys R@20 |
|:-------|:---------:|:---------:|:-----------:|:-----------:|:---------:|:---------:|
| **SASRec** | 0.0134±0.0002 | 0.0374±0.0014 | 0.0254±0.0007 | 0.0711±0.0016 | 0.0241±0.0003 | 0.0651±0.0015 |
| Content-only (TIGER 复刻, 768d) | 0.0275±0.0006 | 0.0666±0.0015 | 0.0358±0.0007 | 0.0811±0.0008 | 0.0336±0.0006 | 0.0785±0.0007 |
| **TA-SID (768d)** | **0.0322±0.0004** | **0.0769±0.0011** | **0.0415±0.0009** | **0.0941±0.0016** | **0.0377±0.0007** | **0.0911±0.0020** |
| *TA-SID vs SASRec* | *+139.5%* | *+105.5%* | *+63.2%* | *+32.3%* | *+56.4%* | *+39.9%* |
| *Content-only vs SASRec* | *+105.2%* | *+78.1%* | *+40.9%* | *+14.1%* | *+39.4%* | *+20.6%* |

> 层次关系：**SASRec < Content-only < TA-SID** 在全部 3 数据集 × 6 指标上一致成立。

### 详细指标（SASRec，3-seed mean±std）

| 数据集 | NDCG@5 | NDCG@10 | NDCG@20 | Rec@5 | Rec@10 | Rec@20 |
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Sport | 0.0064±0.0003 | 0.0096±0.0001 | 0.0134±0.0002 | 0.0121±0.0006 | 0.0220±0.0004 | 0.0374±0.0014 |
| Beauty | 0.0121±0.0005 | 0.0183±0.0007 | 0.0254±0.0007 | 0.0234±0.0008 | 0.0429±0.0014 | 0.0711±0.0016 |
| Toys | 0.0127±0.0004 | 0.0184±0.0001 | 0.0241±0.0003 | 0.0245±0.0011 | 0.0423±0.0006 | 0.0651±0.0015 |

### 逐 seed 明细（N@20）

| 数据集 | seed2025 | seed2026 | seed2027 | best epoch (2025/26/27) |
|:-----|:-----:|:-----:|:-----:|:-----|
| Sport | 0.0137 | 0.0135 | 0.0132 | 100 / 20 / 20 |
| Beauty | 0.0256 | 0.0245 | 0.0262 | 120 / 180 / 180 |
| Toys | 0.0237 | 0.0243 | 0.0244 | 160 / 180 / 160 |

---

## 分析

### 对齐 vs 旧口径的差异

v1 的 off-by-one 使 SASRec 整体虚高（N@20 高估 ~16-21%）。v2 对齐后，**TA-SID 相对 SASRec 的优势反而扩大**（Sport +96.3%→+139.5%、Beauty +29.7%→+63.2%；Toys +27.4%→+56.4%，其中 +56.4% 为 v2 对齐后再经 2026-09-09 Toys 主 codes 订正之值，见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)）：预测最后一项（i_L）比预测倒数第二项（i_{L-1}）更难，TA-SID 借助转移信号在此更占优。

### 稀疏度趋势

TA-SID vs SASRec 的提升幅度与数据稀疏度一致：Sport（最稀疏，avg len 8.32）最大 +139.5%，Beauty +63.2%，Toys（最稠密，avg len 8.63）最小 +56.4%。与 TA-SID vs Content-only 的趋势同向。

### 训练行为

| 数据集 | 典型 best epoch | 说明 |
|:-----|:---------:|:-----|
| Sport | 20-100 | 快速收敛，20 后 val 波动，出现过拟合迹象 |
| Toys | 160-180 | 稳定上升后 plateau |
| Beauty | 120-180 | 稳定上升后 plateau |

---

## 文件位置

- 代码：`tools/run_sasrec.py`（v2：对齐 + 全量用户 + 3-seed + 可复现播种）
- 每 seed 结果：`sasrec_output/{Dataset}/seed{seed}_epochs.txt`、`seed{seed}_best.pth`
- 汇总：`sasrec_output/{Dataset}/summary_3seed.txt`（论文引用来源）
- 运行日志：`sasrec_output/sasrec_3seed_all.log`
