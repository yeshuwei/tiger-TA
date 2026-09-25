# E48 Toys 主 codes provenance 修复与主表数字订正（2026-09-09）

> **本文件是 Toys 主表 codes provenance 问题的权威记录 + 数字订正 + 修改索引。**
> 一句话：主表 Toys codes 文件（`Toys_ta_sid_768d_sid.npy`，L0=62）是**孤儿**——无法从任何现存
> ckpt 复现（行级一致率最高仅 29.3%），主表 Toys 0.0367 建立在它上面 → **该数字作废，不作为 Toys
> 正式结果**。已用存活的健康 best_collision（L0=207）重导出 codes 并重跑 T5 3-seed →
> **Toys 主结果订正为 NDCG@20 = 0.0377±0.0007 / R@20 = 0.0911±0.0020**。
> 各文档凡把 0.0367（或 62/256 塌缩）当作 Toys 效果数字处，以本文档订正为准。
>
> 关联：审计脚本 [audit_main_codes_provenance.py](../rqvae/audit_main_codes_provenance.py)；
> 订正 T5 重跑：`model/logs/tiger_ta_sid_768d_toys_rq2024_seed{2025,2026,2027}_eval.log`；
> 既有裁定 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)；
> seed 隔离实锤 [e47_seed_health_isolation.md](e47_seed_health_isolation.md)。

---

## 1. Provenance 审计结论（只读，2026-09-09）

对每个主表 `*_ta_sid_768d_sid.npy`，用全部现存 ckpt（每训练 dir × best_collision/best_loss）
重算 argmin L0 分配，与 codes 文件 L0 列做行级一致率：

| 数据集 | 主 codes 文件 L0 distinct | 现存 best_collision L0 | **最佳行级一致率** | 判定 |
|---|---|---|---|---|
| Beauty | 204 | 204 | **100.00%**（best_collision） | ✅ 干净，出自现存 ckpt |
| Sport | 249 | 249 | **100.00%**（best_collision） | ✅ 干净，出自现存 ckpt |
| **Toys** | **62** | 207（健康）/ 15（best_loss 塌缩） | **29.26%**（best_collision） | ❌ **孤儿** |

Toys codes 的 L0=62 无法由任何一个现存 ckpt 复现（L1/L2 更低，2.3%/0.6%）。其产生时的
checkpoint 已被覆盖（训练 `save_limit=5`，仅存 best_collision/best_loss + 每 50 epoch 断点）。
→ **主表 Toys 0.0367±0.0006 / 0.0873±0.0016 建立在孤儿 codes 上，不能作为 Toys 正式结果。**

> 补充（对 E45/E46/E47 表述的修正）：E45/E46/E47 把 62/256 塌缩标为 "seed2024 塌缩锚点"。
> Provenance 表明幸存 seed2024 best_collision 实际**健康（207/256）**，62 来自被覆盖的 epoch 快照
> （该轮训练确曾塌缩，best_loss=15 是最接近的幸存证据）。故"塌缩是 seed2024 单次训练的系统属性"
> 应弱化为"TA-SID×Toys 在 legacy seed2024 训练的**部分 epoch/快照**上出现 L0 分配崩坏"；塌缩
> 作为**可复现现象**由 E45（seed2025/2026 健康对照）+ E47（ZeroPad/content-only 全健康）刻画，
> 其"码本健康度决定下游"结论不受影响，反而被下述健康 codes 订正数字（0.0367→0.0377）加强。

---

## 2. 数字订正（旧 → 新）

**Toys 主表（TA-SID 768d，3 T5 seed 2025/2026/2027 mean±std，best-val 测试值）：**

| 指标 | 旧（孤儿 codes, L0=62）| **新（健康 codes, L0=207）** |
|---|---|---|
| NDCG@5 | 0.0221±0.0001 | **0.0225±0.0008** |
| NDCG@10 | 0.0290±0.0004 | **0.0294±0.0005** |
| **NDCG@20** | 0.0367±0.0006 | **0.0377±0.0007** |
| Recall@5 | 0.0351±0.0001 | **0.0365±0.0014** |
| Recall@10 | 0.0565±0.0009 | **0.0579±0.0003** |
| **Recall@20** | 0.0873±0.0016 | **0.0911±0.0020** |
| per-seed N@20 | 0.0360/0.0366/0.0375* | 0.0369/0.0383/0.0380 |

> \* paper_main_results.md 旧明细为 0.0371/0.0362/0.0368，与 experiment_results_summary 的
> 0.0360/0.0366/0.0375 不一致（历史文档抄录歧义）。**一律以 eval log 为准**；新数字三处文档统一用
> 0.0369/0.0383/0.0380。

**由此派生的提升幅度（Toys，相对不变的两个 baseline）：**

| 对比 | 指标 | 旧 | **新** |
|---|---|---|---|
| vs Content-only（0.0336±0.0006 / 0.0785±0.0007） | N@20 | +9.2% | **+12.2%** |
| | R@20 | +11.2% | **+16.1%** |
| vs SASRec（0.0241±0.0003 / 0.0651±0.0015） | N@20 | +52.2% | **+56.4%** |
| | R@20 | +34.0% | **+39.9%** |

**N@20 跨数据集排序不变**：Sport 0.0322 < Toys 0.0377 < Beauty 0.0415；TA-SID vs content-only
提升区间由 +9.2~17.1% 变为 **+12.2~17.1%**。

---

## 3. 什么没变（防止连带误改）

- **Beauty / Sport 主表数字不动**：provenance 100% 干净。
- **E45 判(a)（健康 256 ≈ 512，0.0372≈0.0381）不受影响**：健康 seed2025 codes 数字独立、仍成立；
  本订正（0.0377）与 0.0372/0.0381 同侧，反而加强"健康 ≥ 塌缩"。
- **E46 保留项 R1-R7 / E47 ZeroPad/content-only 全 seed 健康**结论不受影响。
- **创新 1 结论（行为融合输入 → RQ-VAE 增益）方向不变且略增强**：Toys 现 +12.2%（原 +9.2%）。
- content-only / SASRec baseline 数字不动（口径已在 §1.5 / SASRec v2 对齐）。

## 4. 作废声明（读者须知）

凡把下列数字当作 **Toys TA-SID 正式结果**的引用一律**作废、改引 §2 新数字**：
- NDCG@20 = 0.0367±0.0006、R@20 = 0.0873±0.0016；
- 提升 +9.2% / +11.2% / +52.2% / +34.0%；
- per-seed 明细 0.0360/0.0366/0.0375 或 0.0371/0.0362/0.0368。

保留不废的**唯一场景**：E43/E44/E45/E46/E47 中把它显式标为"**塌缩（孤儿 codes）实例**"做
健康 vs 塌缩对照时，可保留作为历史塌缩样本，但需带本文件横幅，勿当 Toys 效果数字引用。

## 5. 修改索引（2026-09-09 已落地）

| 文档 | 改动 |
|---|---|
| **[e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)** | 本文件 |
| [experiment_results_summary.md](experiment_results_summary.md) §1.5 | Toys 行换新数字 + 派生 % + 明细；加订正注 |
| [paper_main_results.md](paper_main_results.md) | Toys 全列换新数字 + 提升 % + 明细；加订正注 |
| [paper_draft_zh_method_exp.md](paper_draft_zh_method_exp.md) / [paper_outline_ta_sid.md](paper_outline_ta_sid.md) / [sasrec_baseline_report.md](sasrec_baseline_report.md) / sec54_56 | Toys 主结果换新数字；塌缩对照处加横幅 |
| [e43_adaptive_experiment.md](e43_adaptive_experiment.md) / [e43_summary.md](e43_summary.md) / [e44_five_dataset_summary.md](e44_five_dataset_summary.md) / [e45_toys_multiseed_collapse.md](e45_toys_multiseed_collapse.md) / [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md) / [e47_seed_health_isolation.md](e47_seed_health_isolation.md) / [e48_sfrq_direction_route.md](e48_sfrq_direction_route.md) | 塌缩锚点 0.0367/62 加"孤儿作废"横幅 + 指向本文档 |
| [e48_step0_sinkhorn_gate0.md](e48_step0_sinkhorn_gate0.md) | 已建（Step-0 探针结果，独立于本订正） |

## 6. 复现

```bash
# provenance 审计（CPU，~数分钟）
python rqvae/audit_main_codes_provenance.py
# 健康 codes 重导出（本订正依据）→ data/Toys/Toys_ta_sid_768d_seed2024_sid.npy（L0=207）
# T5 3-seed 重跑（tag rq2024）→ 上表新数字
bash model/run_t5_multiseed.sh Toys Toys_ta_sid_768d_seed2024_sid.npy rq2024 2025 2026 2027
```

## 相关
[[session-2026-08-03-project-state]]（主参考：实验定稿口径）| [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md) | [e47_seed_health_isolation.md](e47_seed_health_isolation.md) | [e45_toys_multiseed_collapse.md](e45_toys_multiseed_collapse.md)
