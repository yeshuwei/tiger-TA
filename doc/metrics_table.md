# TA-SID 主实验指标汇总（速查版）
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

> 📖 **完整版见** [auto_metrics_table.md](auto_metrics_table.md)（含 Track B、SID 质量、维度对比、种子鲁棒性等全部分析）

---

## 核心结果（seed=2025, 768d）

| Dataset | Baseline N@20 | TA-SID N@20 | 提升 |
|:--------|:-------------:|:-----------:|:----:|
| Beauty | 0.0379 | **0.0418** | **+10.3%** |
| Sport | 0.0207 | **0.0327** | **+58.0%** 🎯 |
| Toys | 0.0305 | **0.0360** | **+18.0%** |

---

## 全版本速查表

| Dataset | Variant | Seed | NDCG@20 | vs Baseline |
|:--------|:--------|:----:|:-------:|:-----------:|
| Beauty | Baseline | 2025 | 0.0379 | — |
| Beauty | TA-SID 768d | 2025 | **0.0418** | +10.3% |
| Beauty | TA-SID 768d | 42 | 0.0404 | +6.6% |
| Beauty | TA-SID 4096d (全量) | 2025 | 0.0382 | +0.8% |
| Beauty | TA-SID 4096d (验证) | 2025 | 0.0368 | -2.9% |
| Sport | Baseline | 2025 | 0.0207 | — |
| Sport | **TA-SID 768d** | **2025** | **0.0327** | **+58.0%** |
| Sport | TA-SID 768d | 42 | 0.0315 | +52.2% |
| Sport | TA-SID 4096d | 2025 | 0.0276 | +33.3% |
| Sport | TA-SID 4096d | 42 | 0.0270 | +30.4% |
| Sport | **Track B-1** (λ=0.1) | 2025 | 0.03046 | +47.1% ❓ |
| Sport | Track B-2 (λ=0.01) | 2025 | 🔄 | — |
| Toys | Baseline | 2025 | 0.0305 | — |
| Toys | **TA-SID 768d** | **2025** | **0.0360** | **+18.0%** |
| Toys | TA-SID 768d | 42 | 0.0348 | +14.1% |
| Toys | TA-SID 4096d | 2025 | 0.0317 | +3.9% |
| Toys | TA-SID 4096d | 42 | 0.0312 | +2.3% |
