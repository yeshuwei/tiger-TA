# E49 行为消融结果（命名陷阱安全版，自动生成）

**生成时间**：2026-09-24 07:43:02　**产出脚本**：`tools/e49_summary.py`　**调度**：`rqvae/e49_finalize.sh`（crontab）

> 本表由**逐 lane 显式文件名**解析，**不使用 tag glob**，故不受 P6 段两处命名陷阱影响：
> ① Toys TA-SID 取**健康 codes**（`tiger_ta_sid_768d_toys_rq2024_seed*`，L0=207）而非作废的孤儿 codes；
> ② Toys Shuffled 取实际 tag `shuffled_768d` 的日志而非 `shuffled_ablation`。
> ⚠️ `doc/e49_behavior_ablation_results.md`（P6 段产物）**未经此修正**，其 Toys TA-SID / Toys Shuffled
> 两行应作废，以本表为准。见 `doc/e48_toys_main_codes_provenance_fix.md`。

**进度**：36/36 路（Random 辅助行单 seed，不计入）。

## 1. NDCG@20（3-seed mean±std）

| 数据集 | 变体 | 2025 | 2026 | 2027 | 均值±std | 数据源 |
|:--|:--|:--:|:--:|:--:|:--:|:--|
| Sport | Content-only | 0.0274 | 0.0283 | 0.0269 | **0.0275±0.0007** | 标准 |
| Sport | Shuffled | 0.0262 | 0.0267 | 0.0262 | **0.0264±0.0003** | 标准 |
| Sport | ZeroPad | 0.0286 | 0.0280 | 0.0274 | **0.0280±0.0006** | 标准 |
| Sport | ItemCF | 0.0298 | 0.0296 | 0.0290 | **0.0295±0.0004** | 标准 |
| Sport | Raw-CoOcc | 0.0311 | 0.0306 | 0.0299 | **0.0305±0.0006** | 标准 |
| Sport | TA-SID | 0.0327 | 0.0317 | 0.0322 | **0.0322±0.0005** | 标准 |
| Toys | Content-only | 0.0341 | 0.0338 | 0.0328 | **0.0336±0.0007** | 标准 |
| Toys | Shuffled | 0.0285 | 0.0276 | 0.0292 | **0.0284±0.0008** | 标准 |
| Toys | ZeroPad | 0.0331 | 0.0324 | 0.0348 | **0.0334±0.0012** | 标准 |
| Toys | ItemCF | 0.0354 | 0.0355 | 0.0357 | **0.0355±0.0002** | 标准 |
| Toys | Raw-CoOcc | 0.0342 | 0.0324 | 0.0354 | **0.0340±0.0015** | 标准 |
| Toys | TA-SID | 0.0369 | 0.0383 | 0.0380 | **0.0377±0.0007** | rq2024 健康 codes |

## 2. Recall@20（3-seed mean）

| 数据集 | 变体 | 均值 |
|:--|:--|:--:|
| Sport | Content-only | 0.0666 |
| Sport | Shuffled | 0.0646 |
| Sport | ZeroPad | 0.0676 |
| Sport | ItemCF | 0.0714 |
| Sport | Raw-CoOcc | 0.0731 |
| Sport | TA-SID | 0.0769 |
| Toys | Content-only | 0.0785 |
| Toys | Shuffled | 0.0670 |
| Toys | ZeroPad | 0.0797 |
| Toys | ItemCF | 0.0850 |
| Toys | Raw-CoOcc | 0.0802 |
| Toys | TA-SID | 0.0911 |

## 3. 辅助行（单 seed，不参与显著性比较）

| 数据集 | 变体 | NDCG@20 |
|:--|:--|:--:|
| Sport | Random | 0.0281 |

## 4. 口径说明（历史/作废值）

- 旧单 seed 2025 = 0.0262，现已被 3-seed 均值取代
- 旧单 seed 2025 = 0.0286，现已被 3-seed 均值取代
- 孤儿 codes(L0=62) 0.0367 已作废（e48），本行用健康 codes(207) 0.0377

## 5. 未完成 lane

- 无，全部 3-seed 齐备。
