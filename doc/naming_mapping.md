# TA-SID 文件命名映射存档（2026-08-03 重命名）

> 本次按 [naming_convention.md](naming_convention.md) 统一**正式实验**文件命名。
> **历史日志类文档（ai-log.md / TA-SID_PLAN.md / scheme_a_summary.md / track_b_experiment_log.md / paper_audit_critique.md）保留旧名作历史记录**，如需定位文件以本表为准。

## 1. TA-SID 主实验（ta_sid_768d）

### T5 ckpt

| 数据集/seed | 旧名 | 新名 |
|:--|:--|:--|
| Beauty / 2025 | `model/ckpt/tiger_ta_sid.pth` | `model/ckpt/tiger_ta_sid_768d_beauty_seed2025.pth` |
| Sport / 2025 | `model/ckpt/tiger_ta_sid_sport_768d.pth` | `model/ckpt/tiger_ta_sid_768d_sport_seed2025.pth` |
| Toys / 2025 | `model/ckpt/tiger_ta_sid_Toys_768d.pth` | `model/ckpt/tiger_ta_sid_768d_toys_seed2025.pth` |
| 全部 / 2026, 2027 | `tiger_ta_sid_768d_{ds}_seed{seed}.pth` | ✅ 已符合规范 |

### SID code

| 数据集 | 旧名 | 新名 |
|:--|:--|:--|
| Sport | `data/Sport/Sport_sid_codes_768d.npy` | `data/Sport/Sport_ta_sid_768d_sid.npy` |
| Beauty | `data/Beauty/Beauty_ta_sid.npy` | `data/Beauty/Beauty_ta_sid_768d_sid.npy` |
| Toys | `data/Toys/Toys_sid_codes_768d.npy` | `data/Toys/Toys_ta_sid_768d_sid.npy` |

### eval log

| 数据集 | 状态 |
|:--|:--|
| Sport/Beauty/Toys / seed2025 | ✅ 已符合规范（`{Ds}_ta_sid_768d_seed2025_eval.log`，2026-08-03 补齐） |
| Sport/Beauty/Toys / seed2026, 2027 | ✅ 已符合规范 |

## 2. Content-only baseline（content768d_baseline）

### T5 ckpt（补 `_seed2025` 后缀）

| 数据集 | 旧名 | 新名 |
|:--|:--|:--|
| Sport / 2025 | `model/ckpt/tiger_content768d_baseline_sport.pth` | `model/ckpt/tiger_content768d_baseline_sport_seed2025.pth` |
| Beauty / 2025 | `model/ckpt/tiger_content768d_baseline_beauty.pth` | `model/ckpt/tiger_content768d_baseline_beauty_seed2025.pth` |
| Toys / 2025 | `model/ckpt/tiger_content768d_baseline_toys.pth` | `model/ckpt/tiger_content768d_baseline_toys_seed2025.pth` |
| 全部 / 2026, 2027 | `tiger_content768d_baseline_{ds}_seed{seed}.pth` | ✅ 已符合规范 |

### eval log（补 `_seed2025` 后缀）

| 数据集 | 旧名 | 新名 |
|:--|:--|:--|
| Sport / 2025 | `model/logs/Sport_content768d_baseline_eval.log` | `model/logs/Sport_content768d_baseline_seed2025_eval.log` |
| Beauty / 2025 | `model/logs/Beauty_content768d_baseline_eval.log` | `model/logs/Beauty_content768d_baseline_seed2025_eval.log` |
| Toys / 2025 | `model/logs/Toys_content768d_baseline_eval.log` | `model/logs/Toys_content768d_baseline_seed2025_eval.log` |
| 全部 / 2026, 2027 | `{Ds}_content768d_baseline_seed{seed}_eval.log` | ✅ 已符合规范 |

### SID code（baseline）

`data/{Ds}/{Ds}_content768d_baseline_sid.npy` ✅ 已符合规范，未改动。

## 3. 历史 / 弃用文件（不重命名，仅索引）

| 类别 | 文件 | 状态 |
|:--|:--|:--|
| seed42 旧口径 | `tiger_ta_sid_{ds}_seed42.pth`、`tiger_ta_sid_{ds}.pth`(7-15) | 历史归档 |
| 4096d 对照 | `tiger_ta_sid_{ds}_4096d.pth`、`{Ds}_ta_sid_4096d*.npy` | 维度对照 |
| Track B（L_trans 无效） | `tiger_ta_sid_sport_track_b*.pth`、`Sport_track_b_sid_codes_768d*.npy` | 已证无效 |
| 消融 | `tiger_{random,shuffled,zeropad}_ablation_sport.pth`、`{Ds}_{tag}_ablation_sid.npy` | 消融 |
| Random SID | `{ds}_random_sid.pth`、`{Ds}_random_sid.npy` | 码质量消融 |
| TIGER 原始 baseline | `tiger.pth`、`sports.pth`、`toys.pth`、`{Ds}_t5_rqvae.npy` | 旧管线参考 |
| 一次性脚本 | `rqvae/generate_code_*.py`（含绝对路径+时间戳） | 已用/一次性 |

## 4. 同步的脚本与文档（2026-08-03）

- ✅ `rqvae/run_multiseed.py`（METHODS code 路径 → 新名；eval 跳过逻辑补 seed2025）
- ✅ `rqvae/run_content_baseline.py`（eval_log → 补 `_seed2025`）
- ✅ `tools/sid_consistency_contrast.py`（ta_code → 新名）
- ✅ `tools/sid_quality.py`（npy_file / log → 新名）
- ✅ `tools/eval_invalid_rate.py`（ckpt / code → 新名）
- ✅ `tools/extract_metrics.py`（日志名 → 新名）
- ✅ `tiger_migration.sh`（check_file → 新名）
- ✅ `doc/experiment_dimension_manifest.md`（文件名索引 → 新名）
- ✅ `doc/naming_convention.md`（§3 对照表 → 标注已完成）
