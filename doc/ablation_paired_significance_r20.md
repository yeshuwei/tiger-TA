# 行为消融主表 · 配对显著性检验（Recall@20）

> 自动生成：`python3 tools/paired_significance_ablation.py --metric R20`（勿手改）
> 数据源复用 `tools/ablation_3datasets_summary.py` 的 LANES；逐 seed 值经 provenance 断言与主表 §1.2 对齐。

## 0. 口径（必读）

配对结构：RQ-VAE 固定 seed=2024（每数据集所有变体共用一份 codes），T5 用同一组 seed 2025/2026/2027 ⇒ **同 seed 两变体之间唯一变量是行为向量**。

- n=3，df=2。配对 t 检验需 **|d_z| > 2.48** 才达 p<0.05。
- ⚠️ 非参数符号检验在 n=3 的双侧最小 p **恒为 0.25**（下界）。故主结论用「逐 seed 一致性 + 效应量」，p 值仅作辅助且**必须与 df 同报**。
- 某层不显著 ≠ 该层为零，只表示 n=3 下无法与 0 区分。

## 1. 配对检验结果（Recall@20）

| 数据集 | 对比 | Δ | t | p (df=2) | d_z | 越界 | 符号检验 p |
|:--|:--|--:|--:|--:|--:|:--:|--:|
| Sport | TA-SID − Content-only | +0.01030 | 5.85 | 0.0280 ✅ | 3.38 | 3/3 | 0.250 |
| Sport | Raw-CoOcc-SVD − Content-only | +0.00647 | 8.82 | 0.0126 ✅ | 5.09 | 3/3 | 0.250 |
| Sport | TA-SID − Raw-CoOcc-SVD | +0.00383 | 3.70 | 0.0660 ³/₃ | 2.14 | 3/3 | 0.250 |
| Sport | ZeroPad − Content-only | +0.00097 | 0.99 | 0.4267 ⚠ | 0.57 | 2/3 | 1.000 |
| Sport | Shuffled − Content-only | -0.00203 | -1.85 | 0.2051 ⚠ | -1.07 | 0/3 | 0.250 |
| Sport | ItemCF-SVD − Content-only | +0.00477 | 6.87 | 0.0205 ✅ | 3.97 | 3/3 | 0.250 |
| Sport | TA-SID − ItemCF-SVD | +0.00553 | 4.25 | 0.0511 ³/₃ | 2.45 | 3/3 | 0.250 |
| Sport | Raw-CoOcc-SVD − ItemCF-SVD | +0.00170 | 4.21 | 0.0521 ³/₃ | 2.43 | 3/3 | 0.250 |
| Toys | TA-SID − Content-only | +0.01257 | 7.50 | 0.0173 ✅ | 4.33 | 3/3 | 0.250 |
| Toys | Raw-CoOcc-SVD − Content-only | +0.00167 | 0.95 | 0.4420 ⚠ | 0.55 | 2/3 | 1.000 |
| Toys | TA-SID − Raw-CoOcc-SVD | +0.01090 | 5.47 | 0.0318 ✅ | 3.16 | 3/3 | 0.250 |
| Toys | ZeroPad − Content-only | +0.00120 | 0.69 | 0.5614 ⚠ | 0.40 | 2/3 | 1.000 |
| Toys | Shuffled − Content-only | -0.01157 | -10.78 | 0.0085 ✅ | -6.22 | 0/3 | 0.250 |
| Toys | ItemCF-SVD − Content-only | +0.00643 | 3.77 | 0.0638 ³/₃ | 2.18 | 3/3 | 0.250 |
| Toys | TA-SID − ItemCF-SVD | +0.00613 | 4.60 | 0.0442 ✅ | 2.65 | 3/3 | 0.250 |
| Toys | Raw-CoOcc-SVD − ItemCF-SVD | -0.00477 | -1.61 | 0.2484 ⚠ | -0.93 | 0/3 | 0.250 |
| Beauty | TA-SID − Content-only | +0.01300 | 12.85 | 0.0060 ✅ | 7.42 | 3/3 | 0.250 |
| Beauty | Raw-CoOcc-SVD − Content-only | +0.00577 | 2.65 | 0.1175 ³/₃ | 1.53 | 3/3 | 0.250 |
| Beauty | TA-SID − Raw-CoOcc-SVD | +0.00723 | 5.10 | 0.0363 ✅ | 2.95 | 3/3 | 0.250 |
| Beauty | ZeroPad − Content-only | +0.00300 | 1.73 | 0.2266 ³/₃ | 1.00 | 3/3 | 0.250 |
| Beauty | Shuffled − Content-only | -0.00380 | -6.90 | 0.0204 ✅ | -3.98 | 0/3 | 0.250 |
| Beauty | ItemCF-SVD − Content-only | +0.00747 | 7.56 | 0.0170 ✅ | 4.37 | 3/3 | 0.250 |
| Beauty | TA-SID − ItemCF-SVD | +0.00553 | 3.84 | 0.0616 ³/₃ | 2.22 | 3/3 | 0.250 |
| Beauty | Raw-CoOcc-SVD − ItemCF-SVD | -0.00170 | -0.90 | 0.4639 ⚠ | -0.52 | 1/3 | 1.000 |

### 1.1 逐 seed 明细

| 数据集 | 对比 | s2025 | s2026 | s2027 | Δ/合并std |
|:--|:--|--:|--:|--:|--:|
| Sport | TA-SID − Content-only | +0.01170 | +0.00680 | +0.01240 | 6.28× |
| Sport | Raw-CoOcc-SVD − Content-only | +0.00720 | +0.00500 | +0.00720 | 4.46× |
| Sport | TA-SID − Raw-CoOcc-SVD | +0.00450 | +0.00180 | +0.00520 | 3.32× |
| Sport | ZeroPad − Content-only | +0.00240 | -0.00090 | +0.00140 | 0.61× |
| Sport | Shuffled − Content-only | -0.00210 | -0.00390 | -0.00010 | -1.53× |
| Sport | ItemCF-SVD − Content-only | +0.00600 | +0.00360 | +0.00470 | 2.77× |
| Sport | TA-SID − ItemCF-SVD | +0.00570 | +0.00320 | +0.00770 | 3.75× |
| Sport | Raw-CoOcc-SVD − ItemCF-SVD | +0.00120 | +0.00140 | +0.00250 | 1.35× |
| Toys | TA-SID − Content-only | +0.00960 | +0.01270 | +0.01540 | 8.09× |
| Toys | Raw-CoOcc-SVD − Content-only | +0.00220 | -0.00160 | +0.00440 | 0.76× |
| Toys | TA-SID − Raw-CoOcc-SVD | +0.00740 | +0.01430 | +0.01100 | 4.32× |
| Toys | ZeroPad − Content-only | +0.00050 | -0.00140 | +0.00450 | 0.61× |
| Toys | Shuffled − Content-only | -0.01210 | -0.01310 | -0.00950 | -9.34× |
| Toys | ItemCF-SVD − Content-only | +0.00320 | +0.00900 | +0.00710 | 3.61× |
| Toys | TA-SID − ItemCF-SVD | +0.00640 | +0.00370 | +0.00830 | 2.81× |
| Toys | Raw-CoOcc-SVD − ItemCF-SVD | -0.00100 | -0.01060 | -0.00270 | -1.79× |
| Beauty | TA-SID − Content-only | +0.01290 | +0.01480 | +0.01130 | 8.62× |
| Beauty | Raw-CoOcc-SVD − Content-only | +0.00330 | +0.01010 | +0.00390 | 2.39× |
| Beauty | TA-SID − Raw-CoOcc-SVD | +0.00960 | +0.00470 | +0.00740 | 2.70× |
| Beauty | ZeroPad − Content-only | +0.00040 | +0.00230 | +0.00630 | 1.63× |
| Beauty | Shuffled − Content-only | -0.00480 | -0.00290 | -0.00370 | -5.27× |
| Beauty | ItemCF-SVD − Content-only | +0.00550 | +0.00830 | +0.00860 | 8.58× |
| Beauty | TA-SID − ItemCF-SVD | +0.00740 | +0.00650 | +0.00270 | 3.80× |
| Beauty | Raw-CoOcc-SVD − ItemCF-SVD | -0.00220 | +0.00180 | -0.00470 | -0.71× |

## 2. 消融阶梯分解

总增益 = 共现层（`Raw-CoOcc − Content`）+ PPMI 层（`TA-SID − Raw-CoOcc`）

| 数据集 | 总增益 | 共现层 | PPMI 层 |
|:--|--:|--:|--:|
| Sport | +0.01030 | +0.00647 ✅ | +0.00383 同向未过线 |
| Toys | +0.01257 | +0.00167 ⚠2/3 | +0.01090 ✅ |
| Beauty | +0.01300 | +0.00577 同向未过线 | +0.00723 ✅ |

## 3. 写作红线（承接 seq_baselines_significance.md §4）

- **禁止**写 "statistically significant (p < 0.05)" 而不交代 n=3 与 df=2。
- **禁止**只报参数检验、不提非参数下界 0.25。
- 推荐首选表述："consistent across all 3 seeds (no crossover); gap = X, which is Y× the pooled between-seed std"。
- 若报 p：必须连 df 与逐 seed 原值一起报，并注明正态性假设无法在 n=3 下检验。
- **不得**因某一层不显著就写"该层贡献为零"——n=3 的检出力不足以支持该断言。
