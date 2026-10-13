# 行为消融主表 · 配对显著性检验（NDCG@20）

> 自动生成：`python3 tools/paired_significance_ablation.py --metric N20`（勿手改）
> 数据源复用 `tools/ablation_3datasets_summary.py` 的 LANES；逐 seed 值经 provenance 断言与主表 §1.1 对齐。

## 0. 口径（必读）

配对结构：RQ-VAE 固定 seed=2024（每数据集所有变体共用一份 codes），T5 用同一组 seed 2025/2026/2027 ⇒ **同 seed 两变体之间唯一变量是行为向量**。

- n=3，df=2。配对 t 检验需 **|d_z| > 2.48** 才达 p<0.05。
- ⚠️ 非参数符号检验在 n=3 的双侧最小 p **恒为 0.25**（下界）。故主结论用「逐 seed 一致性 + 效应量」，p 值仅作辅助且**必须与 df 同报**。
- 某层不显著 ≠ 该层为零，只表示 n=3 下无法与 0 区分。

## 1. 配对检验结果（NDCG@20）

| 数据集 | 对比 | Δ | t | p (df=2) | d_z | 越界 | 符号检验 p |
|:--|:--|--:|--:|--:|--:|:--:|--:|
| Sport | TA-SID − Content-only | +0.00467 | 7.37 | 0.0179 ✅ | 4.25 | 3/3 | 0.250 |
| Sport | Raw-CoOcc-SVD − Content-only | +0.00300 | 7.42 | 0.0177 ✅ | 4.29 | 3/3 | 0.250 |
| Sport | TA-SID − Raw-CoOcc-SVD | +0.00167 | 4.79 | 0.0409 ✅ | 2.77 | 3/3 | 0.250 |
| Sport | ZeroPad − Content-only | +0.00047 | 1.08 | 0.3942 ⚠ | 0.62 | 2/3 | 1.000 |
| Sport | Shuffled − Content-only | -0.00117 | -4.48 | 0.0464 ✅ | -2.59 | 0/3 | 0.250 |
| Sport | ItemCF-SVD − Content-only | +0.00193 | 5.89 | 0.0276 ✅ | 3.40 | 3/3 | 0.250 |
| Sport | TA-SID − ItemCF-SVD | +0.00273 | 8.33 | 0.0141 ✅ | 4.81 | 3/3 | 0.250 |
| Sport | Raw-CoOcc-SVD − ItemCF-SVD | +0.00107 | 8.88 | 0.0125 ✅ | 5.12 | 3/3 | 0.250 |
| Toys | TA-SID − Content-only | +0.00417 | 5.85 | 0.0280 ✅ | 3.38 | 3/3 | 0.250 |
| Toys | Raw-CoOcc-SVD − Content-only | +0.00043 | 0.37 | 0.7460 ⚠ | 0.21 | 2/3 | 1.000 |
| Toys | TA-SID − Raw-CoOcc-SVD | +0.00373 | 3.44 | 0.0749 ³/₃ | 1.99 | 3/3 | 0.250 |
| Toys | ZeroPad − Content-only | -0.00013 | -0.12 | 0.9125 ⚠ | -0.07 | 1/3 | 1.000 |
| Toys | Shuffled − Content-only | -0.00513 | -6.53 | 0.0227 ✅ | -3.77 | 0/3 | 0.250 |
| Toys | ItemCF-SVD − Content-only | +0.00197 | 4.09 | 0.0549 ³/₃ | 2.36 | 3/3 | 0.250 |
| Toys | TA-SID − ItemCF-SVD | +0.00220 | 5.81 | 0.0284 ✅ | 3.35 | 3/3 | 0.250 |
| Toys | Raw-CoOcc-SVD − ItemCF-SVD | -0.00153 | -1.86 | 0.2043 ⚠ | -1.07 | 0/3 | 0.250 |
| Beauty | TA-SID − Content-only | +0.00573 | 7.31 | 0.0182 ✅ | 4.22 | 3/3 | 0.250 |
| Beauty | Raw-CoOcc-SVD − Content-only | +0.00293 | 2.82 | 0.1063 ³/₃ | 1.63 | 3/3 | 0.250 |
| Beauty | TA-SID − Raw-CoOcc-SVD | +0.00280 | 5.46 | 0.0320 ✅ | 3.15 | 3/3 | 0.250 |
| Beauty | ZeroPad − Content-only | +0.00160 | 2.56 | 0.1245 ³/₃ | 1.48 | 3/3 | 0.250 |
| Beauty | Shuffled − Content-only | -0.00123 | -1.98 | 0.1862 ⚠ | -1.14 | 1/3 | 0.500 |
| Beauty | ItemCF-SVD − Content-only | +0.00333 | 3.44 | 0.0753 ³/₃ | 1.98 | 3/3 | 0.250 |
| Beauty | TA-SID − ItemCF-SVD | +0.00240 | 2.09 | 0.1722 ³/₃ | 1.20 | 3/3 | 0.250 |
| Beauty | Raw-CoOcc-SVD − ItemCF-SVD | -0.00040 | -0.45 | 0.6991 ⚠ | -0.26 | 1/3 | 1.000 |

### 1.1 逐 seed 明细

| 数据集 | 对比 | s2025 | s2026 | s2027 | Δ/合并std |
|:--|:--|--:|--:|--:|--:|
| Sport | TA-SID − Content-only | +0.00530 | +0.00340 | +0.00530 | 7.60× |
| Sport | Raw-CoOcc-SVD − Content-only | +0.00370 | +0.00230 | +0.00300 | 4.56× |
| Sport | TA-SID − Raw-CoOcc-SVD | +0.00160 | +0.00110 | +0.00230 | 3.01× |
| Sport | ZeroPad − Content-only | +0.00120 | -0.00030 | +0.00050 | 0.71× |
| Sport | Shuffled − Content-only | -0.00120 | -0.00160 | -0.00070 | -2.15× |
| Sport | ItemCF-SVD − Content-only | +0.00240 | +0.00130 | +0.00210 | 3.32× |
| Sport | TA-SID − ItemCF-SVD | +0.00290 | +0.00210 | +0.00320 | 5.94× |
| Sport | Raw-CoOcc-SVD − ItemCF-SVD | +0.00130 | +0.00100 | +0.00090 | 2.06× |
| Toys | TA-SID − Content-only | +0.00280 | +0.00450 | +0.00520 | 5.87× |
| Toys | Raw-CoOcc-SVD − Content-only | +0.00010 | -0.00140 | +0.00260 | 0.37× |
| Toys | TA-SID − Raw-CoOcc-SVD | +0.00270 | +0.00590 | +0.00260 | 3.14× |
| Toys | ZeroPad − Content-only | -0.00100 | -0.00140 | +0.00200 | -0.13× |
| Toys | Shuffled − Content-only | -0.00560 | -0.00620 | -0.00360 | -6.90× |
| Toys | ItemCF-SVD − Content-only | +0.00130 | +0.00170 | +0.00290 | 3.99× |
| Toys | TA-SID − ItemCF-SVD | +0.00150 | +0.00280 | +0.00230 | 4.13× |
| Toys | Raw-CoOcc-SVD − ItemCF-SVD | -0.00120 | -0.00310 | -0.00030 | -1.43× |
| Beauty | TA-SID − Content-only | +0.00500 | +0.00730 | +0.00490 | 5.70× |
| Beauty | Raw-CoOcc-SVD − Content-only | +0.00120 | +0.00480 | +0.00280 | 2.95× |
| Beauty | TA-SID − Raw-CoOcc-SVD | +0.00380 | +0.00250 | +0.00210 | 2.52× |
| Beauty | ZeroPad − Content-only | +0.00040 | +0.00190 | +0.00250 | 2.32× |
| Beauty | Shuffled − Content-only | -0.00200 | +0.00000 | -0.00170 | -1.49× |
| Beauty | ItemCF-SVD − Content-only | +0.00150 | +0.00370 | +0.00480 | 3.61× |
| Beauty | TA-SID − ItemCF-SVD | +0.00350 | +0.00360 | +0.00010 | 2.28× |
| Beauty | Raw-CoOcc-SVD − ItemCF-SVD | -0.00030 | +0.00110 | -0.00200 | -0.39× |

## 2. 消融阶梯分解

总增益 = 共现层（`Raw-CoOcc − Content`）+ PPMI 层（`TA-SID − Raw-CoOcc`）

| 数据集 | 总增益 | 共现层 | PPMI 层 |
|:--|--:|--:|--:|
| Sport | +0.00467 | +0.00300 ✅ | +0.00167 ✅ |
| Toys | +0.00417 | +0.00043 ⚠2/3 | +0.00373 同向未过线 |
| Beauty | +0.00573 | +0.00293 同向未过线 | +0.00280 ✅ |

## 3. 写作红线（承接 seq_baselines_significance.md §4）

- **禁止**写 "statistically significant (p < 0.05)" 而不交代 n=3 与 df=2。
- **禁止**只报参数检验、不提非参数下界 0.25。
- 推荐首选表述："consistent across all 3 seeds (no crossover); gap = X, which is Y× the pooled between-seed std"。
- 若报 p：必须连 df 与逐 seed 原值一起报，并注明正态性假设无法在 n=3 下检验。
- **不得**因某一层不显著就写"该层贡献为零"——n=3 的检出力不足以支持该断言。
