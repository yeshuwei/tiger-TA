# 配对显著性检验与 std 口径对账

> 自动生成：`python3 tools/paired_significance.py`（勿手改）
> 逐 seed N@20 全部经 provenance 断言核对（`doc/paper_main_results.md` §3）。

## 0. 结论速览

- 逐 seed 数据与已发布值**完全一致**（9/9 provenance 断言通过）。
- 稳健性主张基于 **seed 一致性与效应量**，而非 p 值（理由见 §4）。

## 1. std 口径对账（必读）

仓库内同一量存在两套标准差口径，**并列展示会自相矛盾**：

| 出处 | 口径 |
|:--|:--|
| `sasrec_output/*/summary_3seed.txt`（由 `np.std` 默认值写出） | **ddof=0**（总体标准差） |
| `doc/ablation_main_table_3datasets.md`、图 A、`doc/paper_main_results.md` §1/§3 | **ddof=1**（样本标准差） |

> **注**：`doc/paper_main_results.md` §3 的 std 列（如 Sport SASRec `0.0002`）实为 ddof=0 的舍入结果（ddof=1 会得 `0.0003`），而图 A 中同一格记为 `0.0007`/`0.0009` 等为 ddof=1。**两者需统一。**

| 数据集 | 方法 | 指标 | 逐 seed | mean | std(ddof=0) | std(ddof=1) | 四位小数一致? |
|:--|:--|:--|:--|--:|--:|--:|:--:|
| Sport | SASRec | N20 | 0.0137/0.0135/0.0132 | 0.0134 | 0.000205 | 0.000252 | ❌ |
| Sport | SASRec | R20 | 0.0393/0.0367/0.0362 | 0.0374 | 0.001362 | 0.001668 | ❌ |
| Sport | Content-only | N20 | 0.0274/0.0283/0.0269 | 0.0275 | 0.000579 | 0.000709 | ❌ |
| Sport | Content-only | R20 | 0.0664/0.0686/0.0649 | 0.0666 | 0.001520 | 0.001861 | ❌ |
| Sport | TA-SID | N20 | 0.0327/0.0317/0.0322 | 0.0322 | 0.000408 | 0.000500 | ❌ |
| Sport | TA-SID | R20 | 0.0781/0.0754/0.0773 | 0.0769 | 0.001132 | 0.001387 | ❌ |
| Sport | GRU4Rec | N20 | 0.0126/0.0127/0.0133 | 0.0129 | 0.000304 | 0.000372 | ❌ |
| Sport | GRU4Rec | R20 | 0.0279/0.0280/0.0286 | 0.0282 | 0.000292 | 0.000358 | ❌ |
| Sport | BERT4Rec | N20 | 0.0129/0.0113/0.0112 | 0.0118 | 0.000766 | 0.000938 | ❌ |
| Sport | BERT4Rec | R20 | 0.0312/0.0272/0.0269 | 0.0284 | 0.001963 | 0.002404 | ❌ |
| Toys | SASRec | N20 | 0.0237/0.0243/0.0244 | 0.0241 | 0.000330 | 0.000404 | ❌ |
| Toys | SASRec | R20 | 0.0630/0.0664/0.0661 | 0.0651 | 0.001522 | 0.001864 | ❌ |
| Toys | Content-only | N20 | 0.0341/0.0338/0.0328 | 0.0336 | 0.000556 | 0.000681 | ❌ |
| Toys | Content-only | R20 | 0.0795/0.0784/0.0777 | 0.0785 | 0.000741 | 0.000907 | ❌ |
| Toys | TA-SID | N20 | 0.0369/0.0383/0.0380 | 0.0377 | 0.000602 | 0.000737 | ❌ |
| Toys | TA-SID | R20 | 0.0891/0.0911/0.0931 | 0.0911 | 0.001633 | 0.002000 | ❌ |
| Toys | GRU4Rec | N20 | 0.0194/0.0190/0.0191 | 0.0192 | 0.000188 | 0.000230 | ✅ |
| Toys | GRU4Rec | R20 | 0.0402/0.0382/0.0399 | 0.0394 | 0.000883 | 0.001082 | ❌ |
| Toys | BERT4Rec | N20 | 0.0125/0.0127/0.0144 | 0.0132 | 0.000847 | 0.001037 | ❌ |
| Toys | BERT4Rec | R20 | 0.0302/0.0304/0.0333 | 0.0313 | 0.001411 | 0.001728 | ❌ |
| Beauty | SASRec | N20 | 0.0256/0.0245/0.0262 | 0.0254 | 0.000723 | 0.000885 | ❌ |
| Beauty | SASRec | R20 | 0.0711/0.0692/0.0731 | 0.0711 | 0.001570 | 0.001923 | ❌ |
| Beauty | Content-only | N20 | 0.0368/0.0352/0.0354 | 0.0358 | 0.000712 | 0.000872 | ❌ |
| Beauty | Content-only | R20 | 0.0822/0.0805/0.0806 | 0.0811 | 0.000779 | 0.000954 | ❌ |
| Beauty | TA-SID | N20 | 0.0418/0.0425/0.0403 | 0.0415 | 0.000918 | 0.001124 | ❌ |
| Beauty | TA-SID | R20 | 0.0951/0.0953/0.0919 | 0.0941 | 0.001558 | 0.001908 | ❌ |
| Beauty | GRU4Rec | N20 | 0.0319/0.0312/0.0338 | 0.0323 | 0.001093 | 0.001339 | ❌ |
| Beauty | GRU4Rec | R20 | 0.0611/0.0596/0.0630 | 0.0612 | 0.001390 | 0.001703 | ❌ |
| Beauty | BERT4Rec | N20 | 0.0230/0.0240/0.0229 | 0.0233 | 0.000507 | 0.000620 | ❌ |
| Beauty | BERT4Rec | R20 | 0.0533/0.0537/0.0547 | 0.0539 | 0.000573 | 0.000702 | ❌ |

四位小数下即不同的格子：**29 / 30**。

**处置建议**：论文正文/图表统一采 **ddof=1**（样本标准差，与消融表、图 A 现有口径一致），逐 seed 产物文件保持 ddof=0 不动（逐位 diff 门禁要求），在 §5.1 实验设置加一句 "standard deviation is computed with Bessel's correction (n−1)"。

## 2. 配对显著性检验（n=3, df=2）

| 数据集 | 对比 | Δ (mean) | t | p (paired t) | p (Wilcoxon) | p (符号检验) | Cohen's d_z | 越界 seed |
|:--|:--|--:|--:|--:|--:|--:|--:|:--:|
| Sport | TA-SID − SASRec | +0.0188 | 74.17 | 0.000 | 0.250 | 0.250 | 42.82 | 0/3 |
| Sport | TA-SID − Content-only | +0.0047 | 7.37 | 0.018 | 0.250 | 0.250 | 4.25 | 0/3 |
| Sport | TA-SID − GRU4Rec | +0.0193 | 52.35 | 0.000 | 0.250 | 0.250 | 30.22 | 0/3 |
| Sport | Content-only − GRU4Rec | +0.0147 | 24.99 | 0.002 | 0.250 | 0.250 | 14.43 | 0/3 |
| Sport | TA-SID − BERT4Rec | +0.0204 | 60.19 | 0.000 | 0.250 | 0.250 | 34.75 | 0/3 |
| Sport | Content-only − BERT4Rec | +0.0157 | 22.07 | 0.002 | 0.250 | 0.250 | 12.74 | 0/3 |
| Toys | TA-SID − SASRec | +0.0136 | 66.22 | 0.000 | 0.250 | 0.250 | 38.23 | 0/3 |
| Toys | TA-SID − Content-only | +0.0042 | 5.85 | 0.028 | 0.250 | 0.250 | 3.38 | 0/3 |
| Toys | TA-SID − GRU4Rec | +0.0186 | 33.26 | 0.001 | 0.250 | 0.250 | 19.20 | 0/3 |
| Toys | Content-only − GRU4Rec | +0.0144 | 42.60 | 0.001 | 0.250 | 0.250 | 24.60 | 0/3 |
| Toys | TA-SID − BERT4Rec | +0.0245 | 43.40 | 0.001 | 0.250 | 0.250 | 25.06 | 0/3 |
| Toys | Content-only − BERT4Rec | +0.0204 | 20.55 | 0.002 | 0.250 | 0.250 | 11.86 | 0/3 |
| Beauty | TA-SID − SASRec | +0.0161 | 14.11 | 0.005 | 0.250 | 0.250 | 8.15 | 0/3 |
| Beauty | TA-SID − Content-only | +0.0057 | 7.31 | 0.018 | 0.250 | 0.250 | 4.22 | 0/3 |
| Beauty | TA-SID − GRU4Rec | +0.0092 | 6.50 | 0.023 | 0.250 | 0.250 | 3.75 | 0/3 |
| Beauty | Content-only − GRU4Rec | +0.0035 | 3.59 | 0.070 | 0.250 | 0.250 | 2.07 | 0/3 |
| Beauty | TA-SID − BERT4Rec | +0.0182 | 44.24 | 0.001 | 0.250 | 0.250 | 25.54 | 0/3 |
| Beauty | Content-only − BERT4Rec | +0.0125 | 16.69 | 0.004 | 0.250 | 0.250 | 9.64 | 0/3 |

### 2.1 逐 seed 明细

| 数据集 | 对比 | seed2025 | seed2026 | seed2027 | 效应量(Δ / 合并std) |
|:--|:--|--:|--:|--:|--:|
| Sport | TA-SID − SASRec | +0.0190 | +0.0182 | +0.0190 | 47.38× |
| Sport | TA-SID − Content-only | +0.0053 | +0.0034 | +0.0053 | 7.60× |
| Sport | TA-SID − GRU4Rec | +0.0201 | +0.0190 | +0.0189 | 43.83× |
| Sport | Content-only − GRU4Rec | +0.0148 | +0.0156 | +0.0136 | 25.87× |
| Sport | TA-SID − BERT4Rec | +0.0198 | +0.0204 | +0.0210 | 27.15× |
| Sport | Content-only − BERT4Rec | +0.0145 | +0.0170 | +0.0157 | 18.93× |
| Toys | TA-SID − SASRec | +0.0132 | +0.0140 | +0.0136 | 22.91× |
| Toys | TA-SID − Content-only | +0.0028 | +0.0045 | +0.0052 | 5.87× |
| Toys | TA-SID − GRU4Rec | +0.0175 | +0.0193 | +0.0189 | 34.02× |
| Toys | Content-only − GRU4Rec | +0.0147 | +0.0148 | +0.0137 | 28.36× |
| Toys | TA-SID − BERT4Rec | +0.0244 | +0.0256 | +0.0236 | 27.26× |
| Toys | Content-only − BERT4Rec | +0.0216 | +0.0211 | +0.0184 | 23.21× |
| Beauty | TA-SID − SASRec | +0.0162 | +0.0180 | +0.0141 | 15.92× |
| Beauty | TA-SID − Content-only | +0.0050 | +0.0073 | +0.0049 | 5.70× |
| Beauty | TA-SID − GRU4Rec | +0.0099 | +0.0113 | +0.0065 | 7.48× |
| Beauty | Content-only − GRU4Rec | +0.0049 | +0.0040 | +0.0016 | 3.11× |
| Beauty | TA-SID − BERT4Rec | +0.0188 | +0.0185 | +0.0174 | 20.10× |
| Beauty | Content-only − BERT4Rec | +0.0138 | +0.0112 | +0.0125 | 16.54× |

## 3. 写作口径（可直接进论文）

可用表述：

- **Sport: TA-SID vs SASRec** — gap = +0.0188 NDCG@20，是方法间合并 seed std 的 47.4 倍，3/3 seed 同向，零交叉。
- **Sport: TA-SID vs Content-only** — gap = +0.0047 NDCG@20，是方法间合并 seed std 的 7.6 倍，3/3 seed 同向，零交叉。
- **Sport: TA-SID vs GRU4Rec** — gap = +0.0193 NDCG@20，是方法间合并 seed std 的 43.8 倍，3/3 seed 同向，零交叉。
- **Sport: Content-only vs GRU4Rec** — gap = +0.0147 NDCG@20，是方法间合并 seed std 的 25.9 倍，3/3 seed 同向，零交叉。
- **Sport: TA-SID vs BERT4Rec** — gap = +0.0204 NDCG@20，是方法间合并 seed std 的 27.2 倍，3/3 seed 同向，零交叉。
- **Sport: Content-only vs BERT4Rec** — gap = +0.0157 NDCG@20，是方法间合并 seed std 的 18.9 倍，3/3 seed 同向，零交叉。
- **Toys: TA-SID vs SASRec** — gap = +0.0136 NDCG@20，是方法间合并 seed std 的 22.9 倍，3/3 seed 同向，零交叉。
- **Toys: TA-SID vs Content-only** — gap = +0.0042 NDCG@20，是方法间合并 seed std 的 5.9 倍，3/3 seed 同向，零交叉。
- **Toys: TA-SID vs GRU4Rec** — gap = +0.0186 NDCG@20，是方法间合并 seed std 的 34.0 倍，3/3 seed 同向，零交叉。
- **Toys: Content-only vs GRU4Rec** — gap = +0.0144 NDCG@20，是方法间合并 seed std 的 28.4 倍，3/3 seed 同向，零交叉。
- **Toys: TA-SID vs BERT4Rec** — gap = +0.0245 NDCG@20，是方法间合并 seed std 的 27.3 倍，3/3 seed 同向，零交叉。
- **Toys: Content-only vs BERT4Rec** — gap = +0.0204 NDCG@20，是方法间合并 seed std 的 23.2 倍，3/3 seed 同向，零交叉。
- **Beauty: TA-SID vs SASRec** — gap = +0.0161 NDCG@20，是方法间合并 seed std 的 15.9 倍，3/3 seed 同向，零交叉。
- **Beauty: TA-SID vs Content-only** — gap = +0.0057 NDCG@20，是方法间合并 seed std 的 5.7 倍，3/3 seed 同向，零交叉。
- **Beauty: TA-SID vs GRU4Rec** — gap = +0.0092 NDCG@20，是方法间合并 seed std 的 7.5 倍，3/3 seed 同向，零交叉。
- **Beauty: Content-only vs GRU4Rec** — gap = +0.0035 NDCG@20，是方法间合并 seed std 的 3.1 倍，3/3 seed 同向，零交叉。
- **Beauty: TA-SID vs BERT4Rec** — gap = +0.0182 NDCG@20，是方法间合并 seed std 的 20.1 倍，3/3 seed 同向，零交叉。
- **Beauty: Content-only vs BERT4Rec** — gap = +0.0125 NDCG@20，是方法间合并 seed std 的 16.5 倍，3/3 seed 同向，零交叉。

## 4. n=3 的措辞红线（必须遵守）

### 4.1 事实澄清（先纠正一个常见误判）

n=3 下 p 值**并非不可达小**：df=2 时 t=74 对应 p≈1.8e-4。所以问题不是 "算不出显著"，而是 **p 值依赖无法检验的正态性假设**——n=3 时既不能检验正态性，也无法抵抗单个 seed 的偏移。**非参数**符号检验的精确最小双侧 p 恒为 2·(1/2)³ = **0.25**，即任何纯非参数主张在 n=3 下都不可能低于 0.25。

**禁止**：

- 写 "statistically significant (p < 0.05)" 而**不交代** n=3 与 df=2。
- 报 p 值但不写 df、或不同时报逐 seed 原值。
- 只报参数检验、不提非参数下界 0.25（会被审稿人视为选择性汇报）。

**允许且推荐**（首选第一项）：

- "consistent across all 3 seeds (no crossover)" + 效应量（"gap = X, which is Y× the pooled between-seed std"）。
- 若要报 p，必须连 df 与逐 seed 值一起报，并注明假设：*"paired t-test with df = 2 (p = …), which assumes normality that cannot be verified at n = 3; the corresponding non-parametric sign test floors at p = 0.25."*
- 显式免责一句：*"With n = 3 seeds we do not claim statistical significance; we report per-seed consistency and effect size instead, as the sign test's minimum attainable two-sided p-value at n = 3 is 0.25."*

## 5. 复现

```bash
python3 tools/paired_significance.py \
    --out-md doc/seq_baselines_significance.md \
    --out-csv doc/seq_baselines_significance.csv
```
