# E49 变体 × 数据集 量化端诊断网格（6×2，统一口径）

**生成时间**：2026-09-23　**产出脚本**：`rqvae/run_e49grid_diag.sh`
**产物**：`doc/e49grid/{Sport,Toys}_{content,shuffled,zeropad,itemcf,rawcooc,ta_sid}.md`（12 份）
**日志**：`rqvae/logs/e49grid_diag.log`

## 0. 这份文档解决什么

把 `xq[full]` 的转移一致性 AUC 与 E49 的 NDCG@20 **对齐到同一口径**，使
"**这个诊断指标到底能不能预测下游性能**"成为可检验的问题。

在此之前，AUC 与 NDCG 的出处是**分散且口径不一**的：

| 来源 | 覆盖 | n_random | 问题 |
|:--|:--|:--:|:--|
| `per_layer_transition_diagnostic.md`（2026-08-13）| TA-SID / Content-only | 5000 | 属**第二创新**线（已被 E46/E47 推翻），非消融产物 |
| `doc/e49_*.md`（2026-09-22/23，E49 脚本 D 步）| 4 新变体，且只成功 5 格 | 3000 | ZeroPad/Shuffled 的 Sport 格静默失败（空表）|

本轮**统一重跑 12 格**（6 变体 × 2 数据集），口径：`--n_random 5000 --skip_sweep`。

## 1. 配对验证（先做的，12/12 通过）

诊断需要 (RQ-VAE checkpoint, SID codes 文件) 配对正确，否则 `rebuild_xq = Σ codebook_l[code_l]`
重建出来的向量无意义。用 e48 审计法验证：**从 ckpt 重建码本 → 重算 L0 的 argmin 分配 →
与 codes 的 L0 列比行级一致率**（L0 是 z 上的直接 argmin，判据有效；L1/L2 在残差上量化，
直接 argmin 无判据性）。

结果 **12/12 一致率 = 100.00%**。过程中查出并订正 `tools/sid_consistency_contrast.py`
`CONFIG` 里三处错误：

| 位置 | 原值 | 实际 |
|:--|:--|:--|
| Sport Shuffled ckpt | `Sport_Shuffled_768d` | `Sport_Shuffled_Ablation` |
| Sport ZeroPad ckpt | `Sport_Zeropad_768d` | `Sport_Zeropad_Ablation` |
| Toys Shuffled codes | `Toys_shuffled_ablation_sid.npy` | `Toys_shuffled_768d_sid.npy` |

并把 Toys 的 TA-SID 从**孤儿 codes**（`Toys_ta_sid_768d_sid.npy`，L0=62，e48 已作废）
换成**健康 codes**（`Toys_ta_sid_768d_seed2024_sid.npy`，L0=207）。

## 2. 结果（AUC，n_random=5000）

| 数据集 | 变体 | z (量化前) | xq[L0] | xq[L0+L1] | xq[full] | **NDCG@20** |
|:--|:--|:--:|:--:|:--:|:--:|:--:|
| Sport | Content-only | 0.8983 | 0.8881 | 0.8951 | **0.8999** | 0.0275 |
| Sport | Shuffled | 0.8785 | 0.8739 | 0.8739 | **0.8755** | 0.0264 |
| Sport | ZeroPad | 0.9049 | 0.8875 | 0.8987 | **0.9021** | 0.0280 |
| Sport | ItemCF | 0.9328 | 0.9146 | 0.9270 | **0.9308** | 0.0295 |
| Sport | Raw-CoOcc | 0.9382 | 0.9210 | 0.9371 | **0.9414** | 0.0305 |
| Sport | TA-SID | 0.9645 | 0.9459 | 0.9607 | **0.9621** | 0.0322 |
| Toys | Content-only | 0.8601 | 0.8469 | 0.8612 | **0.8632** | 0.0336 |
| Toys | Shuffled | 0.8033 | 0.7524 | 0.7866 | **0.8014** | 0.0284 |
| Toys | ZeroPad | 0.8541 | 0.8487 | 0.8550 | **0.8579** | 0.0334 |
| Toys | ItemCF | 0.9325 | 0.9077 | 0.9361 | **0.9417** | 0.0355 |
| Toys | Raw-CoOcc | 0.9490 | 0.9033 | 0.9414 | **0.9488** | 0.0340 |
| Toys | TA-SID | 0.9652 | 0.9288 | 0.9550 | **0.9599** | 0.0377 |

NDCG 取自 [e49_results_corrected.md](e49_results_corrected.md)（3-seed 均值，权威表）。

## 3. 三个结构性发现

### 3.1 诊断指标与下游排序在**数据集内**高度一致

Spearman ρ(xq[full], NDCG@20)：

| 口径 | xq[full] | z | xq[L0] |
|:--|:--:|:--:|:--:|
| **Sport（n=6）** | **+1.000** | +1.000 | +0.943 |
| **Toys（n=6）** | **+0.943** | +0.943 | +0.943 |
| 合并 12 格 | +0.490 | +0.448 | +0.343 |

排序逐项对比：

```
Sport  xq[full]: TA-SID > Raw-CoOcc > ItemCF > ZeroPad > Content > Shuffled
Sport  NDCG    : TA-SID > Raw-CoOcc > ItemCF > ZeroPad > Content > Shuffled   ✅ 6/6 同序
Toys   xq[full]: TA-SID > Raw-CoOcc > ItemCF > Content > ZeroPad > Shuffled
Toys   NDCG    : TA-SID > ItemCF  > Raw-CoOcc > Content > ZeroPad > Shuffled  ⚠ 4/6（ItemCF/Raw-CoOcc 互换）
```

- **两端稳健**：`TA-SID` 恒第一、`Shuffled` 恒最后，跨两个数据集成立。
- **唯一失手**：Toys 上 ItemCF 与 Raw-CoOcc 的**相邻互换**——与 E49 已知的行为空间
  闭环失效（[e49_behavior_space_analysis.md](e49_behavior_space_analysis.md) §5）**同一处**。
- ⚠️ **必须按数据集分开算**：合并 12 格 ρ 掉到 0.49，因为 Toys 的 NDCG 整体高于
  Sport（同变体跨数据集不可比）——合并会因尺度差异把相关性稀释掉。这是个口径陷阱。

### 3.2 量化环节无系统损失（12/12）

`z → xq[full]` 差值：范围 **[−0.0053, +0.0092]**，均值 **+0.0003**，**符号 5/12 正**。

→ 差异全部落在噪声地板内（同一脚本的容量扫描显示 k=512→8192 的自然波动 ≈0.003），
且**方向不一致** → **无系统性量化损失**。这与"量化无损"的通行假设一致，
但与文献主流立论（ReSID / AsymRec 的"离散化必然丢信息"）方向相反。

### 3.3 粗层 L0 是普遍瓶颈（12/12）

`xq[L0]` 在**全部 12 格**都是 (z, L0, L0+L1, full) 中的**最低点**；补上 L1/L2 后回升。
跨数据集、跨变体一致 → **256 格的粗层码是共同的容量瓶颈，细层只做"补差"**。

## 4. 对"方向1（行为信息保不保留）"的意义

**保留**，且证据比"数值大小"更硬：

1. **诊断能区分变体**（Content → … → TA-SID 单调升），且**排序与下游 NDCG 同序** →
   离散码的几何确实携带"行为信息有多少"。
2. **量化前后排序能力相当**（z 与 xq[full] 的 ρ 同为 ~1.0）→ **瓶颈不在量化**。

## 5. 局限（勿外推）

1. **n=6/数据集**：Sport 的 1.000 vs Toys 的 0.943 差异**只是相邻一次互换**，不构成统计证据。
2. **Shuffled 拉大了跨度**：它是个"明显坏掉"的变体（两数据集都最低），
   相关性有很大一部分由它贡献；**去掉 Shuffled 后是否仍成立，未检验**。
3. **探针只覆盖 head**：top-1000 高频相邻对，tail 行为未测。
4. **未测 T5 token embedding 层**：链条止于 `xq`，下游 `xq → token → T5 embedding` 仍是空的。
5. **只有 2 个数据集** → 不足以支撑"跨数据集通用"的结论。
6. **Toys 用 seed2024 健康 codes**；Toys 的塌缩是 TA-SID×Toys×seed2024 特有（E47，1/3），
   本格反映的是该 seed 的码本，不宜当作 Toys 的"平均"行为。

## 6. 复现

```bash
# 配对验证（CPU，~3 min，L0 一致率应为 100%）
python3 rqvae/verify_variant_pairing.py   # 12 格 (ckpt, codes) 配对验证
# 12 格诊断（CPU，~1 min；统一 --n_random 5000 --skip_sweep）
bash rqvae/run_e49grid_diag.sh
```

## 7. 本次改动清单

| 文件 | 改动 |
|:--|:--|
| `tools/sid_capacity_diagnostic.py` | 修标题硬编码 "(TA-SID)" 标签 → 回显实际 ckpt/codes；汇总表加 `ckpt_tag` + `xq[L0+L1]` 列；新增 `--skip_sweep` |
| `tools/sid_consistency_contrast.py` | 订正 CONFIG 三处错误配对 + Toys TA-SID 换健康 codes |
| `rqvae/run_e49grid_diag.sh` | 新建：12 格统一口径驱动 |
| `doc/e49_sport_{shuffled,zeropad}_ablation.md` | **删除**（21 行空表；`if [ ! -f ]` 会永久阻塞重跑） |

## 相关

[e49_ablation_full_conclusion.md](e49_ablation_full_conclusion.md)（消融总汇总）|
[e49_results_corrected.md](e49_results_corrected.md)（权威 NDCG 表）|
[e49_behavior_space_analysis.md](e49_behavior_space_analysis.md)（行为空间判据，已知在 Toys 失效）|
[e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)（孤儿 codes 裁定）
