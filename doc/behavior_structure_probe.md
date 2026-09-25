# Behavior Structure Preservation vs Recommendation Performance

**运行时间**：2026-09-23 23:07:45　**产出脚本**：`tools/behavior_structure_probe.py`

## 0. 口径

- **E 图（正样本来源）**：`train.parquet` 的 `history+[target]` 序列 → 滑窗 `window=3`（即 ±1 相邻对）共现矩阵，与 TA-SID behavior 向量**同源**；**所有 variant 共用同一张 E**。
- **正样本两轴**：`topk` = 频次最高 1000 条边（`count≥6`，强证据，＝ `doc/e49grid/` 口径）；`all` = 从 E 均匀采样 2×10⁴ 条（覆盖全图，含长尾弱边）。
- **负样本两轴**：`degmatch` = 度分布受控（按 ⌊log₂deg⌋ 分桶，候选取自**与正样本端点同桶**，排除 E 内边与自环）—— **主口径**，堵住『热门商品彼此相似』的流行度捷径；`uniform` = 均匀随机（排除自环与 E 内边）—— 与 `doc/e49grid/` 的**近似**参照臂（e49grid 不排除 E 内边，故数值系统性偏低约 0.013；排序一致）。
- **SID 表示**：`xq[L0]` = 第 1 层码本向量；`xq[full]` = 3 层码本向量之和（`e_dim=32`）；`z` = 编码器量化前 latent（容量天花板）。
- **离散 token**：`Hamming` = 前 3 个 token 的差异位数（越小越相似）；`L0-match` = 首 token 是否相同。
- **NDCG@20**：从 `model/logs/*_eval.log` **直接解析**的 3-seed 均值（generator seed；SID codes 固定不随其变化）—— **不抄表**。

## 1. 主表（xq[full]）

**xq[full] · pos=topk · Sport**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Sport | Content-only | 0.0275 | 0.9080 | 0.9105 | 0.6856 | 0.6849 |
| Sport | ZeroPad | 0.0280 | 0.9069 | 0.9076 | 0.6590 | 0.6558 |
| Sport | Shuffled | 0.0264 | 0.8763 | 0.8823 | 0.6253 | 0.6283 |
| Sport | ItemCF | 0.0295 | 0.9397 | 0.9361 | 0.7374 | 0.7179 |
| Sport | Raw-CoOcc | 0.0305 | 0.9469 | 0.9378 | 0.6881 | 0.6688 |
| Sport | TA-SID | 0.0322 | 0.9647 | 0.9623 | 0.7498 | 0.7450 |

**xq[full] · pos=topk · Toys**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Toys | Content-only | 0.0336 | 0.8751 | 0.8948 | 0.7172 | 0.7135 |
| Toys | ZeroPad | 0.0334 | 0.8716 | 0.8843 | 0.6957 | 0.6910 |
| Toys | Shuffled | 0.0284 | 0.8127 | 0.8260 | 0.6225 | 0.6215 |
| Toys | ItemCF | 0.0355 | 0.9455 | 0.9586 | 0.7796 | 0.7735 |
| Toys | Raw-CoOcc | 0.0340 | 0.9543 | 0.9623 | 0.7655 | 0.7620 |
| Toys | TA-SID | 0.0377 | 0.9651 | 0.9708 | 0.8416 | 0.8370 |

**xq[full] · pos=topk · Beauty**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Beauty | Content-only | 0.0358 | 0.8365 | 0.8370 | 0.6584 | 0.6435 |
| Beauty | TA-SID | 0.0415 | 0.9560 | 0.9539 | 0.8847 | 0.8655 |

**xq[full] · pos=all · Sport**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Sport | Content-only | 0.0275 | 0.7377 | 0.7484 | 0.5519 | 0.5507 |
| Sport | ZeroPad | 0.0280 | 0.7329 | 0.7450 | 0.5406 | 0.5394 |
| Sport | Shuffled | 0.0264 | 0.7195 | 0.7320 | 0.5327 | 0.5322 |
| Sport | ItemCF | 0.0295 | 0.7593 | 0.7618 | 0.5536 | 0.5492 |
| Sport | Raw-CoOcc | 0.0305 | 0.7639 | 0.7604 | 0.5397 | 0.5366 |
| Sport | TA-SID | 0.0322 | 0.7718 | 0.7775 | 0.5545 | 0.5526 |

**xq[full] · pos=all · Toys**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Toys | Content-only | 0.0336 | 0.6744 | 0.6889 | 0.5540 | 0.5534 |
| Toys | ZeroPad | 0.0334 | 0.6724 | 0.6817 | 0.5519 | 0.5510 |
| Toys | Shuffled | 0.0284 | 0.6323 | 0.6454 | 0.5304 | 0.5293 |
| Toys | ItemCF | 0.0355 | 0.7345 | 0.7427 | 0.5721 | 0.5675 |
| Toys | Raw-CoOcc | 0.0340 | 0.7349 | 0.7336 | 0.5616 | 0.5579 |
| Toys | TA-SID | 0.0377 | 0.7623 | 0.7704 | 0.5855 | 0.5843 |

**xq[full] · pos=all · Beauty**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Beauty | Content-only | 0.0358 | 0.6680 | 0.6620 | 0.5524 | 0.5504 |
| Beauty | TA-SID | 0.0415 | 0.7364 | 0.7151 | 0.5766 | 0.5739 |

## 2. 相关分析（Behavior AUC vs NDCG@20）

| pos_regime | scope | metric | n | pearson | p_pearson | spearman | p_spearman |
|---|---|---|---|---|---|---|---|
| topk | Sport | auc_cos_degmatch | 6 | 0.9775 | 0.0008 | 0.9429 | 0.0048 |
| topk | Toys | auc_cos_degmatch | 6 | 0.8933 | 0.0165 | 0.9429 | 0.0048 |
| topk | Pooled | auc_cos_degmatch | 12 | 0.4807 | 0.1136 | 0.4825 | 0.1121 |
| topk | Sport | auc_cos_uniform | 6 | 0.9782 | 0.0007 | 0.9429 | 0.0048 |
| topk | Toys | auc_cos_uniform | 6 | 0.9012 | 0.0142 | 0.9429 | 0.0048 |
| topk | Pooled | auc_cos_uniform | 12 | 0.5909 | 0.0431 | 0.6434 | 0.0240 |
| topk | Sport | auc_ham_degmatch | 6 | 0.8339 | 0.0391 | 0.8857 | 0.0188 |
| topk | Toys | auc_ham_degmatch | 6 | 0.9624 | 0.0021 | 1.0000 | 0.0000 |
| topk | Pooled | auc_ham_degmatch | 12 | 0.8714 | 0.0002 | 0.8741 | 0.0002 |
| topk | Sport | auc_l0match_degmatch | 6 | 0.7963 | 0.0580 | 0.7143 | 0.1108 |
| topk | Toys | auc_l0match_degmatch | 6 | 0.9589 | 0.0025 | 1.0000 | 0.0000 |
| topk | Pooled | auc_l0match_degmatch | 12 | 0.8781 | 0.0002 | 0.8462 | 0.0005 |
| all | Sport | auc_cos_degmatch | 6 | 0.9649 | 0.0018 | 0.9429 | 0.0048 |
| all | Toys | auc_cos_degmatch | 6 | 0.9055 | 0.0130 | 0.9429 | 0.0048 |
| all | Pooled | auc_cos_degmatch | 12 | 0.1472 | 0.6481 | 0.1818 | 0.5717 |
| all | Sport | auc_cos_uniform | 6 | 0.9681 | 0.0015 | 0.8857 | 0.0188 |
| all | Toys | auc_cos_uniform | 6 | 0.9275 | 0.0077 | 1.0000 | 0.0000 |
| all | Pooled | auc_cos_uniform | 12 | 0.1274 | 0.6932 | 0.1189 | 0.7129 |
| all | Sport | auc_ham_degmatch | 6 | 0.5476 | 0.2608 | 0.6000 | 0.2080 |
| all | Toys | auc_ham_degmatch | 6 | 0.9810 | 0.0005 | 1.0000 | 0.0000 |
| all | Pooled | auc_ham_degmatch | 12 | 0.8658 | 0.0003 | 0.8531 | 0.0004 |
| all | Sport | auc_l0match_degmatch | 6 | 0.4913 | 0.3223 | 0.4857 | 0.3287 |
| all | Toys | auc_l0match_degmatch | 6 | 0.9846 | 0.0004 | 1.0000 | 0.0000 |
| all | Pooled | auc_l0match_degmatch | 12 | 0.8648 | 0.0003 | 0.8671 | 0.0003 |
| topk | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9354 | 0.0000 | — | — |
| topk | Pooled(z-scored) | auc_cos_uniform | 12 | 0.9397 | 0.0000 | — | — |
| topk | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.8982 | 0.0001 | — | — |
| topk | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.8776 | 0.0002 | — | — |
| all | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9352 | 0.0000 | — | — |
| all | Pooled(z-scored) | auc_cos_uniform | 12 | 0.9478 | 0.0000 | — | — |
| all | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.7643 | 0.0038 | — | — |
| all | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.7380 | 0.0061 | — | — |

## 3. 全空间相关分析（xq[L0] / z）

| space | pos_regime | scope | metric | n | pearson | spearman | p_spearman |
|---|---|---|---|---|---|---|---|
| xq[L0] | topk | Sport | auc_cos_degmatch | 6 | 0.9906 | 0.9429 | 0.0048 |
| xq[L0] | topk | Toys | auc_cos_degmatch | 6 | 0.9566 | 0.8857 | 0.0188 |
| xq[L0] | topk | Pooled | auc_cos_degmatch | 12 | 0.3318 | 0.3357 | 0.2861 |
| xq[L0] | topk | Sport | auc_cos_uniform | 6 | 0.9815 | 0.9429 | 0.0048 |
| xq[L0] | topk | Toys | auc_cos_uniform | 6 | 0.9598 | 1.0000 | 0.0000 |
| xq[L0] | topk | Pooled | auc_cos_uniform | 12 | 0.4390 | 0.4615 | 0.1309 |
| xq[L0] | topk | Sport | auc_ham_degmatch | 6 | 0.8339 | 0.8857 | 0.0188 |
| xq[L0] | topk | Toys | auc_ham_degmatch | 6 | 0.9624 | 1.0000 | 0.0000 |
| xq[L0] | topk | Pooled | auc_ham_degmatch | 12 | 0.8714 | 0.8741 | 0.0002 |
| xq[L0] | topk | Sport | auc_l0match_degmatch | 6 | 0.7963 | 0.7143 | 0.1108 |
| xq[L0] | topk | Toys | auc_l0match_degmatch | 6 | 0.9589 | 1.0000 | 0.0000 |
| xq[L0] | topk | Pooled | auc_l0match_degmatch | 12 | 0.8781 | 0.8462 | 0.0005 |
| xq[L0] | all | Sport | auc_cos_degmatch | 6 | 0.9790 | 0.9429 | 0.0048 |
| xq[L0] | all | Toys | auc_cos_degmatch | 6 | 0.9514 | 0.9429 | 0.0048 |
| xq[L0] | all | Pooled | auc_cos_degmatch | 12 | -0.0188 | -0.1049 | 0.7456 |
| xq[L0] | all | Sport | auc_cos_uniform | 6 | 0.9540 | 0.8857 | 0.0188 |
| xq[L0] | all | Toys | auc_cos_uniform | 6 | 0.9696 | 1.0000 | 0.0000 |
| xq[L0] | all | Pooled | auc_cos_uniform | 12 | -0.0547 | -0.1049 | 0.7456 |
| xq[L0] | all | Sport | auc_ham_degmatch | 6 | 0.5476 | 0.6000 | 0.2080 |
| xq[L0] | all | Toys | auc_ham_degmatch | 6 | 0.9810 | 1.0000 | 0.0000 |
| xq[L0] | all | Pooled | auc_ham_degmatch | 12 | 0.8658 | 0.8531 | 0.0004 |
| xq[L0] | all | Sport | auc_l0match_degmatch | 6 | 0.4913 | 0.4857 | 0.3287 |
| xq[L0] | all | Toys | auc_l0match_degmatch | 6 | 0.9846 | 1.0000 | 0.0000 |
| xq[L0] | all | Pooled | auc_l0match_degmatch | 12 | 0.8648 | 0.8671 | 0.0003 |
| z | topk | Sport | auc_cos_degmatch | 6 | 0.9691 | 0.9429 | 0.0048 |
| z | topk | Toys | auc_cos_degmatch | 6 | 0.8975 | 0.9429 | 0.0048 |
| z | topk | Pooled | auc_cos_degmatch | 12 | 0.4926 | 0.4965 | 0.1006 |
| z | topk | Sport | auc_cos_uniform | 6 | 0.9798 | 1.0000 | 0.0000 |
| z | topk | Toys | auc_cos_uniform | 6 | 0.9106 | 0.9429 | 0.0048 |
| z | topk | Pooled | auc_cos_uniform | 12 | 0.6075 | 0.6294 | 0.0283 |
| z | topk | Sport | auc_ham_degmatch | 6 | 0.8339 | 0.8857 | 0.0188 |
| z | topk | Toys | auc_ham_degmatch | 6 | 0.9624 | 1.0000 | 0.0000 |
| z | topk | Pooled | auc_ham_degmatch | 12 | 0.8714 | 0.8741 | 0.0002 |
| z | topk | Sport | auc_l0match_degmatch | 6 | 0.7963 | 0.7143 | 0.1108 |
| z | topk | Toys | auc_l0match_degmatch | 6 | 0.9589 | 1.0000 | 0.0000 |
| z | topk | Pooled | auc_l0match_degmatch | 12 | 0.8781 | 0.8462 | 0.0005 |
| z | all | Sport | auc_cos_degmatch | 6 | 0.9491 | 0.8857 | 0.0188 |
| z | all | Toys | auc_cos_degmatch | 6 | 0.8961 | 1.0000 | 0.0000 |
| z | all | Pooled | auc_cos_degmatch | 12 | 0.1926 | 0.2867 | 0.3663 |
| z | all | Sport | auc_cos_uniform | 6 | 0.8667 | 0.8857 | 0.0188 |
| z | all | Toys | auc_cos_uniform | 6 | 0.9219 | 1.0000 | 0.0000 |
| z | all | Pooled | auc_cos_uniform | 12 | 0.1443 | 0.1189 | 0.7129 |
| z | all | Sport | auc_ham_degmatch | 6 | 0.5476 | 0.6000 | 0.2080 |
| z | all | Toys | auc_ham_degmatch | 6 | 0.9810 | 1.0000 | 0.0000 |
| z | all | Pooled | auc_ham_degmatch | 12 | 0.8658 | 0.8531 | 0.0004 |
| z | all | Sport | auc_l0match_degmatch | 6 | 0.4913 | 0.4857 | 0.3287 |
| z | all | Toys | auc_l0match_degmatch | 6 | 0.9846 | 1.0000 | 0.0000 |
| z | all | Pooled | auc_l0match_degmatch | 12 | 0.8648 | 0.8671 | 0.0003 |
| xq[L0] | topk | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9736 | — | — |
| xq[L0] | topk | Pooled(z-scored) | auc_cos_uniform | 12 | 0.9707 | — | — |
| xq[L0] | topk | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.8982 | — | — |
| xq[L0] | topk | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.8776 | — | — |
| xq[L0] | all | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9652 | — | — |
| xq[L0] | all | Pooled(z-scored) | auc_cos_uniform | 12 | 0.9618 | — | — |
| xq[L0] | all | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.7643 | — | — |
| xq[L0] | all | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.7380 | — | — |
| z | topk | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9333 | — | — |
| z | topk | Pooled(z-scored) | auc_cos_uniform | 12 | 0.9452 | — | — |
| z | topk | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.8982 | — | — |
| z | topk | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.8776 | — | — |
| z | all | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9226 | — | — |
| z | all | Pooled(z-scored) | auc_cos_uniform | 12 | 0.8943 | — | — |
| z | all | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.7643 | — | — |
| z | all | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.7380 | — | — |

## 4. 产物

- 逐空间长表：`doc/behavior_structure_probe_results.csv`（dataset × variant × space × pos_regime）

<details><summary>原始日志</summary>

```text
====================================================================================================
Behavior Structure Preservation vs Recommendation Performance
运行时间: 2026-09-23 23:07:38
正样本口径: topk = 频次最高 1000 条边（＝ e49grid） | all = 从 E 均匀采样 ≤20000 条（含长尾弱边）
负样本口径: degmatch = 度分布受控（主） | uniform = 均匀随机（参照）
E 图: train.parquet history+[target] → 滑窗 window=3（±1 相邻对）
NDCG@20: 直接从 model/logs/*_eval.log 解析（3-seed 均值），不抄表
只读：不训练、不写实验产物，仅 CSV + 本报告
====================================================================================================

====================================================================================================
▌ Sport
====================================================================================================
  E 图: 18357 items | 180361 条边 (count 2–62) | 35598 条训练序列
  度分布: mean=19.7 median=11 max=994 零度=28
  正样本: topk=1000 条 | all=20000 条
  [topk] 可用正样本 998/1000 | 端点 degree 均值 pos=179.7 degmatch=162.0 (Δ=-17.7) uniform=20.2 (Δ=-159.5)
  [all] 可用正样本 19999/20000 | 端点 degree 均值 pos=65.6 degmatch=63.0 (Δ=-2.6) uniform=19.8 (Δ=-45.8)
  NDCG@20 来源: 6 个变体解析成功
  Content-only   xq[L0]    [topk] AUC cos(degmatch)=0.8967 cos(uniform)=0.8991 | Ham(degmatch)=0.6856 L0match(degmatch)=0.6849 | NDCG@20=0.0275
  Content-only   xq[L0]    [all ] AUC cos(degmatch)=0.7318 cos(uniform)=0.7435 | Ham(degmatch)=0.5519 L0match(degmatch)=0.5507 | NDCG@20=0.0275
  Content-only   xq[full]  [topk] AUC cos(degmatch)=0.9080 cos(uniform)=0.9105 | Ham(degmatch)=0.6856 L0match(degmatch)=0.6849 | NDCG@20=0.0275
  Content-only   xq[full]  [all ] AUC cos(degmatch)=0.7377 cos(uniform)=0.7484 | Ham(degmatch)=0.5519 L0match(degmatch)=0.5507 | NDCG@20=0.0275
  Content-only   z         [topk] AUC cos(degmatch)=0.9089 cos(uniform)=0.9083 | Ham(degmatch)=0.6856 L0match(degmatch)=0.6849 | NDCG@20=0.0275
  Content-only   z         [all ] AUC cos(degmatch)=0.7339 cos(uniform)=0.7461 | Ham(degmatch)=0.5519 L0match(degmatch)=0.5507 | NDCG@20=0.0275
  ZeroPad        xq[L0]    [topk] AUC cos(degmatch)=0.8932 cos(uniform)=0.8935 | Ham(degmatch)=0.6590 L0match(degmatch)=0.6558 | NDCG@20=0.0280
  ZeroPad        xq[L0]    [all ] AUC cos(degmatch)=0.7290 cos(uniform)=0.7419 | Ham(degmatch)=0.5406 L0match(degmatch)=0.5394 | NDCG@20=0.0280
  ZeroPad        xq[full]  [topk] AUC cos(degmatch)=0.9069 cos(uniform)=0.9076 | Ham(degmatch)=0.6590 L0match(degmatch)=0.6558 | NDCG@20=0.0280
  ZeroPad        xq[full]  [all ] AUC cos(degmatch)=0.7329 cos(uniform)=0.7450 | Ham(degmatch)=0.5406 L0match(degmatch)=0.5394 | NDCG@20=0.0280
  ZeroPad        z         [topk] AUC cos(degmatch)=0.9062 cos(uniform)=0.9098 | Ham(degmatch)=0.6590 L0match(degmatch)=0.6558 | NDCG@20=0.0280
  ZeroPad        z         [all ] AUC cos(degmatch)=0.7300 cos(uniform)=0.7441 | Ham(degmatch)=0.5406 L0match(degmatch)=0.5394 | NDCG@20=0.0280
  Shuffled       xq[L0]    [topk] AUC cos(degmatch)=0.8793 cos(uniform)=0.8843 | Ham(degmatch)=0.6253 L0match(degmatch)=0.6283 | NDCG@20=0.0264
  Shuffled       xq[L0]    [all ] AUC cos(degmatch)=0.7194 cos(uniform)=0.7338 | Ham(degmatch)=0.5327 L0match(degmatch)=0.5322 | NDCG@20=0.0264
  Shuffled       xq[full]  [topk] AUC cos(degmatch)=0.8763 cos(uniform)=0.8823 | Ham(degmatch)=0.6253 L0match(degmatch)=0.6283 | NDCG@20=0.0264
  Shuffled       xq[full]  [all ] AUC cos(degmatch)=0.7195 cos(uniform)=0.7320 | Ham(degmatch)=0.5327 L0match(degmatch)=0.5322 | NDCG@20=0.0264
  Shuffled       z         [topk] AUC cos(degmatch)=0.8745 cos(uniform)=0.8806 | Ham(degmatch)=0.6253 L0match(degmatch)=0.6283 | NDCG@20=0.0264
  Shuffled       z         [all ] AUC cos(degmatch)=0.7111 cos(uniform)=0.7270 | Ham(degmatch)=0.5327 L0match(degmatch)=0.5322 | NDCG@20=0.0264
  ItemCF         xq[L0]    [topk] AUC cos(degmatch)=0.9200 cos(uniform)=0.9196 | Ham(degmatch)=0.7374 L0match(degmatch)=0.7179 | NDCG@20=0.0295
  ItemCF         xq[L0]    [all ] AUC cos(degmatch)=0.7506 cos(uniform)=0.7611 | Ham(degmatch)=0.5536 L0match(degmatch)=0.5492 | NDCG@20=0.0295
  ItemCF         xq[full]  [topk] AUC cos(degmatch)=0.9397 cos(uniform)=0.9361 | Ham(degmatch)=0.7374 L0match(degmatch)=0.7179 | NDCG@20=0.0295
  ItemCF         xq[full]  [all ] AUC cos(degmatch)=0.7593 cos(uniform)=0.7618 | Ham(degmatch)=0.5536 L0match(degmatch)=0.5492 | NDCG@20=0.0295
  ItemCF         z         [topk] AUC cos(degmatch)=0.9432 cos(uniform)=0.9366 | Ham(degmatch)=0.7374 L0match(degmatch)=0.7179 | NDCG@20=0.0295
  ItemCF         z         [all ] AUC cos(degmatch)=0.7588 cos(uniform)=0.7583 | Ham(degmatch)=0.5536 L0match(degmatch)=0.5492 | NDCG@20=0.0295
  Raw-CoOcc      xq[L0]    [topk] AUC cos(degmatch)=0.9277 cos(uniform)=0.9234 | Ham(degmatch)=0.6881 L0match(degmatch)=0.6688 | NDCG@20=0.0305
  Raw-CoOcc      xq[L0]    [all ] AUC cos(degmatch)=0.7549 cos(uniform)=0.7555 | Ham(degmatch)=0.5397 L0match(degmatch)=0.5366 | NDCG@20=0.0305
  Raw-CoOcc      xq[full]  [topk] AUC cos(degmatch)=0.9469 cos(uniform)=0.9378 | Ham(degmatch)=0.6881 L0match(degmatch)=0.6688 | NDCG@20=0.0305
  Raw-CoOcc      xq[full]  [all ] AUC cos(degmatch)=0.7639 cos(uniform)=0.7604 | Ham(degmatch)=0.5397 L0match(degmatch)=0.5366 | NDCG@20=0.0305
  Raw-CoOcc      z         [topk] AUC cos(degmatch)=0.9493 cos(uniform)=0.9368 | Ham(degmatch)=0.6881 L0match(degmatch)=0.6688 | NDCG@20=0.0305
  Raw-CoOcc      z         [all ] AUC cos(degmatch)=0.7583 cos(uniform)=0.7461 | Ham(degmatch)=0.5397 L0match(degmatch)=0.5366 | NDCG@20=0.0305
  TA-SID         xq[L0]    [topk] AUC cos(degmatch)=0.9474 cos(uniform)=0.9489 | Ham(degmatch)=0.7498 L0match(degmatch)=0.7450 | NDCG@20=0.0322
  TA-SID         xq[L0]    [all ] AUC cos(degmatch)=0.7644 cos(uniform)=0.7721 | Ham(degmatch)=0.5545 L0match(degmatch)=0.5526 | NDCG@20=0.0322
  TA-SID         xq[full]  [topk] AUC cos(degmatch)=0.9647 cos(uniform)=0.9623 | Ham(degmatch)=0.7498 L0match(degmatch)=0.7450 | NDCG@20=0.0322
  TA-SID         xq[full]  [all ] AUC cos(degmatch)=0.7718 cos(uniform)=0.7775 | Ham(degmatch)=0.5545 L0match(degmatch)=0.5526 | NDCG@20=0.0322
  TA-SID         z         [topk] AUC cos(degmatch)=0.9655 cos(uniform)=0.9651 | Ham(degmatch)=0.7498 L0match(degmatch)=0.7450 | NDCG@20=0.0322
  TA-SID         z         [all ] AUC cos(degmatch)=0.7684 cos(uniform)=0.7703 | Ham(degmatch)=0.5545 L0match(degmatch)=0.5526 | NDCG@20=0.0322

====================================================================================================
▌ Toys
====================================================================================================
  E 图: 11924 items | 102151 条边 (count 2–58) | 19412 条训练序列
  度分布: mean=17.1 median=11 max=273 零度=24
  正样本: topk=1000 条 | all=20000 条
  [topk] 可用正样本 1000/1000 | 端点 degree 均值 pos=59.0 degmatch=55.7 (Δ=-3.2) uniform=16.6 (Δ=-42.3)
  [all] 可用正样本 20000/20000 | 端点 degree 均值 pos=38.9 degmatch=37.4 (Δ=-1.5) uniform=17.1 (Δ=-21.8)
  NDCG@20 来源: 6 个变体解析成功
  Content-only   xq[L0]    [topk] AUC cos(degmatch)=0.8559 cos(uniform)=0.8787 | Ham(degmatch)=0.7172 L0match(degmatch)=0.7135 | NDCG@20=0.0336
  Content-only   xq[L0]    [all ] AUC cos(degmatch)=0.6637 cos(uniform)=0.6778 | Ham(degmatch)=0.5540 L0match(degmatch)=0.5534 | NDCG@20=0.0336
  Content-only   xq[full]  [topk] AUC cos(degmatch)=0.8751 cos(uniform)=0.8948 | Ham(degmatch)=0.7172 L0match(degmatch)=0.7135 | NDCG@20=0.0336
  Content-only   xq[full]  [all ] AUC cos(degmatch)=0.6744 cos(uniform)=0.6889 | Ham(degmatch)=0.5540 L0match(degmatch)=0.5534 | NDCG@20=0.0336
  Content-only   z         [topk] AUC cos(degmatch)=0.8788 cos(uniform)=0.8958 | Ham(degmatch)=0.7172 L0match(degmatch)=0.7135 | NDCG@20=0.0336
  Content-only   z         [all ] AUC cos(degmatch)=0.6732 cos(uniform)=0.6860 | Ham(degmatch)=0.5540 L0match(degmatch)=0.5534 | NDCG@20=0.0336
  ZeroPad        xq[L0]    [topk] AUC cos(degmatch)=0.8561 cos(uniform)=0.8699 | Ham(degmatch)=0.6957 L0match(degmatch)=0.6910 | NDCG@20=0.0334
  ZeroPad        xq[L0]    [all ] AUC cos(degmatch)=0.6607 cos(uniform)=0.6699 | Ham(degmatch)=0.5519 L0match(degmatch)=0.5510 | NDCG@20=0.0334
  ZeroPad        xq[full]  [topk] AUC cos(degmatch)=0.8716 cos(uniform)=0.8843 | Ham(degmatch)=0.6957 L0match(degmatch)=0.6910 | NDCG@20=0.0334
  ZeroPad        xq[full]  [all ] AUC cos(degmatch)=0.6724 cos(uniform)=0.6817 | Ham(degmatch)=0.5519 L0match(degmatch)=0.5510 | NDCG@20=0.0334
  ZeroPad        z         [topk] AUC cos(degmatch)=0.8735 cos(uniform)=0.8874 | Ham(degmatch)=0.6957 L0match(degmatch)=0.6910 | NDCG@20=0.0334
  ZeroPad        z         [all ] AUC cos(degmatch)=0.6689 cos(uniform)=0.6782 | Ham(degmatch)=0.5519 L0match(degmatch)=0.5510 | NDCG@20=0.0334
  Shuffled       xq[L0]    [topk] AUC cos(degmatch)=0.7630 cos(uniform)=0.7729 | Ham(degmatch)=0.6225 L0match(degmatch)=0.6215 | NDCG@20=0.0284
  Shuffled       xq[L0]    [all ] AUC cos(degmatch)=0.6136 cos(uniform)=0.6267 | Ham(degmatch)=0.5304 L0match(degmatch)=0.5293 | NDCG@20=0.0284
  Shuffled       xq[full]  [topk] AUC cos(degmatch)=0.8127 cos(uniform)=0.8260 | Ham(degmatch)=0.6225 L0match(degmatch)=0.6215 | NDCG@20=0.0284
  Shuffled       xq[full]  [all ] AUC cos(degmatch)=0.6323 cos(uniform)=0.6454 | Ham(degmatch)=0.5304 L0match(degmatch)=0.5293 | NDCG@20=0.0284
  Shuffled       z         [topk] AUC cos(degmatch)=0.8151 cos(uniform)=0.8270 | Ham(degmatch)=0.6225 L0match(degmatch)=0.6215 | NDCG@20=0.0284
  Shuffled       z         [all ] AUC cos(degmatch)=0.6345 cos(uniform)=0.6461 | Ham(degmatch)=0.5304 L0match(degmatch)=0.5293 | NDCG@20=0.0284
  ItemCF         xq[L0]    [topk] AUC cos(degmatch)=0.9071 cos(uniform)=0.9274 | Ham(degmatch)=0.7796 L0match(degmatch)=0.7735 | NDCG@20=0.0355
  ItemCF         xq[L0]    [all ] AUC cos(degmatch)=0.7046 cos(uniform)=0.7177 | Ham(degmatch)=0.5721 L0match(degmatch)=0.5675 | NDCG@20=0.0355
  ItemCF         xq[full]  [topk] AUC cos(degmatch)=0.9455 cos(uniform)=0.9586 | Ham(degmatch)=0.7796 L0match(degmatch)=0.7735 | NDCG@20=0.0355
  ItemCF         xq[full]  [all ] AUC cos(degmatch)=0.7345 cos(uniform)=0.7427 | Ham(degmatch)=0.5721 L0match(degmatch)=0.5675 | NDCG@20=0.0355
  ItemCF         z         [topk] AUC cos(degmatch)=0.9416 cos(uniform)=0.9536 | Ham(degmatch)=0.7796 L0match(degmatch)=0.7735 | NDCG@20=0.0355
  ItemCF         z         [all ] AUC cos(degmatch)=0.7368 cos(uniform)=0.7424 | Ham(degmatch)=0.5721 L0match(degmatch)=0.5675 | NDCG@20=0.0355
  Raw-CoOcc      xq[L0]    [topk] AUC cos(degmatch)=0.9128 cos(uniform)=0.9213 | Ham(degmatch)=0.7655 L0match(degmatch)=0.7620 | NDCG@20=0.0340
  Raw-CoOcc      xq[L0]    [all ] AUC cos(degmatch)=0.7047 cos(uniform)=0.7025 | Ham(degmatch)=0.5616 L0match(degmatch)=0.5579 | NDCG@20=0.0340
  Raw-CoOcc      xq[full]  [topk] AUC cos(degmatch)=0.9543 cos(uniform)=0.9623 | Ham(degmatch)=0.7655 L0match(degmatch)=0.7620 | NDCG@20=0.0340
  Raw-CoOcc      xq[full]  [all ] AUC cos(degmatch)=0.7349 cos(uniform)=0.7336 | Ham(degmatch)=0.5616 L0match(degmatch)=0.5579 | NDCG@20=0.0340
  Raw-CoOcc      z         [topk] AUC cos(degmatch)=0.9565 cos(uniform)=0.9634 | Ham(degmatch)=0.7655 L0match(degmatch)=0.7620 | NDCG@20=0.0340
  Raw-CoOcc      z         [all ] AUC cos(degmatch)=0.7338 cos(uniform)=0.7287 | Ham(degmatch)=0.5616 L0match(degmatch)=0.5579 | NDCG@20=0.0340
  TA-SID         xq[L0]    [topk] AUC cos(degmatch)=0.9390 cos(uniform)=0.9444 | Ham(degmatch)=0.8416 L0match(degmatch)=0.8370 | NDCG@20=0.0377
  TA-SID         xq[L0]    [all ] AUC cos(degmatch)=0.7396 cos(uniform)=0.7487 | Ham(degmatch)=0.5855 L0match(degmatch)=0.5843 | NDCG@20=0.0377
  TA-SID         xq[full]  [topk] AUC cos(degmatch)=0.9651 cos(uniform)=0.9708 | Ham(degmatch)=0.8416 L0match(degmatch)=0.8370 | NDCG@20=0.0377
  TA-SID         xq[full]  [all ] AUC cos(degmatch)=0.7623 cos(uniform)=0.7704 | Ham(degmatch)=0.5855 L0match(degmatch)=0.5843 | NDCG@20=0.0377
  TA-SID         z         [topk] AUC cos(degmatch)=0.9697 cos(uniform)=0.9761 | Ham(degmatch)=0.8416 L0match(degmatch)=0.8370 | NDCG@20=0.0377
  TA-SID         z         [all ] AUC cos(degmatch)=0.7643 cos(uniform)=0.7700 | Ham(degmatch)=0.5855 L0match(degmatch)=0.5843 | NDCG@20=0.0377

====================================================================================================
▌ Beauty
====================================================================================================
  E 图: 12101 items | 111649 条边 (count 2–142) | 22363 条训练序列
  度分布: mean=18.5 median=11 max=457 零度=33
  正样本: topk=1000 条 | all=20000 条
  [topk] 可用正样本 1000/1000 | 端点 degree 均值 pos=85.1 degmatch=79.5 (Δ=-5.6) uniform=18.5 (Δ=-66.5)
  [all] 可用正样本 20000/20000 | 端点 degree 均值 pos=50.3 degmatch=48.3 (Δ=-2.0) uniform=18.5 (Δ=-31.8)
  NDCG@20 来源: 2 个变体解析成功
  Content-only   xq[L0]    [topk] AUC cos(degmatch)=0.8155 cos(uniform)=0.8229 | Ham(degmatch)=0.6584 L0match(degmatch)=0.6435 | NDCG@20=0.0358
  Content-only   xq[L0]    [all ] AUC cos(degmatch)=0.6647 cos(uniform)=0.6624 | Ham(degmatch)=0.5524 L0match(degmatch)=0.5504 | NDCG@20=0.0358
  Content-only   xq[full]  [topk] AUC cos(degmatch)=0.8365 cos(uniform)=0.8370 | Ham(degmatch)=0.6584 L0match(degmatch)=0.6435 | NDCG@20=0.0358
  Content-only   xq[full]  [all ] AUC cos(degmatch)=0.6680 cos(uniform)=0.6620 | Ham(degmatch)=0.5524 L0match(degmatch)=0.5504 | NDCG@20=0.0358
  Content-only   z         [topk] AUC cos(degmatch)=0.8334 cos(uniform)=0.8353 | Ham(degmatch)=0.6584 L0match(degmatch)=0.6435 | NDCG@20=0.0358
  Content-only   z         [all ] AUC cos(degmatch)=0.6650 cos(uniform)=0.6595 | Ham(degmatch)=0.5524 L0match(degmatch)=0.5504 | NDCG@20=0.0358
  TA-SID         xq[L0]    [topk] AUC cos(degmatch)=0.9320 cos(uniform)=0.9286 | Ham(degmatch)=0.8847 L0match(degmatch)=0.8655 | NDCG@20=0.0415
  TA-SID         xq[L0]    [all ] AUC cos(degmatch)=0.7240 cos(uniform)=0.7015 | Ham(degmatch)=0.5766 L0match(degmatch)=0.5739 | NDCG@20=0.0415
  TA-SID         xq[full]  [topk] AUC cos(degmatch)=0.9560 cos(uniform)=0.9539 | Ham(degmatch)=0.8847 L0match(degmatch)=0.8655 | NDCG@20=0.0415
  TA-SID         xq[full]  [all ] AUC cos(degmatch)=0.7364 cos(uniform)=0.7151 | Ham(degmatch)=0.5766 L0match(degmatch)=0.5739 | NDCG@20=0.0415
  TA-SID         z         [topk] AUC cos(degmatch)=0.9623 cos(uniform)=0.9622 | Ham(degmatch)=0.8847 L0match(degmatch)=0.8655 | NDCG@20=0.0415
  TA-SID         z         [all ] AUC cos(degmatch)=0.7337 cos(uniform)=0.7123 | Ham(degmatch)=0.5766 L0match(degmatch)=0.5739 | NDCG@20=0.0415

====================================================================================================
主表（xq[full]，Behavior AUC vs NDCG@20）
====================================================================================================

**xq[full] · pos=topk · Sport**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Sport | Content-only | 0.0275 | 0.9080 | 0.9105 | 0.6856 | 0.6849 |
| Sport | ZeroPad | 0.0280 | 0.9069 | 0.9076 | 0.6590 | 0.6558 |
| Sport | Shuffled | 0.0264 | 0.8763 | 0.8823 | 0.6253 | 0.6283 |
| Sport | ItemCF | 0.0295 | 0.9397 | 0.9361 | 0.7374 | 0.7179 |
| Sport | Raw-CoOcc | 0.0305 | 0.9469 | 0.9378 | 0.6881 | 0.6688 |
| Sport | TA-SID | 0.0322 | 0.9647 | 0.9623 | 0.7498 | 0.7450 |

**xq[full] · pos=topk · Toys**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Toys | Content-only | 0.0336 | 0.8751 | 0.8948 | 0.7172 | 0.7135 |
| Toys | ZeroPad | 0.0334 | 0.8716 | 0.8843 | 0.6957 | 0.6910 |
| Toys | Shuffled | 0.0284 | 0.8127 | 0.8260 | 0.6225 | 0.6215 |
| Toys | ItemCF | 0.0355 | 0.9455 | 0.9586 | 0.7796 | 0.7735 |
| Toys | Raw-CoOcc | 0.0340 | 0.9543 | 0.9623 | 0.7655 | 0.7620 |
| Toys | TA-SID | 0.0377 | 0.9651 | 0.9708 | 0.8416 | 0.8370 |

**xq[full] · pos=topk · Beauty**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Beauty | Content-only | 0.0358 | 0.8365 | 0.8370 | 0.6584 | 0.6435 |
| Beauty | TA-SID | 0.0415 | 0.9560 | 0.9539 | 0.8847 | 0.8655 |

**xq[full] · pos=all · Sport**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Sport | Content-only | 0.0275 | 0.7377 | 0.7484 | 0.5519 | 0.5507 |
| Sport | ZeroPad | 0.0280 | 0.7329 | 0.7450 | 0.5406 | 0.5394 |
| Sport | Shuffled | 0.0264 | 0.7195 | 0.7320 | 0.5327 | 0.5322 |
| Sport | ItemCF | 0.0295 | 0.7593 | 0.7618 | 0.5536 | 0.5492 |
| Sport | Raw-CoOcc | 0.0305 | 0.7639 | 0.7604 | 0.5397 | 0.5366 |
| Sport | TA-SID | 0.0322 | 0.7718 | 0.7775 | 0.5545 | 0.5526 |

**xq[full] · pos=all · Toys**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Toys | Content-only | 0.0336 | 0.6744 | 0.6889 | 0.5540 | 0.5534 |
| Toys | ZeroPad | 0.0334 | 0.6724 | 0.6817 | 0.5519 | 0.5510 |
| Toys | Shuffled | 0.0284 | 0.6323 | 0.6454 | 0.5304 | 0.5293 |
| Toys | ItemCF | 0.0355 | 0.7345 | 0.7427 | 0.5721 | 0.5675 |
| Toys | Raw-CoOcc | 0.0340 | 0.7349 | 0.7336 | 0.5616 | 0.5579 |
| Toys | TA-SID | 0.0377 | 0.7623 | 0.7704 | 0.5855 | 0.5843 |

**xq[full] · pos=all · Beauty**

| dataset | variant | ndcg20 | auc_cos_degmatch | auc_cos_uniform | auc_ham_degmatch | auc_l0match_degmatch |
|---|---|---|---|---|---|---|
| Beauty | Content-only | 0.0358 | 0.6680 | 0.6620 | 0.5524 | 0.5504 |
| Beauty | TA-SID | 0.0415 | 0.7364 | 0.7151 | 0.5766 | 0.5739 |

相关分析（Behavior AUC vs NDCG@20）

| pos_regime | scope | metric | n | pearson | p_pearson | spearman | p_spearman |
|---|---|---|---|---|---|---|---|
| topk | Sport | auc_cos_degmatch | 6 | 0.9775 | 0.0008 | 0.9429 | 0.0048 |
| topk | Toys | auc_cos_degmatch | 6 | 0.8933 | 0.0165 | 0.9429 | 0.0048 |
| topk | Pooled | auc_cos_degmatch | 12 | 0.4807 | 0.1136 | 0.4825 | 0.1121 |
| topk | Sport | auc_cos_uniform | 6 | 0.9782 | 0.0007 | 0.9429 | 0.0048 |
| topk | Toys | auc_cos_uniform | 6 | 0.9012 | 0.0142 | 0.9429 | 0.0048 |
| topk | Pooled | auc_cos_uniform | 12 | 0.5909 | 0.0431 | 0.6434 | 0.0240 |
| topk | Sport | auc_ham_degmatch | 6 | 0.8339 | 0.0391 | 0.8857 | 0.0188 |
| topk | Toys | auc_ham_degmatch | 6 | 0.9624 | 0.0021 | 1.0000 | 0.0000 |
| topk | Pooled | auc_ham_degmatch | 12 | 0.8714 | 0.0002 | 0.8741 | 0.0002 |
| topk | Sport | auc_l0match_degmatch | 6 | 0.7963 | 0.0580 | 0.7143 | 0.1108 |
| topk | Toys | auc_l0match_degmatch | 6 | 0.9589 | 0.0025 | 1.0000 | 0.0000 |
| topk | Pooled | auc_l0match_degmatch | 12 | 0.8781 | 0.0002 | 0.8462 | 0.0005 |
| all | Sport | auc_cos_degmatch | 6 | 0.9649 | 0.0018 | 0.9429 | 0.0048 |
| all | Toys | auc_cos_degmatch | 6 | 0.9055 | 0.0130 | 0.9429 | 0.0048 |
| all | Pooled | auc_cos_degmatch | 12 | 0.1472 | 0.6481 | 0.1818 | 0.5717 |
| all | Sport | auc_cos_uniform | 6 | 0.9681 | 0.0015 | 0.8857 | 0.0188 |
| all | Toys | auc_cos_uniform | 6 | 0.9275 | 0.0077 | 1.0000 | 0.0000 |
| all | Pooled | auc_cos_uniform | 12 | 0.1274 | 0.6932 | 0.1189 | 0.7129 |
| all | Sport | auc_ham_degmatch | 6 | 0.5476 | 0.2608 | 0.6000 | 0.2080 |
| all | Toys | auc_ham_degmatch | 6 | 0.9810 | 0.0005 | 1.0000 | 0.0000 |
| all | Pooled | auc_ham_degmatch | 12 | 0.8658 | 0.0003 | 0.8531 | 0.0004 |
| all | Sport | auc_l0match_degmatch | 6 | 0.4913 | 0.3223 | 0.4857 | 0.3287 |
| all | Toys | auc_l0match_degmatch | 6 | 0.9846 | 0.0004 | 1.0000 | 0.0000 |
| all | Pooled | auc_l0match_degmatch | 12 | 0.8648 | 0.0003 | 0.8671 | 0.0003 |
| topk | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9354 | 0.0000 | — | — |
| topk | Pooled(z-scored) | auc_cos_uniform | 12 | 0.9397 | 0.0000 | — | — |
| topk | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.8982 | 0.0001 | — | — |
| topk | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.8776 | 0.0002 | — | — |
| all | Pooled(z-scored) | auc_cos_degmatch | 12 | 0.9352 | 0.0000 | — | — |
| all | Pooled(z-scored) | auc_cos_uniform | 12 | 0.9478 | 0.0000 | — | — |
| all | Pooled(z-scored) | auc_ham_degmatch | 12 | 0.7643 | 0.0038 | — | — |
| all | Pooled(z-scored) | auc_l0match_degmatch | 12 | 0.7380 | 0.0061 | — | — |
```

</details>
