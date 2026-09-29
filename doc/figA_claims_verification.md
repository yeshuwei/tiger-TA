# 图 A 校验报告（自动生成，勿手改）

生成命令：`python3 tools/verify_figA_claims.py`

**汇总：PASS 18 / WARN 0 / FAIL 0**

## 逐项结果

| 层 | 检查项 | 结果 | 说明 |
|:--|:--|:--|:--|
| L1 | 18 格全部有源 | **PASS** | 18/18 |
| L1 | 独立源格数 ≥ 6 | **PASS** | 独立 6 格（Beauty 2 双源 + 4 仅 Tier-1）；另有 12 格为 tier2 表从 probe CSV 转抄，不构成交叉印证 |
| L1 | 双源独立格一致（n=2） | **PASS** | 最大差 0.00e+00 |
| L1 | N@20 上游表可从 eval 日志逐字节重生成 | **PASS** | ablation_3datasets_summary.py 重跑结果与 doc/ablation_main_table_3datasets.md 一致 |
| L2 | 回读点数 == 18 | **PASS** | 读回 18 |
| L2 | 18 格 x/y/yerr 全部与源表一致 | **PASS** | 最大偏差 3.25e-18（容差 1e-09） |
| L2 | 点未串面板（gid 数据集 == 面板标题） | **PASS** | [] |
| L3 | 面板数 == 3 | **PASS** | 3 |
| L3 | 面板标题 == DATASETS | **PASS** | ['Beauty', 'Sport', 'Toys'] |
| L3 | 每面板 6 点且变体集完整 | **PASS** | Beauty:6 / Sport:6 / Toys:6 |
| L3 | 6 变体 (marker,color) 全局唯一 | **PASS** | Content=D ItemCF=s Raw-CoOcc=^ Shuffled=v TA-SID=o ZeroPad=X |
| L3 | marker 形状本身即全局唯一（不依赖颜色） | **PASS** | Content=D ItemCF=s Raw-CoOcc=^ Shuffled=v TA-SID=o ZeroPad=X |
| L3 | 存在 figure 级图例 | **PASS** | fig.legends=1 |
| L3 | 图例条目数 == 6 | **PASS** | ['Content-only', 'ZeroPad', 'Shuffled', 'ItemCF-SVD', 'Raw-CoOcc-SVD', 'TA-SID'] |
| L3 | 图例标签集 == VARIANTS | **PASS** |  |
| L3 | 图例包围盒占画布面积 > 2% | **PASS** | 666.3×22.6 px = 4.09% |
| L3 | 图例包围盒在画布内 | **PASS** | fig.bbox=(0, 0, 1050, 350) |
| L3 | 所有点与误差棒端点均在坐标范围内 | **PASS** | 越界：[] |

## L2 逐格对账（artist 回读 vs 权威表）

| Dataset | Variant | 源 AUC | 图中 AUC | 源 N@20 | 图中 N@20 | 源 std | 图中 yerr | Δ |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| Sport | Content-only | 0.907980 | 0.907980 | 0.0275 | 0.0275 | 0.0007 | 0.0007 | 7.6e-19 |
| Sport | ZeroPad | 0.906870 | 0.906870 | 0.0280 | 0.0280 | 0.0006 | 0.0006 | 1.1e-19 |
| Sport | Shuffled | 0.876267 | 0.876267 | 0.0264 | 0.0264 | 0.0003 | 0.0003 | 1.7e-18 |
| Sport | ItemCF-SVD | 0.939723 | 0.939723 | 0.0295 | 0.0295 | 0.0004 | 0.0004 | 1.0e-18 |
| Sport | Raw-CoOcc-SVD | 0.946925 | 0.946925 | 0.0305 | 0.0305 | 0.0006 | 0.0006 | 1.1e-19 |
| Sport | TA-SID | 0.964747 | 0.964747 | 0.0322 | 0.0322 | 0.0005 | 0.0005 | 4.3e-19 |
| Toys | Content-only | 0.875086 | 0.875086 | 0.0336 | 0.0336 | 0.0007 | 0.0007 | 7.6e-19 |
| Toys | ZeroPad | 0.871581 | 0.871581 | 0.0334 | 0.0334 | 0.0012 | 0.0012 | 2.2e-19 |
| Toys | Shuffled | 0.812701 | 0.812701 | 0.0284 | 0.0284 | 0.0008 | 0.0008 | 1.4e-18 |
| Toys | ItemCF-SVD | 0.945494 | 0.945494 | 0.0355 | 0.0355 | 0.0002 | 0.0002 | 1.2e-18 |
| Toys | Raw-CoOcc-SVD | 0.954332 | 0.954332 | 0.0340 | 0.0340 | 0.0015 | 0.0015 | 1.3e-18 |
| Toys | TA-SID | 0.965060 | 0.965060 | 0.0377 | 0.0377 | 0.0007 | 0.0007 | 7.6e-19 |
| Beauty | Content-only | 0.836513 | 0.836513 | 0.0358 | 0.0358 | 0.0009 | 0.0009 | 2.0e-18 |
| Beauty | TA-SID | 0.956025 | 0.956025 | 0.0415 | 0.0415 | 0.0011 | 0.0011 | 3.3e-18 |
| Beauty | ItemCF-SVD | 0.946298 | 0.946298 | 0.0391 | 0.0391 | 0.0010 | 0.0010 | 8.7e-19 |
| Beauty | Raw-CoOcc-SVD | 0.937674 | 0.937674 | 0.0387 | 0.0387 | 0.0011 | 0.0011 | 3.3e-18 |
| Beauty | ZeroPad | 0.845559 | 0.845559 | 0.0374 | 0.0374 | 0.0004 | 0.0004 | 2.4e-18 |
| Beauty | Shuffled | 0.797425 | 0.797425 | 0.0346 | 0.0346 | 0.0008 | 0.0008 | 2.1e-18 |

## L1 数据来源矩阵

| Variant | Beauty | Sport | Toys |
|:--|:--|:--|:--|
| Content | 双源独立 | 同源转抄 | 同源转抄 |
| ZeroPad | 仅 Tier-1 | 同源转抄 | 同源转抄 |
| Shuffled | 仅 Tier-1 | 同源转抄 | 同源转抄 |
| ItemCF | 仅 Tier-1 | 同源转抄 | 同源转抄 |
| Raw-CoOcc | 仅 Tier-1 | 同源转抄 | 同源转抄 |
| TA-SID | 双源独立 | 同源转抄 | 同源转抄 |

> **来源类型说明**：`双源独立` = probe CSV 与 Tier-1 两次独立运行均产出且一致；`仅 Tier-1` = 只有 Tier-1 产出（Beauty 后补的 4 变体）；`同源转抄` = `beauty_tier2_summary.py` §九 直接从 probe CSV 复制，**不构成交叉印证**。
