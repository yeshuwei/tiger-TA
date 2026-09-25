```
====================================================================================================
Beauty Tier-1 — RQ-VAE encoder(z) 与量化端点(xq[full]) 的行为结构保持诊断
运行时间: 2026-09-23 23:59:03
空间: z (32d, encoder 输出) | xq[full] (32d, 码本重建)　—— 不含 fused 896d
口径: 与 tools/behavior_structure_probe.py 逐位一致（E 图 window=3 / degmatch 负样本 / cosine AUC）
====================================================================================================

▌ 变体资源核验
  Content-only   ckpt ✅ Beauty_Content768d_Baseline      codes ✅ Beauty_content768d_baseline_sid.npy
  ZeroPad        ckpt ✅ Beauty_Zeropad_768d              codes ✅ Beauty_zeropad_768d_sid.npy
  Shuffled       ckpt ✅ Beauty_Shuffled_768d             codes ✅ Beauty_shuffled_768d_sid.npy
  ItemCF         ckpt ✅ Beauty_ItemCF_768d               codes ✅ Beauty_itemcf_768d_sid.npy
  Raw-CoOcc      ckpt ✅ Beauty_RawCoOcc_768d             codes ✅ Beauty_rawcooc_768d_sid.npy
  TA-SID         ckpt ✅ Beauty_TA_SID                    codes ✅ Beauty_ta_sid_768d_sid.npy

E 图: 12101 items | 111649 条边 (count 2–142) | 22363 条训练序列
度分布: mean=18.5 median=11 max=457
[topk] 正样本 1000/1000 | 端点 degree pos=85.1 degmatch=79.5 (Δ=-5.6) uniform=18.5 (Δ=-66.5)
[all] 正样本 20000/20000 | 端点 degree pos=50.3 degmatch=48.3 (Δ=-2.0) uniform=18.5 (Δ=-31.8)

▌ Content-only  (L0distinct=83)
    xq[full]  [topk] AUC degmatch=0.8365 uniform=0.8370
    xq[full]  [all ] AUC degmatch=0.6680 uniform=0.6620
    z         [topk] AUC degmatch=0.8334 uniform=0.8353
    z         [all ] AUC degmatch=0.6650 uniform=0.6595

▌ ZeroPad  (L0distinct=156)
    xq[full]  [topk] AUC degmatch=0.8456 uniform=0.8457
    xq[full]  [all ] AUC degmatch=0.6771 uniform=0.6697
    z         [topk] AUC degmatch=0.8514 uniform=0.8503
    z         [all ] AUC degmatch=0.6739 uniform=0.6677

▌ Shuffled  (L0distinct=214)
    xq[full]  [topk] AUC degmatch=0.7974 uniform=0.8077
    xq[full]  [all ] AUC degmatch=0.6533 uniform=0.6478
    z         [topk] AUC degmatch=0.7841 uniform=0.7887
    z         [all ] AUC degmatch=0.6457 uniform=0.6409

▌ ItemCF  (L0distinct=153)
    xq[full]  [topk] AUC degmatch=0.9463 uniform=0.9537
    xq[full]  [all ] AUC degmatch=0.7252 uniform=0.7120
    z         [topk] AUC degmatch=0.9535 uniform=0.9572
    z         [all ] AUC degmatch=0.7277 uniform=0.7105

▌ Raw-CoOcc  (L0distinct=233)
    xq[full]  [topk] AUC degmatch=0.9377 uniform=0.9267
    xq[full]  [all ] AUC degmatch=0.7163 uniform=0.6833
    z         [topk] AUC degmatch=0.9413 uniform=0.9242
    z         [all ] AUC degmatch=0.7136 uniform=0.6706

▌ TA-SID  (L0distinct=204)
    xq[full]  [topk] AUC degmatch=0.9560 uniform=0.9539
    xq[full]  [all ] AUC degmatch=0.7364 uniform=0.7151
    z         [topk] AUC degmatch=0.9623 uniform=0.9622
    z         [all ] AUC degmatch=0.7337 uniform=0.7123

→ CSV: doc/beauty_tier1_results.csv  (24 行)

▌ Beauty Tier-1 结果表（topk 正样本 × degmatch 负样本）
    Variant            z AUC   xq[full] AUC
    Content-only      0.8334         0.8365
    ZeroPad           0.8514         0.8456
    Shuffled          0.7841         0.7974
    ItemCF            0.9535         0.9463
    Raw-CoOcc         0.9413         0.9377
    TA-SID            0.9623         0.9560

▌ Beauty Tier-1 结果表（all 正样本 × degmatch 负样本）
    Variant            z AUC   xq[full] AUC
    Content-only      0.6650         0.6680
    ZeroPad           0.6739         0.6771
    Shuffled          0.6457         0.6533
    ItemCF            0.7277         0.7252
    Raw-CoOcc         0.7136         0.7163
    TA-SID            0.7337         0.7364

▌ z → xq[full] 排序一致性（Spearman，越大越说明量化保序）
    Beauty   ρ = +1.0000  (p = 0, n = 6)
    Sport    ρ = +1.0000  (p = 0, n = 6)   ← 取自 behavior_structure_probe_results.csv
    Toys     ρ = +1.0000  (p = 0, n = 6)   ← 取自 behavior_structure_probe_results.csv

▌ Variant 排序（按 xq[full] AUC 降序）
    1. TA-SID         0.9560
    2. ItemCF         0.9463
    3. Raw-CoOcc      0.9377
    4. ZeroPad        0.8456
    5. Content-only   0.8365
    6. Shuffled       0.7974

    实测: TA-SID > ItemCF > Raw-CoOcc > ZeroPad > Content-only > Shuffled
    参照: TA-SID > Raw-CoOcc > ItemCF > Content-only > ZeroPad > Shuffled   （Sport/Toys 的 xq[full] 实测排序）
    成对顺序保持: 13/15
    两端: TA-SID 第 1 名 | Shuffled 第 6 名

====================================================================================================
▌ 结论
====================================================================================================
Q1 Beauty 中 RQ-VAE encoder 是否形成行为结构？
   → 是：6 变体 z(32d) AUC 均值 0.8877（范围 0.7841–0.9623），远高于随机 0.5；且变体间存在稳定区分度（极差 0.1783）。

Q2 量化是否破坏行为结构？
   → 未破坏：z → xq[full] Spearman ρ = +1.0000；xq[full] AUC 均值 0.8866（z 为 0.8877，差 -0.0011）。

Q3 Beauty 的 variant 排序是否与 Sport/Toys 一致？
   → 实测: TA-SID > ItemCF > Raw-CoOcc > ZeroPad > Content-only > Shuffled
     参照: TA-SID > Raw-CoOcc > ItemCF > {Content-only, ZeroPad} > Shuffled
     成对顺序保持 13/15；两端 TA-SID/Shuffled 位置见上。
```
