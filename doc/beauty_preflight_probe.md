```
====================================================================================================
Beauty 补齐变体 — 量化前（fused 896d）行为结构 pre-flight 诊断
运行时间: 2026-09-23 23:24:38
⚠ 本表是量化前口径，绝对 AUC 与主表（xq[full]）不可比，勿混引
E 图: train.parquet history+[target] → 滑窗 window=3（±1 相邻对）
====================================================================================================

E 图: 12101 items | 111649 条边 (count 2–142) | 22363 条训练序列
度分布: mean=18.5 median=11 max=457 零度=33
[topk] 可用正样本 1000/1000 | 端点 degree pos=85.1 degmatch=79.5 (Δ=-5.6) uniform=18.5 (Δ=-66.5)
[all] 可用正样本 20000/20000 | 端点 degree pos=50.3 degmatch=48.3 (Δ=-2.0) uniform=18.5 (Δ=-31.8)

▌ 融合输入与行序核验
  行序基准 = fused_item_emb_zeropad.parquet
  ✅ Content-only   fused_item_emb_zeropad.parquet     dim=896
  ✅ ZeroPad        fused_item_emb_zeropad.parquet     dim=896
  ✅ Shuffled       fused_item_emb_shuffled.parquet    dim=896
  ✅ ItemCF         fused_item_emb_itemcf.parquet      dim=896
  ✅ Raw-CoOcc      fused_item_emb_rawcooc.parquet     dim=896
  ✅ TA-SID         fused_item_emb.parquet             dim=896

→ CSV: doc/beauty_preflight_results.csv  (24 行)

▌ 量化前 Behavior AUC（fused 896d）

--- pos=topk × neg=degmatch ---
    TA-SID         AUC=0.9939   (cos_pos=0.894 cos_neg=0.740)   NDCG@20=0.0415
    ItemCF         AUC=0.9782   NDCG@20=—
    Raw-CoOcc      AUC=0.9088   NDCG@20=—
    Content-only   AUC=0.8368   (cos_pos=0.911 cos_neg=0.860)   NDCG@20=0.0358
    ZeroPad        AUC=0.8368   NDCG@20=—
    Shuffled       AUC=0.8227   NDCG@20=—

--- pos=topk × neg=uniform ---
    TA-SID         AUC=0.9946   (cos_pos=0.894 cos_neg=0.737)   NDCG@20=0.0415
    ItemCF         AUC=0.9855   NDCG@20=—
    Content-only   AUC=0.8499   (cos_pos=0.911 cos_neg=0.857)   NDCG@20=0.0358
    ZeroPad        AUC=0.8499   NDCG@20=—
    Shuffled       AUC=0.8350   NDCG@20=—
    Raw-CoOcc      AUC=0.8304   NDCG@20=—

--- pos=all × neg=degmatch ---
    ItemCF         AUC=0.7644   NDCG@20=—
    TA-SID         AUC=0.7463   (cos_pos=0.778 cos_neg=0.737)   NDCG@20=0.0415
    Raw-CoOcc      AUC=0.6975   NDCG@20=—
    Content-only   AUC=0.6556   (cos_pos=0.876 cos_neg=0.857)   NDCG@20=0.0358
    ZeroPad        AUC=0.6556   NDCG@20=—
    Shuffled       AUC=0.6390   NDCG@20=—

--- pos=all × neg=uniform ---
    TA-SID         AUC=0.7462   (cos_pos=0.778 cos_neg=0.737)   NDCG@20=0.0415
    ItemCF         AUC=0.7359   NDCG@20=—
    Content-only   AUC=0.6622   (cos_pos=0.876 cos_neg=0.857)   NDCG@20=0.0358
    ZeroPad        AUC=0.6622   NDCG@20=—
    Shuffled       AUC=0.6456   NDCG@20=—
    Raw-CoOcc      AUC=0.6098   NDCG@20=—

▌ 排序比对（pos=topk × neg=degmatch，主口径）
    量化前 AUC 排序: TA-SID > ItemCF > Raw-CoOcc > Content-only > ZeroPad > Shuffled
    Sport/Toys 期望 : TA-SID > Raw-CoOcc > ItemCF > Content-only > ZeroPad > Shuffled
    与期望排序的相邻正确对数: 4/5

▌ 补零等价性核验（Content-only：896d 补零 vs 原生 768d）
    ItemID 行序一致: True
    [topk] AUC 896d(补零)=0.836845  vs  768d(原生)=0.836844  Δ=+1.50e-06
    [all] AUC 896d(补零)=0.655622  vs  768d(原生)=0.655622  Δ=+1.50e-08

⚠️ 口径提醒：本报告为**量化前**结果，绝对 AUC 高于主表（xq[full]）属正常，两者不可混引；本报告只用于判断**排序方向** go/no-go。
```
