> ⚠️⚠️ **2026-09-27 订正：本文件（2026-09-23 原始输出，下文原样保留）中 Toys 的 TA-SID 行已作废。**
> 根因：TA-SID 被喂了 `fused_item_emb.parquet` —— 该文件在 Sport/Toys 上是 **4224d**（4096d 内容 ⊕ 128d 行为），
> 不是其余 5 个变体所用的 896d。故下文「`TA-SID  fused_item_emb.parquet  dim=4224`」、
> 「`TA-SID AUC=0.7835`」以及由此得出的「量化前 AUC 排序把 TA-SID 判为末位」**全部是这一个维度 bug 的产物**。
> 正确 896d 文件（`fused_item_emb_768d.parquet`）下：**TA-SID 是第 1 名，AUC = 0.9828**（主口径 topk × degmatch）。
> **其余 5 个变体的数值逐位有效**（已被 `tools/stage_probe.py --audit_a` 独立复现）。
> **⇒ 勿引用本文件的 TA-SID 行、"dim=4224" 与"量化前排序"结论。** CSV 中对应 4 行已加 `space_dim_ok=False` 标注。
> 权威记录：`notes/15_g2_stage_seed_robustness_plan.md` §10。脚本已修：`tools/beauty_preflight_probe.py`。

```
====================================================================================================
Beauty 补齐变体 — 量化前（fused 896d）行为结构 pre-flight 诊断
运行时间: 2026-09-23 23:25:07
⚠ 本表是量化前口径，绝对 AUC 与主表（xq[full]）不可比，勿混引
E 图: train.parquet history+[target] → 滑窗 window=3（±1 相邻对）
====================================================================================================

E 图: 11924 items | 102151 条边 (count 2–58) | 19412 条训练序列
度分布: mean=17.1 median=11 max=273 零度=24
[topk] 可用正样本 1000/1000 | 端点 degree pos=59.0 degmatch=55.7 (Δ=-3.2) uniform=16.6 (Δ=-42.3)
[all] 可用正样本 20000/20000 | 端点 degree pos=38.9 degmatch=37.4 (Δ=-1.5) uniform=17.1 (Δ=-21.8)

▌ 融合输入与行序核验
  行序基准 = fused_item_emb_zeropad.parquet
  ✅ Content-only   fused_item_emb_zeropad.parquet     dim=896
  ✅ ZeroPad        fused_item_emb_zeropad.parquet     dim=896
  ✅ Shuffled       fused_item_emb_shuffled.parquet    dim=896
  ✅ ItemCF         fused_item_emb_itemcf.parquet      dim=896
  ✅ Raw-CoOcc      fused_item_emb_rawcooc.parquet     dim=896
  ✅ TA-SID         fused_item_emb.parquet             dim=4224

→ CSV: doc/preflight_control_toys.csv  (24 行)

▌ 量化前 Behavior AUC（fused 896d）

--- pos=topk × neg=degmatch ---
    ItemCF         AUC=0.9621   (cos_pos=0.896 cos_neg=0.797)   NDCG@20=0.0355
    Raw-CoOcc      AUC=0.9359   (cos_pos=0.840 cos_neg=0.748)   NDCG@20=0.0340
    Content-only   AUC=0.8579   (cos_pos=0.915 cos_neg=0.866)   NDCG@20=0.0336
    ZeroPad        AUC=0.8579   (cos_pos=0.915 cos_neg=0.866)   NDCG@20=0.0334
    Shuffled       AUC=0.8423   (cos_pos=0.787 cos_neg=0.745)   NDCG@20=0.0284
    TA-SID         AUC=0.8070   (cos_pos=0.824 cos_neg=0.678)   NDCG@20=0.0377

--- pos=topk × neg=uniform ---
    ItemCF         AUC=0.9702   (cos_pos=0.896 cos_neg=0.788)   NDCG@20=0.0355
    Raw-CoOcc      AUC=0.9297   (cos_pos=0.840 cos_neg=0.750)   NDCG@20=0.0340
    Content-only   AUC=0.8864   (cos_pos=0.915 cos_neg=0.859)   NDCG@20=0.0336
    ZeroPad        AUC=0.8864   (cos_pos=0.915 cos_neg=0.859)   NDCG@20=0.0334
    Shuffled       AUC=0.8662   (cos_pos=0.787 cos_neg=0.738)   NDCG@20=0.0284
    TA-SID         AUC=0.8055   (cos_pos=0.824 cos_neg=0.687)   NDCG@20=0.0377

--- pos=all × neg=degmatch ---
    ItemCF         AUC=0.7631   (cos_pos=0.834 cos_neg=0.791)   NDCG@20=0.0355
    Raw-CoOcc      AUC=0.7152   (cos_pos=0.780 cos_neg=0.747)   NDCG@20=0.0340
    Content-only   AUC=0.6662   (cos_pos=0.882 cos_neg=0.863)   NDCG@20=0.0336
    ZeroPad        AUC=0.6662   (cos_pos=0.882 cos_neg=0.863)   NDCG@20=0.0334
    Shuffled       AUC=0.6500   (cos_pos=0.759 cos_neg=0.742)   NDCG@20=0.0284
    TA-SID         AUC=0.6322   (cos_pos=0.729 cos_neg=0.680)   NDCG@20=0.0377

--- pos=all × neg=uniform ---
    ItemCF         AUC=0.7646   (cos_pos=0.834 cos_neg=0.789)   NDCG@20=0.0355
    Content-only   AUC=0.6878   (cos_pos=0.882 cos_neg=0.860)   NDCG@20=0.0336
    ZeroPad        AUC=0.6878   (cos_pos=0.882 cos_neg=0.860)   NDCG@20=0.0334
    Raw-CoOcc      AUC=0.6859   (cos_pos=0.780 cos_neg=0.751)   NDCG@20=0.0340
    Shuffled       AUC=0.6710   (cos_pos=0.759 cos_neg=0.740)   NDCG@20=0.0284
    TA-SID         AUC=0.6280   (cos_pos=0.729 cos_neg=0.686)   NDCG@20=0.0377

▌ 排序比对（pos=topk × neg=degmatch，主口径）
    量化前 AUC 排序: ItemCF > Raw-CoOcc > Content-only > ZeroPad > Shuffled > TA-SID
    Sport/Toys 期望 : TA-SID > Raw-CoOcc > ItemCF > Content-only > ZeroPad > Shuffled
    与期望排序的相邻正确对数: 3/5

▌ 补零等价性核验（Content-only：896d 补零 vs 原生 768d）
    ItemID 行序一致: True
    [topk] AUC 896d(补零)=0.857900  vs  768d(原生)=0.857900  Δ=+0.00e+00
    [all] AUC 896d(补零)=0.666246  vs  768d(原生)=0.666246  Δ=+0.00e+00

⚠️ 口径提醒：本报告为**量化前**结果，绝对 AUC 高于主表（xq[full]）属正常，两者不可混引；本报告只用于判断**排序方向** go/no-go。
```
