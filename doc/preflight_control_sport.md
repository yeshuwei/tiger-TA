> ⚠️⚠️ **2026-09-27 订正：本文件（2026-09-23 原始输出，下文原样保留）中 Sport 的 TA-SID 行已作废。**
> 根因：TA-SID 被喂了 `fused_item_emb.parquet` —— 该文件在 Sport/Toys 上是 **4224d**（4096d 内容 ⊕ 128d 行为），
> 不是其余 5 个变体所用的 896d。故下文「`TA-SID  fused_item_emb.parquet  dim=4224`」、
> 「`TA-SID AUC=0.7835`」以及由此得出的「量化前 AUC 排序把 TA-SID 判为末位」**全部是这一个维度 bug 的产物**。
> 正确 896d 文件（`fused_item_emb_768d.parquet`）下：**TA-SID 是第 1 名，AUC = 0.9617**（主口径 topk × degmatch）。
> **其余 5 个变体的数值逐位有效**（已被 `tools/stage_probe.py --audit_a` 独立复现）。
> **⇒ 勿引用本文件的 TA-SID 行、"dim=4224" 与"量化前排序"结论。** CSV 中对应 4 行已加 `space_dim_ok=False` 标注。
> 权威记录：`notes/15_g2_stage_seed_robustness_plan.md` §10。脚本已修：`tools/beauty_preflight_probe.py`。

```
====================================================================================================
Beauty 补齐变体 — 量化前（fused 896d）行为结构 pre-flight 诊断
运行时间: 2026-09-23 23:25:02
⚠ 本表是量化前口径，绝对 AUC 与主表（xq[full]）不可比，勿混引
E 图: train.parquet history+[target] → 滑窗 window=3（±1 相邻对）
====================================================================================================

E 图: 18357 items | 180361 条边 (count 2–62) | 35598 条训练序列
度分布: mean=19.7 median=11 max=994 零度=28
[topk] 可用正样本 998/1000 | 端点 degree pos=179.7 degmatch=162.0 (Δ=-17.7) uniform=20.2 (Δ=-159.5)
[all] 可用正样本 19999/20000 | 端点 degree pos=65.6 degmatch=63.0 (Δ=-2.6) uniform=19.8 (Δ=-45.8)

▌ 融合输入与行序核验
  行序基准 = fused_item_emb_zeropad.parquet
  ✅ Content-only   fused_item_emb_zeropad.parquet     dim=896
  ✅ ZeroPad        fused_item_emb_zeropad.parquet     dim=896
  ✅ Shuffled       fused_item_emb_shuffled.parquet    dim=896
  ✅ ItemCF         fused_item_emb_itemcf.parquet      dim=896
  ✅ Raw-CoOcc      fused_item_emb_rawcooc.parquet     dim=896
  ✅ TA-SID         fused_item_emb.parquet             dim=4224

→ CSV: doc/preflight_control_sport.csv  (24 行)

▌ 量化前 Behavior AUC（fused 896d）

--- pos=topk × neg=degmatch ---
    ItemCF         AUC=0.9585   (cos_pos=0.883 cos_neg=0.777)   NDCG@20=0.0295
    Raw-CoOcc      AUC=0.9498   (cos_pos=0.822 cos_neg=0.730)   NDCG@20=0.0305
    Content-only   AUC=0.8964   (cos_pos=0.908 cos_neg=0.847)   NDCG@20=0.0275
    ZeroPad        AUC=0.8964   (cos_pos=0.908 cos_neg=0.847)   NDCG@20=0.0280
    Shuffled       AUC=0.8781   (cos_pos=0.781 cos_neg=0.729)   NDCG@20=0.0264
    TA-SID         AUC=0.7835   (cos_pos=0.799 cos_neg=0.679)   NDCG@20=0.0322

--- pos=topk × neg=uniform ---
    ItemCF         AUC=0.9552   (cos_pos=0.883 cos_neg=0.783)   NDCG@20=0.0295
    Raw-CoOcc      AUC=0.9370   (cos_pos=0.822 cos_neg=0.734)   NDCG@20=0.0305
    Content-only   AUC=0.9136   (cos_pos=0.908 cos_neg=0.841)   NDCG@20=0.0275
    ZeroPad        AUC=0.9136   (cos_pos=0.908 cos_neg=0.841)   NDCG@20=0.0280
    Shuffled       AUC=0.8941   (cos_pos=0.781 cos_neg=0.724)   NDCG@20=0.0264
    TA-SID         AUC=0.7854   (cos_pos=0.799 cos_neg=0.679)   NDCG@20=0.0322

--- pos=all × neg=degmatch ---
    ItemCF         AUC=0.7666   (cos_pos=0.824 cos_neg=0.779)   NDCG@20=0.0295
    Raw-CoOcc      AUC=0.7454   (cos_pos=0.767 cos_neg=0.730)   NDCG@20=0.0305
    Content-only   AUC=0.7186   (cos_pos=0.871 cos_neg=0.844)   NDCG@20=0.0275
    ZeroPad        AUC=0.7186   (cos_pos=0.871 cos_neg=0.844)   NDCG@20=0.0280
    Shuffled       AUC=0.7041   (cos_pos=0.750 cos_neg=0.726)   NDCG@20=0.0264
    TA-SID         AUC=0.6246   (cos_pos=0.722 cos_neg=0.676)   NDCG@20=0.0322

--- pos=all × neg=uniform ---
    ItemCF         AUC=0.7499   (cos_pos=0.824 cos_neg=0.782)   NDCG@20=0.0295
    Content-only   AUC=0.7480   (cos_pos=0.871 cos_neg=0.840)   NDCG@20=0.0275
    ZeroPad        AUC=0.7480   (cos_pos=0.871 cos_neg=0.840)   NDCG@20=0.0280
    Shuffled       AUC=0.7292   (cos_pos=0.750 cos_neg=0.723)   NDCG@20=0.0264
    Raw-CoOcc      AUC=0.7238   (cos_pos=0.767 cos_neg=0.733)   NDCG@20=0.0305
    TA-SID         AUC=0.6348   (cos_pos=0.722 cos_neg=0.672)   NDCG@20=0.0322

▌ 排序比对（pos=topk × neg=degmatch，主口径）
    量化前 AUC 排序: ItemCF > Raw-CoOcc > Content-only > ZeroPad > Shuffled > TA-SID
    Sport/Toys 期望 : TA-SID > Raw-CoOcc > ItemCF > Content-only > ZeroPad > Shuffled
    与期望排序的相邻正确对数: 3/5

▌ 补零等价性核验（Content-only：896d 补零 vs 原生 768d）
    ItemID 行序一致: True
    [topk] AUC 896d(补零)=0.896381  vs  768d(原生)=0.896381  Δ=+0.00e+00
    [all] AUC 896d(补零)=0.718558  vs  768d(原生)=0.718558  Δ=+0.00e+00

⚠️ 口径提醒：本报告为**量化前**结果，绝对 AUC 高于主表（xq[full]）属正常，两者不可混引；本报告只用于判断**排序方向** go/no-go。
```
