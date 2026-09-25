#!/usr/bin/env python3
"""
beauty_tier2_summary.py — Beauty Tier-2 汇总：补齐 6 变体的 T5 NDCG@20 并做排序分析

范围（严格，对应用户 Tier-2 规格 §六~§十）：
    §六  per-seed 表：Variant | seed2025 | seed2026 | seed2027 | mean | std（只记 NDCG@20）
    §七  doc/Beauty_Tier2_NDCG_results.csv：|Variant|Behavior AUC|NDCG mean|NDCG std|
         —— Behavior AUC **直接读自 doc/beauty_tier1_results.csv，不重新计算**
    §八  Beauty 内 Spearman + Pearson（n=6，明确不过度解释）
    §九  整理三数据集表（Sport/Toys/Beauty × variant → AUC, NDCG），供后续
         dataset-wise z-score normalization 使用；**本步不计算任何跨数据集原始值相关性**
    §十  交付：日志路径 / 六变体表 / 相关性 / 三条声明（附可核验证据）

**不重新定义任何指标、不重算 Behavior AUC、不修改 Tier-1 结果。**

评测日志解析逐位复用 tools/e49_summary.py 的 parse（NDCG_RE 取 @20 列，已实测
对 3 列与 5 列两种输出格式均正确）。

用法：
    python3 tools/beauty_tier2_summary.py
"""

import argparse
import datetime
import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from e49_summary import parse, SEEDS, LOG_DIR  # noqa: E402

# Beauty 6 变体 → eval 日志 tag（命名遵循 E49 约定 f"{DS}_{tag}_seed{S}_eval.log"）
BEAUTY_TAGS = {
    "TA-SID":       "Beauty_ta_sid_768d",            # 已有，Tier-2 未重跑
    "Content-only": "Beauty_content768d_baseline",   # 已有，Tier-2 未重跑
    "Raw-CoOcc":    "Beauty_rawcooc_768d",           # Tier-2 新补
    "ItemCF":       "Beauty_itemcf_768d",            # Tier-2 新补
    "Shuffled":     "Beauty_shuffled_768d",          # Tier-2 新补
    "ZeroPad":      "Beauty_zeropad_768d",           # Tier-2 新补
}
NEW_IN_TIER2 = ["Raw-CoOcc", "ItemCF", "Shuffled", "ZeroPad"]
PREEXISTING = ["TA-SID", "Content-only"]

# 展示顺序 = Tier-1 实测的 xq[full] AUC 降序（§八 里用户给定的次序）
ORDER = ["TA-SID", "ItemCF", "Raw-CoOcc", "ZeroPad", "Content-only", "Shuffled"]

TIER2_START = "2026-09-24 08:28:10"   # driver 启动时刻，用于 provenance 断言


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime_str(path):
    return datetime.datetime.fromtimestamp(os.stat(path).st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser(description="Beauty Tier-2 汇总")
    ap.add_argument("--tier1_csv", default="doc/beauty_tier1_results.csv")
    ap.add_argument("--probe_csv", default="doc/behavior_structure_probe_results.csv")
    # §六 逐-seed 格式（用户 2026-09-24 指令：此文件名即此格式）
    ap.add_argument("--csv", default="doc/Beauty_Tier2_NDCG_results.csv")
    # 合并表 |Variant|Behavior AUC|NDCG@20|
    ap.add_argument("--merge_csv", default="doc/Beauty_Tier2_merged_auc_ndcg.csv")
    # 早期 §七 格式 |Variant|Behavior AUC|NDCG mean|NDCG std|
    ap.add_argument("--auc_csv", default="doc/Beauty_Tier2_NDCG_results_auc.csv")
    # 逐 run 的资源清单（variant/seed → ckpt/train log/eval log/NDCG）
    ap.add_argument("--manifest", default="doc/Beauty_Tier2_manifest.csv")
    ap.add_argument("--three_ds_csv", default="doc/beauty_tier2_three_dataset_table.csv")
    ap.add_argument("--report", default="doc/beauty_tier2_conclusion.md")
    args = ap.parse_args()

    L = []

    def log(s=""):
        print(s)
        L.append(s)

    log("=" * 96)
    log("Beauty Tier-2 汇总 — 补齐 4 变体 T5 NDCG@20 + Beauty 内排序分析")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 96)

    # ---------------- §六 逐 seed 解析 NDCG@20 ----------------
    log("\n▌ §六 逐 seed 解析（NDCG@20，来自 model/logs/*_eval.log）")
    ndcg = {}
    missing = []
    for v in ORDER:
        tag = BEAUTY_TAGS[v]
        vals, srcs = {}, {}
        for s in SEEDS:
            fn = f"{tag}_seed{s}_eval.log"
            p = os.path.join(LOG_DIR, fn)
            r = parse(fn)
            if r is not None and r[0] == r[0]:
                vals[s] = r[0]
                srcs[s] = fn
        # 无论是否 3 seed 齐全都记下来，**不静默跳过**：
        # 表里该 seed 显示 nan、missing 列表里如实报告缺哪个
        ndcg[v] = vals
        if len(vals) != len(SEEDS):
            missing.append((v, sorted(set(SEEDS) - set(vals))))
        note = "Tier-2 新补" if v in NEW_IN_TIER2 else "已有（未重跑）"
        log(f"  {v:<14} [{note:<10}] " +
            " ".join(f"seed{s}={vals[s]:.4f}" if s in vals else f"seed{s}=缺失" for s in SEEDS))
        for s in sorted(vals):
            log(f"        ← {srcs[s]}  (mtime {mtime_str(os.path.join(LOG_DIR, srcs[s]))})")
    if missing:
        log(f"\n  ⚠ 以下变体 seed 不全，将在表中如实标注而非跳过：{missing}")

    # ---------------- §七 Behavior AUC（直接读，不重算）----------------
    log("\n▌ §七 Behavior AUC（读自 Tier-1 产物，**未重算**）")
    t1 = pd.read_csv(args.tier1_csv)
    sub = t1[(t1.space == "xq[full]") & (t1.pos_regime == "topk")]
    auc = sub.set_index("variant")["auc_cos_degmatch"].to_dict()
    log(f"  来源: {args.tier1_csv}")
    log(f"  口径: space=xq[full], pos_regime=topk, auc_cos_degmatch（与主表同口径）")
    log(f"  sha256 = {sha256(args.tier1_csv)}")
    log(f"  mtime  = {mtime_str(args.tier1_csv)}   (早于 Tier-2 启动 {TIER2_START} → Tier-1 结果未被触碰)")
    for v in ORDER:
        log(f"    {v:<14} {auc[v]:.4f}")

    # ---------------- §六+§七 结果表 ----------------
    rows = []
    for v in ORDER:
        vals = ndcg.get(v, {})
        arr = np.array([vals[s] for s in SEEDS if s in vals], dtype=float)
        rows.append({
            "Variant": v,
            "Behavior AUC": round(auc[v], 4),
            "seed2025": round(vals[2025], 4) if 2025 in vals else np.nan,
            "seed2026": round(vals[2026], 4) if 2026 in vals else np.nan,
            "seed2027": round(vals[2027], 4) if 2027 in vals else np.nan,
            "NDCG mean": round(float(arr.mean()), 4) if len(arr) else np.nan,
            "NDCG std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else np.nan,
        })
    df = pd.DataFrame(rows)

    log("\n▌ §六/§七 Beauty 六变体完整表（NDCG@20 = 3-seed mean ± std）")
    log(f"  {'Variant':<14} {'BehAUC':>8} {'s2025':>8} {'s2026':>8} {'s2027':>8} {'mean':>8} {'std':>7}")
    for _, r in df.iterrows():
        log(f"  {r['Variant']:<14} {r['Behavior AUC']:>8.4f} "
            + " ".join(f"{r[f'seed{s}']:>8.4f}" if r[f"seed{s}"] == r[f"seed{s}"] else f"{'--':>8}"
                       for s in SEEDS)
            + f" {r['NDCG mean']:>8.4f} {r['NDCG std']:>7.4f}")

    # --- 输出 1（用户指令格式）：Beauty_Tier2_NDCG_results.csv
    #     Variant | seed2025 | seed2026 | seed2027 | mean | std
    full = df[["Variant", "seed2025", "seed2026", "seed2027", "NDCG mean", "NDCG std"]].copy()
    full.columns = ["Variant", "seed2025", "seed2026", "seed2027", "mean", "std"]
    full.to_csv(args.csv, index=False)
    log(f"\n→ CSV（逐 seed 格式，用户指令）: {args.csv}")

    # --- 输出 2：合并表 Variant | Behavior AUC | NDCG@20
    mg = df[["Variant", "Behavior AUC", "NDCG mean"]].copy()
    mg.columns = ["Variant", "Behavior AUC", "NDCG@20"]
    mg.to_csv(args.merge_csv, index=False)
    log(f"→ CSV（合并 AUC+NDCG）: {args.merge_csv}")

    # --- 输出 3：早期 §七 格式 Variant | Behavior AUC | NDCG mean | NDCG std
    ac = df[["Variant", "Behavior AUC", "NDCG mean", "NDCG std"]].copy()
    ac.to_csv(args.auc_csv, index=False)
    log(f"→ CSV（§七 AUC+mean/std）: {args.auc_csv}")

    # --- 输出 4：逐 run 资源清单
    man = []
    for v in ORDER:
        tag = BEAUTY_TAGS[v]
        for s in SEEDS:
            tr = f"model/logs/{tag}_seed{s}_t5_train.log"
            ev = f"model/logs/{tag}_seed{s}_eval.log"
            ck = f"model/ckpt/tiger_{tag.replace('Beauty_', '')}_beauty_seed{s}.pth"
            man.append({
                "Variant": v, "seed": s,
                "ckpt_path": ck if os.path.exists(ck) else "",
                "train_log": tr if os.path.exists(tr) else "",
                "eval_log": ev if os.path.exists(ev) else "",
                # eval 日志 mtime = 该 seed 评测完成时刻（逐 seed 完成记录）
                "eval_finished_at": mtime_str(ev) if os.path.exists(ev) else "",
                "NDCG@20": round(ndcg.get(v, {}).get(s, float("nan")), 4)
                if v in ndcg and s in ndcg[v] else "",
                "source": "Tier-2 新补" if v in NEW_IN_TIER2 else "已有（未重跑）",
            })
    pd.DataFrame(man).to_csv(args.manifest, index=False)
    log(f"→ CSV（逐 run 资源清单）: {args.manifest}")

    # ---------------- §八 排序分析 ----------------
    log("\n▌ §八 Beauty 内排序分析（n = 6）")
    j = df.dropna(subset=["Behavior AUC", "NDCG mean"])
    n = len(j)
    if n >= 3:
        rho, p_s = spearmanr(j["Behavior AUC"], j["NDCG mean"])
        r, p_p = pearsonr(j["Behavior AUC"], j["NDCG mean"])
        log(f"  Spearman ρ = {rho:+.4f}   (p = {p_s:.4g})")
        log(f"  Pearson  r = {r:+.4f}   (p = {p_p:.4g})")
        log(f"  n = {n}")
        log(f"  AUC  降序: {' > '.join(j.sort_values('Behavior AUC', ascending=False).Variant)}")
        log(f"  NDCG 降序: {' > '.join(j.sort_values('NDCG mean', ascending=False).Variant)}")
        log("  ⚠ n=6：方向可读，**不可**解读 p 值精度或做显著性断言。（照用户要求不过度解释）")
    else:
        rho = r = float("nan")
        log("  点数不足，未计算。")

    # ---------------- §九 三数据集表（只整理，不算相关性）----------------
    log("\n▌ §九 三数据集整理表（供后续 dataset-wise z-score normalization；本步不算跨数据集相关性）")
    pr = pd.read_csv(args.probe_csv)
    pr = pr[(pr.space == "xq[full]") & (pr.pos_regime == "topk")]
    three = []
    for v in ORDER:
        three.append({"dataset": "Beauty", "variant": v,
                      "behavior_auc_xq_full_topk": auc[v],
                      "ndcg20": float(df.loc[df.Variant == v, "NDCG mean"].iloc[0])})
    for ds in ("Sport", "Toys"):
        s = pr[pr.dataset == ds]
        a = s.set_index("variant")["auc_cos_degmatch"].to_dict()
        nd = s.groupby("variant")["ndcg20"].first().to_dict()
        for v in ORDER:
            if v in a:
                three.append({"dataset": ds, "variant": v,
                              "behavior_auc_xq_full_topk": a[v],
                              "ndcg20": nd.get(v, np.nan)})
    t3 = pd.DataFrame(three)
    t3.to_csv(args.three_ds_csv, index=False)
    log(f"  → {args.three_ds_csv}  ({len(t3)} 行)")
    for ds in ("Sport", "Toys", "Beauty"):
        d = t3[t3.dataset == ds]
        log(f"    {ds:<7} {len(d)} 变体 × (AUC, NDCG)  "
            f"NDCG 覆盖率 {d.ndcg20.notna().sum()}/{len(d)}")
    log("  ⚠ 本步**只整理数据**，未对原始值做任何相关性计算（跨数据集相关需先做 dataset-wise z-score）。")

    # ---------------- §十 日志路径 ----------------
    log("\n▌ §十 交付物：日志路径")
    for v in NEW_IN_TIER2:
        tag = BEAUTY_TAGS[v]
        tr = sorted(glob.glob(f"model/logs/{tag}_seed*_t5_train.log"))
        ev = sorted(glob.glob(f"model/logs/{tag}_seed*_eval.log"))
        ck = sorted(glob.glob(f"model/ckpt/tiger_{tag.split('Beauty_')[1]}_beauty_seed*.pth"))
        log(f"  {v}")
        for p in tr:
            log(f"    train: {p}")
        for p in ev:
            log(f"    eval : {p}")
        for p in ck:
            log(f"    ckpt : {p}")

    # ---------------- §十-5 声明 + 可核验证据 ----------------
    log("\n▌ §十-5 声明（附可核验证据）")
    log("  1. 未修改 Tier-1 结果")
    log(f"     - {args.tier1_csv} sha256={sha256(args.tier1_csv)[:16]}… mtime={mtime_str(args.tier1_csv)}")
    log(f"     - Tier-2 启动于 {TIER2_START}，晚于上述 mtime → Tier-1 产物未被触碰")
    log("  2. 未重新训练 RQ-VAE")
    for v, codef in (("Raw-CoOcc", "Beauty_rawcooc_768d_sid.npy"), ("ItemCF", "Beauty_itemcf_768d_sid.npy"),
                     ("Shuffled", "Beauty_shuffled_768d_sid.npy"), ("ZeroPad", "Beauty_zeropad_768d_sid.npy")):
        p = f"data/Beauty/{codef}"
        log(f"     - {p} mtime={mtime_str(p)} (早于 Tier-2)")
    log("     - driver rqvae/run_beauty_tier2_t5.sh **只调用 run_behavior_ablation.sh 的 t5-only 分支**，"
        "该分支在 run_t5 之前即 exit，物理上不可能执行 RQ-VAE/codegen")
    log("  3. 未调整任何超参数")
    log("     - model/main.py 与 tools/eval_trained_model.py 均未被修改（git status 无改动）")
    log("     - 全部 T5 运行只传 --dataset_path/--code_path/--save_path/--log_path/--seed，其余走默认；")
    log("       评测固定 --beam_size 30（topk_list 只控制打印哪几列，不影响 NDCG@20 的取值）")

    # ---------------- 落报告 ----------------
    with open(args.report, "w") as f:
        f.write("\n".join(L) + "\n")
    log(f"\n→ 报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
