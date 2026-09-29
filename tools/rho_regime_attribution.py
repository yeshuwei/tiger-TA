#!/usr/bin/env python3
"""rho_regime_attribution.py — ρ(A,B) 的口径敏感性归因（2026-09-27）

背景
----
订正文件维度 bug 后，主口径（pos=topk × neg=degmatch）给 ρ(A输入,B z) = 0.928 / 0.928 / 0.986
（Sport/Toys/Beauty）。但换正/负样本口径后区间达 **0.638–0.986** —— 摆动比种子敏感性（=0）大一个数量级。
若论文只报 0.928，审稿人换口径重算得到 0.638，会被判定为挑数字。
故必须先回答两件事：

    Q-a  0.638 是**哪些变体对的相对顺序**翻了？（即：是整体乱序，还是个别相邻对抖动？）
    Q-b  主口径是不是**方法论上本来就该选**的那个？（否则"主口径"只是事后挑出来的）

本脚本做两件事
--------------
**Part 1（离线，只读 `doc/g2_stage_probe_results.csv`，不重算、不加载模型）**
    四个口径组合 × 三个空间对 × 三数据集，算出完整的 ρ 表；
    对 ρ < 0.9 的单元，逐对列出 A 空间与 B 空间的排序，并**打印出翻转的具体变体对**。

**Part 2（重建样本，但**不加载任何模型** —— 纯热度计算）**
    对每个口径，测**只用"端点热度"就能拿到的 AUC**（两个基线：
    热度之和 = "两端都热门"，热度相近 = "两端热度匹配"）。
    ⇒ 若某口径下热度基线自己就有 ~0.9 的 AUC，那么该口径下的表示 AUC 里混着热度，
       ρ 低是**测量被污染**的结果，而不是"排序真的变了"。
    这是判定 Q-b 的决定性证据：口径是否控制了热度，是可以用纯 CPU 量出来的。

用法（仓库根，CPU，分钟级）：
    python3 tools/rho_regime_attribution.py
    python3 tools/rho_regime_attribution.py --datasets Sport Beauty --seeds 2024
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from behavior_structure_probe import (  # noqa: E402
    DATA_DIR, N_POS, N_NEG, SEED, TOPK_POS, auc_of,
)
from stage_probe import build_samples  # noqa: E402

VARIANT_ORDER = ["Content-only", "ZeroPad", "Shuffled", "ItemCF", "Raw-CoOcc", "TA-SID"]
SPACES = ["A", "B", "C"]
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]
# 四个口径组合：(pos_regime, 负样本列名, 简称)
REGIMES = [
    ("topk", "auc_cos_degmatch", "topk × degmatch  ★主口径"),
    ("topk", "auc_cos_uniform",  "topk × uniform"),
    ("all",  "auc_cos_degmatch", "all  × degmatch"),
    ("all",  "auc_cos_uniform",  "all  × uniform"),
]

LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


# ---------------------------------------------------------------------------
# Part 1：从 CSV 离线算 ρ，并定位翻转的变体对
# ---------------------------------------------------------------------------
def space_vector(df, ds, seed, space, pos, negcol):
    """取 (数据集,种子,空间,口径) 下的 6 变体 AUC 向量（按 VARIANT_ORDER）。"""
    s = df[(df.dataset == ds) & (df.rqvae_seed == seed) & (df.space == space) &
           (df.pos_regime == pos)]
    if s.empty:
        return None
    v = s.set_index("variant")[negcol]
    if not all(k in v.index for k in VARIANT_ORDER):
        return None
    return v.reindex(VARIANT_ORDER).astype(float).values


def rank_of(vec):
    """降序名次：分数最高者名次 1。"""
    order = np.argsort(-np.asarray(vec))
    r = np.empty(len(vec), dtype=int)
    r[order] = np.arange(1, len(vec) + 1)
    return r


def discordant_pairs(va, vb):
    """返回 A/B 两空间下名次不一致的变体对 [(v_i, v_j, 名次A, 名次B)]。"""
    ra, rb = rank_of(va), rank_of(vb)
    out = []
    for i in range(len(VARIANT_ORDER)):
        for j in range(i + 1, len(VARIANT_ORDER)):
            if (ra[i] < ra[j]) != (rb[i] < rb[j]):
                out.append((VARIANT_ORDER[i], VARIANT_ORDER[j], ra[i], ra[j], rb[i], rb[j]))
    return out


def part1(df, datasets, seeds):
    log("=" * 108)
    log("Part 1 — 四个口径下的 ρ 表（离线，读 doc/g2_stage_probe_results.csv）")
    log("=" * 108)

    rows = []
    for ds in datasets:
        for seed in seeds:
            log(f"\n▌ {ds}   rqvae_seed={seed}")
            log(f"    {'口径':<26s} " + "".join(f"{f'ρ({a},{b})':>12s}" for a, b in PAIRS))
            for pos, negcol, label in REGIMES:
                vecs = {sp: space_vector(df, ds, seed, sp, pos, negcol) for sp in SPACES}
                if any(v is None for v in vecs.values()):
                    log(f"    {label:<26s}   （数据缺失）")
                    continue
                cells = []
                for a, b in PAIRS:
                    rho, _ = spearmanr(vecs[a], vecs[b])
                    cells.append(rho)
                    rows.append({"dataset": ds, "rqvae_seed": seed, "regime": label,
                                 "space_pair": f"{a}-{b}", "rho": rho})
                flag = ""
                if min(cells) < 0.9:
                    flag = f"   ⚠ 最低 {min(cells):+.3f}"
                log(f"    {label:<26s} " + "".join(f"{c:>+12.3f}" for c in cells) + flag)

            # 对 ρ(A,B) < 0.9 的口径，逐对定位翻转
            log(f"\n    ── ρ(A,B) 的翻转定位（主口径为参照）")
            ref = space_vector(df, ds, seed, "B", "topk", "auc_cos_degmatch")
            ref_rank = rank_of(ref)
            for pos, negcol, label in REGIMES:
                va = space_vector(df, ds, seed, "A", pos, negcol)
                vb = space_vector(df, ds, seed, "B", pos, negcol)
                if va is None or vb is None:
                    continue
                rho, _ = spearmanr(va, vb)
                d = discordant_pairs(va, vb)
                log(f"       {label:<26s} ρ(A,B)={rho:+.3f}  不一致变体对 {len(d)}/15")
                for vi, vj, ra_i, ra_j, rb_i, rb_j in d:
                    log(f"           · {vi} vs {vj}:  A 空间 {vi}#{ra_i} / {vj}#{ra_j}"
                        f"   →   B 空间 {vi}#{rb_i} / {vj}#{rb_j}")
            log(f"       参照：B 空间主口径名次  " +
                " > ".join(f"{VARIANT_ORDER[i]}({ref_rank[i]})" for i in np.argsort(ref_rank)))
    return rows


# ---------------------------------------------------------------------------
# Part 2：纯热度基线（不加载模型）—— 口径是否控制住了热度
# ---------------------------------------------------------------------------
def deg_scores(pairs, deg, mode):
    """只用端点热度给一对 item 打分（不涉及任何表示）。"""
    a = deg[pairs[:, 0]].astype(np.float64)
    b = deg[pairs[:, 1]].astype(np.float64)
    la, lb = np.log2(a + 1.0), np.log2(b + 1.0)
    if mode == "sum":          # 两端都热门 → 分数高（topk 正样本的特征）
        return la + lb
    if mode == "similar":      # 两端热度接近 → 分数高（degmatch 负样本的特征）
        return -np.abs(la - lb)
    raise ValueError(mode)


def part2(datasets, data_dir, window_size, n_pos, n_neg, sample_seed):
    log("\n" + "=" * 108)
    log("Part 2 — 纯热度基线：该口径下的 AUC 有多少是「端点热度」就能拿到的？（不加载模型）")
    log("=" * 108)
    log("  读法：若某口径的热度基线 AUC 很高，则该口径下**任何**表示的 AUC 都混着热度，")
    log("        ρ 降低是测量被污染，而非排序真的变了。理想口径的热度基线应 ≈ 0.5。")

    rows = []
    for ds in datasets:
        n_items, n_edges, negcg = build_samples(ds, data_dir, window_size, n_pos, n_neg, sample_seed)
        log(f"\n▌ {ds}   E 图 {n_items} items / {n_edges} 条边")
        log(f"    {'口径':<26s} {'n_pos':>7s} {'n_neg':>7s} "
            f"{'pos 度中位':>10s} {'neg 度中位':>10s} {'度差':>7s} "
            f"{'热度和基线':>11s} {'热度近基线':>11s}")
        for pname, pos_label in (("topk", "topk"), ("all", "all")):
            pos_ok, neg_deg, neg_unif = negcg[pname]
            for negname, neg in (("degmatch", neg_deg), ("uniform", neg_unif)):
                label = f"{pos_label:<4s} × {negname}"
                deg = DEG_CACHE[ds]
                dpos = np.r_[deg[pos_ok[:, 0]], deg[pos_ok[:, 1]]]
                dneg = np.r_[deg[neg[:, 0]], deg[neg[:, 1]]]
                s_sum = auc_of(deg_scores(pos_ok, deg, "sum"),
                               deg_scores(neg,    deg, "sum"))
                s_sim = auc_of(deg_scores(pos_ok, deg, "similar"),
                               deg_scores(neg,    deg, "similar"))
                log(f"    {label:<26s} {len(pos_ok):>7d} {len(neg):>7d} "
                    f"{np.median(dpos):>10.0f} {np.median(dneg):>10.0f} "
                    f"{np.median(dpos)-np.median(dneg):>+7.0f} "
                    f"{s_sum:>11.4f} {s_sim:>11.4f}")
                rows.append({"dataset": ds, "pos_regime": pos_label, "neg_regime": negname,
                             "n_pos": len(pos_ok), "n_neg": len(neg),
                             "deg_median_pos": float(np.median(dpos)),
                             "deg_median_neg": float(np.median(dneg)),
                             "deg_median_gap": float(np.median(dpos) - np.median(dneg)),
                             "auc_degsum_baseline": s_sum,
                             "auc_degsimilar_baseline": s_sim})
    return rows


DEG_CACHE = {}


def main():
    ap = argparse.ArgumentParser(description="ρ 的口径敏感性归因")
    ap.add_argument("--datasets", nargs="+", default=["Sport", "Toys", "Beauty"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--sample_seed", type=int, default=SEED,
                    help="正负样本采样 RNG 种子（非 RQ-VAE 种子），必须固定")
    ap.add_argument("--results_csv", type=str, default="doc/g2_stage_probe_results.csv")
    ap.add_argument("--csv", type=str, default="doc/rho_regime_attribution.csv")
    ap.add_argument("--log", type=str, default="doc/rho_regime_attribution.log")
    args = ap.parse_args()

    log("=" * 108)
    log("ρ(A输入, B z) 的口径敏感性归因")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("口径 = 正样本 {topk, all} × 负样本 {degmatch, uniform}；主口径 = topk × degmatch")
    log(f"样本采样 seed = {args.sample_seed}（与主 probe 同路径，逐位可复现）")
    log("=" * 108)

    if not os.path.exists(args.results_csv):
        log(f"\n❌ 缺 {args.results_csv} —— 请先跑 tools/stage_probe.py --delta_auc")
        return 1
    df = pd.read_csv(args.results_csv)

    rows1 = part1(df, args.datasets, args.seeds)

    # Part 2 需要端点度数：从 E 图取一次并缓存
    from behavior_structure_probe import build_graph  # noqa: E402
    for ds in args.datasets:
        G = build_graph(ds, args.data_dir, args.window_size)
        DEG_CACHE[ds] = G["deg"].astype(np.float64)
    rows2 = part2(args.datasets, args.data_dir, args.window_size,
                  args.n_pos, args.n_neg, args.sample_seed)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log(f"\n→ 日志: {args.log}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["part"] + sorted({k for r in (rows1 + rows2) for k in r}))
            w.writeheader()
            for r in rows1:
                w.writerow({"part": "rho", **r})
            for r in rows2:
                w.writerow({"part": "degree_baseline", **r})
        log(f"→ CSV: {args.csv}  ({len(rows1) + len(rows2)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
