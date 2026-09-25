#!/usr/bin/env python3
"""
plot_behavior_auc_vs_ndcg.py — Behavior AUC vs NDCG@20 散点图

读 `tools/behavior_structure_probe.py` 产出的长表 CSV，画两张图：
    ① xq[full] AUC vs NDCG@20
    ② xq[L0]   AUC vs NDCG@20

每个面板 = 一个数据集（单系列，故不设图例，标题即系列名）；每个点 = 一个 SID variant，
**逐点直标 variant 名**（识别不依赖颜色）。y 轴误差棒 = NDCG@20 的 3-seed std（噪声地板）。
面板副标给出该面板的 Spearman ρ / Pearson r。

Beauty 仅 2 个 variant → 只展示、标注"不做相关"。

配色：单一强调色 #2a78d6（categorical slot 1）。单系列面板不需图例；
该色对浅色底对比度 >= 3:1（已用 dataviz 的 validate_palette.js 验证）。

用法：
    python3 tools/plot_behavior_auc_vs_ndcg.py \
        [--csv doc/behavior_structure_probe_results.csv] \
        [--outdir doc/figures] [--pos_regime topk] [--neg_mode degmatch]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ACCENT = "#2a78d6"        # categorical slot 1
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"

SHORT = {
    "Content-only": "Content", "TA-SID": "TA-SID", "Raw-CoOcc": "Raw-CoOcc",
    "ItemCF": "ItemCF", "ZeroPad": "ZeroPad", "Shuffled": "Shuffled",
}


def load(csv, space, pos_regime):
    df = pd.read_csv(csv)
    d = df[(df.space == space) & (df.pos_regime == pos_regime)]
    out = {}
    for ds, g in d.groupby("dataset"):
        g = g.dropna(subset=[f"auc_cos_{NEG}", "ndcg20"]).copy()
        # NDCG 的 3-seed std（误差棒 = 噪声地板）
        g["ndcg_std"] = g["ndcg_seeds"].apply(
            lambda s: float(np.std([float(x) for x in str(s).split(";") if x], ddof=1))
            if isinstance(s, str) and s.count(";") >= 1 else 0.0)
        out[ds] = g.sort_values(f"auc_cos_{NEG}").reset_index(drop=True)
    return out


def repel_labels(ax, xs, ys, labels, fontsize=8):
    """贪心避让：每个标签在 8 个候选方位里选一个与已放置标签/数据点重叠最小的。"""
    ax.figure.canvas.draw()
    placed = []
    for x, y, t in zip(xs, ys, labels):
        best, best_cost = (0, 8), 1e9
        for dx, dy in [(0, 9), (0, -14), (7, -4), (-7, -4), (9, 4), (-9, 4), (0, 16), (0, -21)]:
            cx, cy = ax.transData.transform((x, y))
            bb = matplotlib.transforms.Bbox.from_bounds(
                cx + dx - 5 * len(t), cy + dy - 7, 10 * len(t), 14)
            cost = sum(bb.overlaps(p) for p in placed) * 100
            # 与所有数据点的距离惩罚
            for px, py in zip(xs, ys):
                qx, qy = ax.transData.transform((px, py))
                if abs(qx - cx) < 34 and abs(qy - cy) < 16:
                    cost += 1
            if cost < best_cost:
                best_cost, best = cost, (dx, dy)
        dx, dy = best
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha="center", va="center", fontsize=fontsize, color=INK_PRIMARY)
        cx, cy = ax.transData.transform((x, y))
        placed.append(matplotlib.transforms.Bbox.from_bounds(
            cx + dx - 5 * len(t), cy + dy - 7, 10 * len(t), 14))


def panel(ax, g, space, ds, xlabel):
    x = g[f"auc_cos_{NEG}"].values
    y = g["ndcg20"].values
    ye = g["ndcg_std"].values

    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d5d4cf")

    ax.errorbar(x, y, yerr=ye, fmt="none", ecolor="#c9c8c2", elinewidth=1.2,
                capsize=3, capthick=1.2, zorder=1)
    ax.scatter(x, y, s=90, c=ACCENT, edgecolors=SURFACE, linewidths=2, zorder=3)

    if len(g) >= 3:
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, k * xs + b, color=ACCENT, linewidth=2, alpha=0.45, zorder=2)
        rho, p_rho = spearmanr(x, y)
        r, p_r = pearsonr(x, y)
        sub = (f"Spearman ρ = {rho:.2f}  (p = {p_rho:.3f})\n"
               f"Pearson r = {r:.2f}  (p = {p_r:.3f})   n = {len(g)}")
    else:
        sub = f"n = {len(g)} — too few variants for correlation"

    repel_labels(ax, x, y, [SHORT.get(v, v) for v in g["variant"]], fontsize=8)

    ax.set_title(ds, fontsize=11, color=INK_PRIMARY, pad=8, fontweight="semibold")
    ax.text(0.03, 0.97, sub, transform=ax.transAxes, fontsize=8, color=INK_SECONDARY,
            va="top", ha="left", linespacing=1.5)
    ax.tick_params(labelsize=8, colors=INK_MUTED)


def main():
    global NEG
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="doc/behavior_structure_probe_results.csv")
    ap.add_argument("--outdir", default="doc/figures")
    ap.add_argument("--pos_regime", default="topk", choices=["topk", "all"])
    ap.add_argument("--neg_mode", default="degmatch", choices=["degmatch", "uniform"])
    args = ap.parse_args()
    NEG = args.neg_mode

    os.makedirs(args.outdir, exist_ok=True)
    suffix = "" if (args.pos_regime == "topk" and NEG == "degmatch") \
        else f"_{args.pos_regime}_{NEG}"
    specs = [("xq[full]", "Full SID AUC vs NDCG@20", f"behavior_auc_vs_ndcg_full{suffix}"),
             ("xq[L0]", "L0 SID AUC vs NDCG@20", f"behavior_auc_vs_ndcg_l0{suffix}")]

    for space, title, stem in specs:
        per_ds = load(args.csv, space, args.pos_regime)
        order = [d for d in ["Sport", "Toys", "Beauty"] if d in per_ds]
        fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 4.2),
                                 facecolor=SURFACE)
        if len(order) == 1:
            axes = [axes]
        for ax, ds in zip(axes, order):
            panel(ax, per_ds[ds], space, ds, None)
            ax.set_ylabel("NDCG@20 (3-seed mean)" if ax is axes[0] else "", fontsize=9,
                          color=INK_SECONDARY)
        fig.suptitle(title, fontsize=13, color=INK_PRIMARY, y=1.03, fontweight="semibold")
        fig.supxlabel(f"Behavior AUC — cosine similarity in {space}  "
                      f"({args.pos_regime} positive edges / {NEG}-matched negatives)",
                      fontsize=9, color=INK_SECONDARY)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = os.path.join(args.outdir, f"{stem}.{ext}")
            fig.savefig(p, dpi=200, bbox_inches="tight", facecolor=SURFACE)
            print(f"✓ {p}")
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
