#!/usr/bin/env python3
"""plot_checkpoint_trajectory.py — Checkpoint Trajectory 实验的 trajectory plots

读 `doc/checkpoint_trajectory_results.csv`（由 tools/checkpoint_trajectory_probe.py 产出），
出 5 张图到 `doc/figures/checkpoint_trajectory/`：
    F1 auc_vs_epoch      AUC(z) / AUC(xq) 随 epoch（3 数据集 × 2 测度）
    F2 zgap_vs_epoch     z-gap 主口径 与 原生口径（审计）随 epoch
    F3 recon_vs_zgap     关键图：recon loss vs z-gap 散点（点标注 epoch）
    F4 losses_vs_epoch   recon / quant / total loss 随 epoch
    F5 codebook_vs_epoch L0/L1/L2 utilization 与 prefix2/prefix3/full 碰撞率随 epoch

配色：dataviz skill 的参考调色板 categorical slot 1–3（blue/orange/aqua），
已用 `scripts/validate_palette.js --pairs all` 在 light/dark 两模式验证通过
（worst all-pairs CVD ΔE 9.2 light / 9.4 dark；normal-vision 24.0 / 20.9）。
aqua 在 light 面对比度 2.74 (<3:1) ⇒ 触发 relief 规则：每张图都带**线端直接标注**
（seed 名）且完整数值另有 CSV 表视图。颜色恒绑定 seed（entity），不随排序变化。

x 轴：4 个 epoch 快照按**等距分类位置**摆放（50/500/1000/2000 在训练时间上不等距，
等距摆放避免误读为线性时间；刻度标签写明真实 epoch）。

用法（仓库根）：
    python3 tools/plot_checkpoint_trajectory.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

# 中文字形：与 tools/make_fig5x_taxonomy.py 同一字体栈（缺字会渲染成方框）
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EPOCHS = [50, 500, 1000, 2000]
DATASETS = ["Beauty", "Sport", "Toys"]
SEEDS = [2024, 2025, 2026]

# ---- dataviz 参考调色板（categorical slot 1–3），两模式均为「选定值」，非自动翻转 ----
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#d8d7d2", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
              "grid": "#3a3a38", "series": ["#3987e5", "#d95926", "#199e70"]},
}
TEXT = {"title": 12, "label": 10, "tick": 9, "note": 8, "direct": 8}
LW = 2.0          # 2px lines
MS = 8.0          # >=8px markers


def style_ax(ax, th, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(th["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["grid"])
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=th["ink2"], labelsize=TEXT["tick"], length=3, width=1.0)
    ax.grid(True, color=th["grid"], linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=th["ink"], fontsize=TEXT["title"], loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=th["ink2"], fontsize=TEXT["label"])
    if ylabel:
        ax.set_ylabel(ylabel, color=th["ink2"], fontsize=TEXT["label"])


def cat_x(ax, values=EPOCHS):
    """4 个 epoch 快照 → 等距分类位置，刻度标签写真实 epoch。"""
    pos = list(range(len(values)))
    ax.set_xticks(pos)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_xlim(-0.25, len(values) - 0.75 + 1.15)      # 右侧留位给线端直接标注
    return pos


def series_for(df, ds, seed, ycol):
    g = df[(df["dataset"] == ds) & (df["seed"] == seed)].sort_values("epoch")
    ys = []
    for e in EPOCHS:
        r = g[g["epoch"] == e]
        ys.append(float(r[ycol].iloc[0]) if len(r) else np.nan)
    return ys


def dodge(ends, min_sep):
    """把末端数值相近的直接标注按 y 方向撑开，避免互相压叠（数值相等时会完全重合）。"""
    order = sorted(range(len(ends)), key=lambda i: ends[i])
    out = list(ends)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < min_sep:
            out[j] = out[i] + min_sep
    return out


def line_panel(ax, df, th, ycol, ds=None, title=None, ylabel=None, seeds=SEEDS,
               direct_labels=True, ylim=None):
    """在 ax 上画 3 个 seed 的折线 + 线端直接标注（relief 规则）。"""
    pos = cat_x(ax)
    ends = []
    for i, seed in enumerate(seeds):
        ys = series_for(df, ds, seed, ycol)
        c = th["series"][i % len(th["series"])]
        ax.plot(pos, ys, color=c, linewidth=LW, marker="o", markersize=MS / 2,
                markeredgecolor=th["surface"], markeredgewidth=2, zorder=3,
                label="seed %d" % seed)
        ends.append(ys[-1] if not np.isnan(ys[-1]) else np.nan)

    style_ax(ax, th, title=title, ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)

    if direct_labels:
        # 避让量必须按**轴的显示范围**算，不能按数值跨度算：Toys 的 y 轴跨 0.033 而
        # 三条末值只差 0.002，按跨度算出的间距落到屏幕上不足 3px，等于没避让。
        y0, y1 = ax.get_ylim()
        ends = dodge(ends, 0.062 * (y1 - y0))
        finite = [e for e in ends if not np.isnan(e)]
        if finite:                      # 撑开后的标注可能越出 ylim → 补边距，避免被切
            if max(finite) > y1:
                ax.set_ylim(y0, max(finite) + 0.02 * (y1 - y0))
            if min(finite) < y0:
                ax.set_ylim(min(finite) - 0.02 * (y1 - y0), ax.get_ylim()[1])
        for i, seed in enumerate(seeds):
            if np.isnan(ends[i]):
                continue
            ax.annotate("seed %d" % seed, xy=(pos[-1], ends[i]),
                        xytext=(6, 0), textcoords="offset points",
                        color=th["ink2"], fontsize=TEXT["direct"], va="center")
    return ax


def fig_frame(th, figsize, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=th["surface"])
    return fig, axes


def save(fig, outdir, name, mode, th):
    fig.tight_layout()
    p = os.path.join(outdir, "%s_%s.png" % (name, mode))
    fig.savefig(p, dpi=170, facecolor=th["surface"])
    plt.close(fig)
    print("→", p)


# --------------------------------------------------------------------------- F1
def f1_auc(df, outdir, mode, th):
    fig, axes = fig_frame(th, (13, 9), 3, 2)
    measures = [("auc_z", "AUC(z)"), ("auc_xq", "AUC(xq)")]
    for r, ds in enumerate(DATASETS):
        for c, (col, lab) in enumerate(measures):
            ax = axes[r][c]
            vals = np.concatenate([series_for(df, ds, s, col) for s in SEEDS])
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = max(0.006, (hi - lo) * 0.25)
            line_panel(ax, df, th, col, ds=ds,
                       title="%s — %s" % (ds, lab), ylabel="cosine AUC",
                       ylim=(lo - pad, hi + pad))
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=TEXT["note"], labelcolor=th["ink2"],
                          loc="lower right")
    fig.suptitle("F1  AUC(z) 与 AUC(xq) 随训练 epoch（同一次 run 内；正 topk × 负 degmatch）",
                 color=th["ink"], fontsize=TEXT["title"] + 1, x=0.01, ha="left")
    save(fig, outdir, "auc_vs_epoch", mode, th)


# --------------------------------------------------------------------------- F2
def f2_zgap(df, outdir, mode, th):
    fig, axes = fig_frame(th, (13, 6.6), 2, 3)
    rows = [("z_gap", "主口径 = AUC(z) − AUC(xq)"),
            ("z_gap_native", "原生口径（审计：未做 Sinkhorn 缓解）")]
    for r, (col, lab) in enumerate(rows):
        for c, ds in enumerate(DATASETS):
            ax = axes[r][c]
            vals = np.concatenate([series_for(df, ds, s, col) for s in SEEDS])
            lo, hi = min(0.0, np.nanmin(vals)), max(0.0, np.nanmax(vals))
            pad = max(0.004, (hi - lo) * 0.2)
            # 口径名放 y 轴标签（列 0），面板标题只留数据集名 —— 否则第三列标题被截断
            line_panel(ax, df, th, col, ds=ds, title=ds,
                       ylabel="z-gap · %s" % lab if c == 0 else None,
                       ylim=(lo - pad, hi + pad))
            ax.axhline(0.0, color=th["ink2"], linewidth=1.0, linestyle=":", zorder=1)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=TEXT["note"], labelcolor=th["ink2"],
                          loc="lower right")
    fig.suptitle("F2  z-gap 随训练 epoch（上行主口径 · 下行原生口径审计）",
                 color=th["ink"], fontsize=TEXT["title"] + 1, x=0.01, ha="left")
    save(fig, outdir, "zgap_vs_epoch", mode, th)


# --------------------------------------------------------------------------- F3
def f3_scatter(df, outdir, mode, th):
    fig, axes = fig_frame(th, (13, 4.8), 1, 3)
    for c, ds in enumerate(DATASETS):
        ax = axes[c]
        sub = df[df["dataset"] == ds]
        for i, seed in enumerate(SEEDS):
            g = sub[sub["seed"] == seed].sort_values("epoch")
            ax.plot(g["recon_loss"], g["z_gap"], color=th["series"][i], linewidth=LW,
                    marker="o", markersize=MS, markeredgecolor=th["surface"],
                    markeredgewidth=2, zorder=3, label="seed %d" % seed)
            # 只标两端（ep50 / ep2000）：recon 随 epoch 单调降 ⇒ 点序即 epoch 序，
            # 中间两点可沿线读出。全标 4 个会在收敛簇里互相压叠（label collision）。
            for _, r in g.iterrows():
                if int(r["epoch"]) not in (EPOCHS[0], EPOCHS[-1]):
                    continue
                ax.annotate("ep%d" % int(r["epoch"]), xy=(r["recon_loss"], r["z_gap"]),
                            xytext=(7, 5), textcoords="offset points",
                            color=th["ink"], fontsize=TEXT["note"], zorder=4)
        lo, hi = sub["recon_loss"].min(), sub["recon_loss"].max()
        pad = max(0.02, (hi - lo) * 0.25)
        style_ax(ax, th, title=ds,
                 xlabel="reconstruction loss（← 越低 = epoch 越大）",
                 ylabel="z-gap = AUC(z) − AUC(xq)" if c == 0 else None)
        ax.set_xlim(lo - pad, hi + pad * 1.5)
        ax.axhline(0.0, color=th["ink2"], linewidth=1.0, linestyle=":", zorder=1)
        if c == 0:
            ax.legend(frameon=False, fontsize=TEXT["note"], labelcolor=th["ink2"],
                      loc="lower right", markerscale=1.0)
    fig.suptitle("F3  reconstruction loss 与 z-gap 的关系（点 = 同一次 run 内的 epoch 快照）",
                 color=th["ink"], fontsize=TEXT["title"] + 1, x=0.01, ha="left")
    save(fig, outdir, "recon_vs_zgap", mode, th)


# --------------------------------------------------------------------------- F4
def f4_losses(df, outdir, mode, th):
    fig, axes = fig_frame(th, (13, 9), 3, 3)
    cols = [("recon_loss", "reconstruction loss"), ("quant_loss", "quantization loss"),
            ("total_loss", "total loss = recon + qlw·quant")]
    for r, ds in enumerate(DATASETS):
        for c, (col, lab) in enumerate(cols):
            ax = axes[r][c]
            vals = np.concatenate([series_for(df, ds, s, col) for s in SEEDS])
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = max(0.05, (hi - lo) * 0.2)
            line_panel(ax, df, th, col, ds=ds, title="%s — %s" % (ds, lab),
                       ylabel="loss", ylim=(lo - pad, hi + pad))
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=TEXT["note"], labelcolor=th["ink2"],
                          loc="upper right")
    fig.suptitle("F4  训练损失随 epoch（trainer 同口径：逐 batch reduction='mean' 后求和）",
                 color=th["ink"], fontsize=TEXT["title"] + 1, x=0.01, ha="left")
    save(fig, outdir, "losses_vs_epoch", mode, th)


# --------------------------------------------------------------------------- F5
def f5_codebook(df, outdir, mode, th):
    rows = [("util_L0", "L0 utilization"), ("util_L1", "L1 utilization"),
            ("util_L2", "L2 utilization"),
            ("prefix2_collision_rate", "prefix-2 collision"),
            ("prefix3_collision_rate", "prefix-3 collision"),
            ("full_sid_collision_rate", "full SID collision")]
    fig, axes = fig_frame(th, (13, 16), 6, 3)
    for r, (col, lab) in enumerate(rows):
        for c, ds in enumerate(DATASETS):
            ax = axes[r][c]
            vals = np.concatenate([series_for(df, ds, s, col) for s in SEEDS])
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = max(0.004, (hi - lo) * 0.2)
            line_panel(ax, df, th, col, ds=ds, title="%s — %s" % (ds, lab),
                       ylabel=lab, ylim=(min(0.0, lo - pad), hi + pad))
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=TEXT["note"], labelcolor=th["ink2"],
                          loc="lower right")
    fig.suptitle("F5  码本结构随 epoch（上三行 = 逐层 utilization；下三行 = 逐深度碰撞率）",
                 color=th["ink"], fontsize=TEXT["title"] + 1, x=0.01, ha="left")
    save(fig, outdir, "codebook_vs_epoch", mode, th)


def main():
    ap = argparse.ArgumentParser(description="Checkpoint Trajectory plots")
    ap.add_argument("--csv", type=str, default="doc/checkpoint_trajectory_results.csv")
    ap.add_argument("--outdir", type=str, default="doc/figures/checkpoint_trajectory")
    ap.add_argument("--modes", type=str, nargs="+", default=["light", "dark"])
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    miss = [e for e in EPOCHS if e not in set(df["epoch"])]
    if miss:
        print("⚠ CSV 缺少 epoch %s（图会留空段）" % miss)
    os.makedirs(args.outdir, exist_ok=True)

    for mode in args.modes:
        th = THEMES[mode]
        f1_auc(df, args.outdir, mode, th)
        f2_zgap(df, args.outdir, mode, th)
        f3_scatter(df, args.outdir, mode, th)
        f4_losses(df, args.outdir, mode, th)
        f5_codebook(df, args.outdir, mode, th)


if __name__ == "__main__":
    main()
