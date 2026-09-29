#!/usr/bin/env python3
"""
plot_paper_ablation_figs.py — 论文 §2.4 配图（图 A / 图 C）

图 A：Behavior AUC (xq[full]) vs NDCG@20 散点，三面板（每数据集一个）。
      含义 = 量化空间的转移编码质量与端到端性能在同一数据集内同序，
      且跨数据集稳定复现「两端 + 分层」。点标签做显示空间碰撞检测，不压点、不压误差棒。
图 C：消融分层条形图，三面板 × 6 变体，柱高 = N@20(3-seed 均值)，误差棒 = std。
      **按层级升序排列**（Shuffled → 内容类 → 共现类 → TA-SID），
      使「四层台阶」在三个数据集上肉眼可辨。

数据源（只读，不训练、不评测）：
    AUC  : doc/behavior_structure_probe_results.csv (space=xq[full], pos_regime=topk,
           degmatch 口径) ∪ doc/beauty_tier2_three_dataset_table.csv
           （两份在重合格上已验证一致，脚本内 assert）
    N@20 : doc/ablation_main_table_3datasets.md §1.1（权威表，3-seed 均值±样本 std）

用法：
    python3 tools/plot_paper_ablation_figs.py [--outdir doc/figures] [--dpi 300]
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

# ---------------------------------------------------------------- 常量

DATASETS = ["Beauty", "Sport", "Toys"]          # 出图面板序

VARIANTS = ["Content-only", "ZeroPad", "Shuffled", "ItemCF-SVD", "Raw-CoOcc-SVD", "TA-SID"]  # 表 5-4 行序

# 图 C 柱序：按层级升序，形成台阶
BAR_ORDER = ["Shuffled", "Content-only", "ZeroPad", "ItemCF-SVD", "Raw-CoOcc-SVD", "TA-SID"]

ALIAS = {"ItemCF": "ItemCF-SVD", "Raw-CoOcc": "Raw-CoOcc-SVD"}

SHORT = {
    "Content-only": "Content",
    "ZeroPad": "ZeroPad",
    "Shuffled": "Shuffled",
    "ItemCF-SVD": "ItemCF",
    "Raw-CoOcc-SVD": "Raw-CoOcc",
    "TA-SID": "TA-SID",
}

# 四层结构：0 = 完整方法，3 = 最差
LAYER = {
    "TA-SID": 0,
    "Raw-CoOcc-SVD": 1,
    "ItemCF-SVD": 1,
    "ZeroPad": 2,
    "Content-only": 2,
    "Shuffled": 3,
}
LAYER_COLOR = {0: "#2a78d6", 1: "#2f9e8f", 2: "#9aa0a6", 3: "#d1495b"}
# 6 个变体各用唯一形状：同层同色（如 Content-only vs ZeroPad）仅靠颜色分不开，
# 形状必须全局唯一，配合图例给出唯一映射，避免近邻点归属不清。
MARKER = {
    "TA-SID": "o",
    "ItemCF-SVD": "s",
    "Raw-CoOcc-SVD": "^",
    "Content-only": "D",
    "ZeroPad": "X",
    "Shuffled": "v",
}
TIER_LABEL = {
    0: "TA-SID (full)",
    1: "co-occurrence-based",
    2: "content-based",
    3: "misaligned behavior",
}

INK = "#0b0b0b"
INK_SOFT = "#5a5a57"
GRID = "#e3e3e0"
SURFACE = "#fcfcfb"

PROBE_CSV = "doc/behavior_structure_probe_results.csv"
TIER2_CSV = "doc/beauty_tier2_three_dataset_table.csv"
ABL_MD = "doc/ablation_main_table_3datasets.md"


# ---------------------------------------------------------------- 数据装载

def load_auc():
    """返回 DataFrame[dataset, variant, auc]，degmatch + topk 正样本口径。"""
    probe = pd.read_csv(PROBE_CSV)
    probe = probe[(probe.space == "xq[full]") & (probe.pos_regime == "topk")].copy()
    probe["variant"] = probe["variant"].replace(ALIAS)
    a = probe[["dataset", "variant", "auc_cos_degmatch"]].rename(columns={"auc_cos_degmatch": "auc"})

    t2 = pd.read_csv(TIER2_CSV)
    t2["variant"] = t2["variant"].replace(ALIAS)
    b = t2[["dataset", "variant", "behavior_auc_xq_full_topk"]].rename(
        columns={"behavior_auc_xq_full_topk": "auc"}
    )

    m = a.merge(b, on=["dataset", "variant"], how="outer", suffixes=("_probe", "_tier2"))

    both = m.dropna(subset=["auc_probe", "auc_tier2"])
    if len(both):
        d = (both["auc_probe"] - both["auc_tier2"]).abs().max()
        if d > 1e-4:
            sys.exit(f"[FATAL] 两份 AUC 源在重合格上不一致，最大差 {d:.6f}：\n{both}")
        # 注意：重合格里只有 Beauty 的 2 格是真正独立的双源交叉印证；
        # Sport/Toys 的 12 格是 beauty_tier2_summary.py §九 从 probe CSV 直接复制
        # （a = s["auc_cos_degmatch"]），属同源转抄，不一致才奇怪。别把它当成
        # "14 格交叉验证通过"写进论文。逐格来源见 doc/figA_claims_verification.md。
        print(f"[check] 两份 AUC 源重合格 {len(both)} 格一致（最大差 {d:.2e}）"
              f"，其中仅 Beauty 2 格为独立双源，其余 {len(both) - 2} 格为同源转抄")

    m["auc"] = m["auc_probe"].fillna(m["auc_tier2"])
    m = m.dropna(subset=["auc"])

    missing = [(ds, v) for ds in DATASETS for v in VARIANTS
               if not ((m.dataset == ds) & (m.variant == v)).any()]
    if missing:
        sys.exit(f"[FATAL] AUC 缺失格：{missing}")

    print(f"[load] AUC 覆盖 {len(m)} 格（{m.dataset.nunique()} 数据集 × {m.variant.nunique()} 变体）")
    return m[["dataset", "variant", "auc"]]


def load_ndcg():
    """从权威消融表 §1.1 解析 N@20 均值±std。返回 DataFrame[dataset, variant, mean, std]。"""
    with open(ABL_MD, encoding="utf-8") as f:
        lines = f.read().splitlines()

    start = next(i for i, l in enumerate(lines) if l.strip().startswith("### 1.1"))
    end = next(i for i, l in enumerate(lines) if i > start and l.strip().startswith("### 1.2"))
    block = lines[start:end]

    header = next(l for l in block if l.strip().startswith("| 变体"))
    cols = [c.strip() for c in header.strip().strip("|").split("|")][1:]
    assert sorted(cols) == sorted(DATASETS), f"列名与预期不符：{cols}"

    rows = []
    for l in block:
        l = l.strip()
        if not l.startswith("|") or set(l) <= set("|-: "):
            continue
        cells = [c.strip() for c in l.strip("|").split("|")]
        if len(cells) != len(cols) + 1 or cells[0] == "变体":
            continue
        name = cells[0].replace("**", "").strip()
        for ds, cell in zip(cols, cells[1:]):
            cell = cell.replace("**", "").strip()
            mt = re.match(r"^([\d.]+)±([\d.]+)$", cell)
            if not mt:
                sys.exit(f"[FATAL] 无法解析 {ds}/{name} 的 N@20 单元：{cell!r}")
            rows.append({"dataset": ds, "variant": name,
                         "mean": float(mt.group(1)), "std": float(mt.group(2))})

    df = pd.DataFrame(rows)
    missing = [(ds, v) for ds in DATASETS for v in VARIANTS
               if not ((df.dataset == ds) & (df.variant == v)).any()]
    if missing:
        sys.exit(f"[FATAL] N@20 缺失格：{missing}")
    print(f"[load] N@20 覆盖 {len(df)} 格")
    return df


# ---------------------------------------------------------------- 通用样式

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def place_labels(fig, ax, pts):
    """pts: [(x, y, text, accent), ...]。显示空间贪心避让：不压点、不压误差棒、不互相重叠。"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    occupied = []

    # 预占：每个标记点 + 误差棒竖条
    for x, y, _t, _a, half in pts:
        px, py = ax.transData.transform((x, y))
        _, y_lo = ax.transData.transform((x, y - half))
        _, y_hi = ax.transData.transform((x, y + half))
        occupied.append(Bbox([[px - 7, min(py, y_lo) - 3], [px + 7, max(py, y_hi) + 3]]))

    ax_bb = ax.get_window_extent(r)

    def _draw(x, y, text, accent, dx, dy, leader):
        return ax.annotate(
            text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            ha="center", va="center", fontsize=7.8, zorder=6,
            color=INK if accent else INK_SOFT,
            fontweight="bold" if accent else "normal",
            arrowprops=(dict(arrowstyle="-", color="#c2c2be", lw=0.6,
                             shrinkA=1, shrinkB=4, zorder=5) if leader else None),
        )

    # 按 AUC 升序交替「下 / 上」标注：相邻两点必然分居两侧，解决 ItemCF vs Raw-CoOcc
    # （以及 Content vs ZeroPad）这类近邻点标签归属不清的问题。
    for i, (x, y, text, accent, half) in enumerate(sorted(pts, key=lambda p: p[0])):
        # 偏移量随该点误差棒长度自适应（固定点数在误差棒较长时会压到自己的误差棒）
        _, py = ax.transData.transform((x, y))
        _, py_hi = ax.transData.transform((x, y + half))
        eb = abs(py_hi - py)
        up_n, up_f = eb + 11, eb + 34
        dn_n, dn_f = -(eb + 12), -(eb + 35)

        if i % 2 == 0:      # 偶数位 → 优先标在点下方
            cands = [(0, dn_n, False), (26, -6, False), (-26, -6, False), (0, dn_f, True),
                     (0, up_n, False), (32, 8, True), (-32, 8, True), (0, up_f, True)]
        else:               # 奇数位 → 优先标在点上方
            cands = [(0, up_n, False), (26, 6, False), (-26, 6, False), (0, up_f, True),
                     (0, dn_n, False), (32, -8, True), (-32, -8, True), (0, dn_f, True)]

        chosen = None
        for dx, dy, leader in cands:
            t = _draw(x, y, text, accent, dx, dy, leader)
            bb = t.get_window_extent(renderer=r).expanded(1.08, 1.25)
            if not any(bb.overlaps(o) for o in occupied) and ax_bb.contains(bb.x0, bb.y0) \
                    and ax_bb.contains(bb.x1, bb.y1):
                occupied.append(bb)
                chosen = t
                break
            t.remove()
        if chosen is None:                      # 全部候选都冲突 → 退回首选位带引线，保证不漏标
            dx, dy, _l = cands[0]
            t = _draw(x, y, text, accent, dx, dy, True)
            occupied.append(t.get_window_extent(renderer=r))


# ---------------------------------------------------------------- 图 A

def build_auc_vs_ndcg(df):
    """构建图 A 的图形对象，**不落盘**。返回 (fig, axes, handles, labels)。

    与 fig_auc_vs_ndcg 拆开是为了让 tools/verify_figA_claims.py 能把画出去的
    每个 artist 回读出来，与权威表逐格对账（而非只信任打印出来的中间量）。
    每个数据点用 set_gid(f"figA|{dataset}|{variant}") 打标，便于回读定位。
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    fig.patch.set_facecolor(SURFACE)

    handles, labels = [], []          # ← 必须在数据集循环**外**初始化：
    for ax, ds in zip(axes, DATASETS):  #   图例素材只在 Beauty 面板收集一次
        sub = df[df.dataset == ds].set_index("variant").loc[VARIANTS].reset_index()
        style_axes(ax)

        for _, r in sub.iterrows():
            c = LAYER_COLOR[LAYER[r["variant"]]]
            h = ax.errorbar(
                r["auc"], r["mean"], yerr=r["std"],
                fmt=MARKER[r["variant"]], ms=7.2, mfc=c, mec="white", mew=1.1,
                ecolor=INK_SOFT, elinewidth=0.9, capsize=2.5, capthick=0.9,
                zorder=4 if r["variant"] == "TA-SID" else 3,
            )
            h.lines[0].set_gid(f"figA|{ds}|{r['variant']}")   # 回读用锚点（ErrorbarContainer 本身无 gid）
            if ax is axes[0]:
                handles.append(h)
                labels.append(r["variant"])

        pad_x = (sub["auc"].max() - sub["auc"].min()) * 0.18
        pad_y = (sub["mean"].max() - sub["mean"].min()) * 0.20
        ax.set_xlim(sub["auc"].min() - pad_x, sub["auc"].max() + pad_x)
        ax.set_ylim((sub["mean"] - sub["std"]).min() - pad_y,
                    (sub["mean"] + sub["std"]).max() + pad_y * 0.6)
        ax.set_title(ds, fontsize=10.5, color=INK, pad=7)

    axes[0].set_ylabel("NDCG@20", fontsize=9.5, color=INK)
    for ax in axes:
        ax.set_xlabel("Behavior AUC  ($x_q^{[full]}$)", fontsize=9.5, color=INK)

    # 图例在 builder 内建好：返回的必须是「最终要落盘的那张图」，
    # 否则校验脚本审计的是一个半成品，空图例这类 bug 会漏检。
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               fontsize=8.6, handletextpad=0.5, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.005))

    return fig, axes, handles, labels


def fig_auc_vs_ndcg(df, outdir, dpi):
    fig, axes, handles, labels = build_auc_vs_ndcg(df)
    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"paper_figA_auc_vs_ndcg.{ext}")
        fig.savefig(p, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs


# ---------------------------------------------------------------- 图 A（备选：斜率图）

def _side_labels(fig, ax, items, side):
    """items: [(y, text, accent), ...]。side='left'/'right'。
    标签置于面板**内部**（避开坐标轴刻度），带底色遮住穿过的连线。"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(r)
    occupied = []
    if side == "left":
        dx, ha, x_anchor = 10, "left", 0.0
    else:
        dx, ha, x_anchor = -10, "right", 1.0

    for y, text, accent in sorted(items, key=lambda t: t[0]):
        placed = False
        for dy in (0, 9, -9, 18, -18, 27, -27, 36, -36):
            t = ax.annotate(
                text, xy=(x_anchor, y), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center",
                fontsize=8, zorder=7,
                color=INK if accent else INK_SOFT,
                fontweight="bold" if accent else "normal",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8, alpha=0.92),
            )
            bb = t.get_window_extent(renderer=r).expanded(1.06, 1.15)
            if not any(bb.overlaps(o) for o in occupied) and ax_bb.contains(bb.x0, bb.y0) \
                    and ax_bb.contains(bb.x1, bb.y1):
                occupied.append(bb)
                placed = True
                break
            t.remove()
        if not placed:
            t = ax.annotate(text, xy=(x_anchor, y), xytext=(dx, -36),
                            textcoords="offset points", ha=ha, va="center",
                            fontsize=8, zorder=7, color=INK if accent else INK_SOFT,
                            fontweight="bold" if accent else "normal",
                            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))
            occupied.append(t.get_window_extent(renderer=r))


def fig_auc_slope(df, outdir, dpi):
    """斜率图：左端 = Behavior AUC 名次，右端 = NDCG@20 名次；连线交叉处即名次倒置。"""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    fig.patch.set_facecolor(SURFACE)
    lo, hi = 0.10, 0.90

    for k, (ax, ds) in enumerate(zip(axes, DATASETS)):
        sub = df[df.dataset == ds].set_index("variant").loc[VARIANTS].reset_index()
        style_axes(ax)
        ax.grid(axis="x", visible=False)

        amin, amax = sub["auc"].min(), sub["auc"].max()
        nmin, nmax = (sub["mean"] - sub["std"]).min(), (sub["mean"] + sub["std"]).max()
        a_n = lambda v: lo + (v - amin) / (amax - amin) * (hi - lo)
        n_n = lambda v: lo + (v - nmin) / (nmax - nmin) * (hi - lo)

        left_items, right_items = [], []
        for _, r in sub.iterrows():
            accent = r["variant"] == "TA-SID"
            c = LAYER_COLOR[LAYER[r["variant"]]]
            y0, y1 = a_n(r["auc"]), n_n(r["mean"])
            ax.plot([0, 1], [y0, y1], "-", color=c, lw=1.7, zorder=3,
                    solid_capstyle="round")
            ax.plot([0, 1], [y0, y1], "o", color=c, ms=6.2, mec="white",
                    mew=1.0, zorder=4)
            e = n_n(r["mean"] + r["std"]) - y1
            ax.errorbar([1], [y1], yerr=[e], fmt="none", ecolor=c, elinewidth=1.1,
                        capsize=3, capthick=1.1, zorder=5)
            left_items.append((y0, SHORT[r["variant"]], accent))
            right_items.append((y1, SHORT[r["variant"]], accent))

        ax.set_xlim(-0.46, 1.46)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Behavior AUC", "NDCG@20"], fontsize=9, color=INK)
        ax.tick_params(axis="x", length=0, pad=6)
        aticks = np.linspace(amin, amax, 4)
        ax.set_yticks([a_n(v) for v in aticks])
        ax.set_yticklabels([f"{v:.3f}" for v in aticks])
        ax.set_title(ds, fontsize=10.5, color=INK, pad=8)

        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        nticks = np.linspace(nmin, nmax, 4)
        ax2.set_yticks([n_n(v) for v in nticks])
        ax2.set_yticklabels([f"{v:.4f}" for v in nticks])
        ax2.tick_params(colors=INK_SOFT, labelsize=8, length=3, width=0.8)
        for s in ("top", "left", "bottom"):
            ax2.spines[s].set_visible(False)
        ax2.spines["right"].set_color(GRID)
        ax2.spines["right"].set_linewidth(0.8)

        _side_labels(fig, ax, left_items, "left")
        _side_labels(fig, ax, right_items, "right")

    fig.tight_layout()
    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"paper_figA2_auc_ndcg_slope.{ext}")
        fig.savefig(p, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs


# ---------------------------------------------------------------- 图 C

def fig_ablation_layers(df, outdir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    fig.patch.set_facecolor(SURFACE)

    x = np.arange(len(BAR_ORDER))
    for k, (ax, ds) in enumerate(zip(axes, DATASETS)):
        sub = df[df.dataset == ds].set_index("variant").loc[BAR_ORDER].reset_index()
        style_axes(ax)
        ax.grid(axis="x", visible=False)

        ax.bar(x, sub["mean"], width=0.66,
               color=[LAYER_COLOR[LAYER[v]] for v in sub["variant"]],
               edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar(x, sub["mean"], yerr=sub["std"], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=3, capthick=0.9, zorder=4)
        for xi, (_, r) in zip(x, sub.iterrows()):
            ax.annotate(f"{r['mean']:.4f}", xy=(xi, r["mean"] + r["std"]),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.2,
                        color=INK if r["variant"] == "TA-SID" else INK_SOFT,
                        fontweight="bold" if r["variant"] == "TA-SID" else "normal",
                        zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[v] for v in BAR_ORDER], rotation=34, ha="right", fontsize=8)
        ax.set_ylim(0, (sub["mean"] + sub["std"]).max() * 1.30)
        ax.set_title(ds, fontsize=10.5, color=INK, pad=7)

        if k == 0:      # 层级图例（只画一次）
            handles = [Patch(facecolor=LAYER_COLOR[i], edgecolor="white",
                             label=TIER_LABEL[i]) for i in sorted(TIER_LABEL)]
            ax.legend(handles=handles, loc="upper left", fontsize=6.8, frameon=False,
                      handlelength=1.0, handleheight=1.0, labelspacing=0.42,
                      borderaxespad=0.2)

    axes[0].set_ylabel("NDCG@20", fontsize=9.5, color=INK)
    fig.tight_layout()
    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"paper_figC_ablation_layers.{ext}")
        fig.savefig(p, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="doc/figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--with-slope", action="store_true",
                    help="额外产出斜率图 paper_figA2_auc_ndcg_slope（2026-09-29 决定不入论文，"
                         "默认关闭：它左右两轴各自 min-max 归一化，斜率是归一化造出来的，"
                         "且标签比散点图更易碰撞）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_auc().merge(load_ndcg(), on=["dataset", "variant"], how="inner")

    print("\n=== 出图数据（核对用）===")
    piv = df.pivot(index="variant", columns="dataset", values=["auc", "mean"]).loc[VARIANTS]
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    outs = fig_auc_vs_ndcg(df, args.outdir, args.dpi)
    if args.with_slope:
        outs += fig_auc_slope(df, args.outdir, args.dpi)
    outs += fig_ablation_layers(df, args.outdir, args.dpi)

    for p in outs:
        print(f"[out] {p}  ({os.path.getsize(p)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
