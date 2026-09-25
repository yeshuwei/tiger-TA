#!/usr/bin/env python3
"""
make_fig5x_taxonomy.py — 论文图 5-x: 六数据集行为结构量化损失 (z-gap vs L0 utilization)

内容: 散点 + 状态色 + ε/2ε 阈值线 + 三区域底色 + Adaptive 箭头 (Toys@256→Toys@512)。
数据取自论文表 5-x (K0=256 固定容量) + Toys@512 (Adaptive 结果, e43 §7.2/7.3)。

布局原则: 文字/线/点不重合; 橙色带(ε,2ε]、红色带(2ε,+)内的元素各自错开;
"扩容有效"只出现一次(区域②), 箭头标签只写 "Adaptive: K0 256→512"。

用法:  python3 tools/make_fig5x_taxonomy.py
输出:  doc/figures/paper_fig5x_taxonomy.{png,pdf}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- CJK 字体（中文字面）----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

EPS, X2E = 0.01, 0.02

# --- 六数据集 (表 5-x, K0=256): (名称, L0利用率%, z-gap, 状态, 标签偏移) ---
data = [
    ("Beauty",             80, 0.0001, "Healthy",             (10, -16)),
    ("Sports",             97, 0.0024, "Healthy",             (-12, 14)),
    ("Video_Games",        73, 0.0009, "Healthy",             (10, 12)),
    ("Musical_Instruments", 14, 0.0029, "Collapsed-lossless",  (10, 14)),
    ("Toys",               24, 0.0931, "Insufficient",        (12, 6)),
    ("Electronics",       100, 0.0116, "Boundary",            (8, 2)),
]
TOYS512 = (60, 0.0045)  # Adaptive 结果: Toys@K0=512

state_color = {
    "Healthy":             "#27ae60",
    "Collapsed-lossless": "#2e86c1",
    "Insufficient":       "#c0392b",
    "Boundary":           "#e67e22",
}
state_label = {
    "Healthy":             "Healthy",
    "Collapsed-lossless": "Collapsed-lossless",
    "Insufficient":       "Capacity insufficient",
    "Boundary":           "Boundary",
}

fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=200)

# ---------- 三区域底色 (z-order 最低) ----------
ax.axhspan(2e-5, EPS, color="#27ae60", alpha=0.13, zorder=0)  # ① Lossless
ax.axhspan(EPS, X2E, color="#e67e22", alpha=0.16, zorder=0)   # ③ Boundary
ax.axhspan(X2E, 0.3, color="#c0392b", alpha=0.10, zorder=0)   # ② Insufficient

# ---------- 区域标签 (左侧竖列, 单行: 英文名 + 中文说明, 不压阈值线) ----------
ax.text(2.5, 0.0008, "① Lossless zone · 结构保持充分",
        fontsize=9, color="#1e8449", va="center")
ax.text(2.5, 0.015, "③ Boundary region · 扩容后验证",
        fontsize=9, color="#ca6f1e", va="center")
ax.text(2.5, 0.17, "② Capacity insufficient · 扩容有效",
        fontsize=9, color="#a93226", va="center")

# ---------- 阈值线 ----------
ax.axhline(EPS, color="#c0392b", ls="--", lw=1.8, zorder=1)
ax.text(56, EPS * 1.25, r"$\varepsilon$=0.01", color="#c0392b", fontsize=10, ha="left")
ax.axhline(X2E, color="#7f8c8d", ls=":", lw=1.2, zorder=1)
ax.text(113.5, X2E * 1.5, r"$2\varepsilon$", color="#7f8c8d", fontsize=9, ha="right")

# ---------- 六个点 ----------
for name, util, gap, st, (dx, dy) in data:
    sz = 280 if name == "Toys" else 160          # Toys 突出
    ax.scatter(util, gap, s=sz, marker="o", color=state_color[st],
               edgecolors="black", linewidths=1.2, zorder=3)
    ax.annotate(name, (util, gap), textcoords="offset points", xytext=(dx, dy),
                fontsize=11.5, fontweight="bold", color="#2c3e50", zorder=4)

# Adaptive 结果: Toys@512 (菱形, 区别于六个固定容量点)
ax.scatter(*TOYS512, s=180, marker="D", color="#f5b7b1", edgecolors="#c0392b",
           linewidths=1.6, zorder=3)
ax.annotate("Toys (K0=512)", TOYS512, textcoords="offset points", xytext=(10, 14),
            fontsize=10.5, fontstyle="italic", color="#c0392b", zorder=4)

# ---------- Adaptive 箭头 (Toys@256 → Toys@512, 穿过 ε 线; 标签在箭头上方) ----------
ax.annotate("", xy=TOYS512, xytext=(24, 0.0931),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.2,
                            linestyle="-", shrinkA=22, shrinkB=22), zorder=2)
ax.text(46, 0.06, "Adaptive: K0 256→512", fontsize=9.5,
        color="#c0392b", ha="left", va="center")

# ---------- Electronics: 扩容无效 (文字在红色带内, 箭头跨过 2ε 指向点) ----------
ax.annotate("扩容无效", xy=(100, 0.0116), xytext=(94, 0.03),
            fontsize=10, color="#ca6f1e", ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color="#ca6f1e", lw=1.2))

# ---------- 坐标轴 ----------
ax.set_xlabel("L0 utilization (%)", fontsize=13)
ax.set_ylabel(r"z$-$gap (log scale) = AUC(z) $-$ AUC(x$_{q}$[full])", fontsize=12.5)
ax.set_xlim(0, 122)
ax.set_ylim(2e-5, 0.3)
ax.set_yscale("log")
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
ax.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])

# ---------- 紧凑图例 (右上) ----------
handles = [mpatches.Patch(color=state_color[s], label=state_label[s]) for s in state_color]
ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92,
          borderpad=0.5, handlelength=1.4, handletextpad=0.5, labelspacing=0.45)

plt.tight_layout()
plt.savefig("doc/figures/paper_fig5x_taxonomy.png", bbox_inches="tight", dpi=200)
plt.savefig("doc/figures/paper_fig5x_taxonomy.pdf", bbox_inches="tight")
print("saved doc/figures/paper_fig5x_taxonomy.{png,pdf}")
