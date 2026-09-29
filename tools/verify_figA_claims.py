#!/usr/bin/env python3
"""
verify_figA_claims.py — 图 A（Behavior AUC × NDCG@20 散点）的三层校验

背景：图 A 曾经"看起来画好了"，但图例实际是 0 个条目渲染出的 9.6×9.6 px 空壳
（handles 在数据集循环内被重置）。**肉眼检查抓不到这类 bug**，必须回读 artist。
本脚本因此不信任出图脚本的打印输出，而是重建图形对象、把画出去的每个点读回来
与权威表逐格对账。

三层校验：
  L1 源侧溯源 —— 每格 AUC 到底来自哪个文件、是否只是同源转抄
  L2 回读对账 —— fig 里 artist 的 (x, y, yerr) 是否 == 权威表 (auc, mean, std)
  L3 结构断言 —— 面板/点数/marker 唯一性/图例条目与包围盒/点是否越界

产物：
  doc/figures/verify_figA_annotated.{png,pdf}   带数值标注的审计版图 A（人眼可复核）
  doc/figures/verify_figA_provenance.{png,pdf}  18 格数据来源矩阵
  doc/figA_claims_verification.md               PASS/FAIL 报告（含逐格对账表）

用法：
    python3 tools/verify_figA_claims.py [--report doc/figA_claims_verification.md]

退出码：0 = 全 PASS；1 = 存在 FAIL；2 = 存在 WARN 但无 FAIL。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from plot_paper_ablation_figs import (  # noqa: E402
    ABL_MD, ALIAS, DATASETS, INK, INK_SOFT, LAYER, LAYER_COLOR, MARKER,
    PROBE_CSV, SHORT, SURFACE, TIER2_CSV, VARIANTS,
    build_auc_vs_ndcg, load_auc, load_ndcg, place_labels, style_axes,
)

TOL_DATA = 1e-9          # artist 回读 vs 源表：应当逐位相等
TOL_SRC = 1e-4           # 两份源文件之间
LEGEND_AREA_FLOOR = 0.02  # 图例包围盒至少占画布面积 2%（空图例 ≈ 0.08%）

RESULTS = []             # [(level, name, status, detail)]


def record(level, name, ok, detail="", warn=False):
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((level, name, status, detail))
    mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[status]
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ------------------------------------------------------------ L1 源侧溯源

def check_provenance():
    print("\n▌ L1 源侧溯源（每格 AUC 来自哪个文件；识别同源转抄）")
    probe = pd.read_csv(PROBE_CSV)
    probe = probe[(probe.space == "xq[full]") & (probe.pos_regime == "topk")].copy()
    probe["variant"] = probe["variant"].replace(ALIAS)

    t2 = pd.read_csv(TIER2_CSV)
    t2["variant"] = t2["variant"].replace(ALIAS)

    pmap = probe.set_index(["dataset", "variant"])["auc_cos_degmatch"].to_dict()
    tmap = t2.set_index(["dataset", "variant"])["behavior_auc_xq_full_topk"].to_dict()

    src = {}
    n_both_indep, n_tier1, n_copy = 0, 0, 0
    for ds in DATASETS:
        for v in VARIANTS:
            key = (ds, v)
            in_p, in_t = key in pmap, key in tmap
            if in_p and in_t:
                # Beauty 的两格：probe 与 Tier-1 是**两次独立运行** → 交叉印证
                # Sport/Toys 的十二格：tier2 表由 beauty_tier2_summary.py §九
                #   直接从 probe CSV 复制（a = s["auc_cos_degmatch"]）→ 同源自证
                if ds == "Beauty":
                    src[key] = "双源独立"; n_both_indep += 1
                else:
                    src[key] = "同源转抄"; n_copy += 1
            elif in_t:
                src[key] = "仅 Tier-1"; n_tier1 += 1
            elif in_p:
                src[key] = "仅 probe"
            else:
                src[key] = "缺失"

    missing = [k for k, v in src.items() if v == "缺失"]
    record("L1", "18 格全部有源", not missing, f"缺失 {missing}" if missing else "18/18")

    # 严格来说只有 6 格是真正独立的双源交叉印证
    independent = sum(1 for v in src.values() if v in ("双源独立", "仅 Tier-1"))
    record("L1", "独立源格数 ≥ 6", independent >= 6,
           f"独立 {independent} 格（Beauty 2 双源 + 4 仅 Tier-1）；"
           f"另有 {n_copy} 格为 tier2 表从 probe CSV 转抄，不构成交叉印证", warn=independent < 6)

    ints = [(k, pmap[k], tmap[k]) for k in src
            if src[k] == "双源独立"]
    if ints:
        d = max(abs(a - b) for _, a, b in ints)
        record("L1", f"双源独立格一致（n={len(ints)}）", d <= TOL_SRC, f"最大差 {d:.2e}")
        for k, a, b in ints:
            if abs(a - b) > TOL_SRC:
                record("L1", f"  {k[0]}/{k[1]}", False, f"probe {a:.6f} vs tier2 {b:.6f}")

    # N@20 侧的上游：重跑纯归档脚本，确认权威表 = eval 日志的确定性产物
    import subprocess
    import tempfile
    tmp = tempfile.mktemp(suffix=".md")
    p = subprocess.run([sys.executable, os.path.join(TOOLS_DIR, "ablation_3datasets_summary.py"),
                        "--out", tmp], capture_output=True, text=True)
    if p.returncode != 0:
        record("L1", "N@20 上游表可从 eval 日志重生成", False, p.stderr.strip()[-200:])
    else:
        def _body(path):   # 忽略「生成时间」行（每次重跑必然不同）
            with open(path, encoding="utf-8") as f:
                return [l for l in f if not l.startswith("**生成时间**")]
        same = _body(tmp) == _body(ABL_MD)
        record("L1", "N@20 上游表可从 eval 日志逐字节重生成", same,
               "ablation_3datasets_summary.py 重跑结果与 doc/ablation_main_table_3datasets.md 一致"
               if same else "存在差异 → 权威表与 eval 日志脱节")
        os.unlink(tmp)

    return src


# ------------------------------------------------------------ L2 回读对账

def read_back(fig, axes):
    """从 artist 回读每个点的 (x, y, yerr)。返回 {(dataset, variant): dict}。"""
    got = {}
    for ax in axes:
        ds = ax.get_title()
        for c in ax.containers:
            if not isinstance(c, ErrorbarContainer):
                continue
            line = c.lines[0]
            gid = line.get_gid() or ""
            if not gid.startswith("figA|"):
                continue
            _, g_ds, g_v = gid.split("|", 2)
            x = float(np.asarray(line.get_xdata(), dtype=float)[0])
            y = float(np.asarray(line.get_ydata(), dtype=float)[0])
            seg = c.lines[2][0].get_segments()
            ylo = float(seg[0][0][1])
            yhi = float(seg[0][1][1])
            got[(g_ds, g_v)] = {
                "x": x, "y": y, "yerr_lo": y - ylo, "yerr_hi": yhi - y,
                "axes": ds,
                "marker": line.get_marker(),
                "color": line.get_markerfacecolor(),
            }
    return got


def check_readback(src_df):
    print("\n▌ L2 回读对账（重建 fig，把画出去的 artist 读回来 vs 权威表）")
    fig, axes, handles, labels = build_auc_vs_ndcg(src_df)
    fig.canvas.draw()
    got = read_back(fig, axes)

    record("L2", "回读点数 == 18", len(got) == 18, f"读回 {len(got)}")

    rows, worst = [], 0.0
    for _, r in src_df.iterrows():
        key = (r["dataset"], r["variant"])
        g = got.get(key)
        if g is None:
            rows.append({"dataset": key[0], "variant": key[1], "dx": np.nan,
                         "dy": np.nan, "derr": np.nan, "ok": False})
            continue
        dx = abs(g["x"] - r["auc"])
        dy = abs(g["y"] - r["mean"])
        de = max(abs(g["yerr_lo"] - r["std"]), abs(g["yerr_hi"] - r["std"]))
        worst = max(worst, dx, dy, de)
        rows.append({"dataset": key[0], "variant": key[1],
                     "src_auc": r["auc"], "fig_auc": g["x"],
                     "src_mean": r["mean"], "fig_mean": g["y"],
                     "src_std": r["std"], "fig_yerr": g["yerr_lo"],
                     "dx": dx, "dy": dy, "derr": de,
                     "ok": max(dx, dy, de) <= TOL_DATA})
    rb = pd.DataFrame(rows)

    record("L2", "18 格 x/y/yerr 全部与源表一致", bool(rb["ok"].all()),
           f"最大偏差 {worst:.2e}（容差 {TOL_DATA:.0e}）")
    bad = rb[~rb["ok"]]
    for _, b in bad.iterrows():
        record("L2", f"  {b['dataset']}/{b['variant']}", False, b.to_dict())

    # 面板归属：gid 里的数据集必须与所在 axes 标题一致
    misplaced = [k for k, g in got.items() if k[0] != g["axes"]]
    record("L2", "点未串面板（gid 数据集 == 面板标题）", not misplaced, f"{misplaced}")

    return fig, axes, handles, labels, rb, got


# ------------------------------------------------------------ L3 结构断言

def check_structure(fig, axes, handles, labels, got):
    print("\n▌ L3 结构断言（图例/形状/越界 —— 空图例这类 bug 只能在这一层抓到）")
    r = fig.canvas.get_renderer()

    record("L3", "面板数 == 3", len(axes) == 3, f"{len(axes)}")
    record("L3", "面板标题 == DATASETS", [a.get_title() for a in axes] == DATASETS,
           f"{[a.get_title() for a in axes]}")

    per = {}
    for (ds, v) in got:
        per.setdefault(ds, []).append(v)
    shape_ok = all(sorted(per.get(ds, [])) == sorted(VARIANTS) for ds in DATASETS)
    record("L3", "每面板 6 点且变体集完整", shape_ok,
           " / ".join(f"{ds}:{len(per.get(ds, []))}" for ds in DATASETS))

    # marker 唯一性：同层同色时仅靠颜色分不开，形状必须全局唯一
    sig = {v: (g["marker"], g["color"]) for (ds, v), g in got.items()
           if ds == DATASETS[0]}
    dup = len({s for s in sig.values()}) != len(sig)
    record("L3", f"6 变体 (marker,color) 全局唯一", not dup and len(sig) == 6,
           " ".join(f"{SHORT[v]}={g[0]}" for v, g in sorted(sig.items())))
    marker_uniq = len({g[0] for g in sig.values()}) == 6
    record("L3", "marker 形状本身即全局唯一（不依赖颜色）", marker_uniq,
           " ".join(f"{SHORT[v]}={g[0]}" for v, g in sorted(sig.items())))

    # 图例：条目数 + 标签 + 包围盒（空图例的包围盒 ≈ 一个点大小）
    n_leg = len(fig.legends)
    record("L3", "存在 figure 级图例", n_leg == 1, f"fig.legends={n_leg}")
    if n_leg == 1:
        leg = fig.legends[0]
        texts = [t.get_text() for t in leg.get_texts()]
        record("L3", "图例条目数 == 6", len(texts) == 6, f"{texts}")
        record("L3", "图例标签集 == VARIANTS", set(texts) == set(VARIANTS))
        bb = leg.get_window_extent(r)
        frac = (bb.width * bb.height) / (fig.bbox.width * fig.bbox.height)
        record("L3", f"图例包围盒占画布面积 > {LEGEND_AREA_FLOOR:.0%}", frac > LEGEND_AREA_FLOOR,
               f"{bb.width:.1f}×{bb.height:.1f} px = {frac:.2%}"
               + ("  ← 空图例回归！" if frac <= LEGEND_AREA_FLOOR else ""))
        inb = fig.bbox.contains(bb.x0, bb.y0) and fig.bbox.contains(bb.x1, bb.y1)
        record("L3", "图例包围盒在画布内", inb, f"fig.bbox={tuple(round(v) for v in fig.bbox.bounds)}")

    # 越界：点 + 误差棒端点都必须落在 axes 数据范围内
    out = []
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for (ds, v), g in got.items():
            if ds != ax.get_title():
                continue
            if not (x0 <= g["x"] <= x1 and y0 <= g["y"] - g["yerr_lo"] <= y1
                    and y0 <= g["y"] + g["yerr_hi"] <= y1):
                out.append(f"{ds}/{v}")
    record("L3", "所有点与误差棒端点均在坐标范围内", not out, f"越界：{out}")


# ------------------------------------------------------------ 出图

def fig_annotated(src_df, outdir, dpi):
    """审计版图 A：点旁只标「该面板内 AUC 降序名次」，名次对应数值列在面板下方的表里。

    面板内只放 1 个字符 → 不会像长标签那样互相压住（长标签版实测在 Beauty/Sport
    面板里 ItemCF 与 Raw-CoOcc 的标签会叠在一起，反而无法复核）。复核方式是：
    点应从右到左依次编号 1..6，且下方表的第 i 行的 AUC 必须是该面板第 i 大。
    """
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.1),
                             gridspec_kw={"height_ratios": [2.7, 1.15]})
    fig.patch.set_facecolor(SURFACE)

    for col, ds in enumerate(DATASETS):
        ax, txt = axes[0][col], axes[1][col]
        sub = src_df[src_df.dataset == ds].set_index("variant").loc[VARIANTS].reset_index()
        style_axes(ax)

        for _, r in sub.iterrows():
            v = r["variant"]
            ax.errorbar(r["auc"], r["mean"], yerr=r["std"],
                        fmt=MARKER[v], ms=7.2, mfc=LAYER_COLOR[LAYER[v]], mec="white",
                        mew=1.1, ecolor=INK_SOFT, elinewidth=0.9, capsize=2.5,
                        capthick=0.9, zorder=4 if v == "TA-SID" else 3)

        pad_x = (sub["auc"].max() - sub["auc"].min()) * 0.18
        pad_y = (sub["mean"].max() - sub["mean"].min()) * 0.20
        ax.set_xlim(sub["auc"].min() - pad_x, sub["auc"].max() + pad_x)
        ax.set_ylim((sub["mean"] - sub["std"]).min() - pad_y,
                    (sub["mean"] + sub["std"]).max() + pad_y * 0.6)
        ax.set_title(ds, fontsize=10.5, color=INK, pad=7)

        # 名次 1..6（AUC 降序），标在点上方/下方交替，单字符不会互压
        ranked = sub.sort_values("auc", ascending=False).reset_index(drop=True)
        for i, r in ranked.iterrows():
            dy = 11 + (r["std"] * 0) + 0
            _, py = ax.transData.transform((r["auc"], r["mean"]))
            _, py_hi = ax.transData.transform((r["auc"], r["mean"] + r["std"]))
            off = abs(py_hi - py) + 10
            ax.annotate(f"{i + 1}", xy=(r["auc"], r["mean"]), xytext=(0, off if i % 2 == 0 else -off),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=8.2, color=INK, fontweight="bold", zorder=6)

        # 面板下方数值表：第 i 行 = 名次 i
        txt.axis("off")
        txt.set_facecolor(SURFACE)
        lines = [f"{'#':>2}  {'variant':<13} {'AUC':>7}  {'NDCG@20':>8}"]
        for i, r in ranked.iterrows():
            lines.append(f"{i + 1:>2}  {SHORT[r['variant']]:<13} {r['auc']:.4f}  "
                         f"{r['mean']:.4f}±{r['std']:.4f}")
        txt.text(0.0, 1.0, "\n".join(lines), transform=txt.transAxes,
                 family="monospace", fontsize=6.9, color=INK, va="top", ha="left")

    axes[0][0].set_ylabel("NDCG@20", fontsize=9.5, color=INK)
    for ax in axes[0]:
        ax.set_xlabel("Behavior AUC  ($x_q^{[full]}$)", fontsize=9.0, color=INK)

    fig.suptitle("Figure A audit copy — in-panel digit = within-panel rank by Behavior AUC "
                 "(rightmost point should be #1); values below are read back from the "
                 "authoritative table",
                 fontsize=8.8, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"verify_figA_annotated.{ext}")
        fig.savefig(p, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs


def fig_provenance(src, outdir, dpi):
    """18 格数据来源矩阵：区别「双源独立印证」与「同源转抄」。"""
    codes = {"双源独立": 3, "仅 Tier-1": 2, "同源转抄": 1, "仅 probe": 0, "缺失": -1}
    colors = {3: "#2f9e8f", 2: "#2a78d6", 1: "#e0a33a", 0: "#9aa0a6", -1: "#d1495b"}
    # 图内文字一律 ASCII：matplotlib 默认字体无 CJK 字形，中文会渲染成豆腐块。
    # 每种来源两行（单元格很窄，单行会溢出到相邻格）。
    legend_en = {"双源独立": "dual-source\nindependent",
                 "仅 Tier-1": "Tier-1\nonly",
                 "同源转抄": "copied\nfrom probe",
                 "仅 probe": "probe\nonly",
                 "缺失": "missing"}
    M = np.array([[codes[src[(ds, v)]] for ds in DATASETS] for v in VARIANTS], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(
        [colors[c] for c in (-1, 0, 1, 2, 3)]), vmin=-1.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(DATASETS)), DATASETS, fontsize=9, color=INK)
    ax.set_yticks(range(len(VARIANTS)), [SHORT[v] for v in VARIANTS], fontsize=9, color=INK)
    ax.set_xticks(np.arange(-.5, len(DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(VARIANTS), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="both", length=0)
    for i, v in enumerate(VARIANTS):
        for j, ds in enumerate(DATASETS):
            t = src[(ds, v)]
            light = codes[t] in (3, 2, -1)
            ax.text(j, i - 0.20, legend_en[t], ha="center", va="center", fontsize=6.2,
                    linespacing=1.15, color="white" if light else INK)
            if t != "缺失":
                pm = pd.read_csv(PROBE_CSV)
                pm = pm[(pm.space == "xq[full]") & (pm.pos_regime == "topk")]
                pm["variant"] = pm["variant"].replace(ALIAS)
                row = pm[(pm.dataset == ds) & (pm.variant == v)]
                val = row["auc_cos_degmatch"].iloc[0] if len(row) else np.nan
                if np.isnan(val):
                    t2 = pd.read_csv(TIER2_CSV)
                    t2["variant"] = t2["variant"].replace(ALIAS)
                    val = t2[(t2.dataset == ds) & (t2.variant == v)][
                        "behavior_auc_xq_full_topk"].iloc[0]
                ax.text(j, i + 0.26, f"{val:.4f}", ha="center", va="center",
                        fontsize=7.4, fontweight="bold",
                        color="white" if light else INK_SOFT)
    ax.set_title("Figure A data-provenance matrix (18 cells)\nupper = source type, lower = AUC value",
                 fontsize=10, color=INK, pad=8)
    ax.legend(handles=[Patch(facecolor=colors[c], label=legend_en[k]) for k, c in codes.items()],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    outs = []
    for ext in ("png", "pdf"):
        p = os.path.join(outdir, f"verify_figA_provenance.{ext}")
        fig.savefig(p, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs


# ------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="doc/figures")
    ap.add_argument("--report", default="doc/figA_claims_verification.md")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("图 A 校验：三层对账（源侧溯源 / 回读对账 / 结构断言）")
    print("=" * 78)

    src_map = check_provenance()
    df = load_auc().merge(load_ndcg(), on=["dataset", "variant"], how="inner")
    fig, axes, handles, labels, rb, got = check_readback(df)
    check_structure(fig, axes, handles, labels, got)
    plt.close(fig)

    outs = fig_annotated(df, args.outdir, args.dpi) + fig_provenance(src_map, args.outdir, args.dpi)

    n_pass = sum(1 for _l, _n, s, _d in RESULTS if s == "PASS")
    n_warn = sum(1 for _l, _n, s, _d in RESULTS if s == "WARN")
    n_fail = sum(1 for _l, _n, s, _d in RESULTS if s == "FAIL")

    print("\n" + "=" * 78)
    print(f"汇总：PASS {n_pass} / WARN {n_warn} / FAIL {n_fail}")
    for p in outs:
        print(f"[out] {p}")
    print("=" * 78)

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# 图 A 校验报告（自动生成，勿手改）\n\n")
        f.write(f"生成命令：`python3 tools/verify_figA_claims.py`\n\n")
        f.write(f"**汇总：PASS {n_pass} / WARN {n_warn} / FAIL {n_fail}**\n\n")
        f.write("## 逐项结果\n\n| 层 | 检查项 | 结果 | 说明 |\n|:--|:--|:--|:--|\n")
        for lvl, name, st, det in RESULTS:
            f.write(f"| {lvl} | {name} | **{st}** | {det} |\n")

        f.write("\n## L2 逐格对账（artist 回读 vs 权威表）\n\n")
        f.write("| Dataset | Variant | 源 AUC | 图中 AUC | 源 N@20 | 图中 N@20 | 源 std | 图中 yerr | Δ |\n")
        f.write("|:--|:--|--:|--:|--:|--:|--:|--:|--:|\n")
        for _, r in rb.iterrows():
            if np.isnan(r.get("dx", np.nan)):
                f.write(f"| {r['dataset']} | {r['variant']} | — | — | — | — | — | — | **缺失** |\n")
                continue
            d = max(r["dx"], r["dy"], r["derr"])
            f.write(f"| {r['dataset']} | {r['variant']} | {r['src_auc']:.6f} | {r['fig_auc']:.6f} | "
                    f"{r['src_mean']:.4f} | {r['fig_mean']:.4f} | {r['src_std']:.4f} | "
                    f"{r['fig_yerr']:.4f} | {d:.1e} |\n")

        f.write("\n## L1 数据来源矩阵\n\n| Variant | " + " | ".join(DATASETS) + " |\n")
        f.write("|:--|" + ":--|" * len(DATASETS) + "\n")
        for v in VARIANTS:
            f.write(f"| {SHORT[v]} | " + " | ".join(src_map[(ds, v)] for ds in DATASETS) + " |\n")

        f.write("\n> **来源类型说明**：`双源独立` = probe CSV 与 Tier-1 两次独立运行均产出且一致；"
                "`仅 Tier-1` = 只有 Tier-1 产出（Beauty 后补的 4 变体）；"
                "`同源转抄` = `beauty_tier2_summary.py` §九 直接从 probe CSV 复制，**不构成交叉印证**。\n")

    print(f"[out] {args.report}")
    return 1 if n_fail else (2 if n_warn else 0)


if __name__ == "__main__":
    sys.exit(main())
