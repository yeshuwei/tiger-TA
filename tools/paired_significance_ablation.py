#!/usr/bin/env python3
"""
paired_significance_ablation.py — 行为消融主表（6 变体 × 3 数据集）的配对显著性检验

**这是 G2**（doc/behavior_structure_study_plan.md §10 步骤 1，此前状态 ⬜ 未做）。
此前主表只有 mean±std，对比全凭"±1std 区间是否重叠"这一经验法则——那不是检验。

数据源：
  复用 tools/ablation_3datasets_summary.py 的 LANES（逐 lane 显式文件名、禁 glob）。
  **不重新定义路径**，以免两条陷阱在第二个脚本里复活：
    ① Toys TA-SID 必须取健康 codes 的 tiger_ta_sid_768d_toys_rq2024_seed*
       （裸 Toys_ta_sid_768d_seed* 是孤儿 codes，L0=62，NDCG@20=0.0367 已作废）
    ② Toys Shuffled 的 tag 是 shuffled_768d，不是 shuffled_ablation

配对结构：
  RQ-VAE 固定 seed=2024（每数据集所有变体共用一份 codes），T5 用同一组 seed 2025/2026/2027
  ⇒ 同 seed 的两个变体之间，唯一变量是行为向量。故 seed 可配对，d_s = score_A(s) - score_B(s)。

统计（n=3, df=2）：
  配对 t 检验（scipy）+ Cohen's d_z + 越界计数 + 符号检验。
  ⚠️ n=3 的非参数符号检验双侧最小 p 恒为 2·(1/2)^3 = 0.25（下界）。
  故结论一律以"逐 seed 一致性 + 效应量"为主，p 值为辅且**必须与 df 同报**。

用法：
    python3 tools/paired_significance_ablation.py
    python3 tools/paired_significance_ablation.py --metric R20
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

# ---------------------------------------------------------------- 复用主表的 lane 定义（唯一真源）
import ablation_3datasets_summary as A3  # noqa: E402

A3.LOG_DIR = os.path.join(ROOT, A3.LOG_DIR)   # 固定为绝对路径，去除 cwd 依赖
SEEDS = list(A3.SEEDS)
DATASETS = list(A3.DATASETS)
VARIANTS = list(A3.VARIANTS)

# 与 doc/ablation_main_table_3datasets.md §1.1 (N@20) / §1.2 (R@20) 已发布的 mean 对账
# （容差 5e-5，论文只记 4 位小数）
PUBLISHED_R20_MEAN = {
    ("Sport",  "Content-only"):  0.0666, ("Sport",  "ZeroPad"):       0.0676,
    ("Sport",  "Shuffled"):      0.0646, ("Sport",  "ItemCF-SVD"):    0.0714,
    ("Sport",  "Raw-CoOcc-SVD"): 0.0731, ("Sport",  "TA-SID"):        0.0769,
    ("Toys",   "Content-only"):  0.0785, ("Toys",   "ZeroPad"):       0.0797,
    ("Toys",   "Shuffled"):      0.0670, ("Toys",   "ItemCF-SVD"):    0.0850,
    ("Toys",   "Raw-CoOcc-SVD"): 0.0802, ("Toys",   "TA-SID"):        0.0911,
    ("Beauty", "Content-only"):  0.0811, ("Beauty", "ZeroPad"):       0.0841,
    ("Beauty", "Shuffled"):      0.0773, ("Beauty", "ItemCF-SVD"):    0.0886,
    ("Beauty", "Raw-CoOcc-SVD"): 0.0869, ("Beauty", "TA-SID"):        0.0941,
}

PUBLISHED_N20_MEAN = {
    ("Sport",  "Content-only"):  0.0275, ("Sport",  "ZeroPad"):       0.0280,
    ("Sport",  "Shuffled"):      0.0264, ("Sport",  "ItemCF-SVD"):    0.0295,
    ("Sport",  "Raw-CoOcc-SVD"): 0.0305, ("Sport",  "TA-SID"):        0.0322,
    ("Toys",   "Content-only"):  0.0336, ("Toys",   "ZeroPad"):       0.0334,
    ("Toys",   "Shuffled"):      0.0284, ("Toys",   "ItemCF-SVD"):    0.0355,
    ("Toys",   "Raw-CoOcc-SVD"): 0.0340, ("Toys",   "TA-SID"):        0.0377,
    ("Beauty", "Content-only"):  0.0358, ("Beauty", "ZeroPad"):       0.0374,
    ("Beauty", "Shuffled"):      0.0346, ("Beauty", "ItemCF-SVD"):    0.0391,
    ("Beauty", "Raw-CoOcc-SVD"): 0.0387, ("Beauty", "TA-SID"):        0.0415,
}

# ---------------------------------------------------------------- 对比定义（消融阶梯 + 核心主张）
COMPARISONS = [
    # (a, b, 标签, 分组)
    ("TA-SID",        "Content-only",  "总增益（headline）",            "headline"),
    ("Raw-CoOcc-SVD", "Content-only",  "共现层：只做共现+SVD，不加 PPMI", "ladder"),
    ("TA-SID",        "Raw-CoOcc-SVD", "PPMI 层：在共现之上加 PPMI",     "ladder"),
    ("ZeroPad",       "Content-only",  "维度安慰剂（预期打平）",          "control"),
    ("Shuffled",      "Content-only",  "结构破坏（错配行为 vs 不给行为）", "control"),
    ("ItemCF-SVD",    "Content-only",  "通用 item-level CF 的增量",       "control"),
    ("TA-SID",        "ItemCF-SVD",    "相对通用 CF 的增量",             "control"),
    ("Raw-CoOcc-SVD", "ItemCF-SVD",    "相邻窗 vs 整篮（作用域，非单变量）", "exploratory"),
]


# ---------------------------------------------------------------- 统计
def cohens_dz(diff):
    d = np.asarray(diff, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") if d.mean() != 0 else 0.0
    return float(d.mean() / sd)


def paired_t(diff):
    """(t, p, df)。df=2 解析式：P(|T|>t) = 1 - t/sqrt(2+t^2)（已用 scipy 校验）。"""
    d = np.asarray(diff, dtype=float)
    n, df = len(d), len(d) - 1
    sd = d.std(ddof=1)
    if sd == 0:
        return (float("inf") if d.mean() != 0 else 0.0), (0.0 if d.mean() != 0 else 1.0), df
    t = d.mean() / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t), df))
    except ImportError:
        p = 1 - abs(t) / math.sqrt(2 + t * t) if df == 2 else float("nan")
    return float(t), float(p), df


def sign_test(diff):
    """双侧符号检验；n=3 时最小可能 p = 2*(1/2)^3 = 0.25。"""
    d = np.asarray(diff, dtype=float)
    n = int((d != 0).sum())
    if n == 0:
        return 1.0
    k = int(max((d > 0).sum(), (d < 0).sum()))
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="行为消融主表配对显著性检验（G2）")
    ap.add_argument("--metric", choices=["N20", "R20"], default="N20")
    ap.add_argument("--out-md", default="doc/ablation_paired_significance.md")
    ap.add_argument("--out-csv", default="doc/ablation_paired_significance.csv")
    args = ap.parse_args()

    idx = 0 if args.metric == "N20" else 1
    published = PUBLISHED_N20_MEAN if args.metric == "N20" else PUBLISHED_R20_MEAN
    ref_sec = "§1.1" if args.metric == "N20" else "§1.2"

    # ---------------- 采集 + provenance 断言 ----------------
    print("=" * 78)
    print(f"provenance 断言：逐 seed 解析值 vs doc/ablation_main_table_3datasets.md {ref_sec}")
    print("=" * 78)
    data, bad = {}, []
    for ds in DATASETS:
        for v in VARIANTS:
            files = A3.LANES[ds][v]
            got = {}
            for s in SEEDS:
                r = A3.parse(files[s]) if s in files else None
                if r is None or r[idx] is None:
                    bad.append(f"{ds}/{v}/seed{s} 缺产物 ← {files.get(s)}")
                else:
                    got[s] = r[idx]
            if len(got) == len(SEEDS):
                data[(ds, v)] = np.array([got[s] for s in SEEDS])
                m = float(data[(ds, v)].mean())
                exp = published.get((ds, v))
                if exp is not None and abs(m - exp) > 5e-5:
                    bad.append(f"{ds}/{v}: mean 期望 {exp} 实得 {m:.4f}")
    if bad:
        print("\n❌ provenance 断言失败：")
        for b in bad:
            print(f"   - {b}")
        return 2
    print(f"  ✅ 全部通过（{len(data)}/{len(DATASETS) * len(VARIANTS)} lane × 3 seed 齐备，"
          f"mean 与主表逐格相符）")

    # ---------------- 配对检验 ----------------
    rows = []
    print("\n" + "=" * 78)
    print(f"配对显著性检验（{args.metric}，n=3，df=2）")
    print("=" * 78)
    print(f"  ⚠️ n=3 非参数下界：双侧符号检验最小 p = 0.25；"
          f"配对 t 需 |d_z| > {4.30265 / math.sqrt(3):.2f} 才达 p<0.05\n")
    print(f"  {'数据集':7s} {'对比':44s} {'Δ':>9s} {'t':>7s} {'p':>8s} {'d_z':>7s} {'越界':>5s}")
    for ds in DATASETS:
        for a, b, label, group in COMPARISONS:
            if (ds, a) not in data or (ds, b) not in data:
                continue
            va, vb = data[(ds, a)], data[(ds, b)]
            diff = va - vb
            t, p, df = paired_t(diff)
            dz = cohens_dz(diff)
            cross = int((diff < 0).sum())
            rows.append({
                "dataset": ds, "group": group, "a": a, "b": b, "label": label,
                "a_seeds": va.tolist(), "b_seeds": vb.tolist(), "diff": diff.tolist(),
                "mean_diff": float(diff.mean()), "std_diff": float(diff.std(ddof=1)),
                "t": t, "df": df, "p": p, "dz": dz, "crossover": cross,
                "p_sign": sign_test(diff),
                "pooled_std": float(math.sqrt((va.std(ddof=1) ** 2 + vb.std(ddof=1) ** 2) / 2)),
            })
            flag = ""
            if df == 2 and abs(dz) > 4.30265 / math.sqrt(3):
                flag = " ✅p<.05"
            elif cross == 0:
                flag = " (3/3同向,未过线)"
            else:
                flag = f" ⚠{3 - cross}/3同向"
            print(f"  {ds:7s} {a + ' − ' + b:44s} {diff.mean():+9.5f} {t:7.2f} "
                  f"{p:8.4f} {dz:7.2f} {cross:5d}{flag}")

    # ---------------- 阶梯分解（每数据集） ----------------
    print("\n" + "=" * 78)
    print("消融阶梯分解：总增益 = 共现层 + PPMI 层")
    print("=" * 78)
    print(f"  {'数据集':7s} {'总增益':>10s} {'共现层':>18s} {'PPMI 层':>18s}")
    for ds in DATASETS:
        tot = data[(ds, "TA-SID")].mean() - data[(ds, "Content-only")].mean()
        r1 = next(r for r in rows if r["dataset"] == ds and r["a"] == "Raw-CoOcc-SVD"
                  and r["b"] == "Content-only")
        r2 = next(r for r in rows if r["dataset"] == ds and r["a"] == "TA-SID"
                  and r["b"] == "Raw-CoOcc-SVD")
        def txt(r):
            verdict = ("显著" if abs(r["dz"]) > 4.30265 / math.sqrt(3)
                       else ("同向未过线" if r["crossover"] == 0 else f"{3 - r['crossover']}/3同向"))
            return f"{r['mean_diff']:+.5f} {verdict}"
        print(f"  {ds:7s} {tot:+10.5f} {txt(r1):>18s} {txt(r2):>18s}")
    print("\n  ⚠️ 各层是点位估计；某一层不显著 ≠ 该层为零，只表示 n=3 下无法与 0 区分。")

    # ---------------- 输出 ----------------
    csv_path = os.path.join(ROOT, args.out_csv)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "dataset", "group", "method_a", "method_b", "label",
                    "a_seed2025", "a_seed2026", "a_seed2027",
                    "b_seed2025", "b_seed2026", "b_seed2027",
                    "mean_diff", "std_diff_ddof1", "t", "df", "p_paired_t",
                    "cohens_dz", "crossover_of_3", "p_sign_test", "gap_over_pooled_std"])
        for r in rows:
            w.writerow([args.metric, r["dataset"], r["group"], r["a"], r["b"], r["label"],
                        *[f"{v:.6f}" for v in r["a_seeds"]],
                        *[f"{v:.6f}" for v in r["b_seeds"]],
                        f"{r['mean_diff']:.6f}", f"{r['std_diff']:.6f}",
                        f"{r['t']:.4f}", r["df"], f"{r['p']:.6f}",
                        f"{r['dz']:.4f}", r["crossover"], f"{r['p_sign']:.6f}",
                        f"{r['mean_diff'] / r['pooled_std']:.3f}" if r["pooled_std"] else "inf"])
    print(f"\n  CSV → {os.path.relpath(csv_path, ROOT)}")

    write_md(os.path.join(ROOT, args.out_md), rows, args.metric)
    print(f"  MD  → {args.out_md}")
    return 0


def write_md(path, rows, metric):
    name = "NDCG@20" if metric == "N20" else "Recall@20"
    ref_sec = "§1.1" if metric == "N20" else "§1.2"
    thr = 4.30265 / math.sqrt(3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# 行为消融主表 · 配对显著性检验（{name}）\n\n")
        f.write("> 自动生成：`python3 tools/paired_significance_ablation.py "
                f"--metric {metric}`（勿手改）\n")
        f.write("> 数据源复用 `tools/ablation_3datasets_summary.py` 的 LANES；"
                f"逐 seed 值经 provenance 断言与主表 {ref_sec} 对齐。\n\n")
        f.write("## 0. 口径（必读）\n\n")
        f.write("配对结构：RQ-VAE 固定 seed=2024（每数据集所有变体共用一份 codes），"
                "T5 用同一组 seed 2025/2026/2027 ⇒ **同 seed 两变体之间唯一变量是行为向量**。\n\n")
        f.write(f"- n=3，df=2。配对 t 检验需 **|d_z| > {thr:.2f}** 才达 p<0.05。\n")
        f.write("- ⚠️ 非参数符号检验在 n=3 的双侧最小 p **恒为 0.25**（下界）。"
                "故主结论用「逐 seed 一致性 + 效应量」，p 值仅作辅助且**必须与 df 同报**。\n")
        f.write("- 某层不显著 ≠ 该层为零，只表示 n=3 下无法与 0 区分。\n\n")

        f.write(f"## 1. 配对检验结果（{name}）\n\n")
        f.write("| 数据集 | 对比 | Δ | t | p (df=2) | d_z | 越界 | 符号检验 p |\n")
        f.write("|:--|:--|--:|--:|--:|--:|:--:|--:|\n")
        for r in rows:
            verdict = " ✅" if abs(r["dz"]) > thr else (" ³/₃" if r["crossover"] == 0 else " ⚠")
            f.write(f"| {r['dataset']} | {r['a']} − {r['b']} | {r['mean_diff']:+.5f} | "
                    f"{r['t']:.2f} | {r['p']:.4f}{verdict} | {r['dz']:.2f} | "
                    f"{3 - r['crossover']}/3 | {r['p_sign']:.3f} |\n")

        f.write("\n### 1.1 逐 seed 明细\n\n")
        f.write("| 数据集 | 对比 | s2025 | s2026 | s2027 | Δ/合并std |\n")
        f.write("|:--|:--|--:|--:|--:|--:|\n")
        for r in rows:
            g = f"{r['mean_diff'] / r['pooled_std']:.2f}×" if r["pooled_std"] else "—"
            f.write(f"| {r['dataset']} | {r['a']} − {r['b']} | "
                    f"{r['diff'][0]:+.5f} | {r['diff'][1]:+.5f} | {r['diff'][2]:+.5f} | {g} |\n")

        f.write("\n## 2. 消融阶梯分解\n\n")
        f.write("总增益 = 共现层（`Raw-CoOcc − Content`）+ PPMI 层（`TA-SID − Raw-CoOcc`）\n\n")
        f.write("| 数据集 | 总增益 | 共现层 | PPMI 层 |\n|:--|--:|--:|--:|\n")
        for ds in DATASETS:
            def cell(r):
                if abs(r["dz"]) > thr:
                    return f"{r['mean_diff']:+.5f} ✅"
                return (f"{r['mean_diff']:+.5f} 同向未过线" if r["crossover"] == 0
                        else f"{r['mean_diff']:+.5f} ⚠{3 - r['crossover']}/3")
            r1 = next(r for r in rows if r["dataset"] == ds and r["a"] == "Raw-CoOcc-SVD")
            r2 = next(r for r in rows if r["dataset"] == ds and r["a"] == "TA-SID"
                      and r["b"] == "Raw-CoOcc-SVD")
            ra = next(r for r in rows if r["dataset"] == ds and r["a"] == "TA-SID"
                      and r["b"] == "Content-only")
            f.write(f"| {ds} | {ra['mean_diff']:+.5f} | {cell(r1)} | {cell(r2)} |\n")

        f.write("\n## 3. 写作红线（承接 seq_baselines_significance.md §4）\n\n")
        f.write("- **禁止**写 \"statistically significant (p < 0.05)\" 而不交代 n=3 与 df=2。\n")
        f.write("- **禁止**只报参数检验、不提非参数下界 0.25。\n")
        f.write(f"- 推荐首选表述：\"consistent across all 3 seeds (no crossover); "
                f"gap = X, which is Y× the pooled between-seed std\"。\n")
        f.write("- 若报 p：必须连 df 与逐 seed 原值一起报，并注明正态性假设无法在 n=3 下检验。\n")
        f.write("- **不得**因某一层不显著就写\"该层贡献为零\"——n=3 的检出力不足以支持该断言。\n")


if __name__ == "__main__":
    sys.exit(main())
