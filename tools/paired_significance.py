#!/usr/bin/env python3
"""
配对显著性检验 + std 口径对账（零 GPU，逐 seed 数据现成）

三件事：
  A. **std 口径对账**（横跨论文各处，见下）。论文当前存在**同一量两套口径**：
     `sasrec_output/*/summary_3seed.txt` 用 ddof=0（总体标准差），而
     `doc/ablation_main_table_3datasets.md`（及图 A）用 ddof=1（样本标准差）。
     本脚本把两者并列输出，给出逐格对照，供统一口径。
  B. **配对显著性检验**：paired t-test / Wilcoxon 符号秩 / 符号检验 /
     Cohen's d_z / seed 交叉计数。
  C. **provenance 断言**：把已发布的逐 seed N@20 硬编码为期望值，解析不符即
     非零退出。这一条同时挡住两类事故：
       - 取到 Toys 孤儿 codes 日志（`Toys_ta_sid_768d_seed*` N@20=0.0367 已作废，
         健康源是 `tiger_ta_sid_768d_toys_rq2024_seed*`，N@20=0.0377）
       - 文件缺失被静默当 0

【为何必须显式文件名】不用 tag glob：`model/logs/` 下同名族文件并存且语义不同，
glob 会静默混入作废产物。所有路径逐字写死。

用法：
    python3 tools/paired_significance.py \
        --out-md doc/seq_baselines_significance.md \
        --out-csv doc/seq_baselines_significance.csv

    # 只对账 std / 只跑已有方法（GRU4Rec/BERT4Rec 产物齐了会自动纳入）：
    python3 tools/paired_significance.py --check-std-only
"""

import os
import re
import sys
import csv
import argparse
import math

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [2025, 2026, 2027]

# ---------------------------------------------------------------- 已发布值（权威源：doc/paper_main_results.md §3，已勘误版）
# 解析结果必须与之相符（容差 5e-5，论文只记 4 位小数），否则非零退出。
PUBLISHED_NDCG20 = {
    ("Sport", "SASRec"):       [0.0137, 0.0135, 0.0132],
    ("Sport", "Content-only"): [0.0274, 0.0283, 0.0269],
    ("Sport", "TA-SID"):       [0.0327, 0.0317, 0.0322],
    ("Beauty", "SASRec"):      [0.0256, 0.0245, 0.0262],
    ("Beauty", "Content-only"): [0.0368, 0.0352, 0.0354],
    ("Beauty", "TA-SID"):      [0.0418, 0.0425, 0.0403],
    ("Toys", "SASRec"):        [0.0237, 0.0243, 0.0244],
    ("Toys", "Content-only"):  [0.0341, 0.0338, 0.0328],
    ("Toys", "TA-SID"):        [0.0369, 0.0383, 0.0380],
}

# ---------------------------------------------------------------- 数据源（逐字写死，禁止 glob）
def _t5_log(ds, method, seed):
    """T5 eval 日志路径。Toys TA-SID 必须用 rq2024 健康 codes 版本。"""
    if method == "TA-SID":
        if ds == "Toys":
            # 健康源：L0=207 best_collision codes。裸 Toys_ta_sid_768d_seed* 是孤儿，作废。
            return os.path.join(ROOT, "model/logs",
                                f"tiger_ta_sid_768d_toys_rq2024_seed{seed}_eval.log")
        return os.path.join(ROOT, "model/logs", f"{ds}_ta_sid_768d_seed{seed}_eval.log")
    if method == "Content-only":
        return os.path.join(ROOT, "model/logs",
                            f"{ds}_content768d_baseline_seed{seed}_eval.log")
    raise ValueError(method)


SEQ_SUMMARY_DIRS = {
    "SASRec":   "sasrec_output",
    "GRU4Rec":  "gru4rec_output",
    "BERT4Rec": "bert4rec_output",
}

NDCG_RE = re.compile(r"^\s*NDCG\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M)
RCL_RE = re.compile(r"^\s*Recall\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M)


def parse_t5_log(path):
    """读 T5 eval 日志 → dict(N5,N10,N20,R5,R10,R20) 或 None。取最后一次结果表。

    NDCG/Recall 表各 3 列，依次对应 @5/@10/@20（与 summary_3seed.txt 列序一致）。
    """
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    n = NDCG_RE.findall(txt)
    if not n:
        return None
    r = RCL_RE.findall(txt)
    out = {"N5": float(n[-1][0]), "N10": float(n[-1][1]), "N20": float(n[-1][2])}
    if r:
        out.update({"R5": float(r[-1][0]), "R10": float(r[-1][1]), "R20": float(r[-1][2])})
    else:
        out.update({"R5": None, "R10": None, "R20": None})
    return out


def parse_seq_summary(path):
    """读 {m}_output/{ds}/summary_3seed.txt → {seed: {'N20':, 'R20':, ...}} 或 None。"""
    if not os.path.exists(path):
        return None
    out = {}
    for ln in open(path, errors="ignore"):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("#") or ln.startswith("seed\t") or ln.startswith("mean"):
            continue
        p = ln.split("\t")
        if len(p) < 7:
            continue
        try:
            seed = int(p[0])
        except ValueError:
            continue
        out[seed] = {"N5": float(p[1]), "N10": float(p[2]), "N20": float(p[3]),
                     "R5": float(p[4]), "R10": float(p[5]), "R20": float(p[6])}
    return out or None


def collect(ds, method):
    """→ {seed: {'N20':..., 'R20':...}} 或 None（产物未就绪）"""
    if method in SEQ_SUMMARY_DIRS:
        path = os.path.join(ROOT, SEQ_SUMMARY_DIRS[method], ds, "summary_3seed.txt")
        return parse_seq_summary(path)
    out = {}
    for s in SEEDS:
        rec = parse_t5_log(_t5_log(ds, method, s))
        if rec is None:
            return None
        out[s] = rec
    return out


# ---------------------------------------------------------------- 统计（n=3）
def cohens_dz(diff):
    d = np.asarray(diff, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") if d.mean() != 0 else 0.0
    return float(d.mean() / sd)


def paired_t(diff):
    """返回 (t, p, df)。无 scipy 时手算（df=2 的 t 分布 CDF 有解析式）。"""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    df = n - 1
    sd = d.std(ddof=1)
    if sd == 0:
        return (float("inf") if d.mean() != 0 else 0.0), (0.0 if d.mean() != 0 else 1.0), df
    t = d.mean() / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t), df))
    except ImportError:
        # df=2 解析式：P(|T|>t) = 1 - t/sqrt(2+t^2)
        # ⚠️ 2026-10-07 修正：原写 sqrt(3+t^2) 为笔误（会系统性放大 p，偏保守）。
        # 已用 scipy 1.11.4 逐点校验：t=4.3027 时 2+t² 形式给 0.049999（=临界值），
        # 3+t² 形式给 0.072341（错误）。scipy 在装时本行不触发，故历史数字未受影响。
        # 仅 df=2 有闭式；其余 df 无 scipy 时返回 nan（不猜）。
        p = 1 - abs(t) / math.sqrt(2 + t * t) if df == 2 else float("nan")
    return float(t), float(p), df


def wilcoxon(diff):
    d = np.asarray(diff, dtype=float)
    if np.all(d == 0):
        return None, 1.0
    try:
        from scipy import stats
        res = stats.wilcoxon(d)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        return None, None


def sign_test(diff):
    """双侧符号检验，n=3 最小可能 p = 2*(1/2)^3 = 0.25。"""
    d = np.asarray(diff, dtype=float)
    k = int(max((d > 0).sum(), (d < 0).sum()))
    n = int((d != 0).sum())
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return float(min(1.0, 2 * p))


def std_both(values):
    a = np.asarray(values, dtype=float)
    return float(a.std(ddof=0)), float(a.std(ddof=1))


METRIC_ORDER = [("N5", "NDCG@5"), ("N10", "NDCG@10"), ("N20", "NDCG@20"),
                ("R5", "Recall@5"), ("R10", "Recall@10"), ("R20", "Recall@20")]


def emit_corrected_table(methods):
    """按 ddof=1 重算主表（mean ± std），供 doc/paper_main_results.md §1 统一口径。

    与 §1 现有表格逐格对应（SASRec / Content-only / TA-SID × 3 数据集 × 6 指标）。
    """
    lines = []
    for m in methods:
        for ds in ["Sport", "Beauty", "Toys"]:
            data = collect(ds, m)
            if not data:
                lines.append(f"| {m} | {ds} | (产物缺失) |")
                continue
            cells = []
            for key, _ in METRIC_ORDER:
                vals = [data[s][key] for s in SEEDS]
                if any(v is None for v in vals):
                    cells.append("—")
                    continue
                _, s1 = std_both(vals)
                cells.append(f"{np.mean(vals):.4f}±{s1:.4f}")
            name = m if ds == "Sport" else ""
            lines.append(f"| {name} | {ds} | " + " | ".join(cells) + " |")
    return lines


# ---------------------------------------------------------------- A. std 口径对账
def std_reconciliation(methods):
    """逐 (数据集, 方法) 输出 ddof=0 / ddof=1，并标出论文各处当前用的是哪个。"""
    rows = []
    for ds in ["Sport", "Toys", "Beauty"]:
        for m in methods:
            data = collect(ds, m)
            if not data:
                continue
            for metric in ["N20", "R20"]:
                vals = [data[s][metric] for s in SEEDS]
                if any(v is None for v in vals):
                    continue
                s0, s1 = std_both(vals)
                rows.append({
                    "dataset": ds, "method": m, "metric": metric,
                    "per_seed": vals, "ddof0": s0, "ddof1": s1,
                    "mean": float(np.mean(vals)),
                    "round0": f"{s0:.4f}", "round1": f"{s1:.4f}",
                    "same_when_rounded": f"{s0:.4f}" == f"{s1:.4f}",
                })
    return rows


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="配对显著性检验 + std 口径对账")
    ap.add_argument("--out-md", default="doc/seq_baselines_significance.md")
    ap.add_argument("--out-csv", default="doc/seq_baselines_significance.csv")
    ap.add_argument("--check-std-only", action="store_true",
                    help="只做 std 口径对账 + provenance 断言")
    ap.add_argument("--emit-corrected-table", action="store_true",
                    help="打印 ddof=1 口径的主表（供统一 paper_main_results.md §1）")
    ap.add_argument("--extra-methods", nargs="*", default=[],
                    help="额外纳入对比的方法（如 GRU4Rec BERT4Rec），产物缺失则跳过")
    args = ap.parse_args()

    all_methods = ["SASRec", "Content-only", "TA-SID"] + list(args.extra_methods)

    # ---------------- provenance 断言 ----------------
    print("=" * 70)
    print("provenance 断言：解析值 vs doc/paper_main_results.md §3（已勘误版）")
    print("=" * 70)
    bad = []
    cache = {}
    for (ds, m), exp in PUBLISHED_NDCG20.items():
        data = collect(ds, m)
        if not data:
            bad.append(f"{ds}/{m}: 产物缺失（路径见脚本 _t5_log / SEQ_SUMMARY_DIRS）")
            continue
        cache[(ds, m)] = data
        got = [data[s]["N20"] for s in SEEDS]
        ok = all(abs(g - e) <= 5e-5 for g, e in zip(got, exp))
        flag = "✅" if ok else "❌"
        print(f"  {flag} {ds:7s} {m:13s} 期望 {exp}  实得 {[round(g, 4) for g in got]}")
        if not ok:
            bad.append(f"{ds}/{m}: 期望 {exp} 实得 {[round(g, 4) for g in got]}")
    if bad:
        print("\n❌ provenance 断言失败：")
        for b in bad:
            print(f"   - {b}")
        print("\n（Toys TA-SID 若给出 0.0367，说明取到了孤儿 codes 日志；"
              "健康源应为 tiger_ta_sid_768d_toys_rq2024_seed*，N@20=0.0377）")
        sys.exit(2)
    print("\n✅ provenance 断言全部通过（9/9）")

    # ---------------- A. std 口径对账 ----------------
    std_rows = std_reconciliation(all_methods)
    print("\n" + "=" * 70)
    print("A. std 口径对账（ddof=0 = 总体标准差 / ddof=1 = 样本标准差）")
    print("=" * 70)
    print(f"  {'数据集':8s} {'方法':13s} {'指标':4s} {'ddof=0':>9s} {'ddof=1':>9s} {'四位小数相同?':>12s}")
    n_diff_rounded = 0
    for r in std_rows:
        if not r["same_when_rounded"]:
            n_diff_rounded += 1
        mark = "" if r["same_when_rounded"] else "  ← 并列展示会自相矛盾"
        print(f"  {r['dataset']:8s} {r['method']:13s} {r['metric']:4s} "
              f"{r['ddof0']:9.6f} {r['ddof1']:9.6f} {str(r['same_when_rounded']):>12s}{mark}")
    print(f"\n  四位小数下即不同的格子：{n_diff_rounded} / {len(std_rows)}")

    if args.emit_corrected_table:
        print("\n" + "=" * 70)
        print("C. 统一口径后的主表（ddof=1，样本标准差）")
        print("=" * 70)
        print("| Method | Dataset | NDCG@5 | NDCG@10 | NDCG@20 | Recall@5 | Recall@10 | Recall@20 |")
        print("|:--|:--|--:|--:|--:|--:|--:|--:|")
        for ln in emit_corrected_table(["SASRec", "Content-only", "TA-SID"]):
            print(ln)
        return 0

    if not args.check_std_only:
        # ---------------- B. 配对检验 ----------------
        pairs = [("TA-SID", "SASRec"), ("TA-SID", "Content-only")]
        for m in args.extra_methods:
            if collect("Sport", m):
                pairs += [("TA-SID", m), ("Content-only", m)]

        results = []
        print("\n" + "=" * 70)
        print("B. 配对显著性检验（n=3, df=2）")
        print("=" * 70)
        for ds in ["Sport", "Toys", "Beauty"]:
            for a, b in pairs:
                da, db = collect(ds, a), collect(ds, b)
                if not da or not db:
                    continue
                va = np.array([da[s]["N20"] for s in SEEDS])
                vb = np.array([db[s]["N20"] for s in SEEDS])
                diff = va - vb
                t, p_t, df = paired_t(diff)
                w_stat, p_w = wilcoxon(diff)
                p_sgn = sign_test(diff)
                dz = cohens_dz(diff)
                cross = int((diff < 0).sum())     # A 落后于 B 的 seed 数
                sa0, sa1 = std_both(va)
                sb0, sb1 = std_both(vb)
                pool1 = math.sqrt((sa1 ** 2 + sb1 ** 2) / 2)
                ratio = float(diff.mean() / pool1) if pool1 > 0 else float("inf")
                results.append({
                    "dataset": ds, "a": a, "b": b,
                    "a_seeds": va.tolist(), "b_seeds": vb.tolist(), "diff": diff.tolist(),
                    "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                    "mean_diff": float(diff.mean()),
                    "std_diff_ddof1": float(diff.std(ddof=1)),
                    "t": t, "p_t": p_t, "df": df,
                    "p_wilcoxon": p_w, "p_sign": p_sgn, "dz": dz,
                    "crossover": cross, "gap_over_pooled_std": ratio,
                })
                print(f"  {ds:7s} {a:>12s} − {b:<13s} "
                      f"Δ={diff.mean():+.4f}  t={t:6.2f} p={p_t:.3f} "
                      f"wilcoxon_p={p_w if p_w is None else round(p_w, 3)} "
                      f"sign_p={p_sgn:.3f} d_z={dz:6.2f} 越界={cross}/3")
                if cross > 0:
                    print(f"      ⚠ OVERTAKE: {a} 在 {cross} 个 seed 上低于 {b}")

        # ---------------- 输出 CSV ----------------
        csv_path = os.path.join(ROOT, args.out_csv)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "method_a", "method_b",
                        "seed2025_a", "seed2026_a", "seed2027_a",
                        "seed2025_b", "seed2026_b", "seed2027_b",
                        "mean_a", "mean_b", "mean_diff", "std_diff_ddof1",
                        "t", "df", "p_paired_t", "p_wilcoxon", "p_sign_test",
                        "cohens_dz", "crossover_of_3", "gap_over_pooled_std"])
            for r in results:
                w.writerow([r["dataset"], r["a"], r["b"],
                            *[f"{v:.6f}" for v in r["a_seeds"]],
                            *[f"{v:.6f}" for v in r["b_seeds"]],
                            f"{r['mean_a']:.6f}", f"{r['mean_b']:.6f}",
                            f"{r['mean_diff']:.6f}", f"{r['std_diff_ddof1']:.6f}",
                            f"{r['t']:.4f}", r["df"], f"{r['p_t']:.6f}",
                            "" if r["p_wilcoxon"] is None else f"{r['p_wilcoxon']:.6f}",
                            f"{r['p_sign']:.6f}", f"{r['dz']:.4f}",
                            r["crossover"], f"{r['gap_over_pooled_std']:.3f}"])
        print(f"\n  CSV → {os.path.relpath(csv_path, ROOT)}")

        write_md(os.path.join(ROOT, args.out_md), std_rows, results, all_methods)
        print(f"  MD  → {args.out_md}")

    return 0


def write_md(path, std_rows, results, methods):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# 配对显著性检验与 std 口径对账\n\n")
        f.write("> 自动生成：`python3 tools/paired_significance.py`（勿手改）\n")
        f.write("> 逐 seed N@20 全部经 provenance 断言核对（`doc/paper_main_results.md` §3）。\n\n")

        f.write("## 0. 结论速览\n\n")
        f.write("- 逐 seed 数据与已发布值**完全一致**（9/9 provenance 断言通过）。\n")
        f.write("- 稳健性主张基于 **seed 一致性与效应量**，而非 p 值（理由见 §4）。\n\n")

        f.write("## 1. std 口径对账（必读）\n\n")
        f.write("仓库内同一量存在两套标准差口径，**并列展示会自相矛盾**：\n\n")
        f.write("| 出处 | 口径 |\n|:--|:--|\n")
        f.write("| `sasrec_output/*/summary_3seed.txt`（由 `np.std` 默认值写出） | **ddof=0**（总体标准差） |\n")
        f.write("| `doc/ablation_main_table_3datasets.md`、图 A、`doc/paper_main_results.md` §1/§3 | **ddof=1**（样本标准差） |\n\n")
        f.write("> **注**：`doc/paper_main_results.md` §3 的 std 列（如 Sport SASRec `0.0002`）"
                "实为 ddof=0 的舍入结果（ddof=1 会得 `0.0003`），"
                "而图 A 中同一格记为 `0.0007`/`0.0009` 等为 ddof=1。**两者需统一。**\n\n")
        f.write("| 数据集 | 方法 | 指标 | 逐 seed | mean | std(ddof=0) | std(ddof=1) | 四位小数一致? |\n")
        f.write("|:--|:--|:--|:--|--:|--:|--:|:--:|\n")
        for r in std_rows:
            f.write(f"| {r['dataset']} | {r['method']} | {r['metric']} | "
                    f"{'/'.join(f'{v:.4f}' for v in r['per_seed'])} | {r['mean']:.4f} | "
                    f"{r['ddof0']:.6f} | {r['ddof1']:.6f} | "
                    f"{'✅' if r['same_when_rounded'] else '❌'} |\n")
        ndiff = sum(1 for r in std_rows if not r["same_when_rounded"])
        f.write(f"\n四位小数下即不同的格子：**{ndiff} / {len(std_rows)}**。\n\n")
        f.write("**处置建议**：论文正文/图表统一采 **ddof=1**（样本标准差，与消融表、图 A 现有口径一致），"
                "逐 seed 产物文件保持 ddof=0 不动（逐位 diff 门禁要求），"
                "在 §5.1 实验设置加一句 \"standard deviation is computed with Bessel's correction (n−1)\"。\n\n")

        if results:
            f.write("## 2. 配对显著性检验（n=3, df=2）\n\n")
            f.write("| 数据集 | 对比 | Δ (mean) | t | p (paired t) | p (Wilcoxon) | p (符号检验) | Cohen's d_z | 越界 seed |\n")
            f.write("|:--|:--|--:|--:|--:|--:|--:|--:|:--:|\n")
            for r in results:
                pw = "—" if r["p_wilcoxon"] is None else f"{r['p_wilcoxon']:.3f}"
                f.write(f"| {r['dataset']} | {r['a']} − {r['b']} | {r['mean_diff']:+.4f} | "
                        f"{r['t']:.2f} | {r['p_t']:.3f} | {pw} | {r['p_sign']:.3f} | "
                        f"{r['dz']:.2f} | {r['crossover']}/3 |\n")

            f.write("\n### 2.1 逐 seed 明细\n\n")
            f.write("| 数据集 | 对比 | seed2025 | seed2026 | seed2027 | 效应量(Δ / 合并std) |\n")
            f.write("|:--|:--|--:|--:|--:|--:|\n")
            for r in results:
                f.write(f"| {r['dataset']} | {r['a']} − {r['b']} | "
                        f"{r['diff'][0]:+.4f} | {r['diff'][1]:+.4f} | {r['diff'][2]:+.4f} | "
                        f"{r['gap_over_pooled_std']:.2f}× |\n")

            f.write("\n## 3. 写作口径（可直接进论文）\n\n")
            zero_cross = [r for r in results if r["crossover"] == 0]
            if zero_cross:
                f.write("可用表述：\n\n")
                for r in zero_cross:
                    f.write(f"- **{r['dataset']}: {r['a']} vs {r['b']}** — "
                            f"gap = {r['mean_diff']:+.4f} NDCG@20，"
                            f"是方法间合并 seed std 的 {r['gap_over_pooled_std']:.1f} 倍，"
                            f"3/3 seed 同向，零交叉。\n")
            cross = [r for r in results if r["crossover"] > 0]
            if cross:
                f.write("\n**⚠ 存在越界，禁止写 \"all seeds\" 或 \"consistently\"**：\n\n")
                for r in cross:
                    f.write(f"- {r['dataset']}: {r['a']} 在 {r['crossover']}/3 seed 上低于 "
                            f"{r['b']}（逐 seed Δ = "
                            f"{', '.join(f'{d:+.4f}' for d in r['diff'])}）"
                            f"→ 改写为 \"comparable on {r['dataset']}\"，不得声称全面占优。\n")

            f.write("\n## 4. n=3 的措辞红线（必须遵守）\n\n")
            f.write("### 4.1 事实澄清（先纠正一个常见误判）\n\n")
            f.write("n=3 下 p 值**并非不可达小**：df=2 时 t=74 对应 p≈1.8e-4。"
                    "所以问题不是 \"算不出显著\"，而是 **p 值依赖无法检验的正态性假设**——"
                    "n=3 时既不能检验正态性，也无法抵抗单个 seed 的偏移。"
                    "**非参数**符号检验的精确最小双侧 p 恒为 2·(1/2)³ = **0.25**，"
                    "即任何纯非参数主张在 n=3 下都不可能低于 0.25。\n\n")
            f.write("**禁止**：\n\n")
            f.write("- 写 \"statistically significant (p < 0.05)\" 而**不交代** n=3 与 df=2。\n")
            f.write("- 报 p 值但不写 df、或不同时报逐 seed 原值。\n")
            f.write("- 只报参数检验、不提非参数下界 0.25（会被审稿人视为选择性汇报）。\n\n")
            f.write("**允许且推荐**（首选第一项）：\n\n")
            f.write("- \"consistent across all 3 seeds (no crossover)\" + 效应量"
                    "（\"gap = X, which is Y× the pooled between-seed std\"）。\n")
            f.write("- 若要报 p，必须连 df 与逐 seed 值一起报，并注明假设："
                    "*\"paired t-test with df = 2 (p = …), which assumes normality that cannot be "
                    "verified at n = 3; the corresponding non-parametric sign test floors at p = 0.25.\"*\n")
            f.write("- 显式免责一句：*\"With n = 3 seeds we do not claim statistical significance; "
                    "we report per-seed consistency and effect size instead, as the sign test's "
                    "minimum attainable two-sided p-value at n = 3 is 0.25.\"*\n\n")

        f.write("## 5. 复现\n\n```bash\n")
        f.write("python3 tools/paired_significance.py \\\n"
                "    --out-md doc/seq_baselines_significance.md \\\n"
                "    --out-csv doc/seq_baselines_significance.csv\n```\n")


if __name__ == "__main__":
    sys.exit(main())
