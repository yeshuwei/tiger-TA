#!/usr/bin/env python3
"""
e49_summary.py — E49 行为消融结果汇总（**命名陷阱安全版**）

背景：`rqvae/run_e49_p3_p6.sh` 的 P6 段用「tag 拼接 + glob」定位 eval 日志，对两个变体失效：

  1. **Toys TA-SID**：P6 找 `Toys_ta_sid_768d_seed*_eval.log`，那是**孤儿 codes（L0=62）**的日志，
     NDCG@20 = 0.0367 **已作废**（见 doc/e48_toys_main_codes_provenance_fix.md）。
     权威值 0.0377 在 `tiger_ta_sid_768d_toys_rq2024_seed*_eval.log`（健康 codes，L0=207）。
  2. **Toys Shuffled**：Toys 无 legacy codes，故该变体实际 tag 是 `shuffled_768d`，
     日志名为 `Toys_shuffled_768d_seed*_eval.log`；P6 却找 `Toys_shuffled_ablation_seed*_eval.log`
     （Sport 走 legacy codes 才是这个名字）→ P6 该行会输出为空。

本脚本**逐 lane 显式列出文件名模板**，不使用 glob 拼接，从根上绕开这两类陷阱。
另注：Sport Shuffled/ZeroPad 的 seed2025 是 **legacy 单 seed 文件**（无 seed 后缀），
2026/2027 才是新跑的 `..._seed{S}_eval.log`。

输出：`doc/e49_results_corrected.md`（本脚本专属产物，不改写 P6 生成的任何文件）

用法：
    python tools/e49_summary.py [--out doc/e49_results_corrected.md]
"""

import argparse
import datetime
import os
import re
import statistics as st

SEEDS = (2025, 2026, 2027)

# ---------------------------------------------------------------- lane 定义
# 每项: (数据集, 变体, {seed: 日志文件名})；缺某 seed 即不写该键
def _std(tag):
    return {s: f"{tag}_seed{s}_eval.log" for s in SEEDS}


def _legacy_2025(tag, new_tag):
    """seed2025 用 legacy 无后缀文件名，2026/2027 用带 seed 的新名。"""
    d = {2025: f"{tag}_eval.log"}
    d.update({s: f"{new_tag}_seed{s}_eval.log" for s in (2026, 2027)})
    return d


LANES = [
    # ---------------- Sport（18/18 全齐）----------------
    ("Sport", "Content-only", _std("Sport_content768d_baseline")),
    ("Sport", "Shuffled",     _legacy_2025("Sport_shuffled_ablation", "Sport_shuffled_ablation")),
    ("Sport", "ZeroPad",      _legacy_2025("Sport_zeropad_ablation", "Sport_zeropad_ablation")),
    ("Sport", "ItemCF",       _std("Sport_itemcf_768d")),
    ("Sport", "Raw-CoOcc",    _std("Sport_rawcooc_768d")),
    ("Sport", "TA-SID",       _std("Sport_ta_sid_768d")),
    # ---------------- Toys ----------------
    ("Toys", "Content-only", _std("Toys_content768d_baseline")),
    # ↓ 陷阱 2：实际 tag 是 shuffled_768d，不是 shuffled_ablation
    ("Toys", "Shuffled",     _std("Toys_shuffled_768d")),
    ("Toys", "ZeroPad",      _std("Toys_zeropad_ablation")),
    ("Toys", "ItemCF",       _std("Toys_itemcf_768d")),
    ("Toys", "Raw-CoOcc",    _std("Toys_rawcooc_768d")),
    # ↓ 陷阱 1：权威值来自健康 codes 的 rq2024 重跑，非孤儿 codes 的 Toys_ta_sid_768d_seed*
    ("Toys", "TA-SID",       {s: f"tiger_ta_sid_768d_toys_rq2024_seed{s}_eval.log" for s in SEEDS}),
    # ---------------- 辅助行（单 seed，不计入 36）----------------
    ("Sport", "Random",      {2025: "Sport_random_ablation_eval.log"}),
]

# 已知被取代的历史单 seed 值（仅记录，不参与均值）
SUPERSEDED = {
    ("Sport", "Shuffled"): "旧单 seed 2025 = 0.0262，现已被 3-seed 均值取代",
    ("Sport", "ZeroPad"):  "旧单 seed 2025 = 0.0286，现已被 3-seed 均值取代",
    ("Toys", "TA-SID"):    "孤儿 codes(L0=62) 0.0367 已作废（e48），本行用健康 codes(207) 0.0377",
}

LOG_DIR = "model/logs"

# ⚠️ 项目内两种日志格式并存，**都必须支持**（否则会误报"未跑"）：
#   旧日志（Content-only / TA-SID / legacy Shuffled·ZeroPad / Random）：
#       "  NDCG            0.0160     0.0211     0.0274"          → 3 列 @5,@10,@20
#   E49 新日志（--topk_list 5,10,20,50,100）：
#       "  NDCG            0.0186     0.0242     0.0311     0.0347     0.0347"  → 5 列
#   两种格式的 **NDCG@20 都是第 3 个值**，故只匹配前 3 列即可同时覆盖。
NDCG_RE = re.compile(r"^\s*NDCG\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M)
RCL_RE = re.compile(r"^\s*Recall\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M)


def parse(fname):
    """读 eval 日志，返回 (NDCG@20, Recall@20)；取最后一次出现的结果表。"""
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    n = NDCG_RE.findall(txt)
    r = RCL_RE.findall(txt)
    if not n:
        return None  # 评测尚未写出结果表（tee 已建文件但表在结束时才写）
    ndcg20 = float(n[-1][2])
    rcl20 = float(r[-1][2]) if r else float("nan")
    return ndcg20, rcl20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="doc/e49_results_corrected.md")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows, n_done, n_total = [], 0, 0
    for ds, variant, files in LANES:
        aux = (variant == "Random")
        vals, seeds_used = {}, []
        for s in SEEDS:
            r = parse(files[s]) if s in files else None
            if r:
                vals[s] = r
                seeds_used.append(s)
        if not aux:
            n_total += 3
            n_done += len(vals)
        nd = [v[0] for v in vals.values()]
        rc = [v[1] for v in vals.values() if v[1] == v[1]]
        mean = st.mean(nd) if nd else None
        std = st.stdev(nd) if len(nd) > 1 else None
        rows.append(dict(ds=ds, variant=variant, aux=aux, vals=vals, seeds=seeds_used,
                         mean=mean, std=std, rmean=st.mean(rc) if rc else None,
                         files=files))

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = [f"# E49 行为消融结果（命名陷阱安全版，自动生成）", "",
           f"**生成时间**：{ts}　**产出脚本**：`tools/e49_summary.py`　**调度**：`rqvae/e49_finalize.sh`（crontab）", "",
           "> 本表由**逐 lane 显式文件名**解析，**不使用 tag glob**，故不受 P6 段两处命名陷阱影响：",
           "> ① Toys TA-SID 取**健康 codes**（`tiger_ta_sid_768d_toys_rq2024_seed*`，L0=207）而非作废的孤儿 codes；",
           "> ② Toys Shuffled 取实际 tag `shuffled_768d` 的日志而非 `shuffled_ablation`。",
           "> ⚠️ `doc/e49_behavior_ablation_results.md`（P6 段产物）**未经此修正**，其 Toys TA-SID / Toys Shuffled",
           "> 两行应作废，以本表为准。见 `doc/e48_toys_main_codes_provenance_fix.md`。", "",
           f"**进度**：{n_done}/{n_total} 路（Random 辅助行单 seed，不计入）。", "",
           "## 1. NDCG@20（3-seed mean±std）", "",
           "| 数据集 | 变体 | 2025 | 2026 | 2027 | 均值±std | 数据源 |",
           "|:--|:--|:--:|:--:|:--:|:--:|:--|"]

    def cell(r, s):
        return f"{r['vals'][s][0]:.4f}" if s in r["vals"] else "—"

    for r in rows:
        if r["aux"]:
            continue
        if r["mean"] is None:
            ms, src = "∅ 未跑", "—"
        elif r["std"] is None:
            ms = f"{r['mean']:.4f}（单 seed）"
            src = r["files"][r["seeds"][0]]
        else:
            ms = f"**{r['mean']:.4f}±{r['std']:.4f}**"
            src = "rq2024 健康 codes" if r["ds"] == "Toys" and r["variant"] == "TA-SID" else "标准"
        out.append(f"| {r['ds']} | {r['variant']} | {cell(r,2025)} | {cell(r,2026)} | {cell(r,2027)} | {ms} | {src} |")

    out += ["", "## 2. Recall@20（3-seed mean）", "",
            "| 数据集 | 变体 | 均值 |", "|:--|:--|:--:|"]
    for r in rows:
        if r["aux"] or r["rmean"] is None:
            continue
        out.append(f"| {r['ds']} | {r['variant']} | {r['rmean']:.4f} |")

    out += ["", "## 3. 辅助行（单 seed，不参与显著性比较）", "",
            "| 数据集 | 变体 | NDCG@20 |", "|:--|:--|:--:|"]
    for r in rows:
        if not r["aux"]:
            continue
        v = f"{r['mean']:.4f}" if r["mean"] is not None else "∅ 未跑"
        out.append(f"| {r['ds']} | {r['variant']} | {v} |")

    note = [x for k, x in SUPERSEDED.items() if any(r["ds"] == k[0] and r["variant"] == k[1] for r in rows)]
    if note:
        out += ["", "## 4. 口径说明（历史/作废值）", ""] + [f"- {x}" for x in note]

    missing = [f"{r['ds']} {r['variant']}（缺 seed {sorted(set(SEEDS)-set(r['seeds']))}）"
               for r in rows if not r["aux"] and len(r["seeds"]) < 3]
    out += ["", "## 5. 未完成 lane", ""]
    out += [f"- {x}" for x in missing] if missing else ["- 无，全部 3-seed 齐备。"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(out) + "\n")

    if not args.quiet:
        print("\n".join(out))
    print(f"\n[写出] {args.out}")
    print(f"[进度] {n_done}/{n_total} 路")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
