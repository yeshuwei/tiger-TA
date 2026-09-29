#!/usr/bin/env python3
"""multiseed_delta_auc_check.py — ΔAUC 的多种子自查（替代路线 (b)，2026-09-27）

要回答的质疑
------------
`doc/stage_gain_decomposition_conclusion.md` 的核心数字是 ΔAUC = TA-SID − Content-only
（Toys 主口径 +0.09）。但口径 (a) 下**非 TA-SID 变体被冻结在 seed2024**，于是：

    ΔAUC(seed s) = TA-SID(seed s) − Content-only(seed 2024)

**基线那一项不随 s 变**。若 Content-only 在 seed 2024 上恰好偏低，则 +0.09 可能是
"基线抽签"而不是真实增益。**路线 (b)**（补训 Sport/Beauty 的 content-only + 4 变体 × seed2025/2026，
约 13–20 次 RQ-VAE 训练）就是为堵这个洞而提的，代价是 GPU 时间。

本脚本的替代做法
----------------
Toys 恰好**现成就有** Content-only 的 seed2024/2025/2026 三份产物（+ ZeroPad 两份、
TA-SID 三份）。于是可以直接测：

  1. **基线自身的种子方差**：Content-only 在三个种子上的 AUC 摆幅有多大？
     —— 若摆幅远小于 +0.09，则"基线抽签"解释不了增益。
  2. **逐种子配对 ΔAUC**：ΔAUC(s) = TA-SID(s) − Content-only(s)，**同一 s 配对**，
     消掉基线抽签。看三个 s 上的 ΔAUC 是否稳定同号同量级。
  3. **混算 vs 配对**：把冻结口径（TA-SID(s) − Content-only(2024)）与配对口径并排列出，
     两者之差 = 冻结引入的偏差。若偏差 ≪ ΔAUC，则口径 (a) 无害，路线 (b) 不必跑。

测量空间
--------
- **A 输入**：fused 文件，**与种子无关**（同一份 parquet）→ ΔAUC 在此空间恒为定值，是锚点。
- **B = z**（encoder 输出，需前向）：随种子变。
- **C = xq[full]**（3 层码本重建，纯 CPU）：随种子变。

口径与 `stage_probe.py` 逐位一致（同 `build_samples` 路径、同 sample_seed）。

用法（仓库根，CPU；`--spaces C` 可完全避开前向）：
    python3 tools/multiseed_delta_auc_check.py
    python3 tools/multiseed_delta_auc_check.py --datasets Toys --spaces C
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "rqvae"))

from behavior_structure_probe import (  # noqa: E402
    DATA_DIR, N_POS, N_NEG, SEED, cos_sim, auc_of,
)
from sid_per_layer_diagnostic import (  # noqa: E402
    rebuild_xq_partial, normalize_rows, encode_z,
)
from sid_consistency_contrast import (  # noqa: E402
    find_best_ckpt, load_codebook_weights,
)
from stage_probe import FUSED_896D, VARIANTS, resolve  # noqa: E402

NEG_COLS = {"degmatch": "auc_cos_degmatch", "uniform": "auc_cos_uniform"}
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def load_fused_896(path, n_items):
    df = pd.read_parquet(path)
    emb = np.stack(df["embedding"].values).astype(np.float32)
    if emb.shape[0] != n_items:
        raise ValueError(f"{os.path.basename(path)}: n={emb.shape[0]} != {n_items}")
    return normalize_rows(torch.from_numpy(emb))


def space_A(ds, data_dir, variant, n_items):
    """A 空间 = fused 输入文件，**与 RQ-VAE 种子无关**。"""
    return load_fused_896(os.path.join(data_dir, ds, FUSED_896D[variant]), n_items)


def space_C(ds, data_dir, codef, ckpt):
    codes = np.load(os.path.join(data_dir, ds, codef))
    return normalize_rows(rebuild_xq_partial(codes, load_codebook_weights(ckpt), 3))


def space_B(ckpt):
    zpath = getattr(torch.load(ckpt, map_location="cpu", weights_only=False)["args"],
                    "data_path", None)
    if zpath and not os.path.isabs(zpath):
        zpath = os.path.normpath(os.path.join(ROOT, zpath))
    if not zpath or not os.path.exists(zpath):
        return None
    return encode_z(ckpt, zpath)


def main():
    ap = argparse.ArgumentParser(description="ΔAUC 的多种子自查")
    ap.add_argument("--datasets", nargs="+", default=["Toys"])
    ap.add_argument("--variants", nargs="+", default=["Content-only", "ZeroPad", "TA-SID"],
                    help="要枚举种子的变体；TA-SID 为增益端，其余为基线端")
    ap.add_argument("--baseline", type=str, default="Content-only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--spaces", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--pos", type=str, default="topk", choices=["topk", "all"])
    ap.add_argument("--sample_seed", type=int, default=SEED)
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--csv", type=str, default="doc/multiseed_delta_auc_check.csv")
    ap.add_argument("--log", type=str, default="doc/multiseed_delta_auc_check.log")
    args = ap.parse_args()

    log("=" * 100)
    log("ΔAUC 的多种子自查 —— 「冻结基线抽签」能否解释增益？")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"正样本口径 = {args.pos} × degmatch；采样 seed = {args.sample_seed}（固定）")
    log("空间 A = fused 输入（**与种子无关**）| B = z | C = xq[full]（3 层重建）")
    log("=" * 100)

    all_rows = []

    for ds in args.datasets:
        n_items, n_edges, negcg = build_samples_wrap(ds, args)
        pos_ok, neg_deg, neg_unif = negcg[args.pos]
        negs = {"degmatch": neg_deg, "uniform": neg_unif}
        log(f"\n{'#' * 100}\n# {ds}   E 图 {n_items} items / {n_edges} 边　"
            f"正样本 {len(pos_ok)}　负样本 {len(neg_deg)}\n{'#' * 100}")

        # ---- 逐 (变体, 种子) 取 AUC ----
        auc = {}          # (variant, seed, space, negname) -> auc
        avail = {}        # (variant, seed) -> ckpt tag / "frozen"
        for v in args.variants:
            for s in args.seeds:
                tag, codef, frozen = resolve(ds, v, s)
                if tag is None:
                    continue
                ck = find_best_ckpt(tag)
                if ck is None or codef is None or not os.path.exists(
                        os.path.join(args.data_dir, ds, codef)):
                    log(f"  ⚠ {v} seed{s}: 资源缺失（tag={tag} codes={codef}）")
                    continue
                avail[(v, s)] = (tag, frozen)
                embs = {}
                if "A" in args.spaces:
                    embs["A"] = space_A(ds, args.data_dir, v, n_items)
                if "C" in args.spaces:
                    embs["C"] = space_C(ds, args.data_dir, codef, ck)
                if "B" in args.spaces:
                    embs["B"] = space_B(ck)
                for sp, emb in embs.items():
                    if emb is None:
                        log(f"  ⚠ {v} seed{s}: {sp} 空间缺失")
                        continue
                    for nn, neg in negs.items():
                        auc[(v, s, sp, nn)] = auc_of(cos_sim(emb, pos_ok), cos_sim(emb, neg))

        # ---- 表 1：各变体 × 种子 的绝对 AUC ----
        for nn in ("degmatch", "uniform"):
            log(f"\n▌ 表 1　绝对 AUC（负样本 = {nn}）")
            log(f"  {'变体':<14}{'种子':>7} " + "".join(f"{sp:>12s}" for sp in args.spaces)
                + "   来源")
            for v in args.variants:
                for s in args.seeds:
                    if (v, s) not in avail:
                        continue
                    tag, frozen = avail[(v, s)]
                    cells = "".join(
                        f"{auc.get((v, s, sp, nn), float('nan')):>12.4f}" for sp in args.spaces)
                    src = "冻结→2024" if frozen and s != 2024 else tag
                    log(f"  {v:<14}{s:>7} {cells}   {src}")

        # ---- 表 2：基线自身的种子摆幅 ----
        base_tags = {s: avail[(args.baseline, s)][0] for s in args.seeds
                     if (args.baseline, s) in avail}
        base_frozen = len(set(base_tags.values())) == 1 and len(base_tags) > 1
        log(f"\n▌ 表 2　基线（{args.baseline}）自身的**种子摆幅**")
        if base_frozen:
            log(f"  ⚠ 本数据集基线无多种子产物：三个种子解析到**同一个 ckpt**（{list(base_tags.values())[0]}）。")
            log(f"    故下表「摆幅 0.0000」是**定义**（同一份产物被重复读取），**不是**测量到的种子稳健性。")
        for nn in ("degmatch", "uniform"):
            log(f"  {nn}：")
            for sp in args.spaces:
                vals = [auc[(args.baseline, s, sp, nn)] for s in args.seeds
                        if (args.baseline, s, sp, nn) in auc]
                same = len({round(x, 10) for x in vals}) == 1
                if len(vals) < 2:
                    log(f"    {sp}: 种子不足（{len(vals)} 个）")
                    continue
                if sp == "A" and same:
                    note = "　（A 空间与种子无关，按设计恒定）"
                elif same and base_frozen:
                    note = "　（冻结：同一 ckpt 被重复读取）"
                else:
                    note = ""
                log(f"    {sp}: " + " / ".join(f"{x:.4f}" for x in vals)
                    + f"　摆幅 {max(vals)-min(vals):.4f}{note}")

        # ---- 表 2b：各变体自身的种子摆幅（只统计真实不同的 ckpt） ----
        log(f"\n▌ 表 2b　各变体**自身**的种子摆幅（只统计解析到**不同 ckpt**的种子）")
        log(f"  用途：无多种子基线的数据集（Sport/Beauty）只能靠这个量来估计「RQ-VAE 训练方差」的量级。")
        log(f"  {'变体':<14}{'空间':>6}{'n种子':>7}   值　→　摆幅")
        spread_proxy = {}
        for v in args.variants:
            for sp in args.spaces:
                seen = {}
                for s in args.seeds:
                    if (v, s) not in avail:
                        continue
                    tag = avail[(v, s)][0]
                    if tag in seen:          # 冻结重复，剔除
                        continue
                    a = auc.get((v, s, sp, "degmatch"))
                    if a is not None:
                        seen[tag] = a
                if len(seen) < 2:
                    continue
                vals = list(seen.values())
                spread_proxy[(v, sp)] = max(vals) - min(vals)
                log(f"  {v:<14}{sp:>6}{len(vals):>7}   "
                    + " / ".join(f"{x:.4f}" for x in vals)
                    + f"　→　**{max(vals)-min(vals):.4f}**")

        # ---- 表 3：逐种子配对 ΔAUC vs 冻结混算 ----
        base_tags_v = {s: avail[(args.baseline, s)][0] for s in args.seeds
                       if (args.baseline, s) in avail}
        base_seeds = [s for s in args.seeds if (args.baseline, s) in avail]
        if len(set(base_tags_v.values())) < 2:
            log(f"\n▌ 表 3　**跳过**：基线在本数据集没有多种子产物"
                f"（三个种子解析到同一 ckpt {list(base_tags_v.values())[0]}）→ "
                f"配对与冻结恒等，配对对照在此数据集**不可做**。")
            log(f"  ⇒ 该数据集的「冻结基线抽签」问题**只能靠跨数据集外推**：见 Toys 的结论 + 表 2b 的训练方差代理。")
            continue
        log(f"\n▌ 表 3　ΔAUC = TA-SID − {args.baseline}")
        log(f"  配对口径：ΔAUC(s) = TA-SID(s) − {args.baseline}(s)　← 消掉基线抽签")
        log(f"  冻结口径：ΔAUC(s) = TA-SID(s) − {args.baseline}(2024)　← 现行口径 (a)，基线固定")
        for nn in ("degmatch", "uniform"):
            for sp in args.spaces:
                if sp == "A":
                    log(f"\n  {nn} / {sp}：A 输入与种子无关 → 配对与冻结必然相同（锚点）")
                else:
                    log(f"\n  {nn} / {sp}：")
                paired, frozen_vals = [], []
                for s in args.seeds:
                    if ("TA-SID", s) not in avail:
                        continue
                    ta = auc.get(("TA-SID", s, sp, nn))
                    bs = auc.get((args.baseline, s, sp, nn))
                    b0 = auc.get((args.baseline, base_seeds[0], sp, nn))
                    if ta is None or bs is None or b0 is None:
                        continue
                    paired.append(ta - bs)
                    frozen_vals.append(ta - b0)
                    log(f"    seed {s}:  TA-SID {ta:.4f} − 基线({s}) {bs:.4f} = **{ta-bs:+.4f}**"
                        f"　|　冻结: − 基线({base_seeds[0]}) {b0:.4f} = {ta-b0:+.4f}"
                        f"　(差 {ta-b0-(ta-bs):+.4f})")
                    all_rows.append({"dataset": ds, "space": sp, "neg": nn, "seed": s,
                                     "ta_auc": ta, "baseline_auc": bs,
                                     "baseline_seed_used": base_seeds[0],
                                     "delta_paired": ta - bs, "delta_frozen": ta - b0,
                                     "method": "paired" if s == base_seeds[0] else "mixed"})
                if len(paired) >= 2:
                    log(f"    ⇒ 配对 ΔAUC: " + " / ".join(f"{x:+.4f}" for x in paired)
                        + f"　均值 {np.mean(paired):+.4f}　摆幅 {max(paired)-min(paired):.4f}")
                    log(f"       全部同号: {all(x > 0 for x in paired) or all(x < 0 for x in paired)}"
                        f"　| 最小 |ΔAUC| = {min(abs(x) for x in paired):.4f}")

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log(f"\n→ 日志: {args.log}")
    if args.csv and all_rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        log(f"→ CSV: {args.csv}  ({len(all_rows)} 行)")
    return 0


def build_samples_wrap(ds, args):
    from stage_probe import build_samples
    return build_samples(ds, args.data_dir, args.window_size,
                         args.n_pos, args.n_neg, args.sample_seed)


if __name__ == "__main__":
    raise SystemExit(main())
