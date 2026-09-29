#!/usr/bin/env python3
"""codebook_budget_control.py — 受限码本对照（note 13 §2.1 的硬要求，2026-09-27）

note 13 §2.1 对 `ρ(z, xq) = 1.0` 的补救是**双条**：
    ① 把结论从"量化保序"升级并限定为"**在有限码本预算下量化不破坏行为判别力**"；
    ② **必须给出受限码本对照**（减码本大小 / 减 RQ 层数），证明代价会**随之上升**。
否则该条不能作为贡献（"用秩相关度量量化前后的一致性，接近 1.0 是 RQ 的定义性质"）。

`doc/stage_gain_decomposition_conclusion.md` §4 完成了 ①（给出"码本阶段对增益的代价 ≤7%"这个数），
**本脚本完成 ②**：在**不重训**的前提下，用 `rebuild_xq_partial(codes, cb, n_layers)`
构造 1 层 / 2 层 / 3 层 的 xq，得到"码本预算 ↓ → 行为判别力/增益 ↓"的曲线。

设计要点
--------
- **不需要 GPU，不需要重训**：只截断码本层数（1 层 = 仅 256 个码，2 层 = 256²，3 层 = 256³）。
- 同时报告三样东西，因为"代价"可以从任一处体现：
    (a) 变体的绝对 Behavior AUC（topk × degmatch）
    (b) 增益 ΔAUC = TA-SID − Content-only（分段报告里那个"增益"）
    (c) ρ(A 输入, xq[n_layers])  ← note 13 §2.1 要求的那个"ρ 是否随之下降"
- 口径与 `behavior_structure_probe` / `stage_probe` 逐位一致（同采样路径、同 seed）。

用法（仓库根，CPU）：
    python3 tools/codebook_budget_control.py
    python3 tools/codebook_budget_control.py --datasets Sport Toys
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "rqvae"))

from behavior_structure_probe import (  # noqa: E402
    DATA_DIR, N_POS, N_NEG, SEED, cos_sim, auc_of,
)
from sid_per_layer_diagnostic import rebuild_xq_partial, normalize_rows  # noqa: E402
from sid_consistency_contrast import load_codebook_weights  # noqa: E402
from stage_probe import build_samples, resolve  # noqa: E402

VARIANT_ORDER = ["Content-only", "ZeroPad", "Shuffled", "ItemCF", "Raw-CoOcc", "TA-SID"]
LAYERS = [1, 2, 3]
CKPT_DIR = os.path.join(ROOT, "rqvae", "ckpt")

LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def find_ckpt(tag):
    import glob
    fs = sorted(glob.glob(os.path.join(CKPT_DIR, tag, "*", "best_collision_model.pth")),
                key=os.path.getmtime)
    return fs[-1] if fs else None


def main():
    ap = argparse.ArgumentParser(description="受限码本对照")
    ap.add_argument("--datasets", nargs="+", default=["Sport", "Toys", "Beauty"])
    ap.add_argument("--rqvae_seed", type=int, default=SEED)
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--sample_seed", type=int, default=SEED)
    ap.add_argument("--a_csv", type=str, default="doc/g2_stage_probe_results.csv",
                    help="取 A 空间 AUC 用于 ρ(A, xq[L])")
    ap.add_argument("--csv", type=str, default="doc/codebook_budget_control.csv")
    ap.add_argument("--log", type=str, default="doc/codebook_budget_control.log")
    args = ap.parse_args()

    log("=" * 104)
    log("受限码本对照 —— 码本预算 ↓ 时，行为判别力与增益是否随之下降？（note 13 §2.1 硬要求）")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"方法: 截断 RQ 层数 n_layers ∈ {LAYERS} 重建 xq（不重训）；口径 = topk × degmatch")
    log("=" * 104)

    a_df = pd.read_csv(args.a_csv) if os.path.exists(args.a_csv) else None
    rows = []

    for ds in args.datasets:
        _, _, negcg = build_samples(ds, args.data_dir, args.window_size,
                                    args.n_pos, args.n_neg, args.sample_seed)
        pos_ok, neg_deg, _ = negcg["topk"]
        log(f"\n▌ {ds}   正样本 {len(pos_ok)} | 负样本(degmatch) {len(neg_deg)}")

        # 每个变体在每个 n_layers 下的 AUC
        auc = {}     # (variant, n_layers) -> auc
        for v in VARIANT_ORDER:
            tag, codef, _ = resolve(ds, v, args.rqvae_seed)
            ck = find_ckpt(tag)
            cp = os.path.join(args.data_dir, ds, codef)
            if ck is None or not os.path.exists(cp):
                log(f"  ⚠ {v}: 资源缺失（ckpt={tag} codes={codef}）→ 跳过")
                continue
            codes = np.load(cp)
            cb = load_codebook_weights(ck)
            for L in LAYERS:
                xq = normalize_rows(rebuild_xq_partial(codes, cb, L)).numpy()
                a = auc_of(cos_sim(xq, pos_ok), cos_sim(xq, neg_deg))
                auc[(v, L)] = a

        if not auc:
            continue
        log(f"\n  {'n_layers':<10}" + "".join(f"{v[:11]:>13s}" for v in VARIANT_ORDER)
            + f"{'ΔAUC':>9s}{'ρ(A,xq)':>10s}")
        for L in LAYERS:
            vals = [auc.get((v, L)) for v in VARIANT_ORDER]
            if any(x is None for x in vals):
                continue
            ta = auc[("TA-SID", L)]
            co = auc[("Content-only", L)]
            d = ta - co
            rho = np.nan
            if a_df is not None:
                av = (a_df[(a_df.dataset == ds) & (a_df.rqvae_seed == args.rqvae_seed) &
                           (a_df.space == "A") & (a_df.pos_regime == "topk")]
                      .set_index("variant")["auc_cos_degmatch"].reindex(VARIANT_ORDER).values)
                rho, _ = spearmanr(av, vals)
            log(f"  {L:<10}" + "".join(f"{x:>13.4f}" for x in vals)
                + f"{d:>+9.4f}{rho:>10.3f}")
            for v, x in zip(VARIANT_ORDER, vals):
                rows.append({"dataset": ds, "variant": v, "n_layers": L,
                             "L0_L1_L2_cum": L, "auc_degmatch": x,
                             "delta_auc_vs_content": d if v == "TA-SID" else "",
                             "rho_A_vs_this": rho})

        # 与 3 层的对照：保留率
        log(f"\n  相对 3 层（满预算）的保留率：")
        for L in LAYERS[:-1]:
            r3 = [auc[("TA-SID", 3)], auc[("Content-only", 3)]]
            d3 = r3[0] - r3[1]
            for v in ("TA-SID", "Content-only"):
                lo, hi = auc[(v, L)], auc[(v, 3)]
                log(f"    n_layers={L}  {v:<14} AUC {lo:.4f} vs 满预算 {hi:.4f}"
                    f"  (差 {lo-hi:+.4f})")
            dL = auc[("TA-SID", L)] - auc[("Content-only", L)]
            log(f"    n_layers={L}  增益 ΔAUC {dL:+.4f} vs 满预算 {d3:+.4f}"
                f"  → 保留 {dL/d3*100:.1f}%")

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    log(f"\n→ 日志: {args.log}")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"→ CSV: {args.csv}  ({len(rows)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
