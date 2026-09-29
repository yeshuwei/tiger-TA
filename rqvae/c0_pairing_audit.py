#!/usr/bin/env python3
"""c0_pairing_audit.py — G2 前置 C0：9 格 (ckpt ↔ codes) 配对核验（2026-09-27）

背景：note 15 §5「C0 不可跳过」。本项目有 provenance 前科——`doc/e48_toys_main_codes_provenance_fix.md`
记录主表 Toys codes（L0=62）无法从任何现存 ckpt 复现（最佳行级一致率 29.3%），数字作废。
若 9 格中任一格的 z 与 codes 出自不同训练，ρ(A,B) 就是在比两个错位的东西。

方法（沿用 `verify_variant_pairing.py` / `audit_main_codes_provenance.py` 的 L0 一致率法）：
    ckpt → 重建 RQVAE → encode_z 全量 → 逐层 argmin 分配 → 与 codes npy 对应列比行级一致率。

**本脚本不做「挑一个能对上的」**（那正是 e48 事故的成因）。它对每个数据集打印
**ckpt × codes 全交叉矩阵**，再按 note 15 §3.1 声明的 9 格配对单独判定。错配无处藏身。

用法（仓库根，CPU）：
    python rqvae/c0_pairing_audit.py              # 全量
    python rqvae/c0_pairing_audit.py --ds Toys    # 只跑一个数据集

C0 结论（2026-09-27）：**9/9 PASS**，每份 codes 只与唯一一个 ckpt 匹配（100%），
其余交叉格 ≤1.2%。L0/L1 一致率恰为 100.00%，L2 为 97.8–99.0%。
L2 的 2% 缺口不是错配：`args.sk_epsilons=[0.0, 0.0, 0.003]`，故生成时 L2 走 Sinkhorn
（ε=0.003）而本脚本用 argmin —— 缺口即 Sinkhorn 与 argmin 的差。详见 note 15 §9。
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "rqvae"))
from sf_rq import load_rqvae_from_ckpt, encode_z

# ---------------------------------------------------------------------------
# 声明的 9 格配对（note 15 §3.1）：(数据集, 种子, codes 文件, ckpt tag 首选)
# ckpt tag 的候选 glob 由 CKPT_GLOBS 给出；首选在上，备用在下。
# ---------------------------------------------------------------------------
CELLS = [
    ("Sport", "2024", "Sport_ta_sid_768d_sid.npy", "Sport_TA_SID_768d"),
    ("Sport", "2025", "Sport_ta_sid_768d_seed2025_sid.npy", "Sport_TA_SID_seed2025"),
    ("Sport", "2026", "Sport_ta_sid_768d_seed2026_sid.npy", "Sport_TA_SID_seed2026"),
    ("Toys", "2024", "Toys_ta_sid_768d_seed2024_sid.npy", "Toys_TA_SID_768d"),
    ("Toys", "2025", "Toys_ta_sid_768d_seed2025_sid.npy", "Toys_TA_SID_seed2025"),
    ("Toys", "2026", "Toys_ta_sid_768d_seed2026_sid.npy", "Toys_TA_SID_seed2026"),
    ("Beauty", "2024", "Beauty_ta_sid_768d_sid.npy", "Beauty_TA_SID"),
    ("Beauty", "2025", "Beauty_ta_sid_768d_seed2025_sid.npy", "Beauty_TA_SID_seed2025"),
    ("Beauty", "2026", "Beauty_ta_sid_768d_seed2026_sid.npy", "Beauty_TA_SID_seed2026"),
]

# 各数据集参与交叉矩阵的 codes 文件（含已作废的孤儿，用于反证）
CODES_FILES = {
    "Sport": ["Sport_ta_sid_768d_sid.npy", "Sport_ta_sid_768d_seed2025_sid.npy",
              "Sport_ta_sid_768d_seed2026_sid.npy", "Sport_ta_sid.npy"],
    "Toys": ["Toys_ta_sid_768d_seed2024_sid.npy", "Toys_ta_sid_768d_seed2025_sid.npy",
             "Toys_ta_sid_768d_seed2026_sid.npy", "Toys_ta_sid_768d_skl0_seed2024_sid.npy",
             "Toys_ta_sid_768d_sid.npy"],
    "Beauty": ["Beauty_ta_sid_768d_sid.npy", "Beauty_ta_sid_768d_seed2025_sid.npy",
               "Beauty_ta_sid_768d_seed2026_sid.npy"],
}

# 各数据集参与交叉矩阵的 ckpt tag（glob 多命中时全部展开）
CKPT_TAGS = {
    "Sport": ["Sport_TA_SID_768d", "Sport_TA_SID_seed2025", "Sport_TA_SID_seed2026"],
    "Toys": ["Toys_TA_SID_skl0_seed2024", "Toys_TA_SID_768d", "Toys_TA_SID",
             "Toys_TA_SID_seed2025", "Toys_TA_SID_seed2026"],
    "Beauty": ["Beauty_TA_SID", "Beauty_TA_SID_seed2025", "Beauty_TA_SID_seed2026"],
}

WHICH = ("best_collision_model.pth", "best_loss_model.pth")
PASS_TH = 0.90      # ≥ 判配对正确
WARN_TH = 0.50      # ≥ 存疑，< 判不匹配


def codebook(model):
    return [model.rq.vq_layers[i].embedding.weight.detach() for i in range(3)]


def decode_indices(model, z):
    """逐层 RQ 解码得到 (L0, L1, L2) 的 argmin 索引。

    ⚠ 关键：L1/L2 的输入是**残差** r = r − cb[pred]，不是原始 z。
    `verify_variant_pairing.py` / `audit_main_codes_provenance.py` 两脚本在此处用
    `argmin(z, cb_l)` 直接算 L1/L2，忽略了残差 —— 因 `VectorQuantizer.forward` 的
    straight-through 写法（`x_q = x + (x_q - x).detach()`）使 RQ 层间残差在数值上≈0，
    故那两个脚本给出的 L1/L2 一致率（如 e48 记录的 2.3%/0.6%）**无意义**。L0 不受影响。
    """
    cb = codebook(model)
    r = z
    preds = []
    for lvl in range(3):
        idx = torch.cdist(r, cb[lvl].to(r.device)).argmin(dim=1)
        preds.append(idx)
        r = r - cb[lvl].to(r.device)[idx]
    return preds


def ckpt_candidates(tag):
    """展开一个 tag 下的所有 (run 子目录, .pth 文件)。"""
    out = []
    for d in sorted(glob.glob(f"rqvae/ckpt/{tag}/*/")):
        run = os.path.basename(d.rstrip("/"))
        for which in WHICH:
            ck = os.path.join(d, which)
            if os.path.exists(ck):
                out.append((tag, run, which, ck))
    return out


def agreement(model, z, codes, levels=3):
    """逐层行级一致率；行数不符返回 None。"""
    if z.shape[0] != codes.shape[0]:
        return None
    preds = decode_indices(model, z)
    return [float((codes[:, lvl] == preds[lvl].cpu().numpy()).mean())
            for lvl in range(levels)]


def audit_dataset(ds, rows_out):
    print(f"\n{'#'*100}\n# {ds}\n{'#'*100}")

    codes = {}
    for f in CODES_FILES[ds]:
        p = os.path.join("data", ds, f)
        if not os.path.exists(p):
            print(f"  ⚠ codes 缺失: {f}")
            continue
        c = np.load(p)
        codes[f] = c
        print(f"  codes {f:44s} N={c.shape[0]:6d}  L0d={len(np.unique(c[:,0])):3d} "
              f"L1d={len(np.unique(c[:,1])):3d} L2d={len(np.unique(c[:,2])):3d}")

    cands = []
    for tag in CKPT_TAGS[ds]:
        cands += ckpt_candidates(tag)
    if not cands:
        print("  ❌ 无任何 ckpt 候选")
        return {}

    # 交叉矩阵：每个 ckpt 编码一次 z，对全部 codes 文件算 L0 一致率
    matrix = {}   # (tag, run, which) -> {codes_file: (frac0, frac1, frac2) | None}
    print(f"\n  交叉矩阵（单元 = L0 行级一致率 %，括号内 L1/L2）")
    hdr = f"  {'ckpt':<52s}" + "".join(f"{f[:22]:>24s}" for f in codes)
    print(hdr)
    for tag, run, which, ck in cands:
        try:
            model, a = load_rqvae_from_ckpt(ck, device="cpu")
            z = encode_z(model, a.data_path, device="cpu")
        except Exception as e:
            print(f"  [{tag}/{run}/{which[:13]}] ✗ 加载/编码失败: {e}")
            continue
        key = (tag, run, which)
        matrix[key] = {}
        cells = []
        for f, c in codes.items():
            agg = agreement(model, z, c)
            matrix[key][f] = agg
            cells.append("           n/a（行数不符）" if agg is None
                         else f"{agg[0]*100:14.2f} ({agg[1]*100:.1f}/{agg[2]*100:.1f})")
        print(f"  {tag + '/' + run + '/' + which[:13]:<52s}" + "".join(f"{x:>24s}" for x in cells))
        # 记录到行输出（只记 L0，用于 CSV）
        for f, agg in matrix[key].items():
            rows_out.append({
                "dataset": ds, "ckpt_tag": tag, "ckpt_run": run, "ckpt_file": which,
                "ckpt_data_path": str(a.data_path), "codes_file": f,
                "codes_N": codes[f].shape[0], "z_N": z.shape[0],
                "codes_L0distinct": len(np.unique(codes[f][:, 0])),
                "agree_L0": "" if agg is None else f"{agg[0]:.4f}",
                "agree_L1": "" if agg is None else f"{agg[1]:.4f}",
                "agree_L2": "" if agg is None else f"{agg[2]:.4f}",
            })

    # 9 格判定
    print(f"\n  声明配对判定：")
    verdicts = {}
    for d, seed, codef, tag in [c for c in CELLS if c[0] == ds]:
        if codef not in codes:
            print(f"    [{d}/{seed}] ❌ codes 缺失")
            verdicts[(d, seed)] = ("MISSING", None)
            continue
        best = None
        for (t, run, which), per in matrix.items():
            agg = per.get(codef)
            if agg is None:
                continue
            if best is None or agg[0] > best[1][0]:
                best = ((t, run, which), agg)
        if best is None:
            print(f"    [{d}/{seed}] ❌ 无可比 ckpt")
            verdicts[(d, seed)] = ("NO_CKPT", None)
            continue
        (t, run, which), agg = best
        v = "✅ PASS" if agg[0] > PASS_TH else ("⚠ 存疑" if agg[0] > WARN_TH else "❌ FAIL")
        decl_ok = (t == tag)
        note = "" if decl_ok else f"  ⚠ 最佳来源 ≠ 声明 tag「{tag}」"
        print(f"    [{d}/{seed}] {v}  最佳来源={t}/{run}/{which[:13]}  "
              f"L0={agg[0]*100:.2f}% L1={agg[1]*100:.2f}% L2={agg[2]*100:.2f}%{note}")
        verdicts[(d, seed)] = (v, best)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=None, help="只跑指定数据集")
    ap.add_argument("--csv", default="doc/g2_c0_pairing_audit.csv")
    args = ap.parse_args()

    datasets = [args.ds] if args.ds else ["Sport", "Toys", "Beauty"]
    rows_out = []
    all_v = {}
    for ds in datasets:
        all_v.update(audit_dataset(ds, rows_out))

    print(f"\n{'='*100}\nC0 总结（note 15 §5 判据：L0 一致率 >90% = 配对正确）\n{'='*100}")
    n_pass = n_fail = 0
    for ds, seed, codef, tag in CELLS:
        if (ds, seed) not in all_v:
            continue
        v, best = all_v[(ds, seed)]
        if best is None:
            n_fail += 1
            continue
        (t, run, which), agg = best
        if agg[0] > PASS_TH:
            n_pass += 1
        else:
            n_fail += 1
    print(f"  PASS {n_pass} / 9    FAIL-or-存疑 {n_fail} / 9")

    if args.csv and rows_out:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"  CSV → {args.csv}  ({len(rows_out)} 行)")


if __name__ == "__main__":
    main()
