#!/usr/bin/env python3
"""
threshold_sensitivity.py — 转移对共现阈值鲁棒性检验（robustness to co-occurrence threshold）

背景：§5.5.1 的转移对取"共现频次 ≥ 2 的 top-1000"，其中 Toys 有近半数对仅共现 2 次。
      本文档检验 z-gap（量化转移损失）的结论是否被弱共现对驱动。

方法：在同一批随机对（seed=0, n_random=3000，E44 统一口径）上，以共现阈值 count ≥ {2,3,5} 重算
      AUC(z) 与 AUC(xq[full])，观察 z-gap 的稳定性。

结论（2026-08-21 复核，3000 随机对）：阈值变化不改变任何结论——
  - Toys 量化丢失 ~8-10%（即使只保留 count≥5 的最强 136 对）；
  - Beauty/Sport 在全部阈值下近无损（z-gap ≤ 0.0032）；
  - Adaptive(512) 的修复结论稳健（count≥5 时 z-gap 0.081→0.001）。
  详见 doc/paper_draft_zh_sec54_56.md §5.5.1 第四点。

只读约束：只读 checkpoint / SID codes / parquet，仅输出到 stdout。

用法（在仓库根目录运行）：
    python3 tools/threshold_sensitivity.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)

from sid_consistency_contrast import (
    CONFIG,
    find_best_ckpt,
    load_codebook_weights,
    pairs_to_rows,
    pair_sims,
    random_pair_sims,
    compute_auc,
)
from sid_per_layer_diagnostic import encode_z, normalize_rows, rebuild_xq_partial

DATA_DIR = "data"
THRESH = [2, 3, 5]

# (tag, code_file) 覆盖：Toys 三个 K0 模型
TOYS_MODELS = [
    ("Toys_TA_SID_768d", "Toys_ta_sid_768d_sid.npy"),    # Fixed-small (256)
    ("Toys_TA_SID_L0512", "Toys_ta_sid_L0512_sid.npy"),  # Adaptive (512)
    ("Toys_TA_SID_L0Big", "Toys_ta_sid_L0Big_sid.npy"),  # Fixed-large (1024)
]


def pair_counts(ds):
    """计数相邻共现（无向），按频次降序返回 [(pair, count), ...]。与 top_transition_pairs 同口径。"""
    df = pd.read_parquet(os.path.join(DATA_DIR, ds, "train.parquet"))
    pc = {}
    for seq in df["history"]:
        s = list(seq)
        for i in range(len(s) - 1):
            a, b = int(s[i]), int(s[i + 1])
            if a > b:
                a, b = b, a
            pc[(a, b)] = pc.get((a, b), 0) + 1
    return sorted(pc.items(), key=lambda x: -x[1])


def auc_of_space(xq_n, pair_ab, n_random=3000):
    pos = pair_sims(xq_n, pair_ab)
    neg = random_pair_sims(xq_n, n_pairs=n_random, seed=0)
    return compute_auc(pos, neg)


def load_z_and_xq(ckpt_tag, code_file, ds):
    ckpt = find_best_ckpt(ckpt_tag)
    codes = np.load(os.path.join(DATA_DIR, ds, code_file))
    cb = load_codebook_weights(ckpt)
    xq_full = normalize_rows(rebuild_xq_partial(codes, cb, len(cb)))
    a = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    dp = getattr(a, "data_path", None)
    z = encode_z(ckpt, dp) if (dp and os.path.exists(dp)) else None
    return z, xq_full


def sensitivity(ds_name, pairs, z, xq_full, tag):
    print(f"  [{tag}]")
    print(f'  {"count>=":7s} {"池大小":>6s} {"实际用":>6s} {"min共现":>7s} | {"z AUC":>7s} {"xq AUC":>7s} {"z-gap":>7s}')
    for T in THRESH:
        sel = [(p, c) for p, c in pairs if c >= T][:1000]
        pair_ab = pairs_to_rows(ds_name, sel)
        if len(pair_ab) < 50:
            print(f'  count>={T:2d} 池不足 50 对，跳过')
            continue
        az = auc_of_space(z, pair_ab) if z is not None else float("nan")
        ax = auc_of_space(xq_full, pair_ab)
        print(f'  count>={T:2d}   {len(pairs):6d}  {len(sel):6d}  {sel[-1][1]:7d} | {az:7.4f} {ax:7.4f} {az - ax:7.4f}')
    print()


def main():
    parser = argparse.ArgumentParser(description="Transition pair co-occurrence threshold sensitivity")
    parser.add_argument("--datasets", type=str, default="Toys,Beauty,Sport")
    args = parser.parse_args()

    for ds in [d.strip() for d in args.datasets.split(",")]:
        if ds not in CONFIG:
            continue
        pairs = pair_counts(ds)
        top1000 = pairs[:1000]
        d = {t: sum(1 for _, c in top1000 if c >= t) for t in [3, 5, 10]}
        print(f"===== {ds}  全部相邻对 {len(pairs)} | top-1000 中 count>=3 的 {d[3]} / >=5 的 {d[5]} / >=10 的 {d[10]} =====")
        cfg = CONFIG[ds]
        z, xq_full = load_z_and_xq(cfg["ta_ckpt"], cfg["ta_code"], ds)
        sensitivity(ds, pairs, z, xq_full, f"256 ({ds})")

    print("===== Toys 三个 K0 模型：z-gap 随阈值 =====")
    toys_pairs = pair_counts("Toys")
    for tag, code_file in TOYS_MODELS:
        z, xq_full = load_z_and_xq(tag, code_file, "Toys")
        sensitivity("Toys", toys_pairs, z, xq_full, tag)


if __name__ == "__main__":
    main()
