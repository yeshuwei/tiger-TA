#!/usr/bin/env python3
"""
filter_transition_pairs.py — PPMI 过滤过渡对

过滤原理：
  现有 transition_pairs 包含所有序列相邻对，很多是噪声（A→B 只因流行而非行为相关）。
  通过 PPMI(Pointwise Mutual Information) 过滤：
    - PPMI(A,B) = max(0, log(P(A,B) / (P(A)*P(B))))
    - PPMI 高 → A 和 B 行为显著相关 → 保留
    - PPMI ≈ 0 → A 和 B 是噪声共现 → 过滤

用法：
    conda activate tiger_t5
    cd /data/gtx/project/code/TIGER_minilm

    # Sport 768d: 过滤 transition_pairs，保留 PPMI > 阈值的对
    python3 tools/filter_transition_pairs.py --dataset Sport --suffix _768d --ppmi_threshold 1.0

    # Toys 768d
    python3 tools/filter_transition_pairs.py --dataset Toys --suffix _768d --ppmi_threshold 1.0
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPMI-filter transition pairs for Track B contrastive loss"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (Beauty/Sport/Toys)"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Root data directory"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_768d",
        help="Suffix for embedding/transition files (default: _768d)",
    )
    parser.add_argument(
        "--ppmi_threshold",
        type=float,
        default=1.0,
        help="PPMI threshold (default: 1.0). Higher = fewer but cleaner pairs.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Co-occurrence window size (default: 3)",
    )
    parser.add_argument(
        "--min_cooccur",
        type=int,
        default=2,
        help="Minimum co-occurrence count (default: 2, filters single-occurrence noise)",
    )
    return parser.parse_args()


def build_cooccurrence_matrix(sequences, num_items, window_size):
    """Build symmetric co-occurrence matrix with sliding window."""
    half_window = window_size // 2
    coo = defaultdict(int)

    for seq in sequences:
        L = len(seq)
        for i in range(L):
            left = max(0, i - half_window)
            right = min(L, i + half_window + 1)
            center = seq[i]
            for j in range(left, right):
                if i == j:
                    continue
                a, b = (center, seq[j]) if center < seq[j] else (seq[j], center)
                coo[(a, b)] += 1

    rows, cols, data = [], [], []
    for (a, b), cnt in coo.items():
        rows.append(a)
        cols.append(b)
        data.append(cnt)
        if a != b:
            rows.append(b)
            cols.append(a)
            data.append(cnt)

    return csr_matrix((data, (rows, cols)), shape=(num_items, num_items))


def compute_ppmi(C):
    """PPMI[i][j] = max(log(C[i][j] * sum_all / (row_sum[i] * col_sum[j])), 0)"""
    sum_all = C.sum()
    row_sum = np.array(C.sum(axis=1)).ravel()
    col_sum = np.array(C.sum(axis=0)).ravel()
    row_sum = np.maximum(row_sum, 1e-12)
    col_sum = np.maximum(col_sum, 1e-12)

    C_coo = C.tocoo()
    pmi = np.log(C_coo.data * sum_all / (row_sum[C_coo.row] * col_sum[C_coo.col]) + 1e-30)
    ppmi = np.maximum(pmi, 0.0)

    return csr_matrix((ppmi, (C_coo.row, C_coo.col)), shape=C.shape)


def main():
    args = parse_args()
    ds = args.dataset
    data_root = f"{args.data_dir}/{ds}"
    suffix = args.suffix

    # Paths
    train_path = f"{data_root}/train.parquet"
    item_emb_path = f"{data_root}/item_emb{suffix}.parquet"
    input_pairs_path = f"{data_root}/transition_pairs{suffix}.npy"

    if not os.path.exists(input_pairs_path):
        print(f"❌ 找不到 transition_pairs: {input_pairs_path}")
        print("   请先运行: python3 rqvae/build_transition_graph.py --dataset {ds} --mode pairs --output_suffix {suffix}")
        sys.exit(1)

    print("=" * 60)
    print(f"PPMI 过滤过渡对: {ds}")
    print(f"  输入: {input_pairs_path}")
    print(f"  阈值: PPMI > {args.ppmi_threshold}")
    print(f"  窗口: {args.window_size}")
    print(f"  最小共现: {args.min_cooccur}")
    print("=" * 60)

    # 1. Load canonical item list
    print(f"\n[1/6] 加载商品列表 ...")
    item_df = pd.read_parquet(item_emb_path)
    all_item_ids = item_df["ItemID"].values
    num_items = len(all_item_ids)
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    print(f"  → {num_items} items")

    # 2. Load sequences from training data
    print(f"[2/6] 加载序列 ...")
    train_df = pd.read_parquet(train_path)
    sequences = []
    for _, row in train_df.iterrows():
        seq = list(row["history"]) + [row["target"]]
        mapped = [item_id_to_idx[iid] for iid in seq if iid in item_id_to_idx]
        if mapped:
            sequences.append(mapped)
    print(f"  → {len(sequences)} sequences")

    # 3. Build co-occurrence matrix
    print(f"[3/6] 构建共现矩阵 (window={args.window_size}) ...")
    C = build_cooccurrence_matrix(sequences, num_items, args.window_size)
    if args.min_cooccur > 1:
        C.data[C.data < args.min_cooccur] = 0
        C.eliminate_zeros()
    print(f"  → shape={C.shape}, non-zero={C.nnz}, density={C.nnz / (num_items**2):.6%}")

    # 4. Compute PPMI
    print(f"[4/6] 计算 PPMI ...")
    PPMI = compute_ppmi(C)
    print(f"  → non-zero={PPMI.nnz}")

    # Build a lookup dict for PPMI scores
    ppmi_coo = PPMI.tocoo()
    ppmi_dict = {}
    for i, j, v in zip(ppmi_coo.row, ppmi_coo.col, ppmi_coo.data):
        ppmi_dict[(i, j)] = v

    # 5. Load and filter transition pairs
    print(f"[5/6] 加载 & 过滤 transition pairs ...")
    pairs = np.load(input_pairs_path)
    print(f"  → 原始: {len(pairs)} pairs")

    # Filter
    kept_indices = []
    dropped_no_cooccur = 0
    dropped_low_ppmi = 0

    for i, (src, tgt) in enumerate(pairs):
        # Get PPMI score (undirected: check (src,tgt) or (tgt,src))
        ppmi_val = ppmi_dict.get((src, tgt), ppmi_dict.get((tgt, src), -1.0))

        if ppmi_val < 0:
            dropped_no_cooccur += 1
            continue
        if ppmi_val < args.ppmi_threshold:
            dropped_low_ppmi += 1
            continue
        kept_indices.append(i)

    kept_pairs = pairs[kept_indices]
    print(f"  → 保留: {len(kept_pairs)} pairs")
    print(f"  → 丢弃(无共现): {dropped_no_cooccur} pairs")
    print(f"  → 丢弃(PPMI<阈值): {dropped_low_ppmi} pairs")

    if len(kept_pairs) == 0:
        print("  ⚠️  WARNING: 过滤后没有保留 pair！尝试降低 ppmi_threshold")
        sys.exit(1)

    # 6. Save filtered pairs
    print(f"[6/6] 保存 ...")
    out_path = f"{data_root}/transition_pairs{suffix}_ppmi{args.ppmi_threshold}.npy"
    np.save(out_path, kept_pairs)
    print(f"  → 保存: {out_path}")
    print(f"  → Shape: {kept_pairs.shape}")

    # Summary stats
    print(f"\n{'='*60}")
    print(f"统计摘要")
    print(f"{'='*60}")
    unique_src = len(np.unique(kept_pairs[:, 0]))
    unique_tgt = len(np.unique(kept_pairs[:, 1]))
    unique_both = len(np.unique(kept_pairs))
    print(f"  原始 pairs:  {len(pairs)}")
    print(f"  过滤后 pairs: {len(kept_pairs)} ({len(kept_pairs)/len(pairs)*100:.1f}% 保留)")
    print(f"  唯一 source items: {unique_src}")
    print(f"  唯一 target items: {unique_tgt}")
    print(f"  唯一 items (both): {unique_both}")
    print(f"  Avg pairs/item:  {len(kept_pairs) / max(unique_both, 1):.1f}")
    print(f"  阈值: PPMI > {args.ppmi_threshold}")
    print(f"  [Done]")


if __name__ == "__main__":
    main()
