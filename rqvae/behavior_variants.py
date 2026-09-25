#!/usr/bin/env python3
"""
behavior_variants.py — 生成行为向量 (behavior embedding) 的消融变体

背景（E49 创新点 1 消融扩展）：
    TA-SID 的 behavior 向量 = 窗口=3 相邻共现矩阵 C → PPMI → SVD(128d)。
    本轮新增两个对照，只改行为向量的来源，其余全链路逐位不变：

    rawcooc : SVD(C)，**跳过 PPMI**。严格单变量对照 → 检验 PPMI 是否必要。
    itemcf  : 整篮共现（同一用户序列内所有商品对，无窗口）→ cosine 归一 → SVD。
              同时改变「作用域 + 归一化 + 方向性」，**非单变量实验** →
              检验局部序列上下文共现结构 vs 全局 item-level 协同信号，
              哪个更适合作为 SID augmentation。
    ppmi    : 与 build_transition_graph.py 等价的 PPMI 路径（**仅用于自检**：
              证明本脚本复现了 TA-SID 的矩阵构建，差异仅来自 svds 数值噪声）。

复用（不复制）build_transition_graph.py 的 build_cooccurrence_matrix /
compute_ppmi 与 build_transition_graph._save_output。

输出：
    data/{Dataset}/behavior_emb_{variant}.parquet   (ItemID + embedding float32)

用法：
    python rqvae/behavior_variants.py --dataset Sport --variant rawcooc
    python rqvae/behavior_variants.py --dataset Sport --variant itemcf
    python rqvae/behavior_variants.py --dataset Sport --variant ppmi --output_file behavior_emb_ppmi.parquet
"""

import argparse
import hashlib
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_transition_graph import (  # noqa: E402
    build_cooccurrence_matrix,
    compute_ppmi,
    _save_output,
)

warnings.filterwarnings("ignore")

VARIANTS = ("rawcooc", "itemcf", "ppmi")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build behavior-embedding ablation variants (rawcooc / itemcf)."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (Sport / Toys / Beauty)")
    parser.add_argument("--variant", type=str, required=True, choices=VARIANTS,
                        help="rawcooc = SVD(C) without PPMI; itemcf = SVD(cosine basket co-occ); "
                             "ppmi = reference PPMI path (self-check only)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--data_suffix", type=str, default="_768d",
                        help="Suffix of the canonical item_emb file (default: _768d)")
    parser.add_argument("--svd_dim", type=int, default=128,
                        help="Behavior embedding dimensionality (default: 128)")
    parser.add_argument("--window_size", type=int, default=3,
                        help="Sliding window for rawcooc/ppmi (default: 3)")
    parser.add_argument("--min_cooccur", type=int, default=1,
                        help="Minimum co-occurrence count to keep an edge (default: 1)")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Seed for the deterministic svds starting vector (default: 2025)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output parquet (default: behavior_emb_{variant}.parquet)")
    return parser.parse_args()


def build_basket_matrix(sequences, num_items):
    """整篮无向共现矩阵（full basket, 无窗口截断）。

    与 build_cooccurrence_matrix 同构：同一序列内任意两件商品（按位置对）
    各计一次；键为 (min,max)；重复商品在同位置对时会产生对角元，与窗口版一致。

    Returns:
        scipy.sparse.csr_matrix, shape (num_items, num_items), 对称。
    """
    coo = defaultdict(int)

    for seq in sequences:
        L = len(seq)
        for i in range(L):
            center = seq[i]
            for j in range(i + 1, L):
                other = seq[j]
                a, b = (center, other) if center < other else (other, center)
                coo[(a, b)] += 1

    rows, cols, data = [], [], []
    for (a, b), cnt in coo.items():
        if cnt < 1:
            continue
        rows.append(a)
        cols.append(b)
        data.append(cnt)
        if a != b:
            rows.append(b)
            cols.append(a)
            data.append(cnt)

    return csr_matrix((data, (rows, cols)), shape=(num_items, num_items))


def to_cosine(S):
    """标准 ItemCF 余弦归一：S_ij / sqrt(d_i * d_j)，对角置零。

    d_i = 与 i 共现的事件总数（去掉对角后的行和）。
    """
    S = S.tocsr(copy=True)
    S.setdiag(0.0)
    S.eliminate_zeros()

    d = np.asarray(S.sum(axis=1)).ravel()
    d = np.maximum(d, 1e-12)
    inv = 1.0 / np.sqrt(d)

    n = S.shape[0]
    D = csr_matrix((inv, (np.arange(n), np.arange(n))), shape=S.shape)
    C = (D @ S @ D).tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    return C


def matrix_checksum(M):
    """矩阵指纹：nnz + 元素和 + 排序后 (row,col,data) 的 md5。

    用于断言 rawcooc 路径复用了与 TA-SID **逐位相同**的共现矩阵 C。
    """
    M = M.tocoo()
    order = np.lexsort((M.data, M.col, M.row))
    h = hashlib.md5()
    h.update(np.ascontiguousarray(M.row[order], dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(M.col[order], dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(M.data[order], dtype=np.float64).tobytes())
    return M.nnz, float(M.sum()), h.hexdigest()[:16]


def svd_embed(M, svd_dim, seen_items, num_items, seed):
    """与 build_transition_graph._run_behavior_emb L273-294 同一条 SVD 管线。

    唯一差别：显式传入确定性起始向量 v0，使重复运行结果可复现
    （svds 默认用随机 v0；收敛后的奇异向量仅相差符号）。
    """
    if M.nnz == 0:
        print("  ⚠ 矩阵为空 — 回退为全零向量")
        return np.zeros((num_items, svd_dim), dtype=np.float32)

    k = min(svd_dim, min(M.shape) - 1)
    k = max(k, 1)

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(M.shape))

    U, S, _ = svds(M.astype(np.float64), k=k, v0=v0)
    idx = np.argsort(S)[::-1]
    U, S = U[:, idx], S[idx]

    emb = U * np.sqrt(np.maximum(S, 0.0).reshape(1, -1))

    if k < svd_dim:
        pad = np.zeros((num_items, svd_dim - k), dtype=emb.dtype)
        emb = np.concatenate([emb, pad], axis=1)

    mask = np.fromiter((i in seen_items for i in range(num_items)),
                       dtype=bool, count=num_items)
    emb[~mask] = 0.0

    return emb.astype(np.float32)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    data_root = f"{args.data_dir}/{args.dataset}"
    item_emb_path = f"{data_root}/item_emb{args.data_suffix}.parquet"
    train_path = f"{data_root}/train.parquet"
    output_path = (f"{data_root}/{args.output_file}" if args.output_file
                   else f"{data_root}/behavior_emb_{args.variant}.parquet")

    # ---- 1. canonical item list（与 build_transition_graph 完全一致）----
    print(f"[1/5] Canonical item list: {item_emb_path}")
    item_df = pd.read_parquet(item_emb_path)
    all_item_ids = item_df["ItemID"].values
    num_items = len(all_item_ids)
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    print(f"  → {num_items} items")

    # ---- 2. 序列（与 build_transition_graph L229-234 逐行一致）----
    print(f"[2/5] Sequences: {train_path}")
    train_df = pd.read_parquet(train_path)
    sequences = []
    for _, row in train_df.iterrows():
        seq = list(row["history"]) + [row["target"]]
        mapped = [item_id_to_idx[iid] for iid in seq if iid in item_id_to_idx]
        if mapped:
            sequences.append(mapped)

    seen_items = set()
    for s in sequences:
        seen_items.update(s)
    n_pairs = sum(len(s) * (len(s) - 1) // 2 for s in sequences)
    print(f"  → {len(sequences)} sequences, {len(seen_items)} seen items, "
          f"{num_items - len(seen_items)} cold-start items")
    print(f"  → max seq len = {max(len(s) for s in sequences)}, "
          f"total basket pairs = {n_pairs}")

    # ---- 3. 矩阵构建 ----
    print(f"[3/5] Building matrix (variant={args.variant}) ...")
    if args.variant in ("rawcooc", "ppmi"):
        M = build_cooccurrence_matrix(sequences, num_items, args.window_size)
        if args.min_cooccur > 1:
            M.data[M.data < args.min_cooccur] = 0
            M.eliminate_zeros()
        scoped = f"window={args.window_size} (adjacent)"
    else:  # itemcf
        M = build_basket_matrix(sequences, num_items)
        scoped = "full-basket (all pairs per user)"

    nnz, total, chk = matrix_checksum(M)
    print(f"  → scope={scoped}")
    print(f"  → nnz={nnz}, sum={total:.1f}, checksum={chk}")

    if args.variant == "ppmi":
        M = compute_ppmi(M)
        print(f"  → PPMI applied: nnz={M.nnz}")
    elif args.variant == "itemcf":
        M = to_cosine(M)
        print(f"  → cosine normalised: nnz={M.nnz}, "
              f"max={M.data.max() if M.nnz else 0.0:.6f}")

    # ---- 4. SVD ----
    k = min(args.svd_dim, min(M.shape) - 1)
    print(f"[4/5] Truncated SVD (k={k}, v0 seed={args.seed}) ...")
    emb = svd_embed(M, args.svd_dim, seen_items, num_items, args.seed)

    norms = np.linalg.norm(emb, axis=1)
    nz = norms[norms > 0]
    print(f"  → norm: mean={nz.mean():.4f}, std={nz.std():.4f}, "
          f"zero rows={(norms == 0).sum()}")

    # ---- 5. 保存 ----
    print(f"[5/5] Saving → {output_path}")
    _save_output(output_path, all_item_ids, emb)
    print("[Done]")


if __name__ == "__main__":
    main()
