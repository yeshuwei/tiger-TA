"""
build_transition_graph.py — Step 1 of TA-SID pipeline

Builds a PPMI-weighted item co-occurrence matrix from user behavior sequences,
then applies Truncated SVD to produce 128-dim Behavior Embeddings.

Usage:
    python rqvae/build_transition_graph.py --dataset Beauty
"""

import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Behavior Embeddings via PPMI + SVD from user sequences."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (Beauty / Sports / Toys)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory (relative to project root)",
    )
    parser.add_argument(
        "--svd_dim",
        type=int,
        default=128,
        help="Behavior embedding dimensionality (default: 128)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Co-occurrence sliding window size (default: 3 → left=1, right=1)",
    )
    parser.add_argument(
        "--min_cooccur",
        type=int,
        default=1,
        help="Minimum co-occurrence count to keep an edge (default: 1)",
    )
    return parser.parse_args()


def build_cooccurrence_matrix(
    sequences: list, num_items: int, window_size: int
) -> csr_matrix:
    """Build a symmetric co-occurrence matrix using a sliding window.

    For each position *i* in a sequence, all items within
    [i - window_size//2, i + window_size//2] (excluding *i* itself) are counted
    as co-occurring with the center item.

    Returns a symmetric sparse matrix of shape (num_items, num_items).
    """
    half_window = window_size // 2
    # Use dict for incremental building, then convert to COO
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
                # Store undirected (keep it symmetric)
                a, b = (center, seq[j]) if center < seq[j] else (seq[j], center)
                coo[(a, b)] += 1

    # Unpack to sparse matrix (mirror for symmetry)
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


def compute_ppmi(C: csr_matrix) -> csr_matrix:
    """Positive Pointwise Mutual Information transform.

    PPMI[i][j] = max(log(C[i][j] * sum_all / (row_sum[i] * col_sum[j])), 0)
    """
    sum_all = C.sum()
    row_sum = np.array(C.sum(axis=1)).ravel()
    col_sum = np.array(C.sum(axis=0)).ravel()

    # Guard against division by zero
    row_sum = np.maximum(row_sum, 1e-12)
    col_sum = np.maximum(col_sum, 1e-12)

    C_coo = C.tocoo()
    pmi = np.log(
        C_coo.data * sum_all / (row_sum[C_coo.row] * col_sum[C_coo.col]) + 1e-30
    )
    ppmi = np.maximum(pmi, 0.0)

    return csr_matrix((ppmi, (C_coo.row, C_coo.col)), shape=C.shape)


def main():
    args = parse_args()

    data_root = f"{args.data_dir}/{args.dataset}"
    train_path = f"{data_root}/train.parquet"
    item_emb_path = f"{data_root}/item_emb.parquet"
    output_path = f"{data_root}/behavior_emb.parquet"

    # ---- 1. Load canonical item list from item_emb.parquet ----
    print(f"[1/5] Loading canonical item list from {item_emb_path}")
    item_df = pd.read_parquet(item_emb_path)
    all_item_ids = item_df["ItemID"].values  # canonical ordering
    num_items = len(all_item_ids)
    item_id_to_idx = {iid: idx for idx, iid in enumerate(all_item_ids)}
    print(f"  → {num_items} items total")

    # ---- 2. Extract item sequences from train data ----
    print(f"[2/5] Loading sequences from {train_path}")
    train_df = pd.read_parquet(train_path)

    sequences = []
    for _, row in train_df.iterrows():
        seq = list(row["history"]) + [row["target"]]
        mapped = [item_id_to_idx[iid] for iid in seq if iid in item_id_to_idx]
        if mapped:
            sequences.append(mapped)

    print(f"  → {len(train_df)} users, {len(sequences)} sequences after filtering")

    # Track which items appear (for cold-start zeroing)
    seen_items = set()
    for s in sequences:
        seen_items.update(s)
    print(f"  → {len(seen_items)} items appear in training sequences")
    print(f"  → {num_items - len(seen_items)} cold-start items (zero embedding)")

    # ---- 3. Build co-occurrence matrix ----
    print(f"[3/5] Building co-occurrence matrix (window_size={args.window_size})...")
    C = build_cooccurrence_matrix(sequences, num_items, args.window_size)

    # Apply minimum co-occurrence threshold
    if args.min_cooccur > 1:
        C.data[C.data < args.min_cooccur] = 0
        C.eliminate_zeros()

    print(f"  → shape={C.shape}, non-zero={C.nnz}, density={C.nnz / (num_items**2):.6%}")

    if C.nnz == 0:
        print("  ⚠ WARNING: Co-occurrence matrix is empty — falling back to zero embeddings.")
        behavior_emb = np.zeros((num_items, args.svd_dim), dtype=np.float32)
        _save_output(output_path, all_item_ids, behavior_emb)
        return

    # ---- 4. PPMI transform ----
    print("[4/5] Computing PPMI ...")
    PPMI = compute_ppmi(C)
    print(f"  → non-zero={PPMI.nnz}")

    if PPMI.nnz == 0:
        print("  ⚠ WARNING: PPMI matrix is empty — falling back to zero embeddings.")
        behavior_emb = np.zeros((num_items, args.svd_dim), dtype=np.float32)
        _save_output(output_path, all_item_ids, behavior_emb)
        return

    # ---- 5. Truncated SVD ----
    k = min(args.svd_dim, min(PPMI.shape) - 1)
    k = max(k, 1)  # svds needs at least k=1
    print(f"[5/5] Running Truncated SVD (k={k})...")

    U, S, Vt = svds(PPMI, k=k)
    # svds returns singular values in ascending order; sort descending
    idx = np.argsort(S)[::-1]
    U, S = U[:, idx], S[idx]

    # behavior_emb = U @ diag(sqrt(S))   — standard for PPMI SVD embeddings
    behavior_emb = U * np.sqrt(S.reshape(1, -1))  # (N, k)

    # Pad to svd_dim if k was smaller
    if k < args.svd_dim:
        pad = np.zeros((num_items, args.svd_dim - k), dtype=behavior_emb.dtype)
        behavior_emb = np.concatenate([behavior_emb, pad], axis=1)

    # Zero out cold-start items explicitly
    for i in range(num_items):
        if i not in seen_items:
            behavior_emb[i] = 0.0

    # ---- Save ----
    _save_output(output_path, all_item_ids, behavior_emb)

    # ---- Validation ----
    _validate(behavior_emb, all_item_ids, seen_items)


def _save_output(path: str, item_ids: np.ndarray, embeddings: np.ndarray):
    """Save to parquet matching item_emb.parquet format."""
    num_items = len(item_ids)
    emb_list = [embeddings[i].astype(np.float32) for i in range(num_items)]
    df = pd.DataFrame({"ItemID": item_ids, "embedding": emb_list})
    df.to_parquet(path, index=False)
    print(f"  ✓ Saved {path}  shape=({len(item_ids)}, {embeddings.shape[1]})")


def _validate(
    embeddings: np.ndarray, item_ids: np.ndarray, seen_items: set
):
    """Print top-5 most similar item pairs as a quick sanity check."""
    print("\n[Validation] Top-5 most similar item pairs (cosine similarity):")

    # Only consider items with non-zero embeddings
    pool = sorted(seen_items)[:200]  # first 200 seen items
    if len(pool) < 2:
        print("  (too few items to validate)")
        return

    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_emb = embeddings / np.maximum(norm, 1e-12)
    sim = norm_emb @ norm_emb.T

    pairs = []
    for idx_i, i in enumerate(pool):
        for j in pool[idx_i + 1 :]:
            pairs.append((i, j, sim[i, j]))
    pairs.sort(key=lambda x: -x[2])

    for rank, (i, j, s) in enumerate(pairs[:5]):
        print(f"  #{rank + 1}:  Item {item_ids[i]} <-> Item {item_ids[j]}  cos={s:.4f}")

    # Also compute some basic stats
    non_zero = np.count_nonzero(np.linalg.norm(embeddings, axis=1))
    print(f"  Non-zero embedding rows: {non_zero} / {len(item_ids)}")
    print("[Done]")


if __name__ == "__main__":
    main()
