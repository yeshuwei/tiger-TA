"""
prepare_data.py — Data preparation for TA-SID pipeline

Creates train.parquet, valid.parquet, test.parquet, and item_emb.parquet
from the raw dataset files (inter.json + content_embeddings.pkl).

Usage:
    python rqvae/prepare_data.py --dataset Sport
    python rqvae/prepare_data.py --dataset Toys
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare data files for TA-SID pipeline."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (Sport / Toys)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Output data directory (relative to project root)",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="/data/gtx/project/datasets/tiger_beauty",
        help="Source directory containing raw dataset files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    out_root = f"{args.data_dir}/{dataset}"
    src_dir = f"{args.source_dir}/{dataset}"

    os.makedirs(out_root, exist_ok=True)

    # ====== 1. Load content embeddings ======
    print(f"[1/4] Loading content embeddings from {src_dir}/content_embeddings.pkl")
    with open(f"{src_dir}/content_embeddings.pkl", "rb") as f:
        ckpt = pickle.load(f)

    ckpt_item_ids = ckpt["item_id"]  # list of 0-indexed original item IDs, length N
    ckpt_embeddings = ckpt["embedding"]  # list of 4096-dim embeddings
    num_items = len(ckpt_item_ids)
    print(f"  → {num_items} items total")

    # Build map: original item_id → row index in content_embeddings.pkl
    orig_id_to_idx = {orig_id: idx for idx, orig_id in enumerate(ckpt_item_ids)}

    # ====== 2. Create item_emb.parquet ======
    # item_emb.parquet has 1-indexed ItemIDs [1, 2, ..., N]
    # The embedding for ItemID X should be the embedding for original item_id X-1.
    # Embeddings are stored in the SAME row order as ItemIDs (not content_embeddings order).
    print(f"[2/4] Creating item_emb.parquet → {out_root}/item_emb.parquet")

    item_ids_1based = []  # [1, 2, ..., N]
    emb_list = []  # embeddings in ItemID row order

    for new_id in range(1, num_items + 1):
        orig_id = new_id - 1
        idx = orig_id_to_idx[orig_id]
        item_ids_1based.append(new_id)
        emb_list.append(np.array(ckpt_embeddings[idx], dtype=np.float32))

    item_emb_df = pd.DataFrame({
        "ItemID": item_ids_1based,
        "embedding": emb_list,
    })
    item_emb_df.to_parquet(f"{out_root}/item_emb.parquet", index=False)
    print(f"  ✓ item_emb.parquet saved: ({num_items}, {len(emb_list[0])})")

    # ====== 3. Load inter.json and create train/valid/test ======
    print(f"[3/4] Loading sequences from {src_dir}/inter.json")
    with open(f"{src_dir}/inter.json") as f:
        inter = json.load(f)

    # inter is {user_idx_str: [item_id_0indexed, ...]}
    # Convert to 1-indexed user IDs and 1-indexed item IDs
    all_seq = []
    for user_idx_str, item_seq in inter.items():
        user_id = int(user_idx_str) + 1  # 1-indexed
        item_seq_1based = [it + 1 for it in item_seq]  # 1-indexed items
        all_seq.append((user_id, item_seq_1based))

    # Sort by user_id (not strictly needed but good practice)
    all_seq.sort(key=lambda x: x[0])

    N_users = len(all_seq)
    print(f"  → {N_users} users")

    # Leave-one-out split
    # train: use items[:-2] as the sequence
    # valid: use items[:-1] as the sequence (history=[:-1], target=last)
    # test: use all items (history=[:-1], target=last)
    train_rows, valid_rows, test_rows = [], [], []

    skipped_short = {"train": 0, "valid": 0, "test": 0}

    for user_id, seq in all_seq:
        # Train: need at least 2 items (history + target after sliding window)
        if len(seq) >= 2:
            train_seq = seq[:-2]  # all but last 2
            if len(train_seq) >= 1:
                train_rows.append({
                    "user": user_id,
                    "history": train_seq[:-1],
                    "target": train_seq[-1],
                })
            else:
                skipped_short["train"] += 1
        else:
            skipped_short["train"] += 1

        # Valid: need at least 2 items
        if len(seq) >= 2:
            valid_seq = seq[:-1]
            valid_rows.append({
                "user": user_id,
                "history": valid_seq[:-1],
                "target": valid_seq[-1],
            })
        else:
            skipped_short["valid"] += 1

        # Test: need at least 2 items
        if len(seq) >= 2:
            test_rows.append({
                "user": user_id,
                "history": seq[:-1],
                "target": seq[-1],
            })
        else:
            skipped_short["test"] += 1

    train_df = pd.DataFrame(train_rows)
    valid_df = pd.DataFrame(valid_rows)
    test_df = pd.DataFrame(test_rows)

    print(f"  → Train: {len(train_df)} rows (users with <2 items: {skipped_short['train']})")
    print(f"  → Valid: {len(valid_df)} rows (users with <2 items: {skipped_short['valid']})")
    print(f"  → Test:  {len(test_df)} rows (users with <2 items: {skipped_short['test']})")

    # ====== 4. Save parquet files ======
    print(f"[4/4] Saving parquet files to {out_root}/")
    train_df.to_parquet(f"{out_root}/train.parquet", index=False)
    valid_df.to_parquet(f"{out_root}/valid.parquet", index=False)
    test_df.to_parquet(f"{out_root}/test.parquet", index=False)

    # Confirm
    for name in ["train", "valid", "test"]:
        df = pd.read_parquet(f"{out_root}/{name}.parquet")
        print(f"  ✓ {name}.parquet: {df.shape}")

    # Quick stats
    print(f"\n=== Data Summary ===")
    print(f"  Items:     {num_items}")
    print(f"  Users:     {N_users}")
    print(f"  Embedding: {len(emb_list[0])}d")
    print(f"  ItemIDs:   1 ~ {num_items}")
    print(f"  UserIDs:   1 ~ {N_users}")
    print("[Done]")


if __name__ == "__main__":
    main()
