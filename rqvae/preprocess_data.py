"""
preprocess_data.py — Preprocess Amazon review JSON data for TIGER + TA-SID pipeline

Generates:
  - data/{DATASET}/train.parquet
  - data/{DATASET}/valid.parquet
  - data/{DATASET}/test.parquet
  - data/{DATASET}/item_emb.parquet (from content_embeddings.pkl)

Usage:
    python rqvae/preprocess_data.py --dataset Sport
    python rqvae/preprocess_data.py --dataset Toys
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset for TIGER pipeline.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (Sport / Toys)")
    parser.add_argument("--raw_data_dir", type=str,
                        default="/data/gtx/project/datasets/tiger_beauty",
                        help="Root dir containing raw dataset directories")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory (relative to project root)")
    return parser.parse_args()


def main():
    args = parse_args()
    ds = args.dataset
    raw_dir = os.path.join(args.raw_data_dir, ds)
    out_dir = f"{args.output_dir}/{ds}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- JSON file name mapping ----
    json_name_map = {
        "Sport": "Sports_and_Outdoors_5.json",
        "Toys": "Toys_and_Games_5.json",
    }
    json_path = os.path.join(raw_dir, json_name_map.get(ds, f"{ds}.json"))
    if not os.path.exists(json_path):
        # fallback: try {dataset}_5.json
        json_path = os.path.join(raw_dir, f"{ds}_5.json")

    print(f"[Preprocess] Dataset: {ds}")
    print(f"  Raw JSON: {json_path}")
    print(f"  Output:   {out_dir}/")

    # ==============================
    # 1. Process JSON → train/valid/test parquet
    # ==============================
    print("\n===== Step 1: Process JSON → train/valid/test.parquet =====")

    userID_mapping = {}
    itemID_mapping = {}

    userIDs = []
    itemIDs = []
    timestamps = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
                user_id = review["reviewerID"]
                item_id = review["asin"]
                timestamp = review["unixReviewTime"]
            except (KeyError, json.JSONDecodeError):
                continue

            if user_id not in userID_mapping:
                userID_mapping[user_id] = len(userID_mapping) + 1
            if item_id not in itemID_mapping:
                itemID_mapping[item_id] = len(itemID_mapping) + 1

            userIDs.append(userID_mapping[user_id])
            itemIDs.append(itemID_mapping[item_id])
            timestamps.append(timestamp)

    print(f"  Users: {len(userID_mapping)}, Items: {len(itemID_mapping)}, Reviews: {len(userIDs)}")

    # Group by user, sort by timestamp
    user_item_map = {}
    for uid, iid, ts in zip(userIDs, itemIDs, timestamps):
        user_item_map.setdefault(uid, []).append((iid, ts))

    for uid in user_item_map:
        user_item_map[uid].sort(key=lambda x: x[1])
        user_item_map[uid] = [x[0] for x in user_item_map[uid]]

    # Leave-one-out split (same as original process)
    train_data = {}
    val_data = {}
    test_data = {}
    for uid, seq in user_item_map.items():
        if len(seq) < 3:
            continue  # need at least 3 items for leave-one-out
        train_data[uid] = seq[:-2]
        val_data[uid] = seq[:-1]
        test_data[uid] = seq

    def make_df(data_dict):
        rows = []
        for uid, seq in data_dict.items():
            rows.append({"user": uid, "history": seq[:-1], "target": seq[-1]})
        return pd.DataFrame(rows)

    train_df = make_df(train_data)
    val_df = make_df(val_data)
    test_df = make_df(test_data)

    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "valid.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"  train.parquet: {train_df.shape} rows")
    print(f"  valid.parquet: {val_df.shape} rows")
    print(f"  test.parquet:  {test_df.shape} rows")

    # Save mappings for reference
    np.save(os.path.join(out_dir, "user_mapping.npy"), userID_mapping)
    np.save(os.path.join(out_dir, "item_mapping.npy"), itemID_mapping)

    # ==============================
    # 2. Convert content_embeddings.pkl → item_emb.parquet
    # ==============================
    print("\n===== Step 2: Convert content_embeddings.pkl → item_emb.parquet =====")

    pkl_path = os.path.join(raw_dir, "content_embeddings.pkl")
    if not os.path.exists(pkl_path):
        print(f"  ⚠ {pkl_path} not found — skipping item_emb.parquet generation")
        return

    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    item_ids_pkl = pkl_data["item_id"]     # numpy array of ints
    embeddings = pkl_data["embedding"]      # numpy array of floats

    # The itemID in JSON are ASIN strings; mapping dict has ASIN→int
    # content_embeddings.pkl uses integer item_id that matches the
    # mapped IDs used in train/valid/test parquet files.
    # Need to convert from 0-indexed to 1-indexed (if needed) or
    # ensure alignment with the itemID_mapping from the JSON processing.
    # The pkl uses the same integer IDs as the mapped IDs.

    rows = []
    for idx in range(len(item_ids_pkl)):
        item_id_val = int(item_ids_pkl[idx])
        # item_emb.parquet expects "ItemID" as string (ASIN) — wait,
        # actually in the original TIGER pipeline the item_emb.parquet
        # maps from int item IDs. Let me check:
        # The build_transition_graph.py uses item_df["ItemID"] and
        # then maps through item_id_to_idx. So ItemID can be int or string.
        # The train/valid/test use the mapped integer IDs.
        # So ItemID here should be the integer mapped ID.
        emb_vec = np.array(embeddings[idx], dtype=np.float32)
        rows.append({"ItemID": item_id_val, "embedding": emb_vec.tolist()})

    item_emb_df = pd.DataFrame(rows)
    item_emb_path = os.path.join(out_dir, "item_emb.parquet")
    item_emb_df.to_parquet(item_emb_path, index=False)
    print(f"  item_emb.parquet: {item_emb_df.shape} rows, embedding dim={len(item_emb_df['embedding'].iloc[0])}")
    print("  ✓ Done!")


if __name__ == "__main__":
    main()
