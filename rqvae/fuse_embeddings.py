"""
fuse_embeddings.py — Step 2 of TA-SID pipeline

Fuses Content Embedding (768-dim, from Sentence-T5) with Behavior Embedding
(128-dim, from PPMI+SVD) via:
    LayerNorm(content) ⊕ LayerNorm(behavior)   →   fused_item_emb  (896-dim)

LayerNorm on each modality independently prevents the behavior signal from
being numerically dominated by the larger content signal.

Usage:
    python rqvae/fuse_embeddings.py --dataset Beauty
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse Content + Behavior embeddings via Concat + LayerNorm."
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
    return parser.parse_args()


def main():
    args = parse_args()

    data_root = f"{args.data_dir}/{args.dataset}"
    content_path = f"{data_root}/item_emb.parquet"
    behavior_path = f"{data_root}/behavior_emb.parquet"
    output_path = f"{data_root}/fused_item_emb.parquet"

    # ---- 1. Load embeddings ----
    print(f"[1/3] Loading Content embedding from {content_path}")
    content_df = pd.read_parquet(content_path)
    print(f"       Loading Behavior embedding from {behavior_path}")
    behavior_df = pd.read_parquet(behavior_path)

    # Sanity check: same items, same order
    c_ids = content_df["ItemID"].values
    b_ids = behavior_df["ItemID"].values
    assert len(c_ids) == len(b_ids), "Item count mismatch"
    assert (c_ids == b_ids).all(), "ItemID order mismatch between the two embeddings"

    content_emb = np.stack(content_df["embedding"].values)   # (N, 768)
    behavior_emb = np.stack(behavior_df["embedding"].values)  # (N, 128)

    N = content_emb.shape[0]
    print(f"  → Content:  {content_emb.shape}  (dtype={content_emb.dtype})")
    print(f"  → Behavior: {behavior_emb.shape}  (dtype={behavior_emb.dtype})")

    # ---- 2. Separate Layer Normalization ----
    print("[2/3] Applying Layer Normalization to each modality ...")

    content_t = torch.from_numpy(content_emb).float()    # (N, 768)
    behavior_t = torch.from_numpy(behavior_emb).float()   # (N, 128)

    # Each gets its own LayerNorm so they are normalized independently
    ln_content = nn.LayerNorm(content_t.shape[-1])
    ln_behavior = nn.LayerNorm(behavior_t.shape[-1])

    content_norm = ln_content(content_t)   # (N, 768)
    behavior_norm = ln_behavior(behavior_t)  # (N, 128)

    # Quick check: post-norm stats
    with torch.no_grad():
        print(f"    Content norm  — mean={content_norm.mean():.4f}, std={content_norm.std():.4f}")
        print(f"    Behavior norm — mean={behavior_norm.mean():.4f}, std={behavior_norm.std():.4f}")

    # ---- 3. Concatenate ----
    print(f"[3/3] Concatenating → ({N}, {content_norm.shape[-1] + behavior_norm.shape[-1]}) ...")
    with torch.no_grad():
        fused = torch.cat([content_norm, behavior_norm], dim=-1)  # (N, 896)
        fused_np = fused.numpy().astype(np.float32)
    emb_list = [fused_np[i] for i in range(N)]

    output_df = pd.DataFrame(
        {"ItemID": c_ids, "embedding": emb_list},
    )
    output_df.to_parquet(output_path, index=False)
    print(f"  ✓ Saved fused embedding → {output_path}")
    print(f"    Shape: ({N}, {fused_np.shape[1]})  dtype={fused_np.dtype}")
    print("[Done]")


if __name__ == "__main__":
    main()
