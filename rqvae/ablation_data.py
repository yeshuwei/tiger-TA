"""
ablation_data.py — 生成两个对照实验的融合嵌入：
  1. Shuffled Transition: 行级 shuffle behavior emb → 破坏 item-embedding 对应
  2. Zero Padding: Content emb + 128 个 0 → 增加维度但不增加任何信息

输出：
  data/{Dataset}/behavior_emb_shuffled.parquet  (shuffled transition)
  data/{Dataset}/fused_item_emb_shuffled.parquet (LN(content) + LN(shuffled))
  data/{Dataset}/fused_item_emb_zeropad.parquet  (LN(content) + zeros)

用法：
    python rqvae/ablation_data.py --dataset Sport --seed 2025
    python rqvae/ablation_data.py --dataset Toys  --seed 2025 --variants shuffled
        （--variants 只写其中一项，避免无关的 fused_item_emb_zeropad.parquet 被重写——
          该文件是 E47 的输入，见 doc/e49_toys_zeropad_md5_provenance.md）

注意：本脚本默认用 bare `python`（base env，parquet-cpp-arrow 22.0.0）。换 env 会让
      等数据的 md5 变化（写入器版本），故跨 env 比对须用数据级指纹而非 md5。
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ablation data: shuffled + zero padding."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (Sport / Toys / Beauty)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed")
    parser.add_argument("--variants", type=str, default="both",
                        choices=["shuffled", "zeropad", "both"],
                        help="Which ablation to write (default: both = 原行为)")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = f"{args.data_dir}/{args.dataset}"
    content_path = f"{data_root}/item_emb_768d.parquet"
    real_beh_path = f"{data_root}/behavior_emb.parquet"

    # ===== Load content embedding =====
    print(f"[1] Loading content embedding: {content_path}")
    content_df = pd.read_parquet(content_path)
    content_emb = np.stack(content_df["embedding"].values).astype(np.float32)
    item_ids = content_df["ItemID"].values
    N = len(item_ids)
    print(f"  → Content: {content_emb.shape}, N={N}")

    # ===== Load real behavior embedding =====
    print(f"[2] Loading real behavior embedding: {real_beh_path}")
    real_beh_df = pd.read_parquet(real_beh_path)
    real_beh = np.stack(real_beh_df["embedding"].values).astype(np.float32)
    beh_item_ids = real_beh_df["ItemID"].values
    print(f"  → Behavior: {real_beh.shape}")

    # ================================================================
    # Experiment 1: Shuffled Transition
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Experiment 1: Shuffled Transition Embedding")
    print(f"{'='*60}")

    # Shuffle the ROWS of behavior embedding
    shuffle_idx = np.random.permutation(N)
    shuffled_beh = real_beh[shuffle_idx].copy()
    # shuffled_beh[i] is now a real behavior vector, but from a DIFFERENT item

    # Verify: the shuffled embedding has identical stats
    real_norms = np.linalg.norm(real_beh, axis=1)
    shuffled_norms = np.linalg.norm(shuffled_beh, axis=1)
    print(f"  Real norms:      mean={real_norms.mean():.4f}, std={real_norms.std():.4f}")
    print(f"  Shuffled norms:  mean={shuffled_norms.mean():.4f}, std={shuffled_norms.std():.4f}")
    assert abs(real_norms.mean() - shuffled_norms.mean()) < 1e-5
    print(f"  ✅ Stats preserved (norm distribution identical by construction)")

    # Save shuffled behavior embedding (ItemIDs stay the same, embeddings are shuffled)
    shuffled_list = [shuffled_beh[i] for i in range(N)]
    shuffled_beh_df = pd.DataFrame({
        "ItemID": item_ids,
        "embedding": shuffled_list,
    })
    shuffled_beh_path = f"{data_root}/behavior_emb_shuffled.parquet"
    if args.variants in ("shuffled", "both"):
        shuffled_beh_df.to_parquet(shuffled_beh_path, index=False)
        print(f"  ✓ Saved: {shuffled_beh_path}")
    else:
        print(f"  – 跳过写入（--variants={args.variants}）: {shuffled_beh_path}")

    # Fuse: LN(content) + LN(shuffled) → same as real TA-SID pipeline
    print(f"  Fusing LN(content) + LN(shuffled) ...")
    content_t = torch.from_numpy(content_emb).float()
    shuffled_t = torch.from_numpy(shuffled_beh).float()

    ln_content = nn.LayerNorm(content_t.shape[-1])
    ln_shuffled = nn.LayerNorm(shuffled_t.shape[-1])

    with torch.no_grad():
        content_norm = ln_content(content_t)
        shuffled_norm = ln_shuffled(shuffled_t)
        fused_shuffled = torch.cat([content_norm, shuffled_norm], dim=-1).numpy().astype(np.float32)

    fused_shuffled_list = [fused_shuffled[i] for i in range(N)]
    fused_shuffled_df = pd.DataFrame({
        "ItemID": item_ids,
        "embedding": fused_shuffled_list,
    })
    fused_shuffled_path = f"{data_root}/fused_item_emb_shuffled.parquet"
    if args.variants in ("shuffled", "both"):
        fused_shuffled_df.to_parquet(fused_shuffled_path, index=False)
        print(f"  ✓ Saved fused shuffled: {fused_shuffled_path}")
        print(f"    Shape: ({N}, {fused_shuffled.shape[1]})")
    else:
        print(f"  – 跳过写入（--variants={args.variants}）: {fused_shuffled_path}")

    # ================================================================
    # Experiment 2: Zero Padding
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Experiment 2: Zero Padding (content + 128 zeros)")
    print(f"{'='*60}")

    # Fuse: LN(content) + zeros(128)
    # Key: the zero-padded 128 dimensions receive zero gradient during backprop
    #   because ∂L/∂w = ∂L/∂output * input, and input=0 → gradients are 0
    #   → these dimensions contribute NOTHING to the model's capacity
    print(f"  Zero padding: 128 zeros concatenated after LN(content)")
    print(f"  → Extra 128 dimensions are DEAD (gradient always 0)")

    with torch.no_grad():
        content_norm = ln_content(content_t)  # same LN as above
        zeros = torch.zeros(N, 128)
        fused_zeropad = torch.cat([content_norm, zeros], dim=-1).numpy().astype(np.float32)

    fused_zeropad_list = [fused_zeropad[i] for i in range(N)]
    fused_zeropad_df = pd.DataFrame({
        "ItemID": item_ids,
        "embedding": fused_zeropad_list,
    })
    fused_zeropad_path = f"{data_root}/fused_item_emb_zeropad.parquet"
    if args.variants in ("zeropad", "both"):
        fused_zeropad_df.to_parquet(fused_zeropad_path, index=False)
        print(f"  ✓ Saved fused zeropad: {fused_zeropad_path}")
        print(f"    Shape: ({N}, {fused_zeropad.shape[1]})")
    else:
        print(f"  – 跳过写入（--variants={args.variants}）: {fused_zeropad_path}")
    print(f"    Last 128 dims are ALL zeros: {np.abs(fused_zeropad[0, 768:]).sum():.6f}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Content only:            768d          (TIGER baseline)")
    print(f"  Shuffled:               LN(768) + LN(128 shuffled) → 896d")
    print(f"  Zero Pad:               LN(768) + zeros(128)      → 896d")
    print(f"  TA-SID (real):          LN(768) + LN(128 trans)   → 896d")
    print(f"\n  预期对比:")
    print(f"    Zero Pad ≈ Content-only   → 证明'额外维度'不是提升原因")
    print(f"    Shuffled ≈ Content-only    → 证明'统计性质'不是提升原因")
    print(f"    TA-SID >> 两者             → Transition 结构是唯一原因")
    print(f"\n  验证梯度为零:")
    print(f"    RQ-VAE 第一层: Linear(896 → 512)")
    print(f"    Zero pad 对应的权重 (128×512=65536 个参数) 梯度始终为 0")
    print(f"    → 这些参数永远不会更新 ←→ '增加容量'的说法不成立")
    print("[Done]")


if __name__ == "__main__":
    main()
