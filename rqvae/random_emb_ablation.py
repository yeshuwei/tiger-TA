"""
random_emb_ablation.py — 生成随机行为嵌入 + 融合嵌入

目的：对比 Content+Random 噪声 与 Content+Transition (PPMI+SVD)，证明
     TA-SID 的提升来自 Transition Dynamics 结构，而非 "额外加一个向量"。

方法：
    1. 从真实 behavior_emb.parquet 统计 norm 分布
    2. 生成相同统计特性的随机 128d 向量（同分布随机噪声）
    3. 随机向量也通过 LN → 与 Content LN 后 Concat
    4. 保存为 fused_item_emb_random.parquet

输出：
    data/{Dataset}/behavior_emb_random.parquet  (128d 随机向量)
    data/{Dataset}/fused_item_emb_random.parquet (LN(content) + LN(random) → 896d)

用法：
    cd /data/gtx/project/code/TIGER_minilm
    python rqvae/random_emb_ablation.py --dataset Sport --seed 2025
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random behavior embedding + fused embedding for ablation."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (Sport / Toys / Beauty)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = f"{args.data_dir}/{args.dataset}"
    content_path = f"{data_root}/item_emb_768d.parquet"
    real_beh_path = f"{data_root}/behavior_emb.parquet"
    out_beh_path = f"{data_root}/behavior_emb_random.parquet"
    out_fused_path = f"{data_root}/fused_item_emb_random.parquet"

    # ===== 1. Load content embedding =====
    print(f"[1/4] Loading content embedding from {content_path}")
    content_df = pd.read_parquet(content_path)
    content_emb = np.stack(content_df["embedding"].values).astype(np.float32)
    item_ids = content_df["ItemID"].values
    N = len(item_ids)
    print(f"  → Content: {content_emb.shape}, N={N}")

    # ===== 2. Compute norm statistics from REAL behavior embedding =====
    print(f"[2/4] Computing norm stats from real behavior emb: {real_beh_path}")
    real_beh_df = pd.read_parquet(real_beh_path)
    real_beh = np.stack(real_beh_df["embedding"].values)
    real_norms = np.linalg.norm(real_beh, axis=1)
    mean_norm = real_norms.mean()
    std_norm = real_norms.std()

    # Also compute the directional statistics: SVD singular values
    # This captures the shape of the real embedding distribution
    U, S, Vt = np.linalg.svd(real_beh, full_matrices=False)
    # S is length 128 — the singular values of the behavior embedding space
    print(f"  → Real behavior norm: mean={mean_norm:.4f}, std={std_norm:.4f}")
    print(f"  → Real behavior SVD singular values (first 5): {S[:5]}")
    print(f"  → Real behavior SVD singular values (range): {S.min():.4f} ~ {S.max():.4f}")

    # ===== 3. Generate random embedding =====
    # Strategy: Generate random vectors whose SVD spectrum matches real behavior emb
    # This ensures the random embedding has similar "shape" (distribution in space)
    # but no actual transition structure
    print(f"[3/4] Generating random behavior embedding (matching norm + spectrum) ...")

    # Step 3a: Generate i.i.d. Gaussian noise
    random_raw = np.random.randn(N, 128).astype(np.float32)

    # Step 3b: Match the SVD spectrum of real behavior embedding
    # We do this by decomposing the random matrix and rescaling its singular values
    U_rand, S_rand, Vt_rand = np.linalg.svd(random_raw, full_matrices=False)

    # Rescale: replace random singular values with real ones (preserving random directions)
    S_matched = S.copy()  # real singular values
    random_emb = (U_rand * S_matched) @ Vt_rand  # (N, 128)

    # Step 3c: Match the mean norm (scale)
    current_norms = np.linalg.norm(random_emb, axis=1)
    scale_factor = mean_norm / current_norms.mean()
    random_emb = random_emb * scale_factor

    final_norms = np.linalg.norm(random_emb, axis=1)
    print(f"  → Random emb norms: mean={final_norms.mean():.4f}, std={final_norms.std():.4f}")
    print(f"  → Random emb shape: {random_emb.shape}")

    # Save random behavior embedding
    emb_list = [random_emb[i] for i in range(N)]
    beh_df = pd.DataFrame({
        "ItemID": item_ids,
        "embedding": emb_list,
    })
    beh_df.to_parquet(out_beh_path, index=False)
    print(f"  ✓ Saved random behavior emb → {out_beh_path}")

    # ===== 4. Fuse: LN(content) + LN(random) → Concat =====
    print(f"[4/4] Fusing: LN(content) + LN(random) → Concat ...")

    # Separate LayerNorm for each modality (same as real fusion)
    content_t = torch.from_numpy(content_emb).float()
    random_t = torch.from_numpy(random_emb).float()

    ln_content = nn.LayerNorm(content_t.shape[-1])
    ln_random = nn.LayerNorm(random_t.shape[-1])

    with torch.no_grad():
        content_norm = ln_content(content_t)
        random_norm = ln_random(random_t)
        fused = torch.cat([content_norm, random_norm], dim=-1)  # (N, 896)
        fused_np = fused.numpy().astype(np.float32)

    print(f"    Content norm  — mean={content_norm.mean():.4f}, std={content_norm.std():.4f}")
    print(f"    Random norm   — mean={random_norm.mean():.4f}, std={random_norm.std():.4f}")

    # Save fused embedding
    fused_list = [fused_np[i] for i in range(N)]
    fused_df = pd.DataFrame({
        "ItemID": item_ids,
        "embedding": fused_list,
    })
    fused_df.to_parquet(out_fused_path, index=False)
    print(f"  ✓ Saved fused embedding (content+random) → {out_fused_path}")
    print(f"    Shape: ({N}, {fused_np.shape[1]})")

    # ===== 5. Verify: fused emb should have same dimensions as real TA-SID fused =====
    real_fused_path = f"{data_root}/fused_item_emb_768d.parquet"
    real_fused = pd.read_parquet(real_fused_path)
    real_fused_emb = np.stack(real_fused["embedding"].values)
    print(f"\n  Verification:")
    print(f"    Real  fused emb: {real_fused_emb.shape} (content+behavior)")
    print(f"    Random fused emb: {fused_np.shape} (content+random)")
    print(f"    Dimensions match: {real_fused_emb.shape[1] == fused_np.shape[1]}")

    print("[Done] Random embedding ablation data ready.")


if __name__ == "__main__":
    main()
