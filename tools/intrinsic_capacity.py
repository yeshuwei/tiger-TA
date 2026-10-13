#!/usr/bin/env python3
"""
intrinsic_capacity.py — 4.2 机制:每数据集 z 空间"内在簇数"（需要多少 L0 码）
在已有 TA-SID 模型的 z 上做单层 k-means 扫 k,找 transition AUC 饱和点。
与 E2(post-hoc k-means)同法,但目的是横向对比三数据集 → 验证 adaptive 信号。

输出:每个数据集 z AUC + k→AUC 曲线 + "达到 z AUC 的 η 所需最小 k"(内在簇数)。
"""
import os
import sys
import numpy as np
import torch
from collections import defaultdict

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(TOOLS_DIR, "..", "rqvae"))

from sid_consistency_contrast import CONFIG, find_best_ckpt, top_transition_pairs, pairs_to_rows, pair_sims, random_pair_sims, compute_auc
from sid_per_layer_diagnostic import encode_z, normalize_rows

KS = [64, 128, 192, 256, 320, 384, 512, 768, 1024]
ETA = 0.99  # 达到 z AUC 的 99% 视为"饱和"


def kmeans_quantize_auc(z, k, pair_ab, n_random=3000, seed=0):
    from sklearn.cluster import KMeans
    k = min(k, z.shape[0])
    km = KMeans(n_clusters=k, n_init=3, random_state=seed, max_iter=200).fit(z.numpy())
    xq = torch.from_numpy(km.cluster_centers_[labels := km.labels_].astype(np.float32))
    xq = normalize_rows(xq)
    pos = pair_sims(xq, pair_ab)
    neg = random_pair_sims(xq, n_pairs=n_random, seed=seed)
    return compute_auc(pos, neg)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device}\n")
    rows = []
    for ds in ["Beauty", "Sport", "Toys"]:
        cfg = CONFIG[ds]
        ckpt = find_best_ckpt(cfg["ta_ckpt"])
        if ckpt is None:
            print(f"{ds}: NO CKPT, skip"); continue
        pairs = top_transition_pairs(ds, top_k=1000)
        pair_ab = pairs_to_rows(ds, pairs)

        import torch as _t
        data_path = _t.load(ckpt, map_location="cpu", weights_only=False)["args"].data_path
        z = encode_z(ckpt, data_path, device=device)
        z_auc = None
        # z 本身(无量化)的 AUC —— 用随机对直接在 z 上算
        pos = pair_sims(z, pair_ab); neg = random_pair_sims(z, n_pairs=3000, seed=0)
        z_auc = compute_auc(pos, neg)

        curve = []
        for k in KS:
            a = kmeans_quantize_auc(z, k, pair_ab)
            curve.append(a)
            print(f"  {ds:6s} k={k:5d}  AUC={a:.4f}  (z={z_auc:.4f}, η·z={ETA*z_auc:.4f})")
        # 饱和点:第一个 AUC ≥ η·z 的 k;若 256 前已饱和,则"256 够用";否则需要更大
        target = ETA * z_auc
        sat = next((k for k, a in zip(KS, curve) if a >= target), None)
        needed = "≤256" if sat is not None and sat <= 256 else f">{max(KS)}" if sat is None else str(sat)
        print(f"  → {ds}: z={z_auc:.4f} 饱和k(≥99%z)={sat} 需要容量≈{needed}\n")
        rows.append((ds, z_auc, sat, needed, dict(zip(KS, curve))))

    print("=" * 80)
    print(f"{'Dataset':8s} {'z AUC':8s} {'饱和k(≥99%z)':14s} {'容量需求':10s}")
    for ds, z_auc, sat, needed, _ in rows:
        print(f"{ds:8s} {z_auc:8.4f} {str(sat):14s} {needed:10s}")


if __name__ == "__main__":
    main()
