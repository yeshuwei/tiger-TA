#!/usr/bin/env python3
"""
sid_consistency_contrast.py — Transition Consistency: TA-SID vs Content-only baseline

在【量化输出空间】对比两个 SID 编码器对"转移关系"的编码程度：
    用 SID code 重建量化向量 x_q = Σ_l codebook_l[code[:, l]]
    计算高频转移对的 x_q 余弦相似度均值

  分数高 → 经常一起出现的商品在 SID 量化空间中也靠近 → 转移被编码进 SID。

对比：
    TA-SID (content+behavior)   vs   Content-only baseline (纯 content)

用法：
    python3 tools/sid_consistency_contrast.py [--datasets Beauty,Sport,Toys]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

DATA_DIR = "data"
CKPT_DIR = "rqvae/ckpt"

# 每数据集的 TA-SID 与 baseline 配置
CONFIG = {
    "Beauty": {
        "ta_ckpt": "Beauty_TA_SID",
        "ta_code": "Beauty_ta_sid_768d_sid.npy",
        "base_ckpt": "Beauty_Content768d_Baseline",
        "base_code": "Beauty_content768d_baseline_sid.npy",
    },
    "Sport": {
        "ta_ckpt": "Sport_TA_SID_768d",
        "ta_code": "Sport_ta_sid_768d_sid.npy",
        "base_ckpt": "Sport_Content768d_Baseline",
        "base_code": "Sport_content768d_baseline_sid.npy",
        # E49 行为消融扩展（可选键；缺产物自动跳过，不报错）
        "extra": [
            ("Raw-CoOcc-SVD",  "Sport_RawCoOcc_768d", "Sport_rawcooc_768d_sid.npy"),
            ("ItemCF-SVD",     "Sport_ItemCF_768d",   "Sport_itemcf_768d_sid.npy"),
            # 2026-09-23 订正：目录名实为 *_Ablation，原写的 *_768d 不存在
            ("Shuffled",       "Sport_Shuffled_Ablation", "Sport_shuffled_ablation_sid.npy"),
            ("ZeroPad",        "Sport_Zeropad_Ablation",  "Sport_zeropad_ablation_sid.npy"),
        ],
    },
    "Toys": {
        "ta_ckpt": "Toys_TA_SID_768d",
        # 2026-09-23 订正：原 _sid.npy 是 L0=62 的孤儿 codes（e48 已作废），换健康 codes（L0=207）
        "ta_code": "Toys_ta_sid_768d_seed2024_sid.npy",
        "base_ckpt": "Toys_Content768d_Baseline",
        "base_code": "Toys_content768d_baseline_sid.npy",
        "extra": [
            ("Raw-CoOcc-SVD",  "Toys_RawCoOcc_768d", "Toys_rawcooc_768d_sid.npy"),
            ("ItemCF-SVD",     "Toys_ItemCF_768d",   "Toys_itemcf_768d_sid.npy"),
            # 2026-09-23 订正：Toys 无 legacy，实际 codes 名是 Toys_shuffled_768d_sid.npy
            ("Shuffled",       "Toys_Shuffled_768d", "Toys_shuffled_768d_sid.npy"),
            ("ZeroPad",        "Toys_Zeropad_seed2024", "Toys_zeropad_seed2024_sid.npy"),
        ],
    },
    "Musical_Instruments": {
        "ta_ckpt": "Musical_Instruments_TA_SID",
        "ta_code": "MI_ta_sid_sid.npy",
        "base_ckpt": "Musical_Instruments_TA_SID",
        "base_code": "MI_ta_sid_sid.npy",
    },
    "Electronics": {
        "ta_ckpt": "Electronics_TA_SID",
        "ta_code": "Electronics_ta_sid_sid.npy",
        "base_ckpt": "Electronics_TA_SID",
        "base_code": "Electronics_ta_sid_sid.npy",
    },
    "Video_Games": {
        "ta_ckpt": "Video_Games_TA_SID",
        "ta_code": "Video_Games_ta_sid_sid.npy",
        "base_ckpt": "Video_Games_TA_SID",
        "base_code": "Video_Games_ta_sid_sid.npy",
    },
}


def find_best_ckpt(ckpt_tag):
    ckpt_dir = os.path.join(CKPT_DIR, ckpt_tag)
    if not os.path.exists(ckpt_dir):
        return None
    # 遍历所有子目录找 best_collision_model.pth
    import glob
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*/best_collision_model.pth")),
                   key=lambda p: os.path.getmtime(p))
    return files[-1] if files else None


def load_codebook_weights(ckpt_path, device="cpu"):
    """从 checkpoint 加载 RQ-VAE 码本权重。返回每层的 (256, e_dim) 列表。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dict"]
    codebooks = []
    for layer in range(len(ckpt["args"].num_emb_list)):
        key = f"rq.vq_layers.{layer}.embedding.weight"
        codebooks.append(state[key].float())
    return codebooks


def rebuild_xq(codes, codebooks):
    """用 SID code 前 num_layers 列重建量化向量 x_q。codes: (N, 4) → (N, e_dim)"""
    num_layers = len(codebooks)
    xq = torch.zeros(codes.shape[0], codebooks[0].shape[1])
    for layer in range(num_layers):
        idx = torch.as_tensor(codes[:, layer], dtype=torch.long)
        xq = xq + codebooks[layer][idx]
    return xq


def top_transition_pairs(ds_name, top_k=1000, min_count=2):
    """从 train.parquet 提取高频相邻转移对（无向）。返回 [(a, b, count), ...]"""
    train_path = os.path.join(DATA_DIR, ds_name, "train.parquet")
    train_df = pd.read_parquet(train_path)
    pair_counts = {}
    for seq in train_df["history"]:
        seq_list = list(seq)
        for i in range(len(seq_list) - 1):
            a, b = int(seq_list[i]), int(seq_list[i + 1])
            if a > b:
                a, b = b, a
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
    top = [(p, c) for p, c in sorted_pairs if c >= min_count][:top_k]
    return top


def item_id_to_row(ds_name, item_emb_file="item_emb_768d.parquet"):
    """ItemID → codes / embedding 矩阵的 canonical 行号（0-based）。

    ⚠ 禁止用 `ItemID - 1` 代替本函数。ItemID 虽是 1-based（prepare_data.py /
    generate_content_embeddings.py 生成），但**不保证等于行号 + 1**：
      Beauty 的 ItemID 是 1..N 的一个排列，4158/12101 = 34.36% 的行错位。
      用 `a - 1` 会静默取到**另一个商品**的码 —— 实测 Beauty TA-SID xq[full]
      AUC 0.8687（错）vs 0.9495（对），1000 对里只有 308 对映射正确。
      Sport / Toys 的 ItemID 恰好连续，两法逐位等价，故该 bug 长期未被发现。
    """
    item_emb_path = os.path.join(DATA_DIR, ds_name, item_emb_file)
    ids = pd.read_parquet(item_emb_path)["ItemID"].values
    return {int(iid): idx for idx, iid in enumerate(ids)}


def pairs_to_rows(ds_name, pairs, item_emb_file="item_emb_768d.parquet"):
    """[((ItemID_a, ItemID_b), count), ...] → [(row_a, row_b), ...]。

    所有下游诊断脚本统一的 ItemID → 行号入口；不要各自写
    `[(a - 1, b - 1) for (a, b), c in pairs]`。
    """
    m = item_id_to_row(ds_name, item_emb_file)
    return [(m[a], m[b]) for (a, b), _ in pairs]


def quantized_space(codes, codebooks):
    """重建量化向量并归一化。返回 (N, e_dim) 归一化张量。"""
    xq = rebuild_xq(codes, codebooks)  # (N, e_dim)
    norms = xq.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return xq / norms


def pair_sims(xq_n, pair_ab):
    """计算指定商品对（0-based index）的余弦相似度。"""
    sims = []
    for a, b in pair_ab:
        if a < xq_n.shape[0] and b < xq_n.shape[0]:
            sims.append(float((xq_n[a] * xq_n[b]).sum()))
    return np.array(sims)


def random_pair_sims(xq_n, n_pairs=5000, seed=0):
    """随机采样商品对，作为"普通商品"基线（unimodality 参照）。"""
    rng = np.random.default_rng(seed)
    N = xq_n.shape[0]
    idxs = rng.integers(0, N, size=(n_pairs, 2))
    return (xq_n[idxs[:, 0]] * xq_n[idxs[:, 1]]).sum(dim=1).numpy()


def compute_auc(pos, neg):
    """
    AUC = P(转移对相似度 > 随机对相似度)。0.5=无区分，1=完美。
    用 Mann-Whitney U 统计量（等价于 ROC 曲线下面积）。
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty(len(all_s))
    ranks[order] = np.arange(1, len(all_s) + 1)
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def main():
    parser = argparse.ArgumentParser(description="Transition Consistency: TA-SID vs Content-only (AUC)")
    parser.add_argument("--datasets", type=str, default="Beauty,Sport,Toys")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--n_random", type=int, default=5000)
    args = parser.parse_args()

    print("=" * 88)
    print("Transition Consistency 对比：TA-SID vs Content-only baseline（量化空间 x_q）")
    print("主指标 AUC = P(转移对相似度 > 随机对相似度)，0.5=随机，1=完美")
    print("=" * 88)

    rows = []
    for ds in [d.strip() for d in args.datasets.split(",")]:
        if ds not in CONFIG:
            print(f"⚠ 跳过: {ds}")
            continue
        cfg = CONFIG[ds]

        pairs = top_transition_pairs(ds, top_k=args.topk)
        if not pairs:
            print(f"{ds}: 无转移对")
            continue
        pair_ab = pairs_to_rows(ds, pairs)

        print(f"\n{'─'*88}")
        print(f"📊 {ds}   |   top-{len(pairs)} 高频转移对, {args.n_random} 随机对")
        print(f"{'─'*88}")

        # 主对照两行 + E49 扩展变体（缺产物则跳过，不中断）
        variant_list = [
            ("TA-SID (content+behavior)", cfg["ta_ckpt"], cfg["ta_code"]),
            ("Content-only baseline", cfg["base_ckpt"], cfg["base_code"]),
        ] + [(t, c, f) for (t, c, f) in cfg.get("extra", [])]

        results = {}
        for tag, cktag, codef in variant_list:
            ckpt = find_best_ckpt(cktag)
            if ckpt is None:
                print(f"  ⚠ checkpoint 缺失，跳过: {cktag}")
                continue
            code_path = os.path.join(DATA_DIR, ds, codef)
            if not os.path.exists(code_path):
                print(f"  ⚠ codes 缺失，跳过: {codef}")
                continue
            codes = np.load(code_path)
            cb = load_codebook_weights(ckpt)
            xq_n = quantized_space(codes, cb)

            pos = pair_sims(xq_n, pair_ab)
            neg = random_pair_sims(xq_n, n_pairs=args.n_random)

            results[tag] = {
                "pos_mean": float(pos.mean()),
                "pos_std": float(pos.std()),
                "neg_mean": float(neg.mean()),
                "mean_diff": float(pos.mean() - neg.mean()),
                "med_diff": float(np.median(pos) - np.median(neg)),
                "auc": compute_auc(pos, neg),
            }

        for tag, r in results.items():
            print(f"\n  {tag}:")
            print(f"    转移对 cos: {r['pos_mean']:.4f} ± {r['pos_std']:.4f}")
            print(f"    随机对 cos: {r['neg_mean']:.4f}")
            print(f"    均值差(相对增益): {r['mean_diff']:+.4f} | 中位数差: {r['med_diff']:+.4f}")
            print(f"    AUC: {r['auc']:.4f}")

        if "TA-SID (content+behavior)" in results and "Content-only baseline" in results:
            ta = results["TA-SID (content+behavior)"]
            base = results["Content-only baseline"]
            print(f"\n  ▶ 结论: TA-SID AUC {ta['auc']:.4f} vs Content-only {base['auc']:.4f}"
                  f"   (Δ{ta['auc']-base['auc']:+.4f})")
            rows.append((ds, ta["auc"], base["auc"],
                         ta["mean_diff"], base["mean_diff"]))

    # 汇总
    print("\n" + "=" * 88)
    print("汇总表（AUC 为主指标）")
    print("=" * 88)
    print(f"{'Dataset':10s} {'TA-SID AUC':>12s} {'Base AUC':>10s} {'ΔAUC':>8s} "
          f"{'TA 均值差':>10s} {'Base 均值差':>12s}")
    print("-" * 70)
    for ds, ta_auc, base_auc, ta_md, base_md in rows:
        print(f"{ds:10s} {ta_auc:12.4f} {base_auc:10.4f} {ta_auc-base_auc:+7.4f} "
              f"{ta_md:10.4f} {base_md:12.4f}")
    print("=" * 88)
    print("\n说明: AUC 0.5=随机(转移未编码), 1.0=完美。TA-SID 应一致高于 Content-only。")


if __name__ == "__main__":
    main()
