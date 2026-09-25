#!/usr/bin/env python3
"""
behavior_space_diagnostic.py — E49 行为空间诊断（GPU-free, 分钟级）

在 **RQ-VAE 之前** 的 behavior 向量空间上对比各变体，回答两个问题：

  1. PPMI 是否降低流行度偏差？
     - frequency_leak : Spearman(||behavior_emb_i||, log1p(freq_i))
     - pc1_cos_freq   : |cos(第一主方向, 归一化流行度方向)|

  2. 行为空间是否真正保留序列上下文结构？
     - cons_auc : behavior consistency retrieval AUC
                  = P(cos(高频共现对) > cos(随机对))，0.5 = 随机
                  （与 tools/sid_consistency_contrast.py 同一度量，
                    但作用在 RQ-VAE 之前的原始 behavior 向量上）

另报谱结构：effective rank = exp(entropy(σ²/Σσ²))、top1 energy σ₁²/Σσ²。

用法：
    python3 tools/behavior_space_diagnostic.py --datasets Sport,Toys
    python3 tools/behavior_space_diagnostic.py --datasets Sport --report doc/e49_behavior_space_gate.md
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# 变体名 → behavior_emb 文件名（相对 data/{Ds}/）
VARIANTS = {
    "ta_sid":       "behavior_emb.parquet",
    "ta_sid_rerun": "behavior_emb_ppmi.parquet",
    "random":       "behavior_emb_random.parquet",
    "shuffled":     "behavior_emb_shuffled.parquet",
    "rawcooc":      "behavior_emb_rawcooc.parquet",
    "itemcf":       "behavior_emb_itemcf.parquet",
}

SEED = 2025
N_NEG = 20000
N_POS_CAP = 20000


def load_sequences(data_dir, ds):
    """与 build_transition_graph 一致：history + [target]，按 canonical 顺序映射。"""
    item_df = pd.read_parquet(f"{data_dir}/{ds}/item_emb_768d.parquet")
    ids = item_df["ItemID"].values
    id2idx = {iid: idx for idx, iid in enumerate(ids)}

    train_df = pd.read_parquet(f"{data_dir}/{ds}/train.parquet")
    seqs = []
    for _, row in train_df.iterrows():
        s = list(row["history"]) + [row["target"]]
        m = [id2idx[x] for x in s if x in id2idx]
        if m:
            seqs.append(m)
    return len(ids), seqs


def item_freq(num_items, seqs):
    f = np.zeros(num_items, dtype=np.int64)
    for s in seqs:
        for i in s:
            f[i] += 1
    return f


def pair_auc(emb_norm, pos_pairs, seed=SEED):
    """behavior consistency retrieval AUC：转移对 vs 随机对 的余弦可分性。"""
    rng = np.random.default_rng(seed)
    n = emb_norm.shape[0]

    pos_sim = np.einsum("ij,ij->i", emb_norm[pos_pairs[:, 0]], emb_norm[pos_pairs[:, 1]])
    neg_a = rng.integers(0, n, size=len(pos_pairs))
    neg_b = rng.integers(0, n, size=len(pos_pairs))
    neg_sim = np.einsum("ij,ij->i", emb_norm[neg_a], emb_norm[neg_b])

    y = np.concatenate([np.ones(len(pos_sim)), np.zeros(len(neg_sim))])
    s = np.concatenate([pos_sim, neg_sim])
    return roc_auc_score(y, s), float(pos_sim.mean()), float(neg_sim.mean())


def build_pair_sets(seqs, rng, far_gap=4):
    """构造两组对照 pair（均为 item index 对）：

    adj  : 序列相邻对 (s[i], s[i+1])            —— 局部序列上下文
    far  : 同一序列但相隔 >= far_gap 的对        —— 同篮但非局部（泛共现）
    rand : 全库随机对                            —— 无关联
    """
    adj, far = [], []
    for s in seqs:
        L = len(s)
        for i in range(L - 1):
            adj.append((s[i], s[i + 1]))
        if L > far_gap:
            for i in range(L - far_gap):
                for j in range(i + far_gap, L):
                    far.append((s[i], s[j]))

    adj = np.asarray(adj, dtype=np.int64)
    far = np.asarray(far, dtype=np.int64)

    if len(adj) > N_POS_CAP:
        adj = adj[rng.choice(len(adj), N_POS_CAP, replace=False)]
    if len(far) > N_POS_CAP:
        far = far[rng.choice(len(far), N_POS_CAP, replace=False)]
    return adj, far


def diagnose(ds, name, path, num_items, seqs, freq, adj_pairs, far_pairs, rng):
    """返回该变体的指标 dict；文件缺失或全零则返回 None。"""
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    emb = np.stack(df["embedding"].values).astype(np.float64)
    if emb.shape[0] != num_items:
        print(f"  ⚠ {name}: 行数 {emb.shape[0]} != {num_items}，跳过")
        return None

    norms = np.linalg.norm(emb, axis=1)
    seen = norms > 0
    if seen.sum() == 0:
        return None

    # ---- 1. frequency leakage ----
    rho, _ = spearmanr(norms[seen], np.log1p(freq[seen]))

    E = emb[seen]
    E_centered = E - E.mean(axis=0, keepdims=True)
    # 第一主方向在 **item 轴**上 = 最大奇异值对应的左奇异向量（长度 = #seen items）
    U1, _, _ = svds(E_centered, k=1)
    pc1 = U1[:, 0]

    fv = freq[seen].astype(np.float64)
    fv_dir = fv / max(np.linalg.norm(fv), 1e-12)
    pc1_cos = float(abs(np.dot(pc1, fv_dir)))

    # ---- 2. spectrum ----
    sv_full = np.linalg.svd(E_centered, compute_uv=False)
    energy = sv_full ** 2
    energy = energy / energy.sum()
    eff_rank = float(np.exp(-np.sum(energy * np.log(energy + 1e-30))))
    top1_energy = float(energy[0])

    # ---- 3. behavior consistency retrieval AUC（两档对照）----
    emb_norm = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    # pair 的 item 索引是全局索引，emb_norm 已按 seen 过滤 → 需重映射
    remap = -np.ones(num_items, dtype=np.int64)
    remap[np.where(seen)[0]] = np.arange(int(seen.sum()))

    def cos_of(pairs):
        if len(pairs) == 0:
            return np.zeros(0)
        p = remap[pairs]
        ok = (p[:, 0] >= 0) & (p[:, 1] >= 0)
        p = p[ok]
        if len(p) == 0:
            return np.zeros(0)
        return np.einsum("ij,ij->i", emb_norm[p[:, 0]], emb_norm[p[:, 1]])

    c_adj, c_far = cos_of(adj_pairs), cos_of(far_pairs)
    n_neg = min(len(c_adj), N_NEG)
    neg_a = rng.integers(0, emb_norm.shape[0], size=n_neg)
    neg_b = rng.integers(0, emb_norm.shape[0], size=n_neg)
    c_rand = np.einsum("ij,ij->i", emb_norm[neg_a], emb_norm[neg_b])

    # 档 1：相邻对 vs 随机对 —— "行为空间是否含共现结构"（itemcf 因定义占优）
    auc_rand = float(roc_auc_score(
        np.r_[np.ones(len(c_adj)), np.zeros(len(c_rand))], np.r_[c_adj, c_rand])) \
        if len(c_adj) > 100 else float("nan")
    # 档 2：相邻对 vs 同篮非相邻对 —— "是否区分局部序列上下文 vs 泛共现"（关键判据）
    m = min(len(c_adj), len(c_far))
    auc_local = float(roc_auc_score(
        np.r_[np.ones(m), np.zeros(m)], np.r_[c_adj[:m], c_far[:m]])) \
        if m > 100 else float("nan")

    return {
        "variant": name,
        "dim": emb.shape[1],
        "n_seen": int(seen.sum()),
        "freq_leak": float(rho),
        "pc1_cos_freq": pc1_cos,
        "eff_rank": eff_rank,
        "top1_energy": top1_energy,
        "cons_auc": auc_rand,
        "cons_auc_local": auc_local,
        "pos_cos": float(c_adj.mean()) if len(c_adj) else float("nan"),
        "far_cos": float(c_far.mean()) if len(c_far) else float("nan"),
        "neg_cos": float(c_rand.mean()) if len(c_rand) else float("nan"),
        "norm_mean": float(norms[seen].mean()),
        "norm_std": float(norms[seen].std()),
    }


def main():
    ap = argparse.ArgumentParser(description="Behavior-space diagnostics for E49 variants")
    ap.add_argument("--datasets", type=str, default="Sport,Toys")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--report", type=str, default="doc/e49_behavior_space_gate.md")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    all_rows = []

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        print(f"\n{'='*78}\n[{ds}] behavior space diagnostics\n{'='*78}")
        num_items, seqs = load_sequences(args.data_dir, ds)
        freq = item_freq(num_items, seqs)

        adj_pairs, far_pairs = build_pair_sets(seqs, rng)
        tp_path = f"{args.data_dir}/{ds}/transition_pairs_768d.npy"
        if os.path.exists(tp_path):
            tp = np.load(tp_path)
            print(f"  transition pairs file: {tp.shape}, "
                  f"max idx={tp.max()} (< {num_items}: {tp.max() < num_items})")
        print(f"  pair sets: adj={len(adj_pairs)}, far(同篮非相邻)={len(far_pairs)}")

        print(f"\n  {'variant':<14}{'seen':>7}{'freqleak':>10}{'|PC1·f|':>9}"
              f"{'effrank':>9}{'consAUC':>9}{'AUC_local':>10}")
        for name, fname in VARIANTS.items():
            r = diagnose(ds, name, f"{args.data_dir}/{ds}/{fname}",
                         num_items, seqs, freq, adj_pairs, far_pairs, rng)
            if r is None:
                print(f"  {name:<14}   (缺文件)")
                continue
            r["dataset"] = ds
            all_rows.append(r)
            print(f"  {name:<14}{r['n_seen']:>7}{r['freq_leak']:>10.4f}"
                  f"{r['pc1_cos_freq']:>9.4f}{r['eff_rank']:>9.2f}"
                  f"{r['cons_auc']:>9.4f}{r['cons_auc_local']:>10.4f}")

    # ---- markdown report ----
    if all_rows:
        cols = ["dataset", "variant", "dim", "n_seen", "freq_leak", "pc1_cos_freq",
                "eff_rank", "top1_energy", "cons_auc", "cons_auc_local",
                "pos_cos", "far_cos", "neg_cos", "norm_mean", "norm_std"]
        df = pd.DataFrame(all_rows)[cols]

        def md_table(frame):
            head = "| " + " | ".join(frame.columns) + " |"
            sep = "|" + "|".join(["---"] * len(frame.columns)) + "|"
            body = []
            for _, r in frame.iterrows():
                cells = []
                for c in frame.columns:
                    v = r[c]
                    if isinstance(v, float):
                        cells.append(f"{v:.4f}")
                    elif isinstance(v, (int, np.integer)):
                        cells.append(str(int(v)))
                    else:
                        cells.append(str(v))
                body.append("| " + " | ".join(cells) + " |")
            return "\n".join([head, sep] + body)

        table = md_table(df)
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as f:
            f.write("# E49 行为空间诊断（RQ-VAE 之前，GPU-free）\n\n")
            f.write("指标定义：\n"
                    "- `freq_leak` = Spearman(||behavior_emb||, log1p(item freq))，越高＝越被流行度主导\n"
                    "- `pc1_cos_freq` = |cos(第一主方向, 归一化流行度方向)|\n"
                    "- `eff_rank` = exp(entropy(σ²/Σσ²))，越低＝谱越集中\n"
                    "- `top1_energy` = σ₁²/Σσ²\n"
                    "- `cons_auc` = P(cos(序列相邻对) > cos(随机对))，0.5 = 随机；"
                    "作用在 RQ-VAE 之前的 behavior 向量上；**对 itemcf 系统性偏高**"
                    "（相邻对是整篮对的子集，属定义优势，不可单独引用）\n"
                    "- `cons_auc_local` = P(cos(序列相邻对) > cos(同篮非相邻对))，0.5 = 不区分；"
                    "**关键判据**：行为空间是否区分「局部序列上下文」与「泛共现」\n\n")
            f.write(table + "\n")
        print(f"\n✓ report → {args.report}")
        print("\n" + table)


if __name__ == "__main__":
    main()
