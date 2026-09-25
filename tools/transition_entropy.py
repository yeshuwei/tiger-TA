#!/usr/bin/env python3
"""
transition_entropy.py — 4.2 adaptive-K0 机制:每个数据集的转移复杂度(transition entropy)

定义（与已有 transition 实验一致）:从 train.parquet 的 history 提取连续相邻有向对
  (history[i] → history[i+1])。可选地加上 (history[-1] → target) 变体（T5 预测目标也
  代表"会话下一步"，两种口径都给出）。

指标：
  H(B|A) = −Σ_a P(a) Σ_b P(b|a) log2 P(b|a)   # bits, 加权的条件熵(转移分布的混合熵)
  H_unw   同公式但 P(a) 均匀(每个 anchor 等权)  # 反映"平均每个 item 的转移不确定性"
  avg out-degree  每个 anchor 的平均不同后继数
  perplexity = 2^H(B|A)

预测方向（4.2 假设）:Toys 的 H 应显著高于 Sport/Beauty → 解释为何 Toys 需要更大 K0。
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

DATASETS = ["Beauty", "Sport", "Toys"]


def extract_pairs(train_path, include_target):
    """返回 {a: Counter(b)} 有向转移计数;以及 P(a) 需要的出度总和。"""
    df = pd.read_parquet(train_path)
    trans = defaultdict(lambda: defaultdict(int))
    for seq, tgt in zip(df["history"], df["target"]):
        s = list(seq)
        for i in range(len(s) - 1):
            trans[s[i]][s[i + 1]] += 1
        if include_target and len(s) > 0:
            trans[s[-1]][int(tgt)] += 1
    return trans


def conditional_entropy(trans):
    """返回 (H_weighted_bits, H_unweighted_bits, avg_outdegree, n_pairs, n_anchors)."""
    total_pairs = 0
    per_anchor = {}
    for a, nxt in trans.items():
        out = sum(nxt.values())
        total_pairs += out
        # 条件熵 (bits)
        h = -sum((c / out) * np.log2(c / out) for c in nxt.values())
        per_anchor[a] = (out, h)
    n_anchors = len(per_anchor)

    # 加权混合熵: Σ_a P(a) H(B|a), P(a) = 出度占比
    h_w = sum((out / total_pairs) * h for out, h in per_anchor.values())
    # 均匀:平均每个 anchor 的条件熵
    h_u = np.mean([h for _, h in per_anchor.values()])
    avg_out = np.mean([out for out, _ in per_anchor.values()])
    # 稀疏度:条件熵 vs 上界 log2(n_anchors)
    return h_w, h_u, avg_out, total_pairs, n_anchors


def main():
    include_target = "--with-target" in sys.argv
    rows = []
    for ds in DATASETS:
        train_path = os.path.join(DATA_DIR, ds, "train.parquet")
        if not os.path.exists(train_path):
            print(f"⚠ missing {train_path}", file=sys.stderr)
            continue
        trans = extract_pairs(train_path, include_target)
        h_w, h_u, avg_out, n_pairs, n_anchors = conditional_entropy(trans)
        px = 2 ** h_w
        rows.append((ds, n_pairs, n_anchors, avg_out, h_w, h_u, px))
        print(f"{ds:8s} pairs={n_pairs:>8d} anchors={n_anchors:>6d} "
              f"avg_out={avg_out:6.2f}  H_weighted={h_w:.4f} bits  "
              f"H_unw={h_u:.4f} bits  perplexity={px:7.2f}")

    if rows:
        print("\n加权条件熵 H(B|A) 对比:")
        base = rows[0]
        for ds, n_pairs, n_anchors, avg_out, h_w, h_u, px in rows:
            rel = 1.0 if ds == base[0] else h_w / base[4]
            print(f"  {ds:8s} H={h_w:.4f} bits (×{rel:.2f} vs {base[0]})  "
                  f"perplexity={px:7.2f}")


if __name__ == "__main__":
    main()
