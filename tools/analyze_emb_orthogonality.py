#!/usr/bin/env python3
"""
TA-SID: 内容 vs 行为 embedding 正交性分析

证明 item content embedding 和 behavior (transition) embedding 编码了不同的信息。

策略：因为 content (4096d) 和 behavior (128d) 维度不同，不能直接算余弦。
改用**近邻一致性分析**：
  - 对每个 item，在内容空间找 K 近邻
  - 在 behavior 空间找 K 近邻
  - 如果两个近邻集合高度重叠 → 内容和行为编码了相同信息
  - 如果低重叠 → 内容和行为是互补/正交的

用法:
    conda activate tiger_t5
    python3 tools/analyze_emb_orthogonality.py --dataset Beauty
    python3 tools/analyze_emb_orthogonality.py --dataset Sport
    python3 tools/analyze_emb_orthogonality.py --dataset Toys
"""

import argparse
import os
import random
from collections import Counter
import numpy as np
import pandas as pd

DATA_DIR = "data"


def topk_sims(vecs, idx, k=20):
    """返回 item[idx] 在 vecs 空间中的 top-k 最近邻索引（排除自身）"""
    diff = vecs - vecs[idx]
    dists = np.linalg.norm(diff, axis=1)
    dists[idx] = np.inf  # 排除自身
    return np.argsort(dists)[:k]


def analyze_emb(dataset: str):
    print(f"\n{'='*60}")
    print(f"  {dataset} — Content vs Behavior Embedding 正交性分析")
    print(f"{'='*60}")

    # =========================================
    # 1. 加载数据
    # =========================================
    item_path = f"{DATA_DIR}/{dataset}/item_emb.parquet"
    beh_path = f"{DATA_DIR}/{dataset}/behavior_emb.parquet"

    item_df = pd.read_parquet(item_path)
    beh_df = pd.read_parquet(beh_path)

    # 合并
    item_df = item_df.rename(columns={'embedding': 'content_emb'})
    beh_df = beh_df.rename(columns={'embedding': 'behavior_emb'})
    item_df = item_df.merge(beh_df, on='ItemID', how='inner')

    content_vecs = np.stack(item_df['content_emb'].values)   # (N, d_content)
    behavior_vecs = np.stack(item_df['behavior_emb'].values) # (N, 128)
    n_items = len(item_df)

    print(f"\n  [数据加载]")
    print(f"    Item 总数: {n_items}")
    print(f"    Content dim: {content_vecs.shape[1]}")
    print(f"    Behavior dim: {behavior_vecs.shape[1]}")

    # =========================================
    # 分析 1: 近邻一致性
    # =========================================
    print(f"\n  ─── 分析 1: 内容空间 vs 行为空间的 K 近邻一致性 ───")
    print(f"      (如果两个空间的近邻高度重叠 → 编码了相同信息)")
    print(f"      (如果低重叠 → 编码了互补/正交信息)")

    k = 20
    n_sample = min(500, n_items)
    idx_sample = random.sample(range(n_items), n_sample)

    jaccards = []
    for idx in idx_sample:
        # 找内容空间的 top-k 近邻
        c_neighbors = set(topk_sims(content_vecs, idx, k))
        # 找 behavior 空间的 top-k 近邻
        b_neighbors = set(topk_sims(behavior_vecs, idx, k))
        # Jaccard 相似度
        overlap = len(c_neighbors & b_neighbors)
        jaccard = overlap / k
        jaccards.append(jaccard)

    jaccards = np.array(jaccards)
    print(f"    样本数: {n_sample}, K={k}")
    print(f"    Jaccard 均值:     {np.mean(jaccards):.4f}")
    print(f"    Jaccard 中位数:   {np.median(jaccards):.4f}")
    print(f"    Jaccard std:      {np.std(jaccards):.4f}")

    if np.mean(jaccards) < 0.2:
        print(f"    → 结论: 内容和行为空间的近邻几乎不重叠，两者高度正交 ✅")
        print(f"      (transition 信号编码了内容之外的独立信息)")
    elif np.mean(jaccards) < 0.4:
        print(f"    → 结论: 内容和行为有部分重叠，但总体互补")
    elif np.mean(jaccards) < 0.6:
        print(f"    → 结论: 内容和行为中等重叠")
    else:
        print(f"    → 结论: 内容和行为高度重叠，transition 信号可能没有提供独特信息 ⚠️")

    # =========================================
    # 分析 2: 随机 item 对的跨模态相似度排序一致性
    # =========================================
    print(f"\n  ─── 分析 2: 随机 item 对的跨模态排序一致性 ───")
    print(f"      (如果两种模态对 item 关系的判断一致)")
    print(f"       → 内容相似的商品在 behavior 空间也相似)")
    print(f"       → 说明两种模态信号冗余)")

    n_pairs = 1000
    c_dists, b_dists = [], []
    for _ in range(n_pairs):
        i, j = random.randint(0, n_items - 1), random.randint(0, n_items - 1)
        c_dists.append(np.linalg.norm(content_vecs[i] - content_vecs[j]))
        b_dists.append(np.linalg.norm(behavior_vecs[i] - behavior_vecs[j]))

    c_dists = np.array(c_dists)
    b_dists = np.array(b_dists)

    # Spearman 秩相关（更适合距离比较）
    from scipy.stats import spearmanr
    rho, pval = spearmanr(c_dists, b_dists)
    # Pearson 相关
    corr = np.corrcoef(c_dists, b_dists)[0, 1]

    print(f"    随机对数量: {n_pairs}")
    print(f"    Spearman ρ: {rho:.4f} (p={pval:.2e})")
    print(f"    Pearson r:  {corr:.4f}")

    if abs(rho) < 0.2:
        print(f"    → 结论: 两种模态对 item 关系的判断几乎无关 ✅")
        print(f"      (内容和行为从不同角度描述商品)")
    elif abs(rho) < 0.4:
        print(f"    → 结论: 两种模态弱相关")
    else:
        print(f"    → 结论: 两种模态中度相关")

    # =========================================
    # 分析 3: 高频转移对在 behavior 空间是否更近
    # =========================================
    print(f"\n  ─── 分析 3: 高频转移对 vs 随机对的行为距离 ───")
    print(f"      (用户在序列中经常一起买的商品")
    print(f"       → 在 behavior 空间中应该更近)")

    trans_ok = False
    trans_path = f"{DATA_DIR}/{dataset}/transition_pairs_768d.npy"
    if os.path.exists(trans_path):
        trans_data = np.load(trans_path, allow_pickle=True)
        if isinstance(trans_data, dict) and 'pairs' in trans_data:
            trans_pairs = trans_data['pairs']
        elif isinstance(trans_data, np.ndarray):
            trans_pairs = trans_data
        else:
            trans_pairs = np.array(list(trans_data)) if isinstance(trans_data, dict) else trans_data

        # top-500 高频对
        pair_counter = Counter(tuple(p) for p in trans_pairs[:min(len(trans_pairs), 500000)])
        top_pairs = [p for p, _ in pair_counter.most_common(500)]

        item_ids = item_df['ItemID'].values
        id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

        high_dists = []
        for pair in top_pairs:
            i_id, j_id = pair
            if i_id in id_to_idx and j_id in id_to_idx:
                d = np.linalg.norm(behavior_vecs[id_to_idx[i_id]] - behavior_vecs[id_to_idx[j_id]])
                high_dists.append(d)

        if len(high_dists) > 50:
            high_dists = np.array(high_dists)
            # 随机对基线
            rand_dists = np.array([
                np.linalg.norm(behavior_vecs[random.randint(0, n_items - 1)]
                             - behavior_vecs[random.randint(0, n_items - 1)])
                for _ in range(len(high_dists))
            ])

            print(f"    高频转移对数量: {len(high_dists)}")
            print(f"    高频对 behavior 距离: {np.mean(high_dists):.4f} ± {np.std(high_dists):.4f}")
            print(f"    随机对 behavior 距离: {np.mean(rand_dists):.4f} ± {np.std(rand_dists):.4f}")
            print(f"    差值: {np.mean(rand_dists) - np.mean(high_dists):.4f}")

            if np.mean(high_dists) < np.mean(rand_dists) * 0.95:
                print(f"    → 结论: 高频转移对在 behavior 空间中显著更近 ✅")
                print(f"      (PPMI+SVD 成功编码了 transition 结构)")
            else:
                print(f"    → 结论: 高频转移对在 behavior 空间中没有显著聚集 ⚠️")
            trans_ok = True
        else:
            print(f"    高频对匹配数不足 ({len(high_dists)}), 跳过")
    else:
        print(f"    {trans_path} 不存在, 跳过分析 3")

    # =========================================
    # 分析 4: Embedding 统计
    # =========================================
    print(f"\n  ─── 分析 4: Embedding 范数统计 ───")
    beh_norms = np.linalg.norm(behavior_vecs, axis=1)
    cont_norms = np.linalg.norm(content_vecs, axis=1)
    norm_corr = np.corrcoef(beh_norms, cont_norms)[0, 1]

    print(f"    Behavior emb: mean_norm={np.mean(beh_norms):.4f}, std={np.std(beh_norms):.4f}")
    print(f"    Content emb:  mean_norm={np.mean(cont_norms):.4f}, std={np.std(cont_norms):.4f}")
    print(f"    范数相关系数: {norm_corr:.4f}")

    # =========================================
    # 汇总
    # =========================================
    print(f"\n  {'='*50}")
    print(f"  汇总")
    print(f"  {'='*50}")
    print(f"  1. Top-{k} 近邻 Jaccard:      {np.mean(jaccards):.4f}")
    print(f"  2. 跨模态 Spearman ρ:        {rho:.4f}")
    if trans_ok:
        print(f"  3. 高频转移对 距离差异:      {np.mean(rand_dists) - np.mean(high_dists):.4f}")
    print(f"  4. Behavior norm:            {np.mean(beh_norms):.4f} ± {np.std(beh_norms):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty',
                        help='Dataset name: Sport, Toys, Beauty')
    args = parser.parse_args()
    analyze_emb(args.dataset)


if __name__ == '__main__':
    main()
