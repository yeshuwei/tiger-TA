#!/usr/bin/env python3
"""
TA-SID t-SNE 可视化工具

生成 3 张对比图：Content、Behavior、Fused 三种 embedding 的 t-SNE 散点图，
按 item 流行度着色（高亮 Head vs Tail items），直观展示融合前后的空间变化。

用法：
    conda activate tiger_t5
    cd /data/gtx/project/code/TIGER_minilm
    python3 tools/tsne_visualize.py --dataset Toys
    python3 tools/tsne_visualize.py --dataset Beauty
    python3 tools/tsne_visualize.py --dataset Sport
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无头模式，服务器上运行
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})


# ============================================================
# 数据加载
# ============================================================
def load_emb(emb_path: str):
    """加载 embedding parquet 文件，返回 (item_ids, embeddings)"""
    df = pd.read_parquet(emb_path)
    ids = df["ItemID"].values
    embs = np.stack(df["embedding"].values)
    return ids, embs


def load_item_popularity(train_path: str):
    """从训练数据加载 item 流行度"""
    train_df = pd.read_parquet(train_path)
    all_items = []
    for _, row in train_df.iterrows():
        hist = row["history"]
        all_items.extend(hist.tolist() if isinstance(hist, np.ndarray) else list(hist))
        all_items.append(int(row["target"]))
    unique, counts = np.unique(all_items, return_counts=True)
    freq_dict = dict(zip(unique, counts))

    # 分流行度等级
    sort_idx = np.argsort(-counts)
    sorted_items = unique[sort_idx]
    n = len(sorted_items)
    head_cut = max(1, int(n * 0.2))
    tail_cut = min(n - 1, int(n * 0.8))

    group_dict = {}
    for item in sorted_items[:head_cut]:
        group_dict[int(item)] = "head"
    for item in sorted_items[head_cut:tail_cut]:
        group_dict[int(item)] = "mid"
    for item in sorted_items[tail_cut:]:
        group_dict[int(item)] = "tail"

    return group_dict, freq_dict


# ============================================================
# t-SNE + 绘图
# ============================================================
def plot_tsne(emb: np.ndarray, item_ids: np.ndarray, group_dict: dict,
              title: str, save_path: str, n_sample: int = 2000):
    """
    t-SNE 降维 + 散点图，按流行度着色。

    如果 item 数超过 n_sample，则随机采样。
    """
    n_total = emb.shape[0]
    if n_total > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, n_sample, replace=False)
        emb = emb[idx]
        item_ids = item_ids[idx]
        print(f"  ℹ️  采样 {n_sample}/{n_total} 个点进行 t-SNE")

    print(f"  🔄 t-SNE  降维到 2D ... ", end="", flush=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000,
                verbose=0, learning_rate="auto")
    emb_2d = tsne.fit_transform(emb)
    print("done")

    # 着色
    colors = {"head": "#e74c3c", "mid": "#f39c12", "tail": "#3498db"}
    labels = {"head": "Head (top 20%)", "mid": "Mid (60%)", "tail": "Tail (bottom 20%)"}
    point_groups = {"head": [], "mid": [], "tail": []}

    for i, iid in enumerate(item_ids):
        g = group_dict.get(int(iid), "tail")
        point_groups[g].append(emb_2d[i])

    fig, ax = plt.subplots(figsize=(7, 6))
    for g in ["tail", "mid", "head"]:  # 按此顺序绘制（Tail 先画在最底层）
        pts = np.array(point_groups[g])
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=colors[g], label=labels[g],
                       alpha=0.6, s=8, edgecolors="none")

    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="TA-SID t-SNE 可视化")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Beauty", "Sport", "Toys"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="doc/figures",
                        help="图片输出目录")
    parser.add_argument("--n_sample", type=int, default=2000,
                        help="t-SNE 采样点数（控制运行时间）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = args.dataset
    data_path = os.path.join(args.data_dir, ds)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"t-SNE 可视化: {ds}")
    print("=" * 60)

    # 检查文件
    content_path = os.path.join(data_path, "item_emb.parquet")
    behavior_path = os.path.join(data_path, "behavior_emb.parquet")
    fused_path = os.path.join(data_path, "fused_item_emb.parquet")
    train_path = os.path.join(data_path, "train.parquet")

    for p in [content_path, behavior_path, fused_path, train_path]:
        if not os.path.exists(p):
            print(f"❌ 缺失: {p}")
            sys.exit(1)

    # 加载流行度信息
    print("\n[1/4] 加载 item 流行度信息 ... ", end="", flush=True)
    group_dict, freq_dict = load_item_popularity(train_path)
    print(f"✅ ({len(group_dict)} items)")

    # 加载各种 embedding
    print("\n[2/4] 加载 embedding ...")
    content_ids, content_emb = load_emb(content_path)
    print(f"  Content:   {content_emb.shape} (dim={content_emb.shape[1]})")
    behavior_ids, behavior_emb = load_emb(behavior_path)
    print(f"  Behavior:  {behavior_emb.shape} (dim={behavior_emb.shape[1]})")
    fused_ids, fused_emb = load_emb(fused_path)
    print(f"  Fused:     {fused_emb.shape} (dim={fused_emb.shape[1]})")

    # 取三者共有的 item IDs（不同 embedding 的 item 集合可能不同）
    common_ids = set(content_ids) & set(behavior_ids) & set(fused_ids)
    print(f"  Common items: {len(common_ids)}")

    def filter_emb(ids, emb, common_ids):
        mask = np.array([iid in common_ids for iid in ids])
        return ids[mask], emb[mask]

    content_ids, content_emb = filter_emb(content_ids, content_emb, common_ids)
    behavior_ids, behavior_emb = filter_emb(behavior_ids, behavior_emb, common_ids)
    fused_ids, fused_emb = filter_emb(fused_ids, fused_emb, common_ids)

    # 绘制 3 张 t-SNE
    print("\n[3/4] 绘制 Content t-SNE ...")
    plot_tsne(
        content_emb, content_ids, group_dict,
        title=f"{ds} — Content Embedding (t-SNE)",
        save_path=os.path.join(args.output_dir, f"{ds.lower()}_tsne_content.png"),
        n_sample=args.n_sample,
    )

    print("\n[4/4] 绘制 Behavior t-SNE ...")
    plot_tsne(
        behavior_emb, behavior_ids, group_dict,
        title=f"{ds} — Behavior Embedding (t-SNE)",
        save_path=os.path.join(args.output_dir, f"{ds.lower()}_tsne_behavior.png"),
        n_sample=args.n_sample,
    )

    print("\n[5/4] 绘制 Fused t-SNE ...")
    plot_tsne(
        fused_emb, fused_ids, group_dict,
        title=f"{ds} — Fused Embedding (t-SNE)",
        save_path=os.path.join(args.output_dir, f"{ds.lower()}_tsne_fused.png"),
        n_sample=args.n_sample,
    )

    # ========== 统计摘要 ==========
    print("\n" + "=" * 60)
    print("embedding 范数统计")
    print("=" * 60)
    for name, emb in [("Content", content_emb), ("Behavior", behavior_emb), ("Fused", fused_emb)]:
        norms = np.linalg.norm(emb, axis=1)
        print(f"  {name:10s}: mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")

    print(f"\n✅ 全部完成! 图片保存在 {args.output_dir}/")
    print(f"   {ds.lower()}_tsne_content.png")
    print(f"   {ds.lower()}_tsne_behavior.png")
    print(f"   {ds.lower()}_tsne_fused.png")


if __name__ == "__main__":
    main()
