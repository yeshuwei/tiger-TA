#!/usr/bin/env python3
"""
TA-SID Semantic ID 质量分析工具

计算四个质量指标：
1. Codebook utilization — 每层 codebook 有多少 codeword 被使用
2. Collision rate — 最终 SID 唯一比例
3. Invalid ID rate — T5 生成的 SID 不在有效集合中的比例（从日志解析）
4. Transition consistency — 高频转移对的量化向量余弦相似度

用法：
    conda activate tiger_t5
    cd /data/gtx/project/code/TIGER_minilm
    python3 tools/sid_quality.py [--datasets Beauty,Sport,Toys]
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import torch
import argparse
from scipy.spatial.distance import cosine


# ============================================================
# 配置
# ============================================================
DATA_DIR = "data"
CKPT_DIR = "rqvae/ckpt"
LOG_DIR = "model/logs"

DATASETS = {
    "Beauty": {
        "data_dir": "Beauty",
        "ckpt_tag": "Beauty_TA_SID",
        "npy_file": "Beauty_ta_sid.npy",
        "fused_emb": "fused_item_emb.parquet",
        "baseline_npy": None,  # 没有
    },
    "Sport": {
        "data_dir": "Sport",
        "ckpt_tag": "Sport_TA_SID",
        "npy_file": "Sport_ta_sid.npy",
        "fused_emb": "fused_item_emb.parquet",
        "baseline_npy": None,
    },
    "Toys": {
        "data_dir": "Toys",
        "ckpt_tag": "Toys_TA_SID",
        "npy_file": "Toys_ta_sid.npy",
        "fused_emb": "fused_item_emb.parquet",
        "baseline_npy": None,
    },
}


def find_best_ckpt(ds_config):
    """找到指定数据集的最佳 RQ-VAE checkpoint"""
    ckpt_dir = os.path.join(CKPT_DIR, ds_config["ckpt_tag"])
    if not os.path.exists(ckpt_dir):
        return None
    for root, dirs, files in os.walk(ckpt_dir):
        if "best_collision_model.pth" in files:
            return os.path.join(root, "best_collision_model.pth")
    return None


def compute_utilization(model_path: str, fused_emb_path: str, device: str = "cpu"):
    """
    计算 codebook utilization（每层有多少 codeword 被使用）。

    加载 RQ-VAE checkpoint → 对所有 item 编码 → 统计每层的 codeword 使用计数。
    """
    if not os.path.exists(model_path):
        return None
    if not os.path.exists(fused_emb_path):
        return None

    sys.path.insert(0, "rqvae")
    from models.rqvae import RQVAE
    from datasets import EmbDataset

    # 加载 checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # 加载数据
    data = EmbDataset(fused_emb_path)

    # 初始化模型
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 编码所有 item
    all_data = data[:]
    if isinstance(all_data, list):
        all_data = torch.stack(all_data)
    elif isinstance(all_data, np.ndarray):
        all_data = torch.from_numpy(all_data)
    all_data = all_data.float()

    with torch.no_grad():
        _, _, all_indices = model(all_data, use_sk=False)
        # all_indices: (N, num_layers) — 每个 item 每层的 codeword 索引

    n_layers = all_indices.shape[-1]
    utilization = {}
    for layer in range(n_layers):
        indices = all_indices[:, layer].cpu().numpy()
        unique_used = len(np.unique(indices))
        total = args.num_emb_list[layer]
        utilization[f"Layer_{layer+1}"] = {
            "total": total,
            "used": int(unique_used),
            "utilization": unique_used / total,
        }

    return utilization


def compute_collision_rate(npy_path: str):
    """
    计算 SID 碰撞率。

    碰撞率 = 1 - (unique SID 数 / 总 item 数)
    """
    if not os.path.exists(npy_path):
        return None

    codes = np.load(npy_path)
    unique_codes = np.unique(codes, axis=0)
    n_total = codes.shape[0]
    n_unique = unique_codes.shape[0]
    collision_rate = 1.0 - n_unique / n_total

    return {
        "total_items": n_total,
        "unique_sids": n_unique,
        "collision_rate": collision_rate,
    }


def compute_transition_consistency(ds_name: str, fused_emb_path: str):
    """
    计算 Transition Consistency。

    从 transition matrix 中找出高频转移对，计算它们量化向量的余弦相似度均值。

    高分 → 经常一起出现的商品在编码空间中也靠近。
    """
    if not os.path.exists(fused_emb_path):
        return None

    # 尝试加载 transition matrix 或 behavior_emb
    train_path = os.path.join(DATA_DIR, ds_name, "train.parquet")
    if not os.path.exists(train_path):
        return None

    # 方法：从 fused_emb 中加载向量，从 behavior_emb 或重建 transition pairs
    fused_df = pd.read_parquet(fused_emb_path)
    item_ids = fused_df["ItemID"].values
    embeddings = np.stack(fused_df["embedding"].values)

    # 从用户序列统计高频转移对
    train_df = pd.read_parquet(train_path)
    pair_counts = {}
    for seq in train_df["history"]:
        seq_list = seq if isinstance(seq, (list, np.ndarray)) else [seq]
        for i in range(len(seq_list) - 1):
            a, b = int(seq_list[i]), int(seq_list[i + 1])
            if a > b:
                a, b = b, a
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    if not pair_counts:
        return None

    # 取 top-1000 高频转移对
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
    top_pairs = sorted_pairs[:1000]

    # 过滤掉 count <= 1 的噪声
    top_pairs = [(p, c) for p, c in top_pairs if c > 1]
    if not top_pairs:
        top_pairs = sorted_pairs[:500]

    # 建 item_id → index 映射
    id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}

    # 计算每对的余弦相似度
    similarities = []
    valid_pairs = 0
    for (a, b), count in top_pairs:
        if a in id_to_idx and b in id_to_idx:
            emb_a = embeddings[id_to_idx[a]]
            emb_b = embeddings[id_to_idx[b]]
            sim = 1.0 - cosine(emb_a, emb_b)
            similarities.append((a, b, count, sim))
            valid_pairs += 1

    if not similarities:
        return None

    sim_values = [s[3] for s in similarities]
    avg_sim = np.mean(sim_values)

    return {
        "total_pairs_considered": len(top_pairs),
        "valid_pairs": valid_pairs,
        "avg_cosine_sim": float(avg_sim),
        "std_cosine_sim": float(np.std(sim_values)),
        "top5_pairs": [(int(a), int(b), c, f"{s:.4f}") for (a, b, c, s) in similarities[:5]],
    }


def compute_invalid_id_rate(ds_name: str):
    """
    计算 Invalid ID Rate。

    从 T5 日志中尝试解析无效 ID 相关信息。
    如果日志中没有，则返回提示信息。
    """
    # 查找对应日志文件
    ds_dirs = {
        "Beauty": "tiger_ta_sid.log",
        "Sport": "tiger_ta_sid_sport.log",
        "Toys": "tiger_ta_sid_toys.log",
    }

    log_file = os.path.join(LOG_DIR, ds_dirs.get(ds_name, ""))
    if not os.path.exists(log_file):
        return None

    # 搜索 invalid 相关关键字
    with open(log_file, "r") as f:
        content = f.read()

    invalid_lines = re.findall(r"invalid.*?(?:rate|ratio|count).*?([0-9.]+)", content, re.IGNORECASE)
    if invalid_lines:
        return {"invalid_rate": float(invalid_lines[-1]), "source": "log"}

    return {"message": "日志中未记录 invalid ID rate，需要修改 eval 函数添加此指标", "source": "unavailable"}


def main():
    parser = argparse.ArgumentParser(description="TA-SID Semantic ID 质量分析")
    parser.add_argument("--datasets", type=str, default="Beauty,Sport,Toys",
                        help="数据集列表，逗号分隔")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备 (cpu/cuda)")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    print("=" * 70)
    print("TA-SID Semantic ID 质量分析报告")
    print("=" * 70)

    for ds_name in datasets:
        if ds_name not in DATASETS:
            print(f"\n⚠ 跳过未知数据集: {ds_name}")
            continue

        ds_config = DATASETS[ds_name]
        print(f"\n{'─' * 70}")
        print(f"📊 {ds_name}")
        print(f"{'─' * 70}")

        # 1. Codebook Utilization
        print("\n[1/4] Codebook Utilization ... ", end="", flush=True)
        ckpt_path = find_best_ckpt(ds_config)
        if ckpt_path is None:
            print("❌ 未找到 checkpoint")
        else:
            fused_path = os.path.join(DATA_DIR, ds_config["data_dir"], ds_config["fused_emb"])
            util = compute_utilization(ckpt_path, fused_path, device)
            if util is None:
                print("❌ 计算失败")
            else:
                print("✅")
                for layer, info in util.items():
                    bar = "█" * int(info["utilization"] * 40) + "░" * (40 - int(info["utilization"] * 40))
                    print(f"     {layer}: {info['used']:3d}/{info['total']:3d} ({info['utilization']:.1%}) {bar}")

        # 2. Collision Rate
        print("\n[2/4] Collision Rate ... ", end="", flush=True)
        npy_path = os.path.join(DATA_DIR, ds_config["data_dir"], ds_config["npy_file"])
        collision = compute_collision_rate(npy_path)
        if collision is None:
            print("❌ 未找到 SID 文件")
        else:
            print(f"✅  {collision['total_items']} items → {collision['unique_sids']} unique SIDs")
            if collision['collision_rate'] == 0:
                print(f"     Collision Rate: 0.0000 ✅ (完全无碰撞)")
            else:
                print(f"     Collision Rate: {collision['collision_rate']:.4f} ⚠")

        # 3. Invalid ID Rate
        print("\n[3/4] Invalid ID Rate ... ", end="", flush=True)
        invalid = compute_invalid_id_rate(ds_name)
        if invalid is None:
            print("❌ 未找到日志文件")
        elif "invalid_rate" in invalid:
            print(f"✅  {invalid['invalid_rate']:.4f}")
        else:
            print(f"ℹ️  {invalid['message']}")

        # 4. Transition Consistency
        print("\n[4/4] Transition Consistency ... ", end="", flush=True)
        fused_path = os.path.join(DATA_DIR, ds_config["data_dir"], ds_config["fused_emb"])
        tc = compute_transition_consistency(ds_name, fused_path)
        if tc is None:
            print("❌ 计算失败（缺少数据或序列太短）")
        else:
            print(f"✅  {tc['valid_pairs']} valid pairs")
            print(f"     Avg Cosine Similarity: {tc['avg_cosine_sim']:.4f} ± {tc['std_cosine_sim']:.4f}")
            print(f"     Top-5 transition pairs:")
            for a, b, c, s in tc.get("top5_pairs", []):
                print(f"       Item {a:5d} ↔ Item {b:5d}  (count={c:3d})  cos={s}")

        print()

    # 汇总表
    print("\n" + "=" * 70)
    print("质量指标汇总表")
    print("=" * 70)
    print(f"{'Dataset':12s} {'Util(L1)':>10s} {'Util(L2)':>10s} {'Util(L3)':>10s} "
          f"{'Collision':>12s} {'InvalidID':>10s} {'TransCos':>10s}")
    print("-" * 70)

    for ds_name in datasets:
        if ds_name not in DATASETS:
            continue
        ds_config = DATASETS[ds_name]

        # Utilization
        ckpt_path = find_best_ckpt(ds_config)
        util_str = ["—", "—", "—"]
        if ckpt_path:
            fused_path = os.path.join(DATA_DIR, ds_config["data_dir"], ds_config["fused_emb"])
            util = compute_utilization(ckpt_path, fused_path, device)
            if util:
                for i in range(3):
                    key = f"Layer_{i+1}"
                    if key in util:
                        util_str[i] = f"{util[key]['utilization']:.1%}"

        # Collision
        npy_path = os.path.join(DATA_DIR, ds_config["data_dir"], ds_config["npy_file"])
        collision_str = "—"
        collision = compute_collision_rate(npy_path)
        if collision:
            collision_str = f"{collision['collision_rate']:.4f}"

        # Invalid
        invalid_str = "—"
        invalid = compute_invalid_id_rate(ds_name)
        if invalid and "invalid_rate" in invalid:
            invalid_str = f"{invalid['invalid_rate']:.4f}"

        # Transition Consistency
        tc_str = "—"
        fused_path = os.path.join(DATA_DIR, ds_config["data_dir"], ds_config["fused_emb"])
        tc = compute_transition_consistency(ds_name, fused_path)
        if tc:
            tc_str = f"{tc['avg_cosine_sim']:.4f}"

        print(f"{ds_name:12s} {util_str[0]:>10s} {util_str[1]:>10s} {util_str[2]:>10s} "
              f"{collision_str:>12s} {invalid_str:>10s} {tc_str:>10s}")

    print("\n💡 提示：Invalid ID Rate 需要修改 T5 evaluation 函数记录无效 SID 数量。")
    print("   当前在 evaluation 中生成的 SID 未做无效性校验，日志中无此数据。")
    print("   如需此指标，需要在 model/main.py 的 evaluate() 中添加校验逻辑。\n")


if __name__ == "__main__":
    main()
