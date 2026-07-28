#!/usr/bin/env python3
"""
TA-SID 冷启动分析工具

评估不同频率分组的 item 的预测表现，生成冷启动曲线。
与长尾分析的区别：使用灵活的百分位分组（bottom 5%/10%/20%），
输出性能随频率下降的趋势数据。

不需要重新训练 T5，直接使用已训好的 T5 模型 + TA-SID codes
对指定数据集做推理，统计各组 Recall/NDCG。

用法：
    conda activate tiger_t5
    cd /data/gtx/project/code/TIGER_minilm
    # TA-SID 冷启动
    python3 tools/run_coldstart.py --dataset Toys

对比 Baseline 冷启动:
    python3 tools/run_coldstart.py --dataset Toys --code_path data/Toys/Toys_t5_rqvae.npy \\
        --model_path model/ckpt/toys.pth
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# 项目模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
import dataset as ds_module
import main as tiger_module


# ============================================================
# Item 频率分组（支持灵活百分位）
# ============================================================
def assign_groups_by_percentile(train_path: str, cold_percentiles: list):
    """
    从训练数据统计 item 频率，为每个 cold_percentile 创建分组。

    参数:
        cold_percentiles: list of float, 如 [0.05, 0.10, 0.20]
            → 分别将最冷门的 5%/10%/20% item 标记为 "cold"

    返回:
        group_layers: dict, percentile → item_to_group (dict)
        stats: dict, 分组统计信息
    """
    train_df = pd.read_parquet(train_path)

    all_items = []
    for _, row in train_df.iterrows():
        hist = row["history"]
        all_items.extend(hist.tolist() if isinstance(hist, np.ndarray) else list(hist))
        all_items.append(int(row["target"]))

    unique, counts = np.unique(all_items, return_counts=True)
    sort_idx = np.argsort(-counts)
    sorted_items = unique[sort_idx]
    sorted_counts = counts[sort_idx]

    n = len(sorted_items)

    # 分组：bottom X% = 频率最低的 X%
    # sorted_items 按频率降序：sorted_items[0] 频率最高，sorted_items[-1] 频率最低
    group_layers = {}
    stats = {}
    for pct in cold_percentiles:
        item_to_group = {}
        cold_cut = max(1, n - int(n * pct))  # 从低频端取 pct
        if cold_cut >= n:  # 没有 cold items（极端小数据集）
            cold_cut = n - 1
        for i in range(cold_cut, n):
            item_to_group[int(sorted_items[i])] = "cold"
        for i in range(cold_cut):
            item_to_group[int(sorted_items[i])] = "warm"

        cold_items = n - cold_cut
        warm_items = cold_cut
        stats[pct] = {
            "cold_count": cold_items,
            "cold_freq_range": f"{sorted_counts[min(n-1, cold_cut)]}-{sorted_counts[-1]}" if cold_items > 0 else "N/A",
            "warm_count": warm_items,
            "warm_freq_range": f"{sorted_counts[0]}-{sorted_counts[cold_cut-1]}" if warm_items > 0 else "N/A",
        }
        group_layers[pct] = item_to_group

    return group_layers, stats, sorted_items, sorted_counts


# ============================================================
# 分群评估
# ============================================================
def evaluate_by_group(model, eval_loader, item_to_group, topk_list, beam_size, device):
    """运行评估，分 warm/cold 计算指标"""
    model.eval()

    group_positions = {"warm": [], "cold": []}

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            raw_target_ids = batch["raw_target_ids"].numpy()

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
            )
            preds = preds[:, 1:]
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)

            pos_index = tiger_module.calculate_pos_index(preds, labels, maxk=beam_size)

            for i, raw_id in enumerate(raw_target_ids):
                group = item_to_group.get(int(raw_id), "unknown")
                if group in group_positions:
                    group_positions[group].append(pos_index[i:i+1])

    results = {}
    for group in ["warm", "cold"]:
        if len(group_positions[group]) == 0:
            results[group] = {"count": 0}
            continue
        group_pos = torch.cat(group_positions[group], dim=0)
        n = group_pos.shape[0]
        result = {"count": n}
        for k in topk_list:
            result[f"Recall@{k}"] = float(tiger_module.recall_at_k(group_pos, k).mean().item())
            result[f"NDCG@{k}"] = float(tiger_module.ndcg_at_k(group_pos, k).mean().item())
        results[group] = result

    return results


# ============================================================
# 数据集（与 longtail_analysis.py 共用）
# ============================================================
class ColdStartDataset(Dataset):
    """保留原始 item ID 的评估数据集"""
    def __init__(self, data_path, item_to_code, max_len=20, PAD_TOKEN=0):
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.item_to_code = item_to_code

        df = pd.read_parquet(data_path)
        self.data = []
        missing = 0
        for _, row in df.iterrows():
            seq = list(row["history"]) + [row["target"]]
            history = seq[:-1]
            target = seq[-1]

            if int(target) not in self.item_to_code:
                missing += 1
                continue

            raw_target_id = int(target)

            sid_history = []
            for h in history[:self.max_len]:
                h_int = int(h)
                if h_int in self.item_to_code:
                    sid_history.append(self.item_to_code[h_int])
                else:
                    sid_history.append(np.array([PAD_TOKEN]*4, dtype=np.int64))

            while len(sid_history) < self.max_len:
                sid_history.insert(0, np.array([PAD_TOKEN]*4, dtype=np.int64))
            if len(sid_history) > self.max_len:
                sid_history = sid_history[-self.max_len:]

            target_code = self.item_to_code[int(target)]

            self.data.append({
                "history": sid_history,
                "target_code": target_code,
                "raw_target_id": raw_target_id,
            })

        if missing:
            print(f"  ⚠ {missing} samples skipped (target has no SID)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_for_coldstart(batch, PAD_TOKEN=0):
    """自定义 collate：展平 SID tokens + 保留 raw_target_id"""
    histories = [item["history"] for item in batch]
    targets = [item["target_code"] for item in batch]
    raw_ids = [item["raw_target_id"] for item in batch]

    flattened_hists = []
    for h in histories:
        flat = []
        for sid in h:
            flat.extend(sid.tolist() if hasattr(sid, 'tolist') else list(sid))
        flattened_hists.append(flat)

    max_hist_len = max(len(h) for h in flattened_hists)
    padded_hists = []
    attention_masks = []
    for h in flattened_hists:
        pad_len = max_hist_len - len(h)
        padded = [PAD_TOKEN] * pad_len + h
        mask = [0] * pad_len + [1] * len(h)
        padded_hists.append(padded)
        attention_masks.append(mask)

    return {
        "history": torch.tensor(padded_hists, dtype=torch.int64),
        "target": torch.tensor(targets, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.int64),
        "raw_target_ids": torch.tensor(raw_ids, dtype=torch.int64),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="TA-SID 冷启动分析")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Beauty", "Sport", "Toys"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--code_path", type=str, default=None,
                        help="SID 码文件路径")
    parser.add_argument("--model_path", type=str, default=None,
                        help="T5 模型 checkpoint 路径")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--beam_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--cold_percentiles", type=str, default="0.05,0.10,0.20",
                        help="冷启动百分位阈值，逗号分隔")
    args = parser.parse_args()

    cold_percentiles = sorted([float(p.strip()) for p in args.cold_percentiles.split(",")])

    # 自动检测模型路径
    if args.model_path is None:
        ds_lower = args.dataset.lower()
        candidates = [
            f"model/ckpt/tiger_ta_sid_{ds_lower}.pth",
            f"model/ckpt/{ds_lower}.pth",
            f"model/ckpt/{ds_lower}_t5.pth",
        ]
        # 按优先级选择
        ta_sid_candidates = [c for c in candidates if "ta_sid" in c]
        for c in ta_sid_candidates:
            if os.path.exists(c):
                args.model_path = c
                break
        if args.model_path is None:
            for c in candidates:
                if os.path.exists(c):
                    args.model_path = c
                    break
        if args.model_path is None:
            available = sorted([f for f in os.listdir("model/ckpt")
                               if args.dataset.lower() in f.lower() and f.endswith(".pth")])
            if available:
                args.model_path = f"model/ckpt/{available[0]}"
            else:
                print("❌ 请指定 --model_path")
                sys.exit(1)
        print(f"ℹ️ 自动选择模型: {args.model_path}")

    # 自动检测 code path
    if args.code_path is None:
        ds_lower = args.dataset.lower()
        candidates = [
            f"data/{args.dataset}/{args.dataset}_ta_sid.npy",
            f"data/{args.dataset}/{ds_lower}_ta_sid.npy",
            f"data/{args.dataset}/{args.dataset}_t5_rqvae.npy",
            f"data/{args.dataset}/{ds_lower}_t5_rqvae.npy",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.code_path = c
                break
        if args.code_path is None:
            available = sorted([f for f in os.listdir(f"data/{args.dataset}")
                               if f.endswith(".npy")])
            if available:
                args.code_path = f"data/{args.dataset}/{available[0]}"
            else:
                print("❌ 请指定 --code_path")
                sys.exit(1)
        print(f"ℹ️ 自动选择 SID: {args.code_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 设备: {device}")
    print(f"📂 数据集: {args.dataset}")
    print(f"📄 SID: {args.code_path}")
    print(f"🤖 模型: {args.model_path}")
    print(f"📊 冷启动百分位: {cold_percentiles}")

    # Step 1: Item 频率分组
    train_path = os.path.join(args.data_dir, args.dataset, "train.parquet")
    print("\n[1/4] Item 频率分组 ... ", end="", flush=True)
    group_layers, group_stats, sorted_items, sorted_counts = assign_groups_by_percentile(
        train_path, cold_percentiles
    )
    print("✅")
    for pct, info in group_stats.items():
        print(f"    Cold < {pct*100:.0f}%: {info['cold_count']:5d} items [{info['cold_freq_range']}]")

    # Step 2: 加载 T5 模型
    print("\n[2/4] 加载 T5 模型 ... ", end="", flush=True)
    model_config = {
        "num_layers": 4, "num_decoder_layers": 4,
        "d_model": 128, "d_ff": 1024, "num_heads": 6,
        "d_kv": 64, "dropout_rate": 0.1, "vocab_size": 1025,
        "pad_token_id": 0, "eos_token_id": 0,
        "feed_forward_proj": "relu",
    }
    model = tiger_module.TIGER(model_config)
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    print("✅")

    # Step 3: 加载 item → code 映射
    print("\n[3/4] 加载 SID 映射... ", end="", flush=True)
    item_to_code, _ = ds_module.item2code(args.code_path)
    print(f"✅ ({len(item_to_code)} items)")

    # Step 4: 加载测试数据 + 逐百分位评估
    test_path = os.path.join(args.data_dir, args.dataset, "test.parquet")
    print("\n[4/4] 加载测试数据 + 冷启动评估 ...")
    test_dataset = ColdStartDataset(test_path, item_to_code, args.max_len)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_for_coldstart
    )
    print(f"    Test samples: {len(test_dataset)}")

    topk_list = [5, 10, 20]
    all_results = {}

    for pct in cold_percentiles:
        item_to_group = group_layers[pct]
        print(f"\n── Cold < {pct*100:.0f}% ──")
        results = evaluate_by_group(model, test_loader, item_to_group, topk_list, args.beam_size, device)
        all_results[pct] = results

        for group in ["warm", "cold"]:
            r = results.get(group)
            if r is None or r["count"] == 0:
                print(f"  {group:6s}: {0:>6d} samples — 无数据")
                continue
            print(f"  {group:6s}: {r['count']:>6d} samples", end="")
            for k in [5, 10, 20]:
                print(f"  R@{k}={r[f'Recall@{k}']:.4f}", end="")
            for k in [5, 10, 20]:
                print(f"  N@{k}={r[f'NDCG@{k}']:.4f}", end="")
            print()

    # ========== 汇总表 ==========
    print("\n" + "=" * 80)
    print("冷启动分析汇总表")
    print("=" * 80)

    # 表头
    col_w = 10
    header = f"{'Threshold':>10s} | {'Group':>6s} | {'Count':>8s}"
    for k in [5, 10, 20]:
        header += f" | {'R@'+str(k):>7s}"
    for k in [5, 10, 20]:
        header += f" | {'N@'+str(k):>7s}"
    print(header)
    print("-" * len(header))

    for pct in cold_percentiles:
        r = all_results[pct]
        for group in ["warm", "cold"]:
            gr = r.get(group)
            if gr is None or gr["count"] == 0:
                print(f"{pct*100:>9.0f}% | {group:>6s} | {'-':>8s}")
                continue
            line = f"{pct*100:>9.0f}% | {group:>6s} | {gr['count']:>8d}"
            for k in [5, 10, 20]:
                line += f" | {gr.get(f'Recall@{k}', 0):.4f}"
            for k in [5, 10, 20]:
                line += f" | {gr.get(f'NDCG@{k}', 0):.4f}"
            print(line)

    # ========== 冷启动退化曲线 ==========
    print("\n" + "─" * 80)
    print("冷启动退化曲线（Cold Performance vs Warm Baseline）")
    print("─" * 80)
    print(f"{'Threshold':>10s} | {'Metric':>8s} | {'Warm':>8s} | {'Cold':>8s} | {'Drop':>8s} | {'Drop%':>8s}")
    print("-" * 68)

    for pct in cold_percentiles:
        r = all_results[pct]
        warm_r = r.get("warm", {})
        cold_r = r.get("cold", {})

        if warm_r.get("count", 0) == 0 or cold_r.get("count", 0) == 0:
            continue

        for k in [5, 10, 20]:
            warm_val = warm_r.get(f"Recall@{k}", 0)
            cold_val = cold_r.get(f"Recall@{k}", 0)
            if warm_val > 0:
                drop = warm_val - cold_val
                drop_pct = (cold_val / warm_val - 1) * 100
                print(f"{pct*100:>9.0f}% | {'R@'+str(k):>8s} | {warm_val:.4f} | {cold_val:.4f} | {drop:.4f} | {drop_pct:>+7.1f}%")

        for k in [5, 10, 20]:
            warm_val = warm_r.get(f"NDCG@{k}", 0)
            cold_val = cold_r.get(f"NDCG@{k}", 0)
            if warm_val > 0:
                drop = warm_val - cold_val
                drop_pct = (cold_val / warm_val - 1) * 100
                print(f"{pct*100:>9.0f}% | {'N@'+str(k):>8s} | {warm_val:.4f} | {cold_val:.4f} | {drop:.4f} | {drop_pct:>+7.1f}%")

    print("\n💡 解读: Drop 越大 = 冷启动问题越严重。")
    print(f"   如果 Cold/Warm 差距随着 threshold 增大而增大 → 频率越低，模型越难预测。")
    print(f"\n   对比 Baseline:")
    print(f"     python3 tools/run_coldstart.py --dataset {args.dataset} \\")
    print(f"       --code_path data/{args.dataset}/{args.dataset}_t5_rqvae.npy \\")
    print(f"       --model_path model/ckpt/{args.dataset.lower()}.pth")


if __name__ == "__main__":
    main()
