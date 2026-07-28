#!/usr/bin/env python3
"""
TA-SID 长尾分析工具

按 item 出现频率分三组（Head/Mid/Tail），分别计算 Recall/NDCG。
需要 GPU，加载 T5 模型在测试集上评估。

用法：
    conda activate tiger_t5
    cd /data/gtx/project/code/TIGER_minilm
    # TA-SID 评估
    python3 tools/longtail_analysis.py --dataset Toys --code_path data/Toys/Toys_ta_sid.npy
    # Baseline 评估
    python3 tools/longtail_analysis.py --dataset Toys --code_path data/Toys/Toys_t5_rqvae.npy
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
import dataset as ds_module  # item2code, pad_or_truncate
import main as tiger_module  # TIGER, calculate_pos_index, recall_at_k, ndcg_at_k


# ============================================================
# Item 分群
# ============================================================
def assign_groups(train_path: str, head_ratio: float = 0.2, tail_ratio: float = 0.2):
    """
    从训练数据统计 item 频率，将 item 分为 Head/Mid/Tail 三组。
    item_id 是 0-based 索引（即 np.load 的 SID 码的索引）
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
    head_cut = max(1, int(n * head_ratio))
    tail_cut = min(n - 1, int(n * (1 - tail_ratio)))

    item_to_group = {}
    for item in sorted_items[:head_cut]:
        item_to_group[int(item)] = "head"
    for item in sorted_items[head_cut:tail_cut]:
        item_to_group[int(item)] = "mid"
    for item in sorted_items[tail_cut:]:
        item_to_group[int(item)] = "tail"

    group_stats = {
        "head": {"count": len(sorted_items[:head_cut]),
                 "freq_range": f"{sorted_counts[0]}-{sorted_counts[head_cut-1]}"},
        "mid": {"count": len(sorted_items[head_cut:tail_cut]),
                "freq_range": f"{sorted_counts[head_cut]}-{sorted_counts[tail_cut-1]}"},
        "tail": {"count": len(sorted_items[tail_cut:]),
                 "freq_range": f"{sorted_counts[tail_cut]}-{sorted_counts[-1]}"},
    }
    return item_to_group, group_stats


# ============================================================
# 分群评估
# ============================================================
def evaluate_by_group(model, eval_loader, item_to_group, topk_list, beam_size, device):
    """运行评估，分 group 计算指标"""
    model.eval()

    # 按 group 累积 pos_index
    group_positions = defaultdict(list)

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            raw_target_ids = batch["raw_target_ids"].numpy()  # 原始 item ID

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
                group_positions[group].append(pos_index[i:i+1])

    results = {}
    for group in ["head", "mid", "tail", "unknown"]:
        if group not in group_positions or len(group_positions[group]) == 0:
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
# 数据集：保留原始 item ID
# ============================================================
class GroupEvalDataset(Dataset):
    """与 GenRecDataset 类似，但额外保留 raw_target_id（原始 item ID）"""
    def __init__(self, data_path, item_to_code, max_len=20, PAD_TOKEN=0):
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.item_to_code = item_to_code

        # 读取 parquet
        df = pd.read_parquet(data_path)
        self.data = []
        missing = 0
        for _, row in df.iterrows():
            seq = list(row["history"]) + [row["target"]]
            history = seq[:-1]
            target = seq[-1]

            # 检查 target 是否有 SID
            if int(target) not in self.item_to_code:
                missing += 1
                continue

            # 保存原始 target ID（用于分组）
            raw_target_id = int(target)

            # 将 history 中的 item 转为 SID tokens（偏移后）
            sid_history = []
            for h in history[:self.max_len]:  # 先截断
                h_int = int(h)
                if h_int in self.item_to_code:
                    sid_history.append(self.item_to_code[h_int])
                else:
                    sid_history.append(np.array([PAD_TOKEN]*4, dtype=np.int64))

            # 左填充到 max_len
            while len(sid_history) < self.max_len:
                sid_history.insert(0, np.array([PAD_TOKEN]*4, dtype=np.int64))
            if len(sid_history) > self.max_len:
                sid_history = sid_history[-self.max_len:]

            # target 转 SID
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


def collate_for_group(batch, PAD_TOKEN=0):
    """自定义 collate：展平 SID tokens + 保留 raw_target_id"""
    histories = [item["history"] for item in batch]  # list of list of arrays
    targets = [item["target_code"] for item in batch]
    raw_ids = [item["raw_target_id"] for item in batch]

    # 展平 history: 每个 history 是 [max_len 个 SID], 每个 SID 是 4 个 token
    # 变成连续的一维序列
    flattened_hists = []
    for h in histories:
        flat = []
        for sid in h:
            flat.extend(sid.tolist() if hasattr(sid, 'tolist') else list(sid))
        flattened_hists.append(flat)

    # Padding 到统一长度
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
    parser = argparse.ArgumentParser(description="TA-SID 长尾分析")
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
    args = parser.parse_args()

    # 自动检测模型路径
    if args.model_path is None:
        candidates = [
            f"model/ckpt/tiger_ta_sid_{args.dataset.lower()}.pth",
            f"model/ckpt/{args.dataset.lower()}.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.model_path = c
                break
        if args.model_path is None:
            available = [f for f in os.listdir("model/ckpt") if args.dataset.lower() in f.lower() and f.endswith(".pth")]
            if available:
                args.model_path = f"model/ckpt/{available[0]}"
            else:
                print("❌ 请指定 --model_path")
                sys.exit(1)
        print(f"ℹ️ 自动选择模型: {args.model_path}")

    # 自动检测 code path
    if args.code_path is None:
        candidates = [
            f"data/{args.dataset}/{args.dataset}_ta_sid.npy",
            f"data/{args.dataset}/{args.dataset}_t5_rqvae.npy",
        ]
        for c in candidates:
            if os.path.exists(c):
                args.code_path = c
                break
        if args.code_path is None:
            print("❌ 请指定 --code_path")
            sys.exit(1)
        print(f"ℹ️ 自动选择 SID: {args.code_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🔧 设备: {device}")
    print(f"📂 数据集: {args.dataset}")
    print(f"📄 SID: {args.code_path}")
    print(f"🤖 模型: {args.model_path}")

    # Step 1: item 分群
    train_path = os.path.join(args.data_dir, args.dataset, "train.parquet")
    print("\n[1/4] 计算 item 频率分组 ... ", end="", flush=True)
    item_to_group, group_stats = assign_groups(train_path)
    print("✅")
    for g, info in group_stats.items():
        print(f"    {g:6s}: {info['count']:5d} items [{info['freq_range']}]")

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

    # Step 4: 加载测试数据 + 分群评估
    test_path = os.path.join(args.data_dir, args.dataset, "test.parquet")
    print("\n[4/4] 加载测试数据 + 分群评估 ...")
    test_dataset = GroupEvalDataset(test_path, item_to_code, args.max_len)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_for_group
    )
    print(f"    Test samples: {len(test_dataset)}")

    topk_list = [5, 10, 20]
    results = evaluate_by_group(model, test_loader, item_to_group, topk_list, args.beam_size, device)

    # 输出
    print("\n" + "=" * 70)
    print(f"长尾分析: {args.dataset} ({os.path.basename(args.code_path)})")
    print("=" * 70)
    header = f"{'Group':8s} {'Count':>8s}  {'R@5':>8s} {'R@10':>8s} {'R@20':>8s}  {'N@5':>8s} {'N@10':>8s} {'N@20':>8s}"
    print(header)
    print("-" * len(header))
    for g in ["head", "mid", "tail"]:
        r = results.get(g)
        if r is None or r["count"] == 0:
            print(f"{g:8s} {'-':>8s}")
            continue
        print(f"{g:8s} {r['count']:>8d}  {r['Recall@5']:.4f}  {r['Recall@10']:.4f}  "
              f"{r['Recall@20']:.4f}  {r['NDCG@5']:.4f}  {r['NDCG@10']:.4f}  {r['NDCG@20']:.4f}")

    print("\n💡 对比 Baseline 和 TA-SID:")
    print(f"   Baseline: python3 tools/longtail_analysis.py --dataset {args.dataset} --code_path data/{args.dataset}/{args.dataset}_t5_rqvae.npy")
    print(f"   TA-SID:   python3 tools/longtail_analysis.py --dataset {args.dataset} --code_path data/{args.dataset}/{args.dataset}_ta_sid.npy")


if __name__ == "__main__":
    main()
