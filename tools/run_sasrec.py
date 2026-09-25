#!/usr/bin/env python3
"""
SASRec 基线实验 v2 — 对齐 TA-SID 评估协议（修正 off-by-one）

相对 v1 (2026-07-29) 的修正：
  1. 修复 off-by-one：test 评估 = 从 test.parquet 的 history 预测 target，
     与 TA-SID 的最终测试集严格一致；val 评估 = 从 valid.parquet 预测，
     与 T5 的模型选择验证集一致。
  2. 全量用户评估：去掉 10,000 用户子集。
  3. 多 seed 支持：--seeds 2025 2026 2027，逐 seed 独立训练 + 汇总 mean±std。
  4. WarpSampler 按 seed 播种（worker = base_seed + i），保证同 seed 可复现。

数据切分（TIGER 滚动划分，L = 完整序列长度，target 全为用户真实交互）：
  train: history=[i_1..i_{L-3}], target=i_{L-2}
  valid: history=[i_1..i_{L-2}], target=i_{L-1}
  test : history=[i_1..i_{L-1}], target=i_L
TA-SID T5 最终评估 = test（从 [i_1..i_{L-1}] 预测 i_L）。
SASRec v1 的 "test" 实际是 valid 任务（从 [i_1..i_{L-2}] 预测 i_{L-1}），已修正。

用法：
    cd /data/gtx/project/code/TIGER_minilm
    conda activate tiger_t5
    python3 tools/run_sasrec.py --dataset Sport --seeds 2025 2026 2027
    python3 tools/run_sasrec.py --dataset all --seeds 2025 2026 2027

输出（sasrec_output/{ds}/）：
    seed{seed}_best.pth        每个 seed 的最佳 val 检查点
    seed{seed}_epochs.txt      逐 eval 步的 val/test 指标
    summary_3seed.txt          3-seed test 指标 + mean±std（论文用）

参考：
  - pmixer/SASRec.pytorch (https://github.com/pmixer/SASRec.pytorch)
  - Original SASRec: Kang & McAuley, ICDM 2018
"""

import os
import sys
import time
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# 1. 数据加载与适配
# ============================================================

def load_tiger_data(data_dir: str):
    """加载 TIGER 格式的 parquet 数据，重建 SASRec 格式。

    返回三个字典（user → list）：
      train   : train.parquet 的 history（SASRec 训练采样用）
      valid   : valid.parquet 的 target（T5 验证任务的目标）
      val_seq : valid.parquet 的 history（T5 验证任务的输入）
      test    : test.parquet 的 target（T5 测试任务的目标）
      test_seq: test.parquet 的 history（T5 测试任务的输入）
    """
    train = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    valid = pd.read_parquet(os.path.join(data_dir, "valid.parquet"))
    test = pd.read_parquet(os.path.join(data_dir, "test.parquet"))

    user_train = {}
    user_valid = {}
    user_val_seq = {}
    user_test = {}
    user_test_seq = {}

    itemnum = 0

    for _, row in train.iterrows():
        uid = int(row["user"])
        history = row["history"].tolist() if isinstance(row["history"], np.ndarray) else list(row["history"])
        target = int(row["target"])
        user_train[uid] = [int(x) for x in history]
        itemnum = max(itemnum, max(user_train[uid]) if user_train[uid] else 0, target)

    for _, row in valid.iterrows():
        uid = int(row["user"])
        history = row["history"].tolist() if isinstance(row["history"], np.ndarray) else list(row["history"])
        target = int(row["target"])
        user_val_seq[uid] = [int(x) for x in history]
        user_valid[uid] = [int(target)]
        itemnum = max(itemnum, target)

    for _, row in test.iterrows():
        uid = int(row["user"])
        history = row["history"].tolist() if isinstance(row["history"], np.ndarray) else list(row["history"])
        target = int(row["target"])
        user_test_seq[uid] = [int(x) for x in history]
        user_test[uid] = [int(target)]
        itemnum = max(itemnum, target)

    # 过滤：三个 split 都存在，且 val/test 序列非空
    common = set(user_train) & set(user_val_seq) & set(user_test_seq)
    common = {u for u in common
              if len(user_val_seq[u]) >= 1 and len(user_test_seq[u]) >= 1}

    filtered_train = {u: user_train[u] for u in common}
    filtered_val = {u: user_valid[u] for u in common}
    filtered_val_seq = {u: user_val_seq[u] for u in common}
    filtered_test = {u: user_test[u] for u in common}
    filtered_test_seq = {u: user_test_seq[u] for u in common}

    print(f"  Users: {len(filtered_train)} (filtered from {len(user_train)})")
    print(f"  Items: {itemnum}")
    print(f"  Avg train seq len: {np.mean([len(v) for v in filtered_train.values()]):.2f}")

    return (filtered_train, filtered_val, filtered_val_seq,
            filtered_test, filtered_test_seq, itemnum)


# ============================================================
# 2. SASRec 采样器（从 pmixer/SASRec.pytorch 移植）
# ============================================================

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        while len(user_train[uid]) < 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.array(list(user_train.keys()), dtype=np.int32)
    counter = 0
    while True:
        if counter % len(uids) == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % len(uids)]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler:
    """多进程数据采样器（worker 按 base_seed + i 播种，保证可复现）"""
    def __init__(self, user_train, usernum, itemnum, batch_size=128, maxlen=200,
                 n_workers=3, base_seed=0):
        from multiprocessing import Process, Queue
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(user_train, usernum, itemnum, batch_size, maxlen,
                              self.result_queue, base_seed + i))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# ============================================================
# 3. SASRec 模型（从 pmixer/SASRec.pytorch 移植）
# ============================================================

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        return outputs.transpose(-1, -2)


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            )
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
            seqs = self.attention_layernorms[i](seqs + mha_outputs)
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        """返回 logits，形状 (len(item_indices), B)"""
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


# ============================================================
# 4. 评估（全量物品 × 全量用户，分批）
# ============================================================

@torch.no_grad()
def evaluate_full(model, dataset, args, mode="test"):
    """
    全量物品评估（非 100 负采样），与 TA-SID 评估协议一致。

    mode="val":   seq = valid.parquet 的 history, target = valid.target
                  （与 T5 模型选择的验证集严格一致）
    mode="test":  seq = test.parquet 的 history,  target = test.target
                  （与 T5 最终测试集严格一致）
    """
    if mode == "val":
        seq_dict = dataset["val_seq"]
        target_dict = dataset["valid"]
    else:
        seq_dict = dataset["test_seq"]
        target_dict = dataset["test"]

    users = [u for u in dataset["train"]
             if u in seq_dict and u in target_dict
             and len(seq_dict[u]) >= 1 and len(target_dict[u]) >= 1]
    users = sorted(users)

    NDCG = {5: 0.0, 10: 0.0, 20: 0.0}
    HIT = {5: 0.0, 10: 0.0, 20: 0.0}
    itemnum = dataset["itemnum"]
    all_items = np.arange(1, itemnum + 1)

    eval_batch = 512
    for i in range(0, len(users), eval_batch):
        ub = users[i:i + eval_batch]
        seqs = np.zeros((len(ub), args.maxlen), dtype=np.int32)
        targets = np.zeros(len(ub), dtype=np.int32)
        for j, u in enumerate(ub):
            history = seq_dict[u][-args.maxlen:]
            seqs[j, args.maxlen - len(history):] = history
            targets[j] = target_dict[u][0]

        # logits 形状 (B, itemnum)（matmul 广播：item_embs (itemnum,hidden) × (B,hidden,1) → (B,itemnum,1)）
        logits = model.predict(np.array(ub), seqs, all_items)
        preds = logits.cpu().numpy()

        for j in range(len(ub)):
            rank = (-preds[j]).argsort().argsort()[targets[j] - 1]
            for k in [5, 10, 20]:
                if rank < k:
                    HIT[k] += 1
                    NDCG[k] += 1 / np.log2(rank + 2)

    n = len(users)
    for k in [5, 10, 20]:
        NDCG[k] /= max(n, 1)
        HIT[k] /= max(n, 1)

    return NDCG, HIT, n


# ============================================================
# 5. 单 seed 训练
# ============================================================

def run_seed(args, seed, dataset, out_dir):
    print(f"\n{'=' * 60}\n  {args.dataset} — seed {seed}\n{'=' * 60}", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = SASRec(dataset["usernum"], dataset["itemnum"], args).to(args.device)

    # Xavier 初始化
    for name, param in model.named_parameters():
        try:
            if param.dim() > 1:
                nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    sampler = WarpSampler(dataset["train"], dataset["usernum"], dataset["itemnum"],
                          batch_size=args.batch_size,
                          maxlen=args.maxlen,
                          n_workers=1 if args.device == "cpu" else 3,
                          base_seed=seed)

    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    epoch_log = os.path.join(out_dir, f"seed{seed}_epochs.txt")
    with open(epoch_log, "w") as f:
        f.write("epoch\tval_ndcg@20\ttest_ndcg@20\tval_recall@20\ttest_recall@20\n")

    num_batch = (len(dataset["train"]) - 1) // args.batch_size + 1
    best_val_ndcg = -1.0
    best_epoch = 0
    best_metrics = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        for step in range(num_batch):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param ** 2)
            loss.backward()
            optimizer.step()

        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            model.eval()
            print(f"\n  Epoch {epoch} ({(time.time() - t0) / 60:.1f} min):", flush=True)

            val_ndcg, val_recall, val_users = evaluate_full(model, dataset, args, "val")
            print(f"    Val:  N@5={val_ndcg[5]:.4f} N@10={val_ndcg[10]:.4f} N@20={val_ndcg[20]:.4f} "
                  f"R@20={val_recall[20]:.4f} ({val_users} users)", flush=True)

            test_ndcg, test_recall, test_users = evaluate_full(model, dataset, args, "test")
            print(f"    Test: N@5={test_ndcg[5]:.4f} N@10={test_ndcg[10]:.4f} N@20={test_ndcg[20]:.4f} "
                  f"R@20={test_recall[20]:.4f} ({test_users} users)", flush=True)

            with open(epoch_log, "a") as f:
                f.write(f"{epoch}\t{val_ndcg[20]:.6f}\t{test_ndcg[20]:.6f}\t"
                        f"{val_recall[20]:.6f}\t{test_recall[20]:.6f}\n")

            if val_ndcg[20] > best_val_ndcg:
                best_val_ndcg = val_ndcg[20]
                best_epoch = epoch
                best_metrics = (val_ndcg, val_recall, test_ndcg, test_recall)
                ckpt_path = os.path.join(out_dir, f"seed{seed}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                no_improve = 0
                print(f"    ⭐ Saved best checkpoint @ epoch {epoch}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"    ⏹ Early stop @ epoch {epoch} (no improvement for "
                          f"{args.patience} evals)", flush=True)
                    break

            t0 = time.time()

    sampler.close()
    return best_metrics, best_epoch


# ============================================================
# 6. 主流程
# ============================================================

METRIC_NAMES = ["NDCG@5", "NDCG@10", "NDCG@20", "Rec@5", "Rec@10", "Rec@20"]


def collect_metrics(best_metrics):
    """从 (val_ndcg, val_recall, test_ndcg, test_recall) 提取 test 6 个指标"""
    val_ndcg, val_recall, test_ndcg, test_recall = best_metrics
    return [test_ndcg[5], test_ndcg[10], test_ndcg[20],
            test_recall[5], test_recall[10], test_recall[20]]


def main():
    parser = argparse.ArgumentParser(description="SASRec Baseline for TA-SID (aligned, 3-seed)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Sport", "Toys", "Beauty", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="sasrec_output")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=200, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_interval", default=20, type=int,
                        help="每 N 个 epoch 评估一次")
    parser.add_argument("--patience", default=5, type=int,
                        help="val NDCG@20 连续 N 次评估无改善则早停")
    args = parser.parse_args()

    if args.device != "cpu":
        assert torch.cuda.is_available(), "cuda requested but unavailable"
        args.device = "cuda"

    datasets = ["Sport", "Toys", "Beauty"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        data_path = os.path.join(args.data_dir, ds)
        out_dir = os.path.join(args.output_dir, ds)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 60}\n=== SASRec Baseline (aligned, 3-seed): {ds} ===\n{'=' * 60}")
        print(f"Output: {out_dir}")

        print("\n[1/3] Loading data...")
        (user_train, user_valid, user_val_seq,
         user_test, user_test_seq, itemnum) = load_tiger_data(data_path)

        dataset = {
            "train": user_train,
            "valid": user_valid,
            "val_seq": user_val_seq,
            "test": user_test,
            "test_seq": user_test_seq,
            "usernum": max(user_train.keys()),
            "itemnum": itemnum,
        }

        per_seed = {}
        for seed in args.seeds:
            best_metrics, best_epoch = run_seed(args, seed, dataset, out_dir)
            if best_metrics is not None:
                per_seed[seed] = {
                    "metrics": collect_metrics(best_metrics),
                    "best_epoch": best_epoch,
                }

        # 汇总 mean±std
        print("\n" + "=" * 60)
        print(f"  {ds} — SASRec 3-seed Test Summary (best-val checkpoint)")
        print("=" * 60)
        summary = os.path.join(out_dir, "summary_3seed.txt")
        with open(summary, "w") as f:
            f.write(f"# SASRec {ds} — 3-seed aligned test metrics\n")
            f.write("# 协议: val=valid.parquet, test=test.parquet (对齐 TA-SID T5)\n")
            f.write("# 全量用户, 全量物品排序, best-val NDCG@20 checkpoint\n\n")
            f.write("seed\t" + "\t".join(METRIC_NAMES) + "\n")
            for seed, info in per_seed.items():
                f.write(f"{seed}\t" + "\t".join(f"{m:.6f}" for m in info["metrics"])
                        + f"\t(best epoch {info['best_epoch']})\n")
            if len(per_seed) >= 2:
                arr = np.array([info["metrics"] for info in per_seed.values()])
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                f.write("\nmean\t" + "\t".join(f"{m:.6f}" for m in mean) + "\n")
                f.write("std\t" + "\t".join(f"{m:.6f}" for m in std) + "\n")
                print(f"\n  {'Metric':<10} {'mean':<12} {'std':<12}")
                for name, m, s in zip(METRIC_NAMES, mean, std):
                    print(f"  {name:<10} {m:.4f}±{s:.4f}")

        print(f"\n✅ Summary saved to {summary}")

    print("\n✅ 全部完成.")


if __name__ == "__main__":
    main()
