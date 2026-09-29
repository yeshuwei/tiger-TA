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

【2026-09-29 重构说明】数据加载 / 评估 / 训练骨架已抽到
tools/seq_baselines_common.py，与 GRU4Rec / BERT4Rec 基线共用，保证三方法
数据切分与评估口径严格一致。本文件的网络（含因果 mask）、采样器（WarpSampler）、
损失（BCE）+ 优化器（betas=(0.9,0.98)）仍是 SASRec 专属，**语义零改动**。
重构后必须通过与 sasrec_output/ 的逐位 diff 门禁（见 seq_baselines_common 顶部说明）。

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
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_baselines_common import (  # noqa: E402
    PointWiseFeedForward,
    add_common_args,
    evaluate_full,          # noqa: F401  (re-export 供外部脚本 import)
    resolve_device,
    run_datasets,
    run_seed,
    seed_everything,
    xavier_init_zeros,
)

METHOD_NAME = "SASRec"


# ============================================================
# 1. SASRec 采样器（从 pmixer/SASRec.pytorch 移植）
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
# 2. SASRec 模型（从 pmixer/SASRec.pytorch 移植）
# ============================================================

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
        """返回 logits，形状 (B, len(item_indices))（matmul 广播）"""
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits


class SasrecTrainer:
    """SASRec 专属训练器：WarpSampler 负采样 + BCE + Adam(betas=(0.9,0.98))。

    构造顺序（WarpSampler → criterion → optimizer）与原 run_seed 逐字一致，
    不可调换：WarpSampler 必须在模型 xavier 初始化之后创建。
    """

    def __init__(self, model, dataset, args, seed):
        self.model = model
        self.args = args
        # n_workers 默认沿用原行为（cpu=1 / cuda=3）；显式设为 1 时采样串行化，
        # 整条训练路径变为逐位可复现（多 worker 时 batch 到达顺序由 OS 调度决定，
        # **不可**跨运行复现——这不是重构引入的，原版同样如此，见
        # doc/seq_baselines_report.md「可复现性边界」）。
        self.sampler = WarpSampler(dataset["train"], dataset["usernum"], dataset["itemnum"],
                                   batch_size=args.batch_size,
                                   maxlen=args.maxlen,
                                   n_workers=(args.n_workers if args.n_workers
                                              else (1 if args.device == "cpu" else 3)),
                                   base_seed=seed)
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        self.num_batch = (len(dataset["train"]) - 1) // args.batch_size + 1

    def train_epoch(self, epoch):
        model, args = self.model, self.args
        for step in range(self.num_batch):
            u, seq, pos, neg = self.sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)

            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            self.optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = self.bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += self.bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.sum(param ** 2)
            loss.backward()
            self.optimizer.step()

    def close(self):
        self.sampler.close()


def sasrec_model_factory(args, dataset):
    """建模型 + Xavier 初始化（顺序与原 run_seed L344-354 逐字一致）。"""
    model = SASRec(dataset["usernum"], dataset["itemnum"], args).to(args.device)
    xavier_init_zeros(model, [("pos_emb", 0), ("item_emb", 0)])
    return model


# ============================================================
# 3. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SASRec Baseline for TA-SID (aligned, 3-seed)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["Sport", "Toys", "Beauty", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--output_dir", type=str, default="sasrec_output")
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--n_workers", default=None, type=int,
                        help="WarpSampler 进程数。默认 None = 原行为（cpu 1 / cuda 3）。"
                             "设为 1 使训练路径逐位可复现（重构等价性门禁用）。")
    add_common_args(parser)
    args = parser.parse_args()

    resolve_device(args)
    datasets = ["Sport", "Toys", "Beauty"] if args.dataset == "all" else [args.dataset]

    run_datasets(args, datasets,
                 method_name=METHOD_NAME,
                 model_factory=sasrec_model_factory,
                 trainer_factory=SasrecTrainer)


if __name__ == "__main__":
    main()
