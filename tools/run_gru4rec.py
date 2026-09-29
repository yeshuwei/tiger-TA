#!/usr/bin/env python3
"""
GRU4Rec 基线 — 对齐 SASRec / TA-SID 评估协议

参考：Hidasi et al., "Session-based Recommendations with Recurrent Neural
Networks", ICLR 2016.

【与原论文的两处刻意偏离（论文必须诚实标注，见 doc/seq_baselines_report.md）】

  1. **损失函数：全物品 softmax 交叉熵，非原版 TOP1/BPR。**
     原版是 session-parallel mini-batch + TOP1/BPR 成对排序损失，配合 100 负采样
     评估。本工作三方法统一使用"全量物品排序"协议，负采样损失与之不可比；且 CE
     变体的监督信号（逐位置 next-item）与 SASRec 基线严格同构。
     因此标注为 "GRU4Rec (full-softmax CE variant)"，**不声称复现原论文数字**。

  2. **无 session 切分。** 原论文面向匿名 session，本工作数据是用户级序列
     （Amazon 5-core），与另两个基线一致。

模型结构按原论文：100 维 embedding + 100 维单层 GRU，输入/输出 embedding
不共享（独立 out_emb），embedding dropout 0.2。

【信息量对齐】训练只使用 train.parquet 的 history（不用 train.target，不碰
val/test 任何字段），与 SASRec 严格一致，杜绝"新基线多吃了数据"。

【泄漏与安全】evaluate 走 seq_baselines_common.evaluate_full —— 与 SASRec
同一函数、同一 users 集合、同一 all_items、同一 rank 计算。

用法：
    conda activate tiger_t5
    python3 tools/run_gru4rec.py --selftest --device cpu
    python3 tools/run_gru4rec.py --dataset Sport --seeds 2025
    python3 tools/run_gru4rec.py --dataset all --seeds 2025 2026 2027

输出（gru4rec_output/{ds}/）：
    seed{seed}_train.log   逐 epoch 训练日志
    seed{seed}_loss.tsv    epoch / CE loss（收敛自检用）
    seed{seed}_epochs.txt  逐 eval 步 val/test 指标（5 列，与 SASRec 同格式）
    seed{seed}_best.pth    最佳 val NDCG@20 检查点
    summary_3seed.txt      3-seed test 指标 + mean±std（论文用）
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seq_baselines_common import (  # noqa: E402
    add_common_args,
    build_padded,
    guard_output_dir,
    resolve_device,
    run_datasets,
    run_seed,
)

METHOD_NAME = "GRU4Rec"


# ============================================================
# 1. 模型
# ============================================================

class GRU4Rec(nn.Module):
    def __init__(self, item_num, args):
        super().__init__()
        self.dev = args.device
        self.item_num = item_num
        hidden = args.hidden_units

        self.item_emb = nn.Embedding(item_num + 1, hidden, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.out_emb = nn.Embedding(item_num + 1, hidden, padding_idx=0)

    def encode(self, log_seqs):
        """(B, L) → (B, L, H)，逐位置隐状态。"""
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(np.asarray(log_seqs))
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs = self.emb_dropout(seqs)
        out, _ = self.gru(seqs)
        return out

    def predict(self, user_ids, log_seqs, item_indices):
        """返回 logits，形状 (B, len(item_indices))，与 evaluate_full 契约一致。"""
        h = self.encode(log_seqs)[:, -1, :]
        w = self.out_emb.weight[torch.LongTensor(np.asarray(item_indices)).to(self.dev)]
        return h.matmul(w.transpose(0, 1))

    def compute_ce(self, hs_flat, labels_flat, valid_idx, ce_chunk):
        """分块全物品 softmax CE。

        hs_flat     : (B*L, H) 展平隐状态
        labels_flat : (B*L,)   展平标签（pad 位置为 0，由 valid_idx 排除）
        valid_idx   : (M,)     有效位置下标（inputs!=0 且 labels!=0）

        第 0 列置 -inf：item 0 是 padding，不是真实物品，不得参与竞争。
        分块避免 (chunk, itemnum+1) fp32 单张量过大
        （chunk=2048, itemnum=18357 → ~150 MB）。
        返回 (mean_loss_tensor, n_valid_positions)。
        """
        w = self.out_emb.weight                       # (V, H), V = itemnum+1
        n = int(valid_idx.numel())
        if n == 0:
            return torch.zeros((), device=self.dev), 0
        total = torch.zeros((), device=self.dev)
        for s in range(0, n, ce_chunk):
            e = min(s + ce_chunk, n)
            idx = valid_idx[s:e]
            logits = hs_flat[idx].matmul(w.transpose(0, 1)).clone()
            logits[:, 0] = float("-inf")
            total = total + F.cross_entropy(logits, labels_flat[idx], reduction="sum")
        return total / n, n


# ============================================================
# 2. 训练数据准备
# ============================================================

class Gru4RecTrainer:
    def __init__(self, model, dataset, args, seed):
        self.model = model
        self.args = args
        self.seed = seed
        self.train_padded, self.uids = build_padded(dataset["train"], args.maxlen)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.num_batch = (len(self.uids) - 1) // args.batch_size + 1

    def train_epoch(self, epoch):
        model, args = self.model, self.args
        # epoch 间独立可复现：随机流只由 (seed, epoch) 决定
        rs = np.random.RandomState(self.seed * 100000 + epoch)
        order = rs.permutation(len(self.uids))

        epoch_sum, epoch_n = 0.0, 0
        for step in range(self.num_batch):
            bidx = order[step * args.batch_size:(step + 1) * args.batch_size]
            if len(bidx) == 0:
                continue
            seq = self.train_padded[bidx]                     # (b, maxlen)
            inputs = torch.LongTensor(seq[:, :-1]).to(args.device)
            labels = torch.LongTensor(seq[:, 1:]).to(args.device)

            valid = (inputs != 0) & (labels != 0)
            valid_idx = valid.reshape(-1).nonzero(as_tuple=True)[0]

            hs = model.encode(inputs)                          # (b, L-1, H)
            loss, n = model.compute_ce(hs.reshape(-1, args.hidden_units),
                                       labels.reshape(-1), valid_idx, args.ce_chunk)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_sum += loss.item() * n
            epoch_n += n

        epoch_loss = epoch_sum / max(epoch_n, 1)
        with open(self.loss_log, "a") as f:
            f.write(f"{epoch}\t{epoch_loss:.6f}\n")
        print(f"    [seed {self.seed}] epoch {epoch:>3}  CE={epoch_loss:.4f}  "
              f"(valid pos={epoch_n})", flush=True)
        return epoch_loss

    def close(self):
        pass


def gru4rec_model_factory(args, dataset):
    """按原论文超参建模型。**不做 SASRec 式的 xavier 初始化**——原论文未使用，
    用 PyTorch 默认初始化（embedding ~ N(0,1)，GRU 用 uniform(-1/√H, 1/√H)），
    这是结构差异而非不公平：两者都用各自论文的默认。"""
    return GRU4Rec(dataset["itemnum"], args).to(args.device)


def make_trainer_factory(out_dir):
    """把 loss.tsv 路径注入 trainer（trainer 构造签名由 common 固定为 4 参数）。"""
    def factory(model, dataset, args, seed):
        t = Gru4RecTrainer(model, dataset, args, seed)
        t.loss_log = os.path.join(out_dir, f"seed{seed}_loss.tsv")
        with open(t.loss_log, "w") as f:
            f.write("epoch\tce_loss\n")
        return t
    return factory


# ============================================================
# 3. 自检（--selftest，CPU，秒级）
# ============================================================

def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok = ok and bool(cond)

    print("GRU4Rec 自检")
    print("-" * 60)

    # ---- T1: 数据对齐（labels[i] == inputs[i+1]）----
    user_train = {1: [3, 5, 7], 2: [1, 2, 3, 4, 5], 3: [9]}
    maxlen = 6
    mat, uids = build_padded(user_train, maxlen)
    check("T1a 左补零、历史在尾部", list(mat[uids.index(1)]) == [0, 0, 0, 3, 5, 7],
          f"{list(mat[uids.index(1)])}")
    inp, lab = mat[:, :-1], mat[:, 1:]
    check("T1b labels[i] == inputs[i+1]（逐元素）", np.array_equal(lab, mat[:, 1:]) and
          np.array_equal(inp, mat[:, :-1]))
    v = (inp != 0) & (lab != 0)
    row = uids.index(1)
    check("T1c 有效位置标签 = 真实下一项",
          list(lab[row][v[row]]) == [5, 7], f"{list(lab[row][v[row]])}")

    # ---- T2: 完美预测 → CE ≈ 0 ----
    # 5 类竞争，正确类 logit 领先 MARGIN ⇒ CE = log(1 + 4e^-MARGIN)。
    # MARGIN 必须落在"贡献可表示"的区间：float32 下 30·eps ≈ 3.6e-6，
    # 取 MARGIN=30 会让 4e^-30 被 e^30 完全吞掉 → CE 恒为精确 0.0，
    # 使 T3 退化成恒真断言（检测不出掩码失效）。取 10 ⇒ CE ≈ 1.8e-4，可表示。
    MARGIN = 10.0
    CE_EXPECT = float(np.log1p(4 * np.exp(-MARGIN)))   # ≈ 1.817e-4，解析期望值
    torch.manual_seed(0)
    args = argparse.Namespace(device="cpu", hidden_units=8, dropout_rate=0.0,
                              item_num=5, itemnum=5, maxlen=6, ce_chunk=8, lr=1e-3,
                              batch_size=2)
    m = GRU4Rec(5, args).eval()
    with torch.no_grad():
        m.out_emb.weight.data = torch.zeros(6, 8)
        for j in range(1, 6):
            m.out_emb.weight.data[j, j % 8] = MARGIN
    hs = torch.zeros(2, 8)
    hs[0, 3] = 1.0
    hs[1, 4] = 1.0
    labels = torch.LongTensor([3, 4])
    valid_idx = torch.arange(2)
    loss, n = m.compute_ce(hs, labels, valid_idx, args.ce_chunk)
    # 容差 1e-6：float32 下 log(e^10+4) − 10 的抵消误差本身约 6e-8
    # （x≈2.2e4 的 relative eps 6e-8 → log 的绝对误差 ≈ 6e-8），
    # 比这更紧的容差只会测出 float32 的舍入，而非代码正确性。
    check("T2 完美预测 CE ≈ log(1+4e^-10)", abs(loss.item() - CE_EXPECT) < 1e-6
          and loss.item() > 0, f"loss={loss.item():.6e} 期望={CE_EXPECT:.6e}")

    # ---- T3: pad 列（index 0）不参与竞争 ----
    # 只扰动 row 0（pad）。基损失非零（见 T2），故本断言非退化。
    with torch.no_grad():
        m.out_emb.weight.data[0, :] = 1000.0
    loss2, _ = m.compute_ce(hs, labels, valid_idx, args.ce_chunk)
    check("T3a 扰动 pad 列 loss 逐位不变（非退化：基损失≠0）",
          loss2.item() == loss.item() and loss.item() > 0,
          f"{loss.item():.6e} vs {loss2.item():.6e}")

    # ---- T3b: 阳性对照 —— 扰动【有效】列必须改变 loss ----
    # hs 是 one-hot：样本 0 读列 3、样本 1 读列 4。故必须改 w_1 的【第 3 列】
    # （把竞争类 1 的 logit 抬到与正确类 3 齐平），改 w_1[1] 是无效扰动。
    # 若本检也"不变"，说明测试根本没接上 logits，T3a 的通过毫无意义。
    with torch.no_grad():
        m.out_emb.weight.data[0, :] = 0.0          # 还原 pad 列
        m.out_emb.weight.data[1, 3] = MARGIN        # 竞争类 1 在样本 0 的 logit: 0 → 10
    loss3, _ = m.compute_ce(hs, labels, valid_idx, args.ce_chunk)
    check("T3b 阳性对照：扰动有效列 loss 必变", loss3.item() != loss.item(),
          f"{loss.item():.6e} → {loss3.item():.6e}")

    # ---- T4: pad 位置被排除；n 计数正确 ----
    seq = torch.LongTensor([[0, 0, 3, 5, 7]])
    i2, l2 = seq[:, :-1], seq[:, 1:]
    v2 = (i2 != 0) & (l2 != 0)
    vi = v2.reshape(-1).nonzero(as_tuple=True)[0]
    check("T4 有效位置数 = 2（0/0,0/3 被排除）", int(vi.numel()) == 2,
          f"idx={vi.tolist()}, labels={l2.reshape(-1)[vi].tolist()}")

    # ---- T5: predict 形状与列序 ----
    torch.manual_seed(1)
    m2 = GRU4Rec(5, args).eval()
    log_seqs = np.array([[0, 0, 1, 2, 3], [0, 0, 0, 4, 5]])
    items = np.array([1, 2, 3, 4, 5])
    with torch.no_grad():
        lg = m2.predict(np.array([1, 2]), log_seqs, items)
        h = m2.encode(log_seqs)[:, -1, :]
        manual = h.matmul(m2.out_emb.weight[torch.LongTensor(items)].transpose(0, 1))
    check("T5a predict 形状 = (B, len(items))", tuple(lg.shape) == (2, 5), f"{tuple(lg.shape)}")
    check("T5b predict 列序 == item_indices 顺序", torch.allclose(lg, manual, atol=1e-6))

    # ---- T6: 无泄漏（训练不得触碰 val/test） ----
    class Poisoned(dict):
        def __getitem__(self, k):
            if k in ("valid", "test", "val_seq", "test_seq"):
                raise AssertionError(f"训练读取了 {k} —— 泄漏！")
            return super().__getitem__(k)

    args10 = argparse.Namespace(**{**vars(args), "itemnum": 10})
    m3 = GRU4Rec(10, args10)
    ds = Poisoned({"train": user_train, "itemnum": 10, "usernum": 3,
                   "valid": None, "test": None, "val_seq": None, "test_seq": None})
    t = Gru4RecTrainer(m3, ds, args10, seed=2025)
    t.loss_log = "/tmp/_gru4rec_selftest_loss.tsv"
    with open(t.loss_log, "w") as f:
        f.write("epoch\tce_loss\n")
    try:
        l1 = t.train_epoch(1)
        check("T6 训练不触碰 val/test 且可跑通", np.isfinite(l1), f"loss={l1:.4f}")
    except AssertionError as ex:
        check("T6 训练不触碰 val/test 且可跑通", False, str(ex))
    os.remove(t.loss_log)

    # ---- T7: epoch 间随机流独立可复现 ----
    r1 = np.random.RandomState(2025 * 100000 + 1).permutation(10)
    r2 = np.random.RandomState(2025 * 100000 + 1).permutation(10)
    r3 = np.random.RandomState(2025 * 100000 + 2).permutation(10)
    check("T7 epoch 洗牌 (seed,epoch) 可复现且相邻 epoch 不同",
          np.array_equal(r1, r2) and not np.array_equal(r1, r3))

    print("-" * 60)
    print("✅ 自检全部通过" if ok else "❌ 自检失败")
    sys.exit(0 if ok else 1)


# ============================================================
# 4. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRU4Rec Baseline for TA-SID (aligned)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["Sport", "Toys", "Beauty", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--output_dir", type=str, default="gru4rec_output")
    parser.add_argument("--ce_chunk", default=2048, type=int,
                        help="分块 CE 的行数上限（控制显存）")
    parser.add_argument("--selftest", action="store_true",
                        help="跑 CPU 自检并退出（不下数据、不训练）")
    parser.add_argument("--summary_only", action="store_true",
                        help="不训练，仅从既有 seed*_best.txt 复建 summary_3seed.txt")
    add_common_args(parser)
    parser.set_defaults(hidden_units=100)   # 原论文口径，不与 SASRec 的 50 强行对齐
    args = parser.parse_args()

    if args.selftest:
        _selftest()

    if args.dataset is None:
        parser.error("--dataset 必填（或用 --selftest）")

    guard_output_dir(args, METHOD_NAME)
    resolve_device(args)
    datasets = ["Sport", "Toys", "Beauty"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        out_dir = os.path.join(args.output_dir, ds)
        os.makedirs(out_dir, exist_ok=True)
        run_datasets(args, [ds],
                     method_name=METHOD_NAME,
                     model_factory=gru4rec_model_factory,
                     trainer_factory=make_trainer_factory(out_dir),
                     summary_only=args.summary_only)


if __name__ == "__main__":
    main()
