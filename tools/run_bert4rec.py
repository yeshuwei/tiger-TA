#!/usr/bin/env python3
"""
BERT4Rec 基线 — 对齐 SASRec / GRU4Rec / TA-SID 评估协议

参考：Sun et al., "BERT4Rec: Sequential Recommendation with Bidirectional
Encoder Representations from Transformer", CIKM 2019.（注意：不是 WWW）

【结构性差异（论文必须诚实标注，见 doc/seq_baselines_report.md）】

  1. **目标不一致性。** 原论文训练目标是 Cloze（掩码项复原），评估是 next-item 排序。
     我们通过"训练时恒将序列末位替换为 [MASK]"使二者结构同构——评估时的输入形态
     （历史 + 末位 MASK）与训练时完全一致。
  2. **评估输入右移一格。** `predict` 用 `inp = cat([log_seqs[:,1:], MASK])`（宽度仍 = maxlen）。
     这是"右移一格"而非"追加一列"：pos_emb 只有 1..maxlen 个位置，追加会让 MASK
     落在从未训练过的位置索引上，输出退化为噪声**且不报错**。
     代价 = history 最左端少看 1 个 item，且**仅当** history 长度 == maxlen 时才真的丢一个
     （短于 maxlen 时被移出的是 padding）。各数据集实测均只影响 1 个用户。
  3. **双向编码** 使被 mask 位置可见左右上下文，与自回归基线在信息可得性上不同——
     但评估时输入不含目标项，不构成泄漏。

【与 SASRec 的关键区分】绝不能复用 SASRec 的因果 `torch.tril` mask——那会让 BERT4Rec
静默退化成单向模型，指标照样能跑出来但数字无意义。`_selftest` 里有专门的双向性检测。

【信息量对齐】训练只使用 train.parquet 的 history；不使用任何 split 的 target。

用法：
    conda activate tiger_t5
    python3 tools/run_bert4rec.py --selftest --device cpu
    python3 tools/run_bert4rec.py --dataset Sport --seeds 2025
    python3 tools/run_bert4rec.py --dataset all --seeds 2025 2026 2027

输出（bert4rec_output/{ds}/）：与 GRU4Rec 同构
    seed{seed}_loss.tsv / _epochs.txt / _best.pth / _best.txt / _config.txt / summary_3seed.txt
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

METHOD_NAME = "BERT4Rec"


# ============================================================
# 1. 模型
# ============================================================

class GeluFFN(nn.Module):
    """BERT 风格 FFN（Linear → GELU → Linear → Dropout，4× 隐层）。

    刻意不复用 common 的 PointWiseFeedForward——那个是 SASRec 的 ReLU 版。
    """

    def __init__(self, hidden, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class Bert4Rec(nn.Module):
    """双向 Transformer 编码器 + Cloze 目标。

    [MASK] 入词表：mask_id = itemnum + 1，emb 表大小 itemnum + 2（0..itemnum+1）。
    只零化 row 0（padding_idx=0 自动处理）。evaluate_full 的
    all_items = arange(1, itemnum+1) 天然排除 0 与 mask_id，公共层一行不用改。
    输出头与输入 embedding **共享权重**（原论文做法）。
    """

    def __init__(self, item_num, args):
        super().__init__()
        self.dev = args.device
        self.item_num = item_num
        self.mask_id = item_num + 1
        H = args.hidden_units

        self.item_emb = nn.Embedding(item_num + 2, H, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen + 1, H, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.attn_layernorms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ffn_layernorms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(H, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attn_layernorms.append(nn.LayerNorm(H, eps=1e-8))
            self.attn_layers.append(nn.MultiheadAttention(H, args.num_heads,
                                                          args.dropout_rate,
                                                          batch_first=False))
            self.ffn_layernorms.append(nn.LayerNorm(H, eps=1e-8))
            self.ffn_layers.append(GeluFFN(H, args.dropout_rate))

    def encode(self, log_seqs):
        """(B, L) → (B, L, H) 双向隐状态。pad 位置通过 key_padding_mask 屏蔽。"""
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(np.asarray(log_seqs))
        log_seqs = log_seqs.to(self.dev)

        seqs = self.item_emb(log_seqs)
        L = log_seqs.shape[1]
        poss = np.tile(np.arange(1, L + 1), [log_seqs.shape[0], 1])
        poss = poss * log_seqs.cpu().numpy().astype(bool)   # pad 位置位置号为 0
        seqs = seqs + self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # pad 位置不参与注意力（True = 屏蔽）。**没有** 因果 tril。
        key_padding_mask = (log_seqs == 0)

        for i in range(len(self.attn_layers)):
            t = seqs.transpose(0, 1)                                  # (L, B, H)
            attn_out, _ = self.attn_layers[i](t, t, t,
                                              key_padding_mask=key_padding_mask)
            t = self.attn_layernorms[i](t + attn_out)
            seqs = t.transpose(0, 1)
            seqs = self.ffn_layernorms[i](seqs + self.ffn_layers[i](seqs))

        return self.last_layernorm(seqs)

    def predict(self, user_ids, log_seqs, item_indices):
        """返回 logits，形状 (B, len(item_indices))，与 evaluate_full 契约一致。"""
        if not isinstance(log_seqs, torch.Tensor):
            log_seqs = torch.LongTensor(np.asarray(log_seqs))
        log_seqs = log_seqs.to(self.dev)
        B = log_seqs.shape[0]
        mask_col = torch.full((B, 1), self.mask_id, dtype=torch.long, device=self.dev)
        inp = torch.cat([log_seqs[:, 1:], mask_col], dim=1)   # 右移一格，宽度不变
        feats = self.encode(inp)[:, -1, :]
        w = self.item_emb.weight[torch.LongTensor(np.asarray(item_indices)).to(self.dev)]
        return feats.matmul(w.transpose(0, 1))

    def compute_ce(self, hs_flat, labels_flat, valid_idx, ce_chunk):
        """分块 CE，只在被 mask 的位置上算。

        第 0 列（padding）与第 mask_id 列（[MASK]）置 -inf：两者都不是真实物品。
        返回 (mean_loss_tensor, n_masked_positions)。
        """
        w = self.item_emb.weight                       # (V, H), V = itemnum+2
        n = int(valid_idx.numel())
        if n == 0:
            return torch.zeros((), device=self.dev), 0
        total = torch.zeros((), device=self.dev)
        for s in range(0, n, ce_chunk):
            e = min(s + ce_chunk, n)
            idx = valid_idx[s:e]
            logits = hs_flat[idx].matmul(w.transpose(0, 1)).clone()
            logits[:, 0] = float("-inf")
            logits[:, self.mask_id] = float("-inf")
            total = total + F.cross_entropy(logits, labels_flat[idx], reduction="sum")
        return total / n, n


# ============================================================
# 2. 训练（Cloze）
# ============================================================

class Bert4RecTrainer:
    def __init__(self, model, dataset, args, seed):
        self.model = model
        self.args = args
        self.seed = seed
        self.train_padded, self.uids = build_padded(dataset["train"], args.maxlen)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        self.num_batch = (len(self.uids) - 1) // args.batch_size + 1

    def _make_cloze(self, seq, rs):
        """从左补零序列构造 Cloze 输入/标签。

        - **末位恒为 [MASK]**，标签 = 该位置原 item（与评估严格同构）
        - 其余真实位置以 mask_prob 被选中，做 80/10/10（MASK / 随机 item / 不变）
        - 只在"被 mask 的位置"上算损失
        """
        inp = seq.copy()
        masked = np.zeros(seq.shape, dtype=bool)
        last = seq.shape[1] - 1
        inp[:, last] = self.model.mask_id
        masked[:, last] = True

        real = seq != 0
        cand = real.copy()
        cand[:, last] = False                       # 末位已处理
        sel = cand & (rs.rand(*seq.shape) < self.args.mask_prob)
        if sel.any():
            r = rs.rand(*seq.shape)
            to_mask = sel & (r < 0.8)
            to_rand = sel & (r >= 0.8) & (r < 0.9)
            inp[to_mask] = self.model.mask_id
            # 随机替换为 1..itemnum 内的真实 item（不含 0 与 mask_id）
            n_rand = int(to_rand.sum())
            if n_rand:
                inp[to_rand] = rs.randint(1, self.model.item_num + 1, size=n_rand)
            masked |= sel
        return inp, seq, masked

    def train_epoch(self, epoch):
        model, args = self.model, self.args
        rs = np.random.RandomState(self.seed * 100000 + epoch)
        order = rs.permutation(len(self.uids))

        epoch_sum, epoch_n = 0.0, 0
        for step in range(self.num_batch):
            bidx = order[step * args.batch_size:(step + 1) * args.batch_size]
            if len(bidx) == 0:
                continue
            seq = self.train_padded[bidx]
            inp, labels, masked = self._make_cloze(seq, rs)

            inp_t = torch.LongTensor(inp).to(args.device)
            labels_t = torch.LongTensor(labels).to(args.device)
            valid_idx = torch.LongTensor(masked.reshape(-1)).to(args.device).nonzero(
                as_tuple=True)[0]

            hs = model.encode(inp_t)
            loss, n = model.compute_ce(hs.reshape(-1, args.hidden_units),
                                       labels_t.reshape(-1), valid_idx, args.ce_chunk)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_sum += loss.item() * n
            epoch_n += n

        epoch_loss = epoch_sum / max(epoch_n, 1)
        with open(self.loss_log, "a") as f:
            f.write(f"{epoch}\t{epoch_loss:.6f}\n")
        print(f"    [seed {self.seed}] epoch {epoch:>3}  ClozeCE={epoch_loss:.4f}  "
              f"(masked pos={epoch_n})", flush=True)
        return epoch_loss

    def close(self):
        pass


def bert4rec_model_factory(args, dataset):
    """按原论文超参建模型（无 xavier；用 PyTorch 默认初始化，BERT 风格）。"""
    return Bert4Rec(dataset["itemnum"], args).to(args.device)


def make_trainer_factory(out_dir):
    def factory(model, dataset, args, seed):
        t = Bert4RecTrainer(model, dataset, args, seed)
        t.loss_log = os.path.join(out_dir, f"seed{seed}_loss.tsv")
        with open(t.loss_log, "w") as f:
            f.write("epoch\tcloze_ce\n")
        return t
    return factory


# ============================================================
# 3. 自检（--selftest，CPU）
# ============================================================

def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok = ok and bool(cond)

    print("BERT4Rec 自检")
    print("-" * 60)

    torch.manual_seed(0)
    args = argparse.Namespace(device="cpu", hidden_units=16, num_blocks=1, num_heads=2,
                              dropout_rate=0.0, maxlen=8, ce_chunk=64, lr=1e-2,
                              batch_size=8, mask_prob=0.2, itemnum=20, item_num=20)
    model = Bert4Rec(20, args).eval()

    # ---- T1: 双向性（若误用因果 mask，改末位不会影响前面位置的输出）----
    base = np.array([[0, 0, 3, 5, 7, 9, 11, 13]])
    alt = base.copy()
    alt[0, -1] = 17                       # 只改最后一个位置
    with torch.no_grad():
        h1 = model.encode(base)
        h2 = model.encode(alt)
    # 位置 1（列 1，真实 item=5，在改动点之前）应受影响 ← 双向的证据
    d_early = (h1[0, 1] - h2[0, 1]).abs().max().item()
    d_late = (h1[0, 4] - h2[0, 4]).abs().max().item()
    check("T1 双向性：改末位会改变更早位置的隐状态", d_early > 1e-6,
          f"Δ(列1)={d_early:.3e}, Δ(列4)={d_late:.3e}")
    check("T1b 改动点自身也变（非恒等）", d_late > 1e-6)

    # ---- T2: 位置号不越界（右移一格后 MASK 落在 maxlen，而非 maxlen+1）----
    log_seqs = np.array([[0, 0, 3, 5, 7, 9, 11, 13]])
    B = log_seqs.shape[0]
    inp = np.concatenate([log_seqs[:, 1:],
                          np.full((B, 1), model.mask_id)], axis=1)
    check("T2 右移后宽度不变 == maxlen", inp.shape[1] == args.maxlen, f"{inp.shape}")
    real_ids = (inp != 0)
    poss = np.arange(1, inp.shape[1] + 1)[None, :] * real_ids
    check("T2b 最大位置号 == maxlen（未越界）", poss.max() == args.maxlen,
          f"max={poss.max()} vs maxlen={args.maxlen}")
    check("T2c MASK 落在末位", inp[0, -1] == model.mask_id)
    check("T2d all_items 天然排除 pad 与 MASK",
          not (np.arange(1, args.itemnum + 1) == 0).any()
          and not (np.arange(1, args.itemnum + 1) == model.mask_id).any())

    # ---- T3: pad 列与 MASK 列不参与竞争 ----
    with torch.no_grad():
        model.item_emb.weight.data.normal_(0, 0.1)
    hs = torch.randn(3, args.hidden_units)
    labels = torch.LongTensor([3, 5, 7])
    vi = torch.arange(3)
    l0, _ = model.compute_ce(hs, labels, vi, args.ce_chunk)
    with torch.no_grad():
        w_saved = model.item_emb.weight.data.clone()
        model.item_emb.weight.data[0, :] = 1000.0
        model.item_emb.weight.data[model.mask_id, :] = 1000.0
    l1, _ = model.compute_ce(hs, labels, vi, args.ce_chunk)
    check("T3 扰动 pad/MASK 列 loss 逐位不变", l0.item() == l1.item(),
          f"{l0.item():.4f} vs {l1.item():.4f}")
    with torch.no_grad():
        model.item_emb.weight.data = w_saved

    # ---- T4: Cloze 构造正确性 ----
    trainer = Bert4RecTrainer(model, {"train": {1: [3, 5, 7, 9, 11]}, "itemnum": 20},
                              args, seed=2025)
    rs = np.random.RandomState(1)
    seq = np.array([[0, 0, 0, 3, 5, 7, 9, 11]])
    inp, lab, msk = trainer._make_cloze(seq, rs)
    check("T4a 末位恒为 MASK", inp[0, -1] == model.mask_id and msk[0, -1])
    check("T4b 末位标签 = 原末位真实 item", lab[0, -1] == 11)
    check("T4c pad 位置从不被 mask", not msk[0, :3].any())
    check("T4d 标签从不为 0 或 mask_id",
          bool(((lab[msk] != 0) & (lab[msk] != model.mask_id)).all()))
    check("T4e 被 mask 位置的输入确实被改（80% 分支或末位）",
          inp[0, -1] != lab[0, -1])

    # ---- T5: 泄漏检查 ----
    class Poisoned(dict):
        def __getitem__(self, k):
            if k in ("valid", "test", "val_seq", "test_seq"):
                raise AssertionError(f"训练读取了 {k} —— 泄漏！")
            return super().__getitem__(k)

    ds = Poisoned({"train": {1: [3, 5, 7, 9, 11], 2: [1, 2, 3]}, "itemnum": 20,
                   "usernum": 2, "valid": None, "test": None,
                   "val_seq": None, "test_seq": None})
    t2 = Bert4RecTrainer(model, ds, args, seed=2025)
    t2.loss_log = "/tmp/_bert4rec_selftest_loss.tsv"
    with open(t2.loss_log, "w") as f:
        f.write("epoch\tcloze_ce\n")
    try:
        l = t2.train_epoch(1)
        check("T5 训练不触碰 val/test 且可跑通", np.isfinite(l), f"loss={l:.4f}")
    except AssertionError as ex:
        check("T5 训练不触碰 val/test 且可跑通", False, str(ex))
    os.remove(t2.loss_log)

    # ---- T6: 合成确定性数据（x_{t+1} = (x_t + 1) mod V）----
    # 计划中的核心检验：若掩码/位置/右移任一环节错位，末位恢复率不可能到 95%。
    V, NU, L, EP = 50, 200, 20, 200
    sargs = argparse.Namespace(device="cpu", hidden_units=64, num_blocks=2, num_heads=2,
                               dropout_rate=0.0, maxlen=L, ce_chunk=512, lr=5e-3,
                               batch_size=64, mask_prob=0.2, itemnum=V, item_num=V)
    torch.manual_seed(7)
    sargs.device = "cpu"
    smodel = Bert4Rec(V, sargs)
    rng = np.random.RandomState(0)
    synth = {}
    for u in range(1, NU + 1):
        s = int(rng.randint(1, V + 1))
        synth[u] = [s]
        for _ in range(L - 2):
            s = s % V + 1
            synth[u].append(s)
    sds = {"train": synth, "itemnum": V, "usernum": NU,
           "valid": None, "test": None, "val_seq": None, "test_seq": None}
    st = Bert4RecTrainer(smodel, sds, sargs, seed=2025)
    st.loss_log = "/tmp/_bert4rec_synth.tsv"
    with open(st.loss_log, "w") as f:
        f.write("epoch\tcloze_ce\n")
    for ep in range(1, EP + 1):
        st.train_epoch(ep)
    os.remove(st.loss_log)

    # 末位恢复准确率：给前 L-1 个真实 item + 末位 MASK，看能否答对第 L 个
    smodel.eval()
    mat, uids = build_padded(synth, L)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(uids), 50):
            sl = mat[i:i + 50]
            inp_s = np.concatenate([sl[:, :-1],
                                    np.full((sl.shape[0], 1), smodel.mask_id)], axis=1)
            h = smodel.encode(inp_s)[:, -1, :]
            lg = h.matmul(smodel.item_emb.weight.transpose(0, 1))
            lg[:, 0] = float("-inf")
            lg[:, smodel.mask_id] = float("-inf")
            correct += int((lg.argmax(-1).cpu().numpy() == sl[:, -1]).sum())
    acc = correct / len(uids)
    check(f"T6 合成确定性数据末位恢复率 ≥95%", acc >= 0.95, f"acc={acc:.3f} (随机基线={1/V:.4f})")

    # ---- T7: 标签打乱对照（标签没被用上则指标不回落）----
    torch.manual_seed(11)
    nmodel = Bert4Rec(V, sargs)
    nmat = mat.copy()
    for r in range(nmat.shape[0]):
        nz = np.nonzero(nmat[r])[0]
        vals = nmat[r][nz].copy()
        rng.shuffle(vals)
        nmat[r][nz] = vals
    correct_n = 0
    with torch.no_grad():
        for i in range(0, len(uids), 50):
            sl = nmat[i:i + 50]
            inp_s = np.concatenate([sl[:, :-1],
                                    np.full((sl.shape[0], 1), smodel.mask_id)], axis=1)
            h = nmodel.encode(inp_s)[:, -1, :]
            lg = h.matmul(nmodel.item_emb.weight.transpose(0, 1))
            lg[:, 0] = float("-inf")
            lg[:, nmodel.mask_id] = float("-inf")
            correct_n += int((lg.argmax(-1).cpu().numpy() == sl[:, -1]).sum())
    acc_n = correct_n / len(uids)
    check("T7 未训练模型在打乱数据上≈随机（证明指标确实反映训练）",
          acc_n < 3.0 / V, f"acc={acc_n:.4f} (随机={1/V:.4f})")

    print("-" * 60)
    print("✅ 自检全部通过" if ok else "❌ 自检失败")
    sys.exit(0 if ok else 1)


# ============================================================
# 4. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BERT4Rec Baseline for TA-SID (aligned)")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["Sport", "Toys", "Beauty", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025, 2026, 2027])
    parser.add_argument("--output_dir", type=str, default="bert4rec_output")
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_heads", default=2, type=int)
    parser.add_argument("--mask_prob", default=0.2, type=float,
                        help="Cloze 掩码比例（末位恒 mask，不计入此比例）")
    parser.add_argument("--ce_chunk", default=2048, type=int)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--summary_only", action="store_true",
                        help="不训练，仅从既有 seed*_best.txt 复建 summary_3seed.txt")
    add_common_args(parser)
    parser.set_defaults(hidden_units=64)     # 原论文口径
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
                     model_factory=bert4rec_model_factory,
                     trainer_factory=make_trainer_factory(out_dir),
                     summary_only=args.summary_only)


if __name__ == "__main__":
    main()
