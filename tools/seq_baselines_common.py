#!/usr/bin/env python3
"""
序列推荐基线通用层 — 从 tools/run_sasrec.py 原样抽出的共享组件。

被 SASRec / GRU4Rec / BERT4Rec 三个基线脚本共同 import，保证 **数据切分、
评估口径、训练骨架、产物格式** 三方法完全一致（审稿人可复核同协议）。

【逐位等价纪律】本模块内的代码是从 tools/run_sasrec.py（v2, 2026-08-07 已发表
口径）逐字符搬移而来，**禁止"顺手优化"**：

  1. 不新增 cudnn.deterministic / benchmark / TF32 任何开关。run_sasrec.py 从未
     设置这些，加了会改变 kernel 选择 → 已发表的 SASRec 数字无法逐位复现。
     （注意 rqvae/main.py 设了这两个，但那是另一条链路，不要照抄过来。）
  2. xavier_init_zeros 的 named_parameters() 遍历顺序与 zero 行顺序不可改。
  3. write_summary_3seed 的注释文案与 np.std 默认 ddof=0 不可改
     （ddof=0 = 总体标准差；doc/ablation_main_table_3datasets.md 用的是 ddof=1，
      两者口径不同，见 doc/seq_baselines_report.md 的说明）。
  4. evaluate_full 的 users=sorted(users) / rank 计算 / all_items=arange(1,N+1)
     不可改。

任何修改都必须重跑门禁：
    python3 tools/run_sasrec.py --dataset Sport --seeds 2025 2026 \
            --output_dir sasrec_output_verify
    diff sasrec_output/Sport/seed2025_epochs.txt \
         sasrec_output_verify/Sport/seed2025_epochs.txt
"""

import os
import sys
import time
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# 1. 数据加载与适配（原 run_sasrec.py L54-115）
# ============================================================

def load_tiger_data(data_dir: str):
    """加载 TIGER 格式的 parquet 数据，重建 SASRec 格式。

    返回六个值（user → list）：
      train   : train.parquet 的 history（训练输入）
      valid   : valid.parquet 的 target（T5 验证任务的目标）
      val_seq : valid.parquet 的 history（T5 验证任务的输入）
      test    : test.parquet 的 target（T5 测试任务的目标）
      test_seq: test.parquet 的 history（T5 测试任务的输入）
      itemnum : 最大 item id（item id 为 1..itemnum 连续，0 = padding）
    """
    import pandas as pd

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


def build_dataset(user_train, user_valid, user_val_seq, user_test, user_test_seq, itemnum):
    """组装 dataset dict（原 main() L487-495）。

    注意 usernum = max(user_train.keys()) 而非用户计数，这是原实现的行为，
    改掉会改变 WarpSampler 的 randint 上界。
    """
    return {
        "train": user_train,
        "valid": user_valid,
        "val_seq": user_val_seq,
        "test": user_test,
        "test_seq": user_test_seq,
        "usernum": max(user_train.keys()),
        "itemnum": itemnum,
    }


# ============================================================
# 2. 共享网络块（原 run_sasrec.py L194-207）
# ============================================================

class PointWiseFeedForward(nn.Module):
    """SASRec 的 FFN 块（ReLU 版，Conv1d 实现）。原样保留供 SASRec 使用。"""

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


def xavier_init_zeros(model, zero_rows):
    """Xavier 初始化 + 指定 embedding 行置零（原 run_seed L346-354）。

    zero_rows: [(属性名, 行号), ...]，按给定顺序执行（顺序影响 RNG 消耗，勿改）。
    SASRec 调用时传 [("pos_emb", 0), ("item_emb", 0)]（先 pos 后 item，与原版一致）。
    """
    for name, param in model.named_parameters():
        try:
            if param.dim() > 1:
                nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    for attr, row in zero_rows:
        getattr(model, attr).weight.data[row, :] = 0


def seed_everything(seed):
    """原 run_seed L340-342。故意不含 cudnn.deterministic / benchmark。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_padded(user_train, maxlen):
    """把 {uid: [items]} 压成 (n_users, maxlen) int32 左补零矩阵。

    与 SASRec WarpSampler 的 seq 构造一致：历史放尾部，前面补 0。
    inputs = mat[:, :-1], labels = mat[:, 1:]  ⇒ labels[i] == inputs[i+1]

    GRU4Rec（CE）与 BERT4Rec（Cloze）共用；SASRec 因需负采样不走这里。
    """
    uids = sorted(user_train.keys())
    mat = np.zeros((len(uids), maxlen), dtype=np.int32)
    for i, u in enumerate(uids):
        hist = user_train[u][-maxlen:]
        mat[i, maxlen - len(hist):] = hist
    return mat, uids


# ============================================================
# 3. 评估（全量物品 × 全量用户，分批）（原 run_sasrec.py L277-330）
# ============================================================

@torch.no_grad()
def evaluate_full(model, dataset, args, mode="test", eval_batch=512):
    """
    全量物品评估（非 100 负采样），与 TA-SID 评估协议一致。

    mode="val":   seq = valid.parquet 的 history, target = valid.target
                  （与 T5 模型选择的验证集严格一致）
    mode="test":  seq = test.parquet 的 history,  target = test.target
                  （与 T5 最终测试集严格一致）

    模型契约：model.predict(user_ids, log_seqs, item_indices) -> (B, len(item_indices))
    （注意 run_sasrec.py 的 docstring 把形状写反了，真实返回 (B, itemnum)）
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

    for i in range(0, len(users), eval_batch):
        ub = users[i:i + eval_batch]
        seqs = np.zeros((len(ub), args.maxlen), dtype=np.int32)
        targets = np.zeros(len(ub), dtype=np.int32)
        for j, u in enumerate(ub):
            history = seq_dict[u][-args.maxlen:]
            seqs[j, args.maxlen - len(history):] = history
            targets[j] = target_dict[u][0]

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
# 4. 训练骨架（原 run_seed L337-429 + main 的汇总部分）
# ============================================================

METRIC_NAMES = ["NDCG@5", "NDCG@10", "NDCG@20", "Rec@5", "Rec@10", "Rec@20"]

EPOCH_LOG_HEADER = "epoch\tval_ndcg@20\ttest_ndcg@20\tval_recall@20\ttest_recall@20\n"


def collect_metrics(best_metrics):
    """从 (val_ndcg, val_recall, test_ndcg, test_recall) 提取 test 6 个指标"""
    val_ndcg, val_recall, test_ndcg, test_recall = best_metrics
    return [test_ndcg[5], test_ndcg[10], test_ndcg[20],
            test_recall[5], test_recall[10], test_recall[20]]


def write_seed_config(out_dir, seed):
    """记录本次运行的 CLI 与 git HEAD（幂等判据 + 预登记"超参在看 test 前已冻结"）。"""
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        head = "unknown"
    path = os.path.join(out_dir, f"seed{seed}_config.txt")
    with open(path, "w") as f:
        f.write(f"# git HEAD: {head}\n")
        f.write("# argv: " + " ".join(sys.argv) + "\n")
    return path


def run_seed(args, seed, dataset, out_dir, *,
             model_factory, trainer_factory, method_name="?", extra_console=None):
    """单 seed 训练 + 每 eval_interval 评估 + best-val NDCG@20 存档 + 早停。

    控制流与原 run_sasrec.py L337-429 逐字一致。模型与训练细节通过
    model_factory(args, dataset) / trainer_factory(model, dataset, args, seed)
    注入；trainer 需实现 train_epoch(epoch) 与 close()。
    """
    print(f"\n{'=' * 60}\n  {args.dataset} — seed {seed}\n{'=' * 60}", flush=True)

    seed_everything(seed)

    model = model_factory(args, dataset)
    trainer = trainer_factory(model, dataset, args, seed)
    write_seed_config(out_dir, seed)

    epoch_log = os.path.join(out_dir, f"seed{seed}_epochs.txt")
    with open(epoch_log, "w") as f:
        f.write(EPOCH_LOG_HEADER)

    best_val_ndcg = -1.0
    best_epoch = 0
    best_metrics = None
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        trainer.train_epoch(epoch)

        if epoch % args.eval_interval == 0 or epoch == args.num_epochs:
            model.eval()
            print(f"\n  Epoch {epoch} ({(time.time() - t0) / 60:.1f} min):", flush=True)
            if extra_console is not None:
                extra_console(epoch)

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
                with open(os.path.join(out_dir, f"seed{seed}_best.txt"), "w") as f:
                    f.write("epoch\t" + "\t".join(METRIC_NAMES) + "\n")
                    f.write(f"{epoch}\t" + "\t".join(f"{m:.6f}" for m in collect_metrics(best_metrics)) + "\n")
                no_improve = 0
                print(f"    ⭐ Saved best checkpoint @ epoch {epoch}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"    ⏹ Early stop @ epoch {epoch} (no improvement for "
                          f"{args.patience} evals)", flush=True)
                    break

            t0 = time.time()

    trainer.close()
    return best_metrics, best_epoch


def write_summary_3seed(out_dir, ds, method_name, per_seed):
    """写 summary_3seed.txt（原 main L507-529）。

    std 用 np.std 默认 ddof=0（总体标准差）——与已发表 SASRec 文件逐字节一致所必需。
    """
    print("\n" + "=" * 60)
    print(f"  {ds} — {method_name} 3-seed Test Summary (best-val checkpoint)")
    print("=" * 60)
    summary = os.path.join(out_dir, "summary_3seed.txt")
    with open(summary, "w") as f:
        f.write(f"# {method_name} {ds} — 3-seed aligned test metrics\n")
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
    return summary


def load_summary_3seed(out_dir):
    """从既有产物复建 per_seed 字典（--summary_only 用）。

    优先读 seed{seed}_best.txt（精确）；缺失时回落到解析 seed{seed}_epochs.txt，
    取 val_ndcg@20 的首次最大值行（复刻 run_seed 的 '>' 严格比较语义，平局取更早 epoch）。
    """
    per_seed = {}
    if not os.path.isdir(out_dir):
        return per_seed
    for fn in sorted(os.listdir(out_dir)):
        if not (fn.startswith("seed") and fn.endswith("_epochs.txt")):
            continue
        try:
            seed = int(fn[len("seed"):-len("_epochs.txt")])
        except ValueError:
            continue

        best_txt = os.path.join(out_dir, f"seed{seed}_best.txt")
        if os.path.isfile(best_txt):
            with open(best_txt) as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
            if len(lines) >= 2:
                parts = lines[1].split("\t")
                per_seed[seed] = {
                    "metrics": [float(x) for x in parts[1:7]],
                    "best_epoch": int(parts[0]),
                }
                continue

        # 回落：解析 epochs.txt
        with open(os.path.join(out_dir, fn)) as f:
            rows = [ln.split("\t") for ln in f if ln.strip() and not ln.startswith("epoch")]
        if not rows:
            continue
        best_val, best_epoch, best_row = -1.0, 0, None
        for r in rows:
            if len(r) < 5:
                continue
            v = float(r[1])
            if v > best_val:
                best_val, best_epoch, best_row = v, int(r[0]), r
        if best_row is None:
            continue
        print(f"  ⚠ seed{seed}: 无 best.txt，由 epochs.txt 复建（best epoch={best_epoch}）")
        # epochs.txt 只有 N@20/R@20，其余指标缺失 → 用 NaN 占位并在 summary 中体现
        per_seed[seed] = {
            "metrics": [float("nan"), float("nan"), float(best_row[2]),
                        float("nan"), float("nan"), float(best_row[4])],
            "best_epoch": best_epoch,
        }
    return per_seed


def resolve_device(args):
    if args.device != "cpu":
        assert torch.cuda.is_available(), "cuda requested but unavailable"
        args.device = "cuda"


def guard_output_dir(args, method_name):
    """防止把新基线写进 SASRec 的权威产物目录（那里的数字已进论文）。"""
    leaf = os.path.basename(os.path.normpath(args.output_dir))
    if leaf.startswith("sasrec_output") and method_name != "SASRec":
        sys.exit(f"❌ --output_dir={args.output_dir} 指向 SASRec 产物目录，"
                 f"{method_name} 不得写入。请用独立目录。")


def add_common_args(parser):
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=200, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_interval", default=20, type=int,
                        help="每 N 个 epoch 评估一次")
    parser.add_argument("--patience", default=5, type=int,
                        help="val NDCG@20 连续 N 次评估无改善则早停")
    return parser


def run_datasets(args, datasets, *, method_name, model_factory, trainer_factory,
                 summary_only=False, extra_console=None):
    """数据集循环 + 多 seed 编排 + summary 写入（原 main L475-531）。"""
    for ds in datasets:
        data_path = os.path.join(args.data_dir, ds)
        out_dir = os.path.join(args.output_dir, ds)
        os.makedirs(out_dir, exist_ok=True)

        if summary_only:
            per_seed = load_summary_3seed(out_dir)
            if per_seed:
                write_summary_3seed(out_dir, ds, method_name, per_seed)
            continue

        print(f"\n{'=' * 60}\n=== {method_name} Baseline (aligned, 3-seed): {ds} ===\n{'=' * 60}")
        print(f"Output: {out_dir}")

        print("\n[1/3] Loading data...")
        (user_train, user_valid, user_val_seq,
         user_test, user_test_seq, itemnum) = load_tiger_data(data_path)

        dataset = build_dataset(user_train, user_valid, user_val_seq,
                                user_test, user_test_seq, itemnum)

        per_seed = {}
        for seed in args.seeds:
            best_metrics, best_epoch = run_seed(
                args, seed, dataset, out_dir,
                model_factory=model_factory,
                trainer_factory=trainer_factory,
                method_name=method_name,
                extra_console=extra_console,
            )
            if best_metrics is not None:
                per_seed[seed] = {
                    "metrics": collect_metrics(best_metrics),
                    "best_epoch": best_epoch,
                }

        write_summary_3seed(out_dir, ds, method_name, per_seed)

    print("\n✅ 全部完成.")
