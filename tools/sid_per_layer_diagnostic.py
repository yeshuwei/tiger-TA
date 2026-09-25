#!/usr/bin/env python3
"""
sid_per_layer_diagnostic.py — 逐层转移结构保留诊断 (per-layer transition-structure preservation)

目的：回答 "transition 结构在 residual VQ 的哪一层被保留 / 丢失？"
      这是 "Track B 增强版 / 第二创新方向" 的判决性诊断：
        - 若各层 AUC 均高 → 量化未破坏结构 → 增强损失无立足点
        - 若某层显著丢失 → 存在真实、可定位的 gap → 增强方向值得继续
        - 若 z 高而 xq 低 → 是"量化环节"丢的（而非编码器）
        - 若 input 就低 → 转移信号在表征里本来就不强（信号冗余问题）

约束：纯只读。只读取已生成的 SID codes + checkpoint + 输入 embedding，
      不训练、不写任何实验产物，仅输出一份 markdown 诊断报告（默认 doc/）。

测量空间（每个 数据集×方法）：
    input     : 模型输入 embedding（量化前，理论天花板）
    z         : 编码器输出（量化前）
    xq[L0]    : 量化重建 = 仅第 0 层 codebook
    xq[L0+L1] : 量化重建 = 前两层 codebook 之和
    xq[full]  : 量化重建 = 三层之和（≡ §5.3 口径，用于 sanity check）

指标：AUC = P(转移对余弦相似度 > 随机对余弦相似度)，Mann-Whitney U。
      0.5=随机，1=完美。AUC 是纯排序指标，剥离绝对聚集度干扰（§5.3 的论证）。

用法：
    python3 tools/sid_per_layer_diagnostic.py \
        [--datasets Beauty,Sport,Toys] [--topk 1000] [--n_random 5000] \
        [--report doc/per_layer_transition_diagnostic.md]
"""

import argparse
import os
import sys
import datetime

import numpy as np
import pandas as pd
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)

from sid_consistency_contrast import (
    CONFIG,
    find_best_ckpt,
    load_codebook_weights,
    top_transition_pairs,
    pair_sims,
    random_pair_sims,
    compute_auc,
)

RQEAV_DIR = os.path.abspath(os.path.join(TOOLS_DIR, "..", "rqvae"))
sys.path.insert(0, RQEAV_DIR)

# ---- §5.3 已知结果，用于 sanity check ----
KNOWN_AUC = {
    "Beauty": {"TA-SID": 0.869, "Content-only": 0.753},
    "Sport":  {"TA-SID": 0.962, "Content-only": 0.900},
    "Toys":   {"TA-SID": 0.872, "Content-only": 0.863},
}


def rebuild_xq_partial(codes, codebooks, n_layers):
    """用 SID code 前 n_layers 列重建量化向量（部分和）。codes: (N, 4) → (N, e_dim)"""
    xq = torch.zeros(codes.shape[0], codebooks[0].shape[1])
    for layer in range(n_layers):
        idx = torch.as_tensor(codes[:, layer], dtype=torch.long)
        xq = xq + codebooks[layer][idx]
    return xq


def normalize_rows(v):
    """按行 L2 归一化。"""
    norms = v.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return v / norms


def load_input_emb(data_path):
    """按模型训练口径加载输入 embedding。返回 (N, D) float32 torch 张量。"""
    df = pd.read_parquet(data_path)
    embs = np.stack(df["embedding"].values, axis=0).astype(np.float32)
    return torch.from_numpy(embs)


def encode_z(ckpt_path, data_path, device="cpu"):
    """运行编码器得到 z（量化前 latent）。返回 (N, e_dim) 归一化张量。

    与代码生成脚本同一口径：EmbDataset(data_path) → model.encoder(x)。
    """
    from torch.utils.data import DataLoader
    from datasets import EmbDataset
    from models.rqvae import RQVAE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    data = EmbDataset(data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=a.num_emb_list,
                  e_dim=a.e_dim,
                  layers=a.layers,
                  dropout_prob=a.dropout_prob,
                  bn=a.bn,
                  loss_type=a.loss_type,
                  quant_loss_weight=a.quant_loss_weight,
                  kmeans_init=a.kmeans_init,
                  kmeans_iters=a.kmeans_iters,
                  sk_epsilons=a.sk_epsilons,
                  sk_iters=a.sk_iters)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    loader = DataLoader(data, batch_size=512, shuffle=False, num_workers=0)
    zs = []
    with torch.no_grad():
        for batch in loader:
            zs.append(model.encoder(batch.to(device)).cpu())
    z = torch.cat(zs, dim=0)
    return normalize_rows(z)


def auc_report(pos, neg):
    """返回 (auc, pos_mean, neg_mean, mean_diff) 元组。"""
    auc = compute_auc(pos, neg)
    return auc, float(pos.mean()), float(neg.mean()), float(pos.mean() - neg.mean())


def main():
    parser = argparse.ArgumentParser(description="Per-layer transition-structure preservation diagnostic")
    parser.add_argument("--datasets", type=str, default="Beauty,Sport,Toys")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--n_random", type=int, default=5000)
    parser.add_argument("--report", type=str,
                        default="doc/per_layer_transition_diagnostic.md")
    parser.add_argument("--skip_z", action="store_true",
                        help="跳过 z 空间（编码器前向）计算，只做量化空间")
    args = parser.parse_args()

    report_lines = []
    def log(s=""):
        print(s)
        report_lines.append(s)

    log("=" * 100)
    log("Per-layer 转移结构保留诊断（transition-structure preservation across RQ-VAE layers）")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"转移对: top-{args.topk} 高频相邻对 | 随机对: {args.n_random} (seed=0)")
    log("AUC = P(转移对 cos > 随机对 cos)，0.5=随机，1=完美")
    log("=" * 100)

    summary_rows = []  # (ds, method, input_auc, z_auc, l0, l01, full, note)

    for ds in [d.strip() for d in args.datasets.split(",")]:
        if ds not in CONFIG:
            log(f"⚠ 跳过未知数据集: {ds}")
            continue
        cfg = CONFIG[ds]

        pairs = top_transition_pairs(ds, top_k=args.topk)
        if not pairs:
            log(f"{ds}: 无转移对，跳过")
            continue
        pair_ab = [(a - 1, b - 1) for (a, b), c in pairs]

        log("")
        log(f"### {ds}  |  top-{len(pairs)} 高频转移对")

        for method, cktag, codef in [
            ("TA-SID", cfg["ta_ckpt"], cfg["ta_code"]),
            ("Content-only", cfg["base_ckpt"], cfg["base_code"]),
        ]:
            ckpt = find_best_ckpt(cktag)
            if ckpt is None:
                log(f"  ⚠ checkpoint 缺失: {cktag}")
                continue
            codes = np.load(os.path.join("data", ds, codef))
            cb = load_codebook_weights(ckpt)
            n_layer_full = len(cb)

            log(f"\n  ▸ {method}  ({cktag})")
            log(f"    codes={codes.shape} | codebook {n_layer_full} 层, e_dim={cb[0].shape[1]}")

            # ---- 固定随机对（跨空间可比）----
            row = {}

            # (1) input 空间：模型训练输入
            a = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
            in_path = getattr(a, "data_path", None)
            if in_path and os.path.exists(in_path):
                in_n = normalize_rows(load_input_emb(in_path))
                pos = pair_sims(in_n, pair_ab)
                neg = random_pair_sims(in_n, n_pairs=args.n_random)
                auc, pm, nm, md = auc_report(pos, neg)
                row["input"] = auc
                log(f"    input({os.path.basename(in_path)}): AUC={auc:.4f}  "
                    f"转移cos={pm:.4f} 随机cos={nm:.4f} Δ={md:+.4f}")

            # (2) z 空间：编码器输出（量化前）
            if not args.skip_z:
                try:
                    z_n = encode_z(ckpt, in_path if in_path and os.path.exists(in_path) else os.path.join("data", ds, codef))
                    pos = pair_sims(z_n, pair_ab)
                    neg = random_pair_sims(z_n, n_pairs=args.n_random)
                    auc, pm, nm, md = auc_report(pos, neg)
                    row["z"] = auc
                    log(f"    z (编码器输出): AUC={auc:.4f}  "
                        f"转移cos={pm:.4f} 随机cos={nm:.4f} Δ={md:+.4f}")
                except Exception as e:
                    log(f"    z 计算失败: {e}")

            # (3) 量化空间：逐层部分和
            for n_layers in range(1, n_layer_full + 1):
                xq = rebuild_xq_partial(codes, cb, n_layers)
                xq_n = normalize_rows(xq)
                pos = pair_sims(xq_n, pair_ab)
                neg = random_pair_sims(xq_n, n_pairs=args.n_random)
                auc, pm, nm, md = auc_report(pos, neg)
                tag = "xq[L0]" if n_layers == 1 else \
                      "xq[L0+L1]" if n_layers == 2 else "xq[full]"
                row[tag] = auc
                log(f"    {tag}: AUC={auc:.4f}  "
                    f"转移cos={pm:.4f} 随机cos={nm:.4f} Δ={md:+.4f}")

            # ---- sanity check vs §5.3 ----
            full_auc = row.get("xq[full]")
            known = KNOWN_AUC[ds].get(method)
            note = ""
            if full_auc is not None and known is not None:
                diff = abs(full_auc - known)
                if diff < 0.005:
                    note = "✓ sanity OK"
                else:
                    note = f"⚠ 与§5.3({known:.3f})差 {diff:.3f}"
                    log(f"    ⚠ sanity: xq[full]={full_auc:.4f} vs §5.3 记录 {known:.3f} (Δ{diff:+.4f})")

            summary_rows.append((
                ds, method,
                row.get("input", float("nan")),
                row.get("z", float("nan")),
                row.get("xq[L0]", float("nan")),
                row.get("xq[L0+L1]", float("nan")),
                full_auc if full_auc is not None else float("nan"),
                note,
            ))

    # ================= 汇总表 =================
    log("")
    log("=" * 100)
    log("汇总表（AUC，越高越好；0.5=随机）")
    log("=" * 100)
    hdr = f"{'Dataset':8s} {'Method':12s} {'input':>7s} {'z':>7s} {'xq[L0]':>7s} {'xq[L0+L1]':>9s} {'xq[full]':>9s}  note"
    log(hdr)
    log("-" * len(hdr))
    for ds, method, ia, za, l0, l01, full, note in summary_rows:
        fmt = lambda v: "   -  " if np.isnan(v) else f"{v:.4f}"
        log(f"{ds:8s} {method:12s} {fmt(ia):>7s} {fmt(za):>7s} {fmt(l0):>7s} {fmt(l01):>9s} {fmt(full):>9s}  {note}")

    # ================= 诊断结论 =================
    log("")
    log("=" * 100)
    log("诊断解读（判决第二创新方向是否值得继续）")
    log("=" * 100)
    for ds, method, ia, za, l0, l01, full, note in summary_rows:
        if method != "TA-SID":
            continue
        log(f"\n── {ds} (TA-SID) ──")
        def f(v):
            return float("nan") if np.isnan(v) else v
        if not np.isnan(f(ia)) and not np.isnan(f(full)):
            log(f"  input→xq[full]: {f(ia):.3f} → {f(full):.3f}"
                f"  (Δ{full - ia:+.3f})  {'⚠ 量化显著丢失结构' if full - ia < -0.05 else '✓ 量化基本保留结构' if full - ia > -0.02 else '~ 小幅度变化'}")
        if not np.isnan(f(za)) and not np.isnan(f(full)):
            log(f"  z→xq[full]: {f(za):.3f} → {f(full):.3f}"
                f"  (Δ{full - za:+.3f})  {'⚠ 量化环节丢失结构' if full - za < -0.05 else '✓ 量化环节无丢失' if full - za > -0.02 else '~ 小幅度变化'}")
        if not np.isnan(f(l0)):
            log(f"  分层: L0={f(l0):.3f}  L0+L1={f(l01):.3f}  full={f(full):.3f}"
                f"  → {'结构主要活在粗层' if f(l0) >= f(full) - 0.02 else '结构需细层累加才出现' if f(l0) < 0.7 else '粗层部分保留，细层补充'}")

    log("")
    log("判定规则：")
    log("  [1] 若 xq[L0] 即≈xq[full]（粗层已保留全部结构）→ 不存在'量化破坏'问题 → 增强损失无立足点")
    log("  [2] 若 z 高而 xq[full] 低 → 丢失发生在量化环节 → 增强方向有真实靶点")
    log("  [3] 若 input 就低 → 转移信号在输入表征里本来就弱 → 是信号/冗余问题，非量化问题")

    # ---- 写报告 ----
    report_path = args.report
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Per-layer 转移结构保留诊断\n\n")
        f.write("```text\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")
    print(f"\n报告已写入: {report_path}")


if __name__ == "__main__":
    main()
