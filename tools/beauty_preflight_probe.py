#!/usr/bin/env python3
"""
⚠️ **已退役（2026-09-27）—— 请改用 `tools/stage_probe.py` 作为单一真理源。**
================================================================================
本脚本的全部产出都是 `stage_probe.py` 的**真子集**，且后者口径更完整：

| 本脚本产出 | `stage_probe.py` 中的对应物 |
|---|---|
| 6 变体 × A 空间 AUC（pre-flight 排序） | `doc/g2_stage_probe_results.csv` 中 `space == "A"` 的行 |
| TA-SID legacy 4224d vs 896d 对照 | `python3 tools/stage_probe.py --audit_a` |
| Content-only ≡ ZeroPad 的等价性证明 | 同 A 空间行（两者输入文件相同，AUC 逐位相同） |

**已核验等价**（rqvae_seed=2024、topk × degmatch、A 空间）：
```
              Content-only  ItemCF  Raw-CoOcc  Shuffled  TA-SID  ZeroPad
Beauty              0.8368  0.9782     0.9088    0.8227  0.9939   0.8368
Sport               0.8964  0.9585     0.9498    0.8781  0.9617   0.8964
Toys                0.8579  0.9621     0.9359    0.8423  0.9828   0.8579
```
⇒ 本脚本与 stage_probe **逐位一致**，重复维护两个测同一件事的脚本只会让口径漂移。
**保留本文件仅为历史可追溯**（2026-09-27 之前的所有结论引用它），**不要再运行、不要再改**。

⚠ 并且本文件正是 2026-09-27 那个**文件维度 bug 的源头** —— 其 `FUSED` 表曾把 TA-SID 映射到
`fused_item_emb.parquet`（在 Sport/Toys 上是 **4224d**，只有 Beauty 上恰为 896d），
导致 TA-SID 的 A 空间 AUC 被记成 0.7835 / 0.8070（正确值 0.9617 / 0.9828）。
该映射已于同日修正，但**修好一个已被取代的脚本没有意义** —— 真理源在 stage_probe。

================================================================================
（以下为原始 docstring，保留不改）

beauty_preflight_probe.py — Beauty 补齐变体的 **零训练 pre-flight** 诊断

背景：Beauty 只有 Content-only / TA-SID 两个变体，缺 Raw-CoOcc / ItemCF /
ZeroPad / Shuffled，无法进 Behavior AUC × NDCG 相关性分析。补齐需要
4 × (RQ-VAE ~7min + 3-seed T5 ~3h) ≈ 12.5h GPU。

本脚本在投 GPU **之前**，用纯 CPU 的「量化前」行为结构 AUC 检验 Beauty 上的
变体排序是否符合预期，作为 go/no-go 判据。

⚠ 口径声明（重要）：
    本脚本测的是 **fused 896d（量化前）** 空间；
    主表（doc/behavior_structure_probe.md）测的是 **xq[full]（量化后 32d）**。
    两者的**绝对 AUC 不可比**，本脚本结果**不进主表**，只用于判断排序方向。

E 图 / 正样本 / 负样本 / similarity **逐位复用** tools/behavior_structure_probe.py，
保证与 Sport/Toys 实验口径一致：
    E     : train.parquet history+[target] → 滑窗 window=3（±1 相邻对），i<j
    pos   : topk（频次最高 1000 条边） | all（从 E 均匀采样）
    neg   : degmatch（⌊log2 deg⌋ 分桶，主口径） | uniform（参照）
    sim   : 余弦（行 L2 归一化后内积）

Content-only 的 fused 输入用 `fused_item_emb_zeropad.parquet`（= content ⊕ 128 零），
因为补零不改变余弦 → 与 content-only 的 768d 原生空间**数学等价**。脚本会打印
两者的 AUC 对照以证明该等价性（应逐位相同）。

只读约束：只读 parquet；写 CSV + markdown 报告。

用法：
    python3 tools/beauty_preflight_probe.py
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "rqvae"))

from behavior_structure_probe import (  # noqa: E402
    build_graph,
    edge_keys,
    sample_positives,
    sample_neg_degmatch,
    sample_neg_uniform,
    cos_sim,
    auc_of,
    load_ndcg,
    DATA_DIR,
    N_POS,
    N_NEG,
    SEED,
)
from sid_per_layer_diagnostic import normalize_rows  # noqa: E402

# ---------------------------------------------------------------- 变体 → 量化前输入
# Content-only 用 zeropad 文件（补零不改余弦）；ZeroPad 同名同文件 → 二者必然同值。
# 这本身是结论的一部分：量化前 Content-only ≡ ZeroPad。
# ⚠ 2026-09-27 订正（维度 bug）：TA-SID 原映射到 `fused_item_emb.parquet`，该文件在
# **Sport / Toys 上是 4224d**（4096d content ⊕ 128d 行为），只有 Beauty 上恰为 896d。
# 给 TA-SID 喂错维度的文件，使它在 A 空间被系统性判为末位，进而得出
# 「量化前指标无预测力、ρ(A,B)=+0.058、排序由 encoder 建立」的**错误结论**（见
# doc/beauty_preflight_probe_conclusion.md 顶部 2026-09-27 订正块，权威记录 note15 §10）。
# 正确的行为轴融合文件是 `fused_item_emb_768d.parquet`（768 content ⊕ 128 行为 = 896d，
# 与其余 5 个变体同维），且已核验它才是 TA-SID 各 ckpt 的真实 `args.data_path`。
FUSED = [
    ("Content-only", "fused_item_emb_zeropad.parquet"),
    ("ZeroPad",      "fused_item_emb_zeropad.parquet"),
    ("Shuffled",     "fused_item_emb_shuffled.parquet"),
    ("ItemCF",       "fused_item_emb_itemcf.parquet"),
    ("Raw-CoOcc",    "fused_item_emb_rawcooc.parquet"),
    ("TA-SID",       "fused_item_emb_768d.parquet"),   # ← 订正：原为 fused_item_emb.parquet
]

# 期望排序（来自 Sport/Toys 的 xq[full] 实测）
EXPECTED = ["TA-SID", "Raw-CoOcc", "ItemCF", "Content-only", "ZeroPad", "Shuffled"]


def load_fused(path, n_items):
    """读 fused parquet → (ItemID, 归一化 896d 矩阵)。"""
    df = pd.read_parquet(path)
    ids = df["ItemID"].values
    emb = np.stack(df["embedding"].values).astype(np.float32)
    if len(ids) != n_items:
        raise ValueError(f"{os.path.basename(path)}: n={len(ids)} != {n_items}")
    # normalize_rows 是 torch 实现（与主 probe 共用同一函数，避免口径漂移）
    return ids, normalize_rows(torch.from_numpy(emb)).numpy()


def main():
    print("=" * 96)
    print("⚠️ 本脚本已于 2026-09-27 退役。产出是 `tools/stage_probe.py` 的真子集，且曾是")
    print("   文件维度 bug 的源头。请改用：")
    print("       python3 tools/stage_probe.py            # A/B/C 三空间 × 6 变体 × 3 数据集")
    print("       python3 tools/stage_probe.py --audit_a  # TA-SID legacy 4224d vs 896d 对照")
    print("   本次运行仅为历史可追溯；**不要把它的输出当作结论**。")
    print("=" * 96)
    ap = argparse.ArgumentParser(description="[已退役] Beauty 补齐变体 pre-flight（量化前）诊断")
    ap.add_argument("--dataset", type=str, default="Beauty")
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--csv", type=str, default="doc/beauty_preflight_results.csv")
    ap.add_argument("--report", type=str, default="doc/beauty_preflight_probe.md")
    args = ap.parse_args()

    ds = args.dataset
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    log("=" * 100)
    log("Beauty 补齐变体 — 量化前（fused 896d）行为结构 pre-flight 诊断")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("⚠ 本表是量化前口径，绝对 AUC 与主表（xq[full]）不可比，勿混引")
    log(f"E 图: train.parquet history+[target] → 滑窗 window={args.window_size}（±1 相邻对）")
    log("=" * 100)

    rng = np.random.default_rng(args.seed)
    G = build_graph(ds, args.data_dir, args.window_size)
    n_items, deg = G["n_items"], G["deg"]
    edge_set = set(edge_keys(G["edges"], n_items).tolist())
    log(f"\nE 图: {n_items} items | {len(G['edges'])} 条边 "
        f"(count {G['counts'].min()}–{G['counts'].max()}) | {G['n_seqs']} 条训练序列")
    log(f"度分布: mean={deg.mean():.1f} median={np.median(deg):.0f} "
        f"max={deg.max()} 零度={int((deg == 0).sum())}")

    # ---- 正/负样本：与主 probe 逐位相同的构造 ----
    pos_regimes = {
        "topk": G["top_edges"],
        "all": sample_positives(G["edges"], args.n_pos, rng),
    }
    negcg = {}
    for pname, pos in pos_regimes.items():
        neg_deg, keep = sample_neg_degmatch(pos, deg, n_items, edge_set, rng)
        pos_ok = pos[keep]
        neg_unif = sample_neg_uniform(len(pos_ok), n_items, edge_set, rng)
        d_p = np.r_[deg[pos_ok[:, 0]], deg[pos_ok[:, 1]]]
        d_nd = np.r_[deg[neg_deg[:, 0]], deg[neg_deg[:, 1]]]
        d_nu = np.r_[deg[neg_unif[:, 0]], deg[neg_unif[:, 1]]]
        log(f"[{pname}] 可用正样本 {len(pos_ok)}/{len(pos)} | 端点 degree "
            f"pos={d_p.mean():.1f} degmatch={d_nd.mean():.1f} "
            f"(Δ={d_nd.mean() - d_p.mean():+.1f}) uniform={d_nu.mean():.1f} "
            f"(Δ={d_nu.mean() - d_p.mean():+.1f})")
        negcg[pname] = (pos_ok, neg_deg, neg_unif)

    # ---- 载入 6 个变体的融合输入，并核验 ItemID 行序一致 ----
    log("\n▌ 融合输入与行序核验")
    ref_ids = None
    embs = {}
    for name, fname in FUSED:
        path = os.path.join(args.data_dir, ds, fname)
        if not os.path.exists(path):
            log(f"  ⚠ {name:<14} 缺 {fname} → 跳过")
            continue
        ids, emb = load_fused(path, n_items)
        if ref_ids is None:
            ref_ids = ids
            log(f"  行序基准 = {fname}")
        elif not np.array_equal(ids, ref_ids):
            log(f"  ❌ {name:<14} ItemID 行序与基准不一致 → 跳过（防错位）")
            continue
        embs[name] = emb
        log(f"  ✅ {name:<14} {fname:34s} dim={emb.shape[1]}")

    # ---- 逐变体算 AUC ----
    rows = []
    for name in [n for n, _ in FUSED if n in embs]:
        emb = embs[name]
        for pname, (pos_ok, neg_deg, neg_unif) in negcg.items():
            cs_p = cos_sim(emb, pos_ok)
            for nname, neg in (("degmatch", neg_deg), ("uniform", neg_unif)):
                rows.append({
                    "dataset": ds, "variant": name, "space": "fused[896d]",
                    "pos_regime": pname, "neg_regime": nname,
                    "auc_cos": auc_of(cs_p, cos_sim(emb, neg)),
                    "cos_pos": float(cs_p.mean()),
                    "cos_neg": float(cos_sim(emb, neg).mean()),
                    "n_pos": len(pos_ok), "n_neg": len(neg),
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)
    log(f"\n→ CSV: {args.csv}  ({len(df)} 行)")

    # ---- 报告 ----
    ndcg = load_ndcg(ds)
    log("\n▌ 量化前 Behavior AUC（fused 896d）")
    for pname in ("topk", "all"):
        for nname in ("degmatch", "uniform"):
            sub = df[(df.pos_regime == pname) & (df.neg_regime == nname)] \
                .sort_values("auc_cos", ascending=False)
            if sub.empty:
                continue
            log(f"\n--- pos={pname} × neg={nname} ---")
            for _, r in sub.iterrows():
                nd = ndcg.get(r["variant"], (float("nan"), []))[0]
                log(f"    {r['variant']:<14} AUC={r['auc_cos']:.4f}   "
                    f"(cos_pos={r['cos_pos']:.3f} cos_neg={r['cos_neg']:.3f})   "
                    f"NDCG@20={nd:.4f}" if nd == nd else
                    f"    {r['variant']:<14} AUC={r['auc_cos']:.4f}   NDCG@20=—")

    # ---- 与期望排序比对（只对 degmatch 主口径、且能算出 NDCG 的变体）----
    log("\n▌ 排序比对（pos=topk × neg=degmatch，主口径）")
    sub = df[(df.pos_regime == "topk") & (df.neg_regime == "degmatch")] \
        .sort_values("auc_cos", ascending=False)
    auc_rank = list(sub["variant"])
    log(f"    量化前 AUC 排序: {' > '.join(auc_rank)}")
    log(f"    Sport/Toys 期望 : {' > '.join(EXPECTED)}")
    have = [v for v in EXPECTED if v in auc_rank]
    exp_sub = [v for v in EXPECTED if v in have]
    got_sub = [v for v in auc_rank if v in have]
    if len(have) >= 3:
        rho, p = spearmanr([exp_sub.index(v) for v in got_sub],
                           [got_sub.index(v) for v in got_sub])
        n_match = sum(1 for i in range(len(got_sub) - 1)
                      if exp_sub.index(got_sub[i]) < exp_sub.index(got_sub[i + 1]))
        log(f"    与期望排序的相邻正确对数: {n_match}/{len(got_sub) - 1}")

    # ---- Content-only(896d) vs content-only 原生 768d 的等价性核验 ----
    log("\n▌ 补零等价性核验（Content-only：896d 补零 vs 原生 768d）")
    native = os.path.join(args.data_dir, ds, "item_emb_768d_contentonly.parquet")
    if os.path.exists(native) and "Content-only" in embs:
        ndf = pd.read_parquet(native)
        nemb = normalize_rows(
            torch.from_numpy(np.stack(ndf["embedding"].values).astype(np.float32))).numpy()
        same_order = np.array_equal(ndf["ItemID"].values, ref_ids)
        log(f"    ItemID 行序一致: {same_order}")
        for pname, (pos_ok, neg_deg, _) in negcg.items():
            a896 = auc_of(cos_sim(embs["Content-only"], pos_ok),
                          cos_sim(embs["Content-only"], neg_deg))
            a768 = auc_of(cos_sim(nemb, pos_ok), cos_sim(nemb, neg_deg))
            log(f"    [{pname}] AUC 896d(补零)={a896:.6f}  vs  768d(原生)={a768:.6f}  "
                f"Δ={a896 - a768:+.2e}")

    log("\n⚠️ 口径提醒：本报告为**量化前**结果，绝对 AUC 高于主表（xq[full]）属正常，"
        "两者不可混引；本报告只用于判断**排序方向** go/no-go。")

    with open(args.report, "w") as f:
        f.write("```\n" + "\n".join(log_lines) + "\n```\n")
    log(f"→ 报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
