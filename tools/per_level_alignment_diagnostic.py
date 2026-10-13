#!/usr/bin/env python3
"""
per_level_alignment_diagnostic.py — 逐层「语义 / 协同」对齐诊断（GrIS Table-8 式，跨变体扩展）

目的：回答「SID 的第 l 层码字，是按**内容**分的还是按**行为**分的？」
      并把这个问题从单方法自述（GrIS 只测自己的方法）扩展为 **6 变体 × 3 数据集**的对照。

与既有 `sid_per_layer_diagnostic.py` 的区别（两者互补，不替代）：
    sid_per_layer_diagnostic : 测「转移对 vs 随机对」的 AUC，逐层**累积** xq，只测行为侧
    本脚本                    : 测「同前缀组内 vs 随机对」的 Δcos，逐层**按码字分组**，内容/行为/协同三空间对照
                               —— 即 GrIS Table 8 的 δ_sem vs δ_CF 语义，外加一个外部协同参照 δ_SASRec

测量空间：
    content : data/{ds}/item_emb_768d_contentonly.parquet      (768d, 跨变体共享)
    beh     : data/{ds}/behavior_emb.parquet                   (128d PPMI+SVD, TA-SID 所用信号, 跨变体共享)
    sasrec  : sasrec_output/{ds}/sasrec_{ds}_best.pth          (50d 序列模型嵌入, 外部协同参照)

指标（每个 数据集 × 变体 × 层 l ∈ {1,2,3}）：
    δ_X(l) = within-prefix-group mean cosine − random-pair mean cosine   (X ∈ {content, beh, sasrec})
    gap(l) = δ_beh(l) − δ_content(l)      —— GrIS 的 δ_CF − δ_sem，>0 表示该层更偏行为侧
另报：collision@3 / max_group / util_L0 / Gini@3（码本健康度，与「ID 唯一性」挑战直接相关）

⚠️ 第 4 列 (L3) 是**碰撞消解序号**，不是语义层，不参与分组（见 rqvae/generate_sid_codes_param.py 文件头）。

约束：纯只读。不训练、不改任何实验产物，只输出一份 markdown 报告。

用法：
    python3 tools/per_level_alignment_diagnostic.py \
        [--datasets Beauty,Sport,Toys] [--n_random 5000] \
        [--report doc/per_level_alignment_diagnostic.md]
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)

from sid_consistency_contrast import find_best_ckpt, load_codebook_weights  # noqa: E402

# ---------------------------------------------------------------- 变体配置
# (显示名, ckpt 目录名, codes 文件名) —— 由 rqvae/verify_variant_pairing.py 的配对清单派生
VARIANTS = {
    "Beauty": [
        ("TA-SID",       "Beauty_TA_SID",             "Beauty_ta_sid_768d_sid.npy"),
        ("Content-only", "Beauty_Content768d_Baseline", "Beauty_content768d_baseline_sid.npy"),
        ("ZeroPad",      "Beauty_Zeropad_768d",       "Beauty_zeropad_768d_sid.npy"),
        ("Shuffled",     "Beauty_Shuffled_768d",      "Beauty_shuffled_768d_sid.npy"),
        ("Raw-CoOcc",    "Beauty_RawCoOcc_768d",      "Beauty_rawcooc_768d_sid.npy"),
        ("ItemCF",       "Beauty_ItemCF_768d",        "Beauty_itemcf_768d_sid.npy"),
    ],
    "Sport": [
        ("TA-SID",       "Sport_TA_SID_768d",         "Sport_ta_sid_768d_sid.npy"),
        ("Content-only", "Sport_Content768d_Baseline", "Sport_content768d_baseline_sid.npy"),
        ("ZeroPad",      "Sport_Zeropad_Ablation",    "Sport_zeropad_ablation_sid.npy"),
        ("Shuffled",     "Sport_Shuffled_Ablation",   "Sport_shuffled_ablation_sid.npy"),
        ("Raw-CoOcc",    "Sport_RawCoOcc_768d",       "Sport_rawcooc_768d_sid.npy"),
        ("ItemCF",       "Sport_ItemCF_768d",         "Sport_itemcf_768d_sid.npy"),
    ],
    "Toys": [
        ("TA-SID",       "Toys_TA_SID_768d",          "Toys_ta_sid_768d_seed2024_sid.npy"),
        ("Content-only", "Toys_Content768d_Baseline", "Toys_content768d_baseline_sid.npy"),
        ("ZeroPad",      "Toys_Zeropad_seed2024",     "Toys_zeropad_seed2024_sid.npy"),
        ("Shuffled",     "Toys_Shuffled_768d",        "Toys_shuffled_768d_sid.npy"),
        ("Raw-CoOcc",    "Toys_RawCoOcc_768d",        "Toys_rawcooc_768d_sid.npy"),
        ("ItemCF",       "Toys_ItemCF_768d",          "Toys_itemcf_768d_sid.npy"),
    ],
}

N_SEM_LAYERS = 3   # L0/L1/L2；L3 是碰撞消解序号


# ---------------------------------------------------------------- 工具函数
def normalize_rows(v):
    """按行 L2 归一化（float32 numpy）。"""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.clip(n, 1e-9, None)).astype(np.float32)


def load_parquet_emb(path):
    import pandas as pd
    df = pd.read_parquet(path)
    return normalize_rows(np.stack(df["embedding"].values, axis=0).astype(np.float32))


def load_sasrec_emb(ds):
    """取 SASRec 的 item embedding 表。行 0 是 padding，item i(1-based) → 行 i。"""
    p = os.path.join("sasrec_output", ds, f"sasrec_{ds}_best.pth")
    if not os.path.exists(p):
        return None
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    w = sd["item_emb.weight"].float().numpy()[1:]   # 丢掉 padding 行
    return normalize_rows(w)


def group_by_keys(keys):
    """按 keys（每行一个分组标识）分组，返回 [(indices...), ...]，只保留 size>=2 的组。"""
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    idx_by_group = [[] for _ in range(len(counts))]
    for i, g in enumerate(inv):
        idx_by_group[g].append(i)
    return [np.asarray(g, dtype=np.int64) for g in idx_by_group if len(g) >= 2]


def prefix_groups(codes, l):
    """累积前缀分组：按前 l 列分组（GrIS Table-8 口径）。深层会退化为单例。"""
    return group_by_keys(np.array([tuple(row) for row in codes[:, :l]]))


def level_groups(codes, l):
    """单层码字分组：只看第 l-1 列（1-based l）。各层恒为 256 个非空组，不会退化。"""
    return group_by_keys(codes[:, [l - 1]].astype(np.int64))


def within_group_mean_cos(V, groups):
    """组内平均成对余弦（用 Σv 的平方和精确求得，O(Σ|g|·D)）。"""
    num, den = 0.0, 0
    for idx in groups:
        n = len(idx)
        S = V[idx].sum(axis=0)
        num += float(S @ S) - n          # Σ_{i≠j} v_i·v_j
        den += n * (n - 1)
    return num / den if den > 0 else float("nan")


def random_pair_mean_cos(V, pairs):
    return float((V[pairs[:, 0]] * V[pairs[:, 1]]).sum(axis=1).mean())


def gini(sizes):
    x = np.sort(np.asarray(sizes, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="Per-level semantic/collaborative alignment diagnostic")
    ap.add_argument("--datasets", type=str, default="Beauty,Sport,Toys")
    ap.add_argument("--n_random", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", type=str, default="doc/per_level_alignment_diagnostic.md")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lines = []
    detail_rows = []   # (ds, variant, l, d_content, d_beh, d_sas, gap, cover)  累积前缀
    single_rows = []   # (ds, variant, l, d_content, d_beh, d_sas, gap, n_grp)   单层码字
    health_rows = []   # (ds, variant, N, coll3, maxgrp, util_l0, gini3, l3_distinct)

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 108)
    log("Per-level 语义/协同对齐诊断（GrIS Table-8 式 δ_sem vs δ_CF，跨 6 变体扩展）")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  seed={args.seed}  n_random={args.n_random}")
    log("δ_X(l) = 前缀组内平均成对余弦 − 随机对平均余弦（同一条随机对集跨变体固定，保证可比）")
    log("gap(l) = δ_beh(l) − δ_content(l)；>0 表示第 l 层的分组更偏行为侧")
    log("=" * 108)

    for ds in [d.strip() for d in args.datasets.split(",")]:
        if ds not in VARIANTS:
            log(f"⚠ 跳过未知数据集: {ds}")
            continue

        c_path = os.path.join("data", ds, "item_emb_768d_contentonly.parquet")
        b_path = os.path.join("data", ds, "behavior_emb.parquet")
        if not (os.path.exists(c_path) and os.path.exists(b_path)):
            log(f"⚠ {ds}: 缺 content/behavior parquet，跳过")
            continue

        C = load_parquet_emb(c_path)
        B = load_parquet_emb(b_path)
        S = load_sasrec_emb(ds)
        N = C.shape[0]

        # 固定随机对（跨变体、跨空间同一集合）
        ri = rng.integers(0, N, size=args.n_random)
        rj = rng.integers(0, N, size=args.n_random)
        mask = ri != rj
        rand_pairs = np.stack([ri[mask], rj[mask]], axis=1)
        base_c = random_pair_mean_cos(C, rand_pairs)
        base_b = random_pair_mean_cos(B, rand_pairs)
        base_s = random_pair_mean_cos(S, rand_pairs) if S is not None else float("nan")

        log("")
        log(f"### {ds}  (N={N}, 随机对基线: content={base_c:.4f}  beh={base_b:.4f}  sasrec={base_s:.4f})")

        for name, ckpt_tag, code_file in VARIANTS[ds]:
            code_path = os.path.join("data", ds, code_file)
            if not os.path.exists(code_path):
                log(f"\n  ▸ {name:12s}   ⚠ codes 缺失: {code_file}")
                continue
            codes = np.load(code_path)
            if codes.shape[0] != N:
                log(f"\n  ▸ {name:12s}   ⚠ codes 行数 {codes.shape[0]} ≠ {N}，跳过")
                continue
            ckpt = find_best_ckpt(ckpt_tag)
            cb = load_codebook_weights(ckpt) if ckpt else None
            n_layers = len(cb) if cb else N_SEM_LAYERS

            log(f"\n  ▸ {name}  ({ckpt_tag})")
            n_sem = min(N_SEM_LAYERS, codes.shape[1] - 1)   # 末列是碰撞消解序号

            def measure(groups):
                d_c = within_group_mean_cos(C, groups) - base_c
                d_b = within_group_mean_cos(B, groups) - base_b
                d_s = (within_group_mean_cos(S, groups) - base_s) if S is not None else float("nan")
                return d_c, d_b, d_s, d_b - d_c

            for l in range(1, n_sem + 1):
                # (a) 累积前缀分组（GrIS Table-8 口径；深层会退化）
                pg = prefix_groups(codes, l)
                if pg:
                    d_c, d_b, d_s, gap = measure(pg)
                    cover = sum(len(g) for g in pg) / N
                    detail_rows.append((ds, name, l, d_c, d_b, d_s, gap, cover))
                    flag = " ⚠退化" if cover < 0.05 else ""
                    sas_s = f"{d_s:+.4f}" if S is not None else "   -   "
                    log(f"    [前缀 l={l}] δ_content={d_c:+.4f} δ_beh={d_b:+.4f} δ_sasrec={sas_s} "
                        f"gap={gap:+.4f} | {len(pg)} 组, 覆盖 {cover:.1%}{flag}")
                # (b) 单层码字分组（恒 256 组，不退化）
                lg = level_groups(codes, l)
                d_c1, d_b1, d_s1, gap1 = measure(lg)
                single_rows.append((ds, name, l, d_c1, d_b1, d_s1, gap1, len(lg)))
                sas_s1 = f"{d_s1:+.4f}" if S is not None else "   -   "
                log(f"    [单层 l={l}] δ_content={d_c1:+.4f} δ_beh={d_b1:+.4f} δ_sasrec={sas_s1} "
                    f"gap={gap1:+.4f} | {len(lg)} 组, 覆盖 100.0%")

            # ---- 码本健康度 / ID 唯一性 ----
            from collections import Counter
            sem_keys = [tuple(r) for r in codes[:, :n_sem]]
            cnt = Counter(sem_keys)
            sizes = list(cnt.values())
            coll3 = 1.0 - len(cnt) / N
            util_l0 = len(np.unique(codes[:, 0]))
            l3d = len(np.unique(codes[:, codes.shape[1] - 1])) if codes.shape[1] > n_sem else 0
            g3 = gini(sizes)
            health_rows.append((ds, name, N, coll3, max(sizes), util_l0, g3, l3d))
            log(f"    [健康度] util(L0)={util_l0}/256  collision@3={coll3:.4%}  "
                f"max_group={max(sizes)}  Gini@3={g3:.4f}  L3 distinct={l3d}")

    # ================= 汇总：δ_content / δ_beh 逐层 =================
    log("")
    log("=" * 108)
    log("汇总 A：单层码字分组的对齐度（每个码字一 组，恒 256 组，不退化 —— 主口径）")
    log("δ_X(l) = 组内平均成对余弦 − 随机对基线；gap = δ_beh − δ_content（>0 表示该层更偏行为侧）")
    log("=" * 108)
    for ds in [d.strip() for d in args.datasets.split(",")]:
        rows = [r for r in single_rows if r[0] == ds]
        if not rows:
            continue
        log(f"\n── {ds} ──")
        hdr = f"{'变体':14s} {'l':>2s} {'δ_content':>10s} {'δ_beh':>10s} {'δ_sasrec':>10s} {'gap(beh-con)':>13s}"
        log(hdr)
        log("-" * len(hdr))
        for _, name, l, d_c, d_b, d_s, gap, _n in rows:
            ss = f"{d_s:+.4f}" if not np.isnan(d_s) else "   -   "
            log(f"{name:14s} {l:>2d} {d_c:>+10.4f} {d_b:>+10.4f} {ss:>10s} {gap:>+13.4f}")

    log("")
    log("=" * 108)
    log("汇总 A'：累积前缀分组（GrIS Table-8 口径；覆盖 <5% 的层已退化，不可与其它层并列解读）")
    log("=" * 108)
    for ds in [d.strip() for d in args.datasets.split(",")]:
        rows = [r for r in detail_rows if r[0] == ds]
        if not rows:
            continue
        log(f"\n── {ds} ──")
        hdr = f"{'变体':14s} {'l':>2s} {'δ_content':>10s} {'δ_beh':>10s} {'δ_sasrec':>10s} {'gap':>9s} {'覆盖':>8s}"
        log(hdr)
        log("-" * len(hdr))
        for _, name, l, d_c, d_b, d_s, gap, cov in rows:
            ss = f"{d_s:+.4f}" if not np.isnan(d_s) else "   -   "
            flag = "⚠" if cov < 0.05 else " "
            log(f"{name:14s} {l:>2d} {d_c:>+10.4f} {d_b:>+10.4f} {ss:>10s} {gap:>+9.4f} {cov:>7.1%}{flag}")

    # ================= 汇总：码本健康度 =================
    log("")
    log("=" * 108)
    log("汇总 B：码本健康度 / ID 唯一性（collision@3 = 前 3 层仍无法区分的物品占比）")
    log("=" * 108)
    hdr = f"{'Dataset':8s} {'变体':14s} {'N':>7s} {'collision@3':>12s} {'max_group':>10s} {'util(L0)':>9s} {'Gini@3':>8s} {'L3_uniq':>8s}"
    log(hdr)
    log("-" * len(hdr))
    for ds, name, N, coll3, mx, util, g3, l3d in health_rows:
        log(f"{ds:8s} {name:14s} {N:>7d} {coll3:>12.4%} {mx:>10d} {util:>9d} {g3:>8.4f} {l3d:>8d}")

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Per-level 语义/协同对齐诊断（跨 6 变体）\n\n")
        f.write("```text\n")
        f.write("\n".join(lines))
        f.write("\n```\n")
    print(f"\n报告已写入: {args.report}")


if __name__ == "__main__":
    main()
