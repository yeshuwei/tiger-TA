#!/usr/bin/env python3
"""
beauty_tier1_probe.py — Beauty Tier-1：RQ-VAE encoder + 量化 的行为结构保持诊断

范围（严格）：
    只测 **z（32d，encoder 输出）** 与 **xq[full]（32d，码本重建）** 两个空间。
    **禁止**用 fused 896d 作判断依据（已在 doc/beauty_preflight_probe_conclusion.md
    §2 证明其与真实排序 ρ=+0.058、且把 TA-SID 判为末位 → 无预测力）。
    ⚠ 2026-09-27 订正：该"无预测力"结论**已作废** —— +0.058 是文件维度 bug（Sport/Toys 的
    TA-SID 被喂了 4224d 文件）。修正后 ρ(A输入, C最终) = 0.928 / 0.928 / 0.986，896d 是**有效**
    的判断空间。本脚本**仍只测 z 与 xq[full]**（保持与既有记录可比，且各自结论已独立成立），
    但"禁止用 896d"这一理由不再是"它无预测力"。见 note15 §10。
    **不跑 T5、不算 NDCG**（那属 Tier-2）。

所有可复用逻辑**逐位复用** tools/behavior_structure_probe.py：
    E 图（train.parquet history+[target] → 滑窗 window=3）、
    正样本（topk = 频次最高 1000 条边 / all = 从 E 均匀采样）、
    负样本（degmatch = ⌊log2 deg⌋ 分桶，主口径；uniform = 参照）、
    cosine similarity AUC、以及 z / xq[full] 的构造（normalize_rows / encode_z /
    rebuild_xq_partial）。**未重新定义 Behavior AUC**。

变体定义与 Sport/Toys 完全一致，仅 dataset=Beauty：
    Content-only / ZeroPad / Shuffled / ItemCF / Raw-CoOcc / TA-SID

用法：
    python3 tools/beauty_tier1_probe.py
"""

import argparse
import datetime
import glob
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
    build_graph, edge_keys, sample_positives, sample_neg_degmatch,
    sample_neg_uniform, cos_sim, auc_of, load_ndcg,
    DATA_DIR, N_POS, N_NEG, SEED, TOPK_POS,
)
from sid_per_layer_diagnostic import (  # noqa: E402
    encode_z, normalize_rows, rebuild_xq_partial,
)
from sid_consistency_contrast import (  # noqa: E402
    find_best_ckpt, load_codebook_weights,
)

# Beauty 六变体（ckpt_tag / codes_file 命名遵循 rqvae/run_behavior_ablation.sh 约定）
BEAUTY_VARIANTS = [
    ("Content-only", "Beauty_Content768d_Baseline", "Beauty_content768d_baseline_sid.npy"),
    ("ZeroPad",      "Beauty_Zeropad_768d",         "Beauty_zeropad_768d_sid.npy"),
    ("Shuffled",     "Beauty_Shuffled_768d",        "Beauty_shuffled_768d_sid.npy"),
    ("ItemCF",       "Beauty_ItemCF_768d",          "Beauty_itemcf_768d_sid.npy"),
    ("Raw-CoOcc",    "Beauty_RawCoOcc_768d",        "Beauty_rawcooc_768d_sid.npy"),
    ("TA-SID",       "Beauty_TA_SID",               "Beauty_ta_sid_768d_sid.npy"),
]

# Sport/Toys 的 xq[full] 实测排序（作为一致性参照，不强行期待）
REF_ORDER = ["TA-SID", "Raw-CoOcc", "ItemCF", "Content-only", "ZeroPad", "Shuffled"]


def main():
    ap = argparse.ArgumentParser(description="Beauty Tier-1 probe（z / xq[full]）")
    ap.add_argument("--dataset", type=str, default="Beauty")
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--main_csv", type=str,
                    default="doc/behavior_structure_probe_results.csv",
                    help="Sport/Toys 既有结果（用于跨数据集 z-xq 一致性表）")
    ap.add_argument("--csv", type=str, default="doc/beauty_tier1_results.csv")
    ap.add_argument("--report", type=str, default="doc/beauty_tier1_probe.md")
    args = ap.parse_args()

    ds = args.dataset
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    log("=" * 100)
    log("Beauty Tier-1 — RQ-VAE encoder(z) 与量化端点(xq[full]) 的行为结构保持诊断")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("空间: z (32d, encoder 输出) | xq[full] (32d, 码本重建)　—— 不含 fused 896d")
    log("口径: 与 tools/behavior_structure_probe.py 逐位一致（E 图 window=3 / degmatch 负样本 / cosine AUC）")
    log("=" * 100)

    # ---------------- 缺失报告（§三：缺 ckpt 不训练，先报告） ----------------
    log("\n▌ 变体资源核验")
    missing = []
    for name, tag, codef in BEAUTY_VARIANTS:
        ck = find_best_ckpt(tag)
        cp = os.path.join(args.data_dir, ds, codef)
        ok_c, ok_k = ck is not None, os.path.exists(cp)
        log(f"  {name:<14} ckpt {'✅' if ok_c else '❌'} {tag:<32} "
            f"codes {'✅' if ok_k else '❌'} {codef}")
        if not (ok_c and ok_k):
            missing.append(name)
    if missing:
        log(f"\n  ❌ 缺失变体: {', '.join(missing)} → 无法完成 Tier-1（需先补 RQ-VAE）")

    # ---------------- 正/负样本（与主 probe 同一构造） ----------------
    rng = np.random.default_rng(args.seed)
    G = build_graph(ds, args.data_dir, args.window_size)
    n_items, deg = G["n_items"], G["deg"]
    edge_set = set(edge_keys(G["edges"], n_items).tolist())
    log(f"\nE 图: {n_items} items | {len(G['edges'])} 条边 "
        f"(count {G['counts'].min()}–{G['counts'].max()}) | {G['n_seqs']} 条训练序列")
    log(f"度分布: mean={deg.mean():.1f} median={np.median(deg):.0f} max={deg.max()}")

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
        log(f"[{pname}] 正样本 {len(pos_ok)}/{len(pos)} | 端点 degree pos={d_p.mean():.1f} "
            f"degmatch={d_nd.mean():.1f} (Δ={d_nd.mean()-d_p.mean():+.1f}) "
            f"uniform={d_nu.mean():.1f} (Δ={d_nu.mean()-d_p.mean():+.1f})")
        negcg[pname] = (pos_ok, neg_deg, neg_unif)

    # ---------------- 逐变体：z 与 xq[full] ----------------
    rows = []
    for name, tag, codef in BEAUTY_VARIANTS:
        ckpt = find_best_ckpt(tag)
        code_path = os.path.join(args.data_dir, ds, codef)
        if ckpt is None or not os.path.exists(code_path):
            log(f"\n⚠ {name} 缺资源 → 跳过")
            continue
        codes = np.load(code_path)
        if codes.shape[0] != n_items:
            log(f"\n⚠ {name} codes 行数 {codes.shape[0]} != {n_items} → 跳过")
            continue
        cb = load_codebook_weights(ckpt)

        zpath = getattr(torch.load(ckpt, map_location="cpu", weights_only=False)["args"],
                        "data_path", None)
        if zpath and not os.path.isabs(zpath):
            zpath = os.path.normpath(os.path.join(ROOT, zpath))
        spaces = {"xq[full]": normalize_rows(rebuild_xq_partial(codes, cb, 3))}
        if zpath and os.path.exists(zpath):
            spaces["z"] = encode_z(ckpt, zpath)
        else:
            log(f"\n⚠ {name} z 输入缺失 ({zpath})")

        log(f"\n▌ {name}  (L0distinct={len(np.unique(codes[:,0]))})")
        for sname, emb in spaces.items():
            for pname, (pos_ok, neg_deg, neg_unif) in negcg.items():
                m = {
                    "dataset": ds, "variant": name, "space": sname, "pos_regime": pname,
                    "auc_cos_degmatch": auc_of(cos_sim(emb, pos_ok), cos_sim(emb, neg_deg)),
                    "auc_cos_uniform": auc_of(cos_sim(emb, pos_ok), cos_sim(emb, neg_unif)),
                    "n_pos": len(pos_ok), "n_neg": len(neg_deg),
                    "ckpt_tag": tag, "codes_file": codef,
                }
                rows.append(m)
                log(f"    {sname:<9} [{pname:<4}] AUC degmatch={m['auc_cos_degmatch']:.4f} "
                    f"uniform={m['auc_cos_uniform']:.4f}")

    if not rows:
        log("\n无结果。")
        with open(args.report, "w") as f:
            f.write("```\n" + "\n".join(log_lines) + "\n```\n")
        return 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    df.to_csv(args.csv, index=False)
    log(f"\n→ CSV: {args.csv}  ({len(df)} 行)")

    # ---------------- §九-1 结果表（主口径 topk × degmatch） ----------------
    def tbl(neg, pname="topk"):
        p = df[(df.pos_regime == pname)].pivot(index="variant", columns="space",
                                                values=f"auc_cos_{neg}")
        return p.reindex([n for n, _, _ in BEAUTY_VARIANTS])

    t_main = tbl("degmatch", "topk")      # 主口径：与主表 xq[full] 的排序结论同口径
    for pname in ("topk", "all"):
        t = tbl("degmatch", pname)
        log(f"\n▌ Beauty Tier-1 结果表（{pname} 正样本 × degmatch 负样本）")
        log(f"    {'Variant':<14} {'z AUC':>9} {'xq[full] AUC':>14}")
        for v, r in t.iterrows():
            if pd.isna(r.get("z")) and pd.isna(r.get("xq[full]")):
                continue
            log(f"    {v:<14} {r['z']:>9.4f} {r['xq[full]']:>14.4f}")

    # ---------------- §七-1 z → xq[full] 一致性 ----------------
    log("\n▌ z → xq[full] 排序一致性（Spearman，越大越说明量化保序）")
    cons = []
    t = t_main.dropna()
    if len(t) >= 3:
        rho, p = spearmanr(t["z"], t["xq[full]"])
        cons.append((ds, rho, p, len(t)))
        log(f"    {ds:<8} ρ = {rho:+.4f}  (p = {p:.4g}, n = {len(t)})")
    if os.path.exists(args.main_csv):
        mo = pd.read_csv(args.main_csv)
        mo = mo[mo.pos_regime == "topk"]
        for other in ("Sport", "Toys"):
            s = mo[mo.dataset == other]
            if s.empty:
                continue
            z = s[s.space == "z"].set_index("variant")["auc_cos_degmatch"]
            x = s[s.space == "xq[full]"].set_index("variant")["auc_cos_degmatch"]
            j = pd.concat([z, x], axis=1, keys=["z", "xq"]).dropna()
            if len(j) >= 3:
                rho, p = spearmanr(j["z"], j["xq"])
                cons.append((other, rho, p, len(j)))
                log(f"    {other:<8} ρ = {rho:+.4f}  (p = {p:.4g}, n = {len(j)})"
                    f"   ← 取自 {os.path.basename(args.main_csv)}")

    # ---------------- §七-2 variant ranking ----------------
    log("\n▌ Variant 排序（按 xq[full] AUC 降序）")
    rank = t_main["xq[full]"].sort_values(ascending=False)
    for i, (v, a) in enumerate(rank.items(), 1):
        log(f"    {i}. {v:<14} {a:.4f}")
    got = list(rank.index)
    have = [v for v in REF_ORDER if v in got]
    pair_ok, pair_tot = 0, 0
    log(f"\n    实测: {' > '.join(got)}")
    log(f"    参照: {' > '.join(REF_ORDER)}   （Sport/Toys 的 xq[full] 实测排序）")
    if len(have) >= 3:
        # 逐对比较：参照序中的相对顺序在实测中是否保持
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                a, b = have[i], have[j]
                pair_tot += 1
                if got.index(a) < got.index(b):
                    pair_ok += 1
        log(f"    成对顺序保持: {pair_ok}/{pair_tot}")
    if "TA-SID" in got and "Shuffled" in got:
        log(f"    两端: TA-SID 第 {got.index('TA-SID')+1} 名 | "
            f"Shuffled 第 {got.index('Shuffled')+1} 名")

    # ---------------- §九-4 结论 ----------------
    log("\n" + "=" * 100)
    log("▌ 结论")
    log("=" * 100)
    has_z = "z" in t_main.columns and t_main["z"].notna().any()
    zm = t_main["z"].mean() if has_z else float("nan")
    xm = t_main["xq[full]"].mean()
    log("Q1 Beauty 中 RQ-VAE encoder 是否形成行为结构？")
    if has_z:
        log(f"   → {'是' if zm > 0.6 else '否/弱'}：6 变体 z(32d) AUC 均值 {zm:.4f}"
            f"（范围 {t_main['z'].min():.4f}–{t_main['z'].max():.4f}），远高于随机 0.5；"
            f"且变体间存在稳定区分度（极差 {t_main['z'].max()-t_main['z'].min():.4f}）。")
    else:
        log("   → z 未计算成功，无法回答。")
    rho_b = cons[0][1] if cons else float("nan")
    log(f"\nQ2 量化是否破坏行为结构？")
    log(f"   → {'未破坏' if rho_b > 0.9 else '存在扰动'}：z → xq[full] Spearman ρ = {rho_b:+.4f}"
        f"；xq[full] AUC 均值 {xm:.4f}（z 为 {zm:.4f}，差 {xm-zm:+.4f}）。")
    log(f"\nQ3 Beauty 的 variant 排序是否与 Sport/Toys 一致？")
    log(f"   → 实测: {' > '.join(got)}")
    log("     参照: TA-SID > Raw-CoOcc > ItemCF > {Content-only, ZeroPad} > Shuffled")
    log(f"     成对顺序保持 {pair_ok}/{pair_tot}；两端 TA-SID/Shuffled 位置见上。")

    with open(args.report, "w") as f:
        f.write("```\n" + "\n".join(log_lines) + "\n```\n")
    log(f"\n→ 报告: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
