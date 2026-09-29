#!/usr/bin/env python3
"""
stage_probe.py — G2 阶段×种子探针（note 15 §5 的 C1 + C2）

把 `tools/beauty_tier1_probe.py`（只测 Beauty / 只测 z 与 xq[full]）泛化为：
    3 数据集 × 3 RQ-VAE 种子 × 6 变体 × 3 空间（A 输入 896d / B z / C xq[full]）
并加 `--delta_auc`（Q3：扣 content 地板的行为增量）。

口径逐位复用 `tools/behavior_structure_probe.py`（E 图 window=3 / topk 主口径 /
degmatch 主负样本 / cosine AUC）与 `tools/sid_per_layer_diagnostic.py`
（encode_z / normalize_rows / rebuild_xq_partial）。**未重新定义 Behavior AUC**。

━━ 两种 `--seed` 的区别（勿混） ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    --rqvae_seed      RQ-VAE 训练种子（G2 的自变量）—— 决定 ckpt / codes
    --sample_seed     E 图正负样本采样的 RNG 种子（必须固定，否则 AUC 差异含采样噪声）
本脚本默认 --sample_seed 2024（= behavior_structure_probe.SEED），故与既有数字可比。

━━ 口径 (a)：种子只移动 TA-SID 一列（note 15 §9 后的裁定）━━━━━━━━━━━━
产物覆盖实测：Sport/Beauty 只有 TA-SID 有 seed2025/2026，Toys 只有 TA-SID +
Content-only + ZeroPad。故**其余变体在 seed 2025/2026 上无产物**，本脚本将它们
**冻结在 seed 2024**（CSV 的 `frozen` 列标 True）。因此 seed 2025/2026 的 n=6 ρ
是「5 固定点 + 1 动点」的相关，**不是**整表重算 —— 读法与局限见 note 15 §10。

━━ A 空间（输入）文件订正 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`beauty_preflight_probe.py` 的硬编码 FUSED 表把 TA-SID 指向 `fused_item_emb.parquet`；
该文件在 **Beauty 是 896d，但在 Sport/Toys 是 4224d**（4096 内容 + 128 行为）。
故 Sport/Toys 的 TA-SID「896d」输入 AUC 实为 4224d 所测（0.7835 / 0.8070），
与同列其余 5 个 896d 变体不同维。本脚本统一改用 **`fused_item_emb_768d.parquet`**
（三数据集均为 896d；Beauty 上它是指向 `fused_item_emb.parquet` 的软链，等价）。
订正影响见 `--audit_a`。

用法（仓库根，CPU）：
    python3 tools/stage_probe.py --rqvae_seeds 2024                     # 先验 seed2024
    python3 tools/stage_probe.py --rqvae_seeds 2024 2025 2026 --delta_auc
    python3 tools/stage_probe.py --audit_a                              # A 空间维度订正对照
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
    build_graph, edge_keys, sample_positives, sample_neg_degmatch,
    sample_neg_uniform, cos_sim, auc_of,
    DATA_DIR, N_POS, N_NEG, SEED, TOPK_POS,
)
from sid_per_layer_diagnostic import (  # noqa: E402
    encode_z, normalize_rows, rebuild_xq_partial,
)
from sid_consistency_contrast import (  # noqa: E402
    find_best_ckpt, load_codebook_weights,
)

# ---------------------------------------------------------------------------
# 变体 × 数据集 → {seed: (ckpt_tag, codes_file)}；A 空间 fused 文件（统一 896d）
# seed 2024 = legacy（无后缀）。键缺失 = 该格无产物。
# ---------------------------------------------------------------------------
FUSED_896D = {
    "Content-only": "fused_item_emb_zeropad.parquet",   # content ⊕ 128 零（preflight 口径）
    "ZeroPad":      "fused_item_emb_zeropad.parquet",   # 同上 → 二者输入严格打平
    "Shuffled":     "fused_item_emb_shuffled.parquet",
    "ItemCF":       "fused_item_emb_itemcf.parquet",
    "Raw-CoOcc":    "fused_item_emb_rawcooc.parquet",
    "TA-SID":       "fused_item_emb_768d.parquet",      # ← 订正：原用 fused_item_emb.parquet
}

VARIANTS = {
    "Sport": {
        "Content-only": {2024: ("Sport_Content768d_Baseline", "Sport_content768d_baseline_sid.npy")},
        "ZeroPad":      {2024: ("Sport_Zeropad_Ablation", "Sport_zeropad_ablation_sid.npy")},
        "Shuffled":     {2024: ("Sport_Shuffled_Ablation", "Sport_shuffled_ablation_sid.npy")},
        "ItemCF":       {2024: ("Sport_ItemCF_768d", "Sport_itemcf_768d_sid.npy")},
        "Raw-CoOcc":    {2024: ("Sport_RawCoOcc_768d", "Sport_rawcooc_768d_sid.npy")},
        "TA-SID": {2024: ("Sport_TA_SID_768d", "Sport_ta_sid_768d_sid.npy"),
                   2025: ("Sport_TA_SID_seed2025", "Sport_ta_sid_768d_seed2025_sid.npy"),
                   2026: ("Sport_TA_SID_seed2026", "Sport_ta_sid_768d_seed2026_sid.npy")},
    },
    "Toys": {
        "Content-only": {2024: ("Toys_Content768d_Baseline", "Toys_content768d_baseline_sid.npy"),
                         2025: ("Toys_Content768d_seed2025", "Toys_content768d_seed2025_sid.npy"),
                         2026: ("Toys_Content768d_seed2026", "Toys_content768d_seed2026_sid.npy")},
        "ZeroPad":      {2024: ("Toys_Zeropad_seed2024", "Toys_zeropad_seed2024_sid.npy"),
                         2025: ("Toys_Zeropad_seed2025", "Toys_zeropad_seed2025_sid.npy")},
        "Shuffled":     {2024: ("Toys_Shuffled_768d", "Toys_shuffled_768d_sid.npy")},
        "ItemCF":       {2024: ("Toys_ItemCF_768d", "Toys_itemcf_768d_sid.npy")},
        "Raw-CoOcc":    {2024: ("Toys_RawCoOcc_768d", "Toys_rawcooc_768d_sid.npy")},
        "TA-SID": {2024: ("Toys_TA_SID_768d", "Toys_ta_sid_768d_seed2024_sid.npy"),
                   2025: ("Toys_TA_SID_seed2025", "Toys_ta_sid_768d_seed2025_sid.npy"),
                   2026: ("Toys_TA_SID_seed2026", "Toys_ta_sid_768d_seed2026_sid.npy")},
    },
    "Beauty": {
        "Content-only": {2024: ("Beauty_Content768d_Baseline", "Beauty_content768d_baseline_sid.npy")},
        "ZeroPad":      {2024: ("Beauty_Zeropad_768d", "Beauty_zeropad_768d_sid.npy")},
        "Shuffled":     {2024: ("Beauty_Shuffled_768d", "Beauty_shuffled_768d_sid.npy")},
        "ItemCF":       {2024: ("Beauty_ItemCF_768d", "Beauty_itemcf_768d_sid.npy")},
        "Raw-CoOcc":    {2024: ("Beauty_RawCoOcc_768d", "Beauty_rawcooc_768d_sid.npy")},
        "TA-SID": {2024: ("Beauty_TA_SID", "Beauty_ta_sid_768d_sid.npy"),
                   2025: ("Beauty_TA_SID_seed2025", "Beauty_ta_sid_768d_seed2025_sid.npy"),
                   2026: ("Beauty_TA_SID_seed2026", "Beauty_ta_sid_768d_seed2026_sid.npy")},
    },
}
VARIANT_ORDER = ["Content-only", "ZeroPad", "Shuffled", "ItemCF", "Raw-CoOcc", "TA-SID"]

# 既有（seed2024）记录值，用于自检：note 15 §1 表
RECORDED_RHO = {   # (ds) -> {pair: 记录值}；A 列按**订正后** 896d 重算，故 Sport/Toys 预期会变
    "Beauty": {("A", "B"): 0.986, ("A", "C"): 0.986, ("B", "C"): 1.000},
    "Sport":  {("A", "B"): 0.058, ("A", "C"): 0.058, ("B", "C"): 1.000},
    "Toys":   {("A", "B"): 0.058, ("A", "C"): 0.058, ("B", "C"): 1.000},
}


def load_fused_896(path, n_items):
    df = pd.read_parquet(path)
    emb = np.stack(df["embedding"].values).astype(np.float32)
    if emb.shape[0] != n_items:
        raise ValueError(f"{os.path.basename(path)}: n={emb.shape[0]} != {n_items}")
    return normalize_rows(torch.from_numpy(emb))


def resolve(ds, variant, rqvae_seed):
    """返回 (ckpt_tag, codes_file, frozen)；非 TA-SID 变体回落到 seed2024。"""
    table = VARIANTS[ds][variant]
    if rqvae_seed in table:
        return (*table[rqvae_seed], False)
    if 2024 in table:                       # 口径 (a)：冻结在 seed2024
        return (*table[2024], True)
    return (None, None, None)


def build_samples(ds, data_dir, window_size, n_pos, n_neg, sample_seed):
    """构造 E 图正负样本（与 behavior_structure_probe / beauty_tier1_probe 同一路径）。

    单独抽出是为了让 --audit_a 走**与主流程完全相同**的采样路径 —— 否则 RNG 进度不同，
    抽到的负样本不同，AUC 会有 ~0.01–0.03 的差异，两处数字对不上会误导。
    返回 (n_items, negcg)；negcg[posname] = (pos_ok, neg_deg, neg_unif)。
    """
    rng = np.random.default_rng(sample_seed)
    G = build_graph(ds, data_dir, window_size)
    n_items, deg = G["n_items"], G["deg"]
    edge_set = set(edge_keys(G["edges"], n_items).tolist())
    pos_regimes = {"topk": G["top_edges"], "all": sample_positives(G["edges"], n_pos, rng)}
    negcg = {}
    for pname, pos in pos_regimes.items():
        neg_deg, keep = sample_neg_degmatch(pos, deg, n_items, edge_set, rng)
        pos_ok = pos[keep]
        negcg[pname] = (pos_ok, neg_deg,
                        sample_neg_uniform(len(pos_ok), n_items, edge_set, rng))
    return n_items, len(G["edges"]), negcg


def audit_a(datasets, data_dir, window_size, n_pos, n_neg, sample_seed):
    """A 空间维度订正对照：TA-SID 在 legacy 文件 vs 896d 文件上的输入 AUC。"""
    print("=" * 100)
    print("A 空间订正对照 —— TA-SID 的 fused 输入 AUC（topk × degmatch，与主流程同一采样路径）")
    print("=" * 100)
    print(f"  {'数据集':<8} {'legacy fused_item_emb':>24s} {'896d fused_item_emb_768d':>26s}   差值")
    for ds in datasets:
        n_items, _, negcg = build_samples(ds, data_dir, window_size, n_pos, n_neg, sample_seed)
        pos_ok, neg_deg, _ = negcg["topk"]
        out = []
        for f in ("fused_item_emb.parquet", "fused_item_emb_768d.parquet"):
            e = load_fused_896(os.path.join(data_dir, ds, f), n_items)
            out.append((auc_of(cos_sim(e, pos_ok), cos_sim(e, neg_deg)), e.shape[1]))
        (a_leg, d_leg), (a_new, d_new) = out
        print(f"  {ds:<8} {a_leg:>18.4f} ({d_leg:4d}d) {a_new:>20.4f} ({d_new:4d}d)   {a_new - a_leg:+.4f}")
    print("\n  ⇒ Sport/Toys 的 legacy 文件是 4224d（4096 内容 + 128 行为）；TA-SID 的真实训练输入")
    print("    是三数据集统一的 896d（各 ckpt 的 args.data_path 已逐格核验）。旧记录的")
    print("    0.7835 / 0.8070 出自 4224d 空间，与同列其余 5 个 896d 变体不可比 → 已订正。")


def main():
    ap = argparse.ArgumentParser(description="G2 阶段×种子探针（3 空间 / 多 RQ-VAE seed）")
    ap.add_argument("--datasets", type=str, default="Sport,Toys,Beauty")
    ap.add_argument("--rqvae_seeds", type=int, nargs="+", default=[2024])
    ap.add_argument("--sample_seed", type=int, default=SEED,
                    help="E 图正负样本采样 RNG 种子（固定，勿随 RQ-VAE 种子变）")
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--delta_auc", action="store_true", help="Q3：报 ΔAUC = TA-SID − Content-only")
    ap.add_argument("--audit_a", action="store_true", help="只跑 A 空间维度订正对照后退出")
    ap.add_argument("--csv", type=str, default="doc/g2_stage_probe_results.csv")
    ap.add_argument("--rho_csv", type=str, default="doc/g2_stage_rho.csv")
    ap.add_argument("--report", type=str, default="doc/g2_stage_seed_robustness.md")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.audit_a:
        audit_a(datasets, args.data_dir, args.window_size, args.n_pos, args.n_neg, args.sample_seed)
        return 0

    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    log("=" * 104)
    log("G2 阶段×种子探针 —— 3 数据集 × RQ-VAE 种子 × 6 变体 × 3 空间")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"RQ-VAE 种子: {args.rqvae_seeds}　采样种子(sample_seed): {args.sample_seed}（固定）")
    log("空间: A=fused 896d（seed 不变量） | B=z（encoder 输出 32d） | C=xq[full]（码本重建 32d）")
    log("口径: E 图 window=3 / 正样本 topk(1000) 主 + all 副 / 负样本 degmatch 主 + uniform 副 / cosine AUC")
    log("=" * 104)
    log("\n⚠ 口径 (a)：种子只移动 TA-SID 一列；其余变体无 seed2025/2026 产物 → 冻结在 seed2024（见 frozen 列）")

    rows = []
    for ds in datasets:
        log(f"\n{'#' * 104}\n# {ds}\n{'#' * 104}")

        n_items, n_edges, negcg = build_samples(ds, args.data_dir, args.window_size,
                                                args.n_pos, args.n_neg, args.sample_seed)
        log(f"  E 图: {n_items} items | {n_edges} 边 | 正样本 topk={len(negcg['topk'][0])} "
            f"all={len(negcg['all'][0])}")

        # ---- 先缓存各资源的 A/B/C 空间（冻结变体跨 seed 复用，避免重复编码） ----
        cache = {}
        for rqvae_seed in args.rqvae_seeds:
            for vname in VARIANT_ORDER:
                tag, codef, frozen = resolve(ds, vname, rqvae_seed)
                if tag is None:
                    continue
                key = (tag, codef)
                if key in cache:
                    continue
                ckpt = find_best_ckpt(tag)
                code_path = os.path.join(args.data_dir, ds, codef) if codef else None
                if ckpt is None or not os.path.exists(code_path):
                    cache[key] = None
                    log(f"  ⚠ 缺资源跳过: {vname} tag={tag} codes={codef}")
                    continue
                codes = np.load(code_path)
                cb = load_codebook_weights(ckpt)
                spaces = {"A": load_fused_896(os.path.join(args.data_dir, ds, FUSED_896D[vname]), n_items),
                          "C": normalize_rows(rebuild_xq_partial(codes, cb, 3))}
                zpath = getattr(torch.load(ckpt, map_location="cpu",
                                           weights_only=False)["args"], "data_path", None)
                if zpath and not os.path.isabs(zpath):
                    zpath = os.path.normpath(os.path.join(ROOT, zpath))
                spaces["B"] = encode_z(ckpt, zpath) if zpath and os.path.exists(zpath) else None
                if spaces["B"] is None:
                    log(f"  ⚠ z 输入缺失: {vname} tag={tag} ({zpath})")
                cache[key] = spaces

        # ---- 逐 seed × 变体 × 空间 × 口径 ----
        auc_tab = {}     # (seed, space, vname, posname) -> {neg: auc}
        for rqvae_seed in args.rqvae_seeds:
            for vname in VARIANT_ORDER:
                tag, codef, frozen = resolve(ds, vname, rqvae_seed)
                if tag is None:
                    continue
                spaces = cache.get((tag, codef))
                if spaces is None:
                    continue
                L0d = len(np.unique(np.load(os.path.join(args.data_dir, ds, codef))[:, 0]))
                for sname, emb in spaces.items():
                    if emb is None:
                        continue
                    for pname, (pos_ok, neg_deg, neg_unif) in negcg.items():
                        sp, sn = cos_sim(emb, pos_ok), cos_sim(emb, neg_deg)
                        su = cos_sim(emb, neg_unif)
                        rec = {"dataset": ds, "rqvae_seed": rqvae_seed, "variant": vname,
                               "space": sname, "pos_regime": pname,
                               "auc_cos_degmatch": auc_of(sp, sn), "auc_cos_uniform": auc_of(sp, su),
                               "n_pos": len(pos_ok), "n_neg": len(neg_deg),
                               "frozen": frozen, "ckpt_tag": tag, "codes_file": codef,
                               "codes_L0distinct": L0d}
                        rows.append(rec)
                        auc_tab.setdefault((rqvae_seed, sname, pname), {})[vname] = rec

        # ---- ρ 表（n=6 变体排序相关） ----
        for pname in ("topk", "all"):
            for negcol in ("auc_cos_degmatch", "auc_cos_uniform"):
                log(f"\n  ▌ ρ 表（正样本={pname} × 负样本={negcol.replace('auc_cos_','')}，空间两两）")
                log(f"    {'seed':>6} {'ρ(A输入,B z)':>14} {'ρ(A输入,C xq)':>15} {'ρ(B z,C xq)':>13}   读法")
                for rqvae_seed in args.rqvae_seeds:
                    vals = {}
                    for sname in ("A", "B", "C"):
                        d = auc_tab.get((rqvae_seed, sname, pname), {})
                        vals[sname] = [d[v][negcol] for v in VARIANT_ORDER if v in d]
                    if any(len(v) != 6 for v in vals.values()):
                        log(f"    {rqvae_seed:>6}   跳过（变体不全: "
                            f"{ {k: len(v) for k, v in vals.items()} }）")
                        continue
                    r_ab = spearmanr(vals["A"], vals["B"]).statistic
                    r_ac = spearmanr(vals["A"], vals["C"]).statistic
                    r_bc = spearmanr(vals["B"], vals["C"]).statistic
                    note = "（变体不全，混合口径）" if rqvae_seed != 2024 else ""
                    log(f"    {rqvae_seed:>6} {r_ab:>14.3f} {r_ac:>15.3f} {r_bc:>13.3f}   {note}")
                    rows.append({"dataset": ds, "rqvae_seed": rqvae_seed, "variant": "__RHO__",
                                 "space": "A~B", "pos_regime": pname,
                                 "auc_cos_degmatch": r_ab if negcol.endswith("degmatch") else "",
                                 "auc_cos_uniform": r_ab if negcol.endswith("uniform") else "",
                                 "n_pos": 6, "n_neg": 6, "frozen": "", "ckpt_tag": "", "codes_file": ""})

        # ---- Q3：ΔAUC = TA-SID − Content-only ----
        if args.delta_auc:
            log(f"\n  ▌ Q3 ΔAUC = AUC(TA-SID) − AUC(Content-only)（逐 seed 逐空间）")
            log(f"    {'seed':>6} {'space':>6} {'TA-SID':>9} {'Content':>9} {'ΔAUC':>9}   （正样本=topk × degmatch）")
            for rqvae_seed in args.rqvae_seeds:
                for sname in ("A", "B", "C"):
                    d = auc_tab.get((rqvae_seed, sname, "topk"), {})
                    if "TA-SID" not in d or "Content-only" not in d:
                        continue
                    a_ta = d["TA-SID"]["auc_cos_degmatch"]
                    a_co = d["Content-only"]["auc_cos_degmatch"]
                    log(f"    {rqvae_seed:>6} {sname:>6} {a_ta:>9.4f} {a_co:>9.4f} {a_ta - a_co:>+9.4f}")

    # ---- 自检：seed2024 的 ρ 与 note 15 §1 记录值对照 ----
    log(f"\n{'=' * 104}\n自检：seed2024 ρ 与 note 15 §1 记录值对照（A 列已按订正后 896d 重算）\n{'=' * 104}")
    log(f"  {'数据集':<8} {'ρ 对':>8} {'记录值':>9} {'本次(896d)':>11}   差")
    df_rows = pd.DataFrame([r for r in rows if r["variant"] != "__RHO__"])
    for ds in datasets:
        for pair, ref in RECORDED_RHO.get(ds, {}).items():
            sa, sb = pair
            vals = {}
            for sname in (sa, sb):
                sub = df_rows[(df_rows.dataset == ds) & (df_rows.rqvae_seed == 2024) &
                              (df_rows.space == sname) & (df_rows.pos_regime == "topk")]
                vals[sname] = sub.set_index("variant").reindex(VARIANT_ORDER)["auc_cos_degmatch"].tolist()
            if any(v is None or (isinstance(v, float) and np.isnan(v)) for vv in vals.values() for v in vv):
                log(f"  {ds:<8} {sa}~{sb:>4} {ref:>9.3f}        —    变体不全，跳过")
                continue
            got = spearmanr(vals[sa], vals[sb]).statistic
            flag = "" if abs(got - ref) < 0.02 else "  ← 变了"
            log(f"  {ds:<8} {sa}~{sb:>4} {ref:>9.3f} {got:>11.3f}{flag}")

    if args.csv and df_rows.shape[0]:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        df_rows.to_csv(args.csv, index=False)
        log(f"\n→ CSV: {args.csv}  ({df_rows.shape[0]} 行)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
