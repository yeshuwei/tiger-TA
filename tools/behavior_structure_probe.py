#!/usr/bin/env python3
"""
behavior_structure_probe.py — 行为结构保持度探针（Behavior AUC）

回答的问题（RQ）：
    Semantic ID 中保留的「行为结构信息」，能否解释下游生成推荐性能？
    即：Behavior AUC 是否与 NDCG@20 排序一致，可否作为 SID 质量评价指标？

与已有产物的关系（**不重复造轮子**）：
    - `rqvae/run_e49grid_diag.sh` → `doc/e49grid/*.md` 已用 **top-1000 高频正样本 +
      均匀随机负样本** 算过 xq[L0]/xq[L0+L1]/xq[full]/z 的 AUC（12 格）。
      本脚本的 `pos=topk, neg=uniform` 格是它的**近似复算**：两者负样本构造不同 ——
      e49grid 的负样本**不排除 E 内边与自环**，本脚本排除（更严格），故本脚本数值系统性
      偏高 mean|Δ|≈0.013 / max|Δ|≈0.032（12 格）。变体排序两者一致。
      详见 doc/behavior_structure_probe_conclusion.md §6。
    - 本脚本的**新增**部分：
        ① `neg=degmatch`：度分布受控的负采样（堵住流行度捷径）；
        ② `pos=all`：从**整张 E** 采样（含长尾弱边）的第二个正样本口径；
        ③ 离散 token 的 Hamming / L0-match 指标；
        ④ 与 NDCG@20 的 Pearson / Spearman 相关检验。
    - NDCG@20 **直接从 T5 评测日志解析**（复用 `tools/e49_summary.py` 的 parse/LANES），
      不抄表 —— 本项目有抄表出错的历史（memory: paper_main_results §3 抄错 4 行）。

复用（不复制）：
    tools/sid_consistency_contrast.py  : find_best_ckpt / load_codebook_weights
    tools/sid_per_layer_diagnostic.py  : rebuild_xq_partial / normalize_rows / encode_z
    tools/e49_summary.py               : LANES / parse / SEEDS（NDCG 权威来源）
    rqvae/build_transition_graph.py    : build_cooccurrence_matrix（E 图定义，与 TA-SID 同源）

E 图定义（与 TA-SID 的 behavior 向量完全同源）：
    train.parquet 的 history+[target] 序列 → 滑窗 window=3（即 ±1 相邻对）共现矩阵 C；
    E = {(i,j) : C[i,j] > 0}，i<j。**所有 variant 共用同一张 E**，保证公平。
    注：window 双向计数 ⇒ E 中最小 count = 2；分布长尾（Sport 180361 条边中仅 1448 条 count≥5）。

两轴口径（2×2）：
    正样本  pos=topk : 频次最高的 1000 条边（count≥6，强证据；＝ e49grid 口径）
            pos=all  : 从 E 均匀采样 n_pos 条（覆盖全图，含长尾弱边）
    负样本  neg=degmatch : 度分布受控（主口径）  neg=uniform : 均匀随机（参照）

只读约束：只读 checkpoint / codes / parquet / 评测日志；写 CSV + markdown 报告。

用法：
    python3 tools/behavior_structure_probe.py --datasets Sport,Toys
    python3 tools/behavior_structure_probe.py --datasets Sport,Toys,Beauty \
        --skip_z --csv doc/behavior_structure_probe_results.csv \
        --report doc/behavior_structure_probe.md
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "rqvae"))

from sid_consistency_contrast import (  # noqa: E402
    find_best_ckpt,
    load_codebook_weights,
    DATA_DIR,
)
from sid_per_layer_diagnostic import (  # noqa: E402
    encode_z,
    normalize_rows,
    rebuild_xq_partial,
)
from e49_summary import LANES, parse, SEEDS  # noqa: E402
from build_transition_graph import build_cooccurrence_matrix  # noqa: E402

# ---------------------------------------------------------------- variant 表
# 与 rqvae/run_e49grid_diag.sh 的 CELLS 逐字一致（12 格配对已由
# rqvae/audit_main_codes_provenance.py 用 L0 一致率法验证 12/12 = 100%）。
VARIANTS = {
    "Sport": [
        ("Content-only", "Sport_Content768d_Baseline", "Sport_content768d_baseline_sid.npy"),
        ("ZeroPad",      "Sport_Zeropad_Ablation",     "Sport_zeropad_ablation_sid.npy"),
        ("Shuffled",     "Sport_Shuffled_Ablation",    "Sport_shuffled_ablation_sid.npy"),
        ("ItemCF",       "Sport_ItemCF_768d",          "Sport_itemcf_768d_sid.npy"),
        ("Raw-CoOcc",    "Sport_RawCoOcc_768d",        "Sport_rawcooc_768d_sid.npy"),
        ("TA-SID",       "Sport_TA_SID_768d",          "Sport_ta_sid_768d_sid.npy"),
    ],
    "Toys": [
        ("Content-only", "Toys_Content768d_Baseline",  "Toys_content768d_baseline_sid.npy"),
        ("ZeroPad",      "Toys_Zeropad_seed2024",      "Toys_zeropad_seed2024_sid.npy"),
        ("Shuffled",     "Toys_Shuffled_768d",         "Toys_shuffled_768d_sid.npy"),
        ("ItemCF",       "Toys_ItemCF_768d",           "Toys_itemcf_768d_sid.npy"),
        ("Raw-CoOcc",    "Toys_RawCoOcc_768d",         "Toys_rawcooc_768d_sid.npy"),
        # 健康 codes（L0=207），**不是**作废的孤儿 codes Toys_ta_sid_768d_sid.npy(L0=62)
        ("TA-SID",       "Toys_TA_SID_768d",           "Toys_ta_sid_768d_seed2024_sid.npy"),
    ],
    # Beauty 缺 4/6 变体（无 Raw-CoOcc / ItemCF / ZeroPad / Shuffled 的 ckpt 与 codes）
    # → 仅 2 个点，无法做相关分析，报告为描述性。
    "Beauty": [
        ("Content-only", "Beauty_Content768d_Baseline", "Beauty_content768d_baseline_sid.npy"),
        ("TA-SID",       "Beauty_TA_SID",               "Beauty_ta_sid_768d_sid.npy"),
    ],
}

# Beauty 的 NDCG 日志不在 e49_summary.LANES 里（E49 只覆盖 Sport/Toys），单独取
BEAUTY_LANES = {
    "Content-only": "Beauty_content768d_baseline",
    "TA-SID":       "Beauty_ta_sid_768d",
}

N_POS = 20000          # pos=all 时从 E 采样的正样本数上限
TOPK_POS = 1000        # pos=topk 时的边数（＝ e49grid 口径）
N_NEG = 20000          # 每种负样本 regime 的样本数
MAX_TRIES = 20         # 度匹配负采样拒绝上限
SEED = 2024

DEFAULT_SPACES = "xq[L0],xq[full],z"


# ---------------------------------------------------------------- NDCG 来源
def load_ndcg(ds):
    """从 T5 评测日志解析 NDCG@20（3-seed 均值，generator seed）。返回 {variant: (mean,[per-seed])}"""
    out = {}
    if ds == "Beauty":
        for var, tag in BEAUTY_LANES.items():
            vals = [parse(f"{tag}_seed{s}_eval.log") for s in SEEDS]
            vals = [v[0] for v in vals if v]
            if vals:
                out[var] = (float(np.mean(vals)), vals)
        return out

    for d, variant, files in LANES:
        if d != ds or variant == "Random":
            continue
        vals = []
        for s in SEEDS:
            r = parse(files[s]) if s in files else None
            if r:
                vals.append(r[0])
        if vals:
            out[variant] = (float(np.mean(vals)), vals)
    return out


# ---------------------------------------------------------------- E 图
def build_graph(ds, data_dir, window_size=3):
    """构建行为共现图 E（与 TA-SID behavior 向量同源）。"""
    item_df = pd.read_parquet(f"{data_dir}/{ds}/item_emb_768d.parquet")
    ids = item_df["ItemID"].values
    id2row = {iid: idx for idx, iid in enumerate(ids)}
    n_items = len(ids)

    train_df = pd.read_parquet(f"{data_dir}/{ds}/train.parquet")
    seqs = []
    for _, row in train_df.iterrows():
        s = list(row["history"]) + [row["target"]]
        m = [id2row[x] for x in s if x in id2row]
        if m:
            seqs.append(m)

    C = build_cooccurrence_matrix(seqs, n_items, window_size).tocsr()
    C.setdiag(0.0)          # 自环不计入结构（同一商品重复出现不算"关联"）
    C.eliminate_zeros()
    deg = np.diff(C.indptr).astype(np.int64)

    coo = C.tocoo()
    mask = coo.row < coo.col
    edges = np.stack([coo.row[mask], coo.col[mask]], axis=1).astype(np.int64)
    counts = coo.data[mask]

    order = np.argsort(-counts)
    return {
        "edges": edges, "counts": counts,
        "top_edges": edges[order[:TOPK_POS]],
        "deg": deg, "n_items": n_items, "id2row": id2row, "n_seqs": len(seqs),
    }


def edge_keys(pairs, n_items):
    return pairs[:, 0] * n_items + pairs[:, 1]


def sample_positives(edges, n_pos, rng):
    if len(edges) <= n_pos:
        return edges.copy()
    return edges[rng.choice(len(edges), n_pos, replace=False)]


def sample_neg_uniform(n, n_items, edge_set, rng):
    """均匀随机负样本（＝ e49grid 口径参照臂）。排除自环与 E 内边。"""
    out = []
    while len(out) < n:
        need = n - len(out)
        a = rng.integers(0, n_items, size=need * 2)
        b = rng.integers(0, n_items, size=need * 2)
        for x, y in zip(a, b):
            if x == y:
                continue
            lo, hi = (x, y) if x < y else (y, x)
            if lo * n_items + hi in edge_set:
                continue
            out.append((lo, hi))
            if len(out) >= n:
                break
    return np.asarray(out, dtype=np.int64)


def sample_neg_degmatch(pos, deg, n_items, edge_set, rng, max_tries=MAX_TRIES):
    """度分布受控负采样：按 ⌊log2(deg)⌋ 分桶，候选取自**与正样本端点同桶**。

    目的：堵住「热门商品彼此相似」这条捷径 —— 若负样本端点度分布与正样本不匹配，
    分类器可凭流行度而非行为结构拿分（参见 tools/behavior_space_diagnostic.py 对
    `cons_auc` 系统性定义优势的告警，以及 memory: e49 行为空间闭环失效）。

    返回 (neg_pairs, keep_idx)；拒绝采样失败的条目由 keep_idx 标出，调用方据此
    把两个负样本 regime 都对齐到同一批正样本上，保证可比。
    """
    safe = np.maximum(deg, 1)
    bucket_of = np.floor(np.log2(safe)).astype(np.int64)
    bucket_of[deg == 0] = -1                       # 冷启动物品不参与
    buckets = {int(b): np.where(bucket_of == b)[0] for b in np.unique(bucket_of) if b >= 0}

    out, keep = [], []
    for k, (i, j) in enumerate(pos):
        ci, cj = buckets.get(int(bucket_of[i])), buckets.get(int(bucket_of[j]))
        if ci is None or cj is None:
            continue
        for _ in range(max_tries):
            a, b = ci[rng.integers(len(ci))], cj[rng.integers(len(cj))]
            if a == b:
                continue
            lo, hi = (a, b) if a < b else (b, a)
            if lo * n_items + hi in edge_set:
                continue
            out.append((lo, hi))
            keep.append(k)
            break
    return np.asarray(out, dtype=np.int64), np.asarray(keep, dtype=np.int64)


# ---------------------------------------------------------------- 指标
def cos_sim(emb_n, pairs):
    return np.einsum("ij,ij->i", emb_n[pairs[:, 0]], emb_n[pairs[:, 1]])


def hamming(codes, pairs, n_tokens=3):
    """离散 SID 的 Hamming 距离（前 n_tokens 个 token），越小越相似。"""
    return (codes[pairs[:, 0], :n_tokens] != codes[pairs[:, 1], :n_tokens]).sum(axis=1)


def auc_of(pos_s, neg_s):
    y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
    return float(roc_auc_score(y, np.r_[pos_s, neg_s]))


def eval_space(emb_n, codes, pos_ok, neg_deg, neg_unif):
    r = {}
    for neg_name, neg in (("degmatch", neg_deg), ("uniform", neg_unif)):
        cs_p, cs_n = cos_sim(emb_n, pos_ok), cos_sim(emb_n, neg)
        r[f"auc_cos_{neg_name}"] = auc_of(cs_p, cs_n)
        r[f"cos_pos"] = float(cs_p.mean())
        r[f"cos_neg_{neg_name}"] = float(cs_n.mean())

        if codes is not None:
            h_p, h_n = hamming(codes, pos_ok), hamming(codes, neg)
            r[f"auc_ham_{neg_name}"] = float(roc_auc_score(
                np.r_[np.ones(len(h_p)), np.zeros(len(h_n))], np.r_[-h_p, -h_n].astype(np.float64)))
            r[f"ham_pos"] = float(h_p.mean())
            r[f"ham_neg_{neg_name}"] = float(h_n.mean())

            m_p = (codes[pos_ok[:, 0], 0] == codes[pos_ok[:, 1], 0]).astype(np.float64)
            m_n = (codes[neg[:, 0], 0] == codes[neg[:, 1], 0]).astype(np.float64)
            r[f"auc_l0match_{neg_name}"] = auc_of(m_p, m_n)
            r[f"l0match_pos"] = float(m_p.mean())
            r[f"l0match_neg_{neg_name}"] = float(m_n.mean())
    return r


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="Behavior structure preservation probe")
    ap.add_argument("--datasets", type=str, default="Sport,Toys")
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--spaces", type=str, default=DEFAULT_SPACES,
                    help="要测的空间，逗号分隔（xq[L0],xq[full],z）")
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--csv", type=str, default="doc/behavior_structure_probe_results.csv")
    ap.add_argument("--report", type=str, default="doc/behavior_structure_probe.md")
    args = ap.parse_args()

    want = {s.strip() for s in args.spaces.split(",") if s.strip()}
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    log("=" * 100)
    log("Behavior Structure Preservation vs Recommendation Performance")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"正样本口径: topk = 频次最高 {TOPK_POS} 条边（＝ e49grid） | "
        f"all = 从 E 均匀采样 ≤{args.n_pos} 条（含长尾弱边）")
    log("负样本口径: degmatch = 度分布受控（主） | uniform = 均匀随机（参照）")
    log(f"E 图: train.parquet history+[target] → 滑窗 window={args.window_size}（±1 相邻对）")
    log("NDCG@20: 直接从 model/logs/*_eval.log 解析（3-seed 均值），不抄表")
    log("只读：不训练、不写实验产物，仅 CSV + 本报告")
    log("=" * 100)

    rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if ds not in VARIANTS:
            log(f"\n⚠ 未知数据集，跳过: {ds}")
            continue
        rng = np.random.default_rng(args.seed)

        log("")
        log("=" * 100)
        log(f"▌ {ds}")
        log("=" * 100)

        G = build_graph(ds, args.data_dir, args.window_size)
        n_items, deg = G["n_items"], G["deg"]
        edge_set = set(edge_keys(G["edges"], n_items).tolist())
        log(f"  E 图: {n_items} items | {len(G['edges'])} 条边 (count {G['counts'].min()}–{G['counts'].max()}) "
            f"| {G['n_seqs']} 条训练序列")
        log(f"  度分布: mean={deg.mean():.1f} median={np.median(deg):.0f} max={deg.max()} "
            f"零度={int((deg == 0).sum())}")

        # ---- 两个正样本口径 ----
        pos_regimes = {
            "topk": G["top_edges"],
            "all": sample_positives(G["edges"], args.n_pos, rng),
        }
        log(f"  正样本: topk={len(pos_regimes['topk'])} 条 | all={len(pos_regimes['all'])} 条")

        negcg = {}
        for pname, pos in pos_regimes.items():
            neg_deg, keep = sample_neg_degmatch(pos, deg, n_items, edge_set, rng)
            pos_ok = pos[keep]
            neg_unif = sample_neg_uniform(len(pos_ok), n_items, edge_set, rng)
            # 度匹配有效性验证
            d_p = np.r_[deg[pos_ok[:, 0]], deg[pos_ok[:, 1]]]
            d_nd = np.r_[deg[neg_deg[:, 0]], deg[neg_deg[:, 1]]]
            d_nu = np.r_[deg[neg_unif[:, 0]], deg[neg_unif[:, 1]]]
            log(f"  [{pname}] 可用正样本 {len(pos_ok)}/{len(pos)} | "
                f"端点 degree 均值 pos={d_p.mean():.1f} degmatch={d_nd.mean():.1f} "
                f"(Δ={d_nd.mean() - d_p.mean():+.1f}) uniform={d_nu.mean():.1f} "
                f"(Δ={d_nu.mean() - d_p.mean():+.1f})")
            negcg[pname] = (pos_ok, neg_deg, neg_unif)

        ndcg = load_ndcg(ds)
        log(f"  NDCG@20 来源: {len(ndcg)} 个变体解析成功")

        for name, tag, codef in VARIANTS[ds]:
            ckpt = find_best_ckpt(tag)
            code_path = os.path.join(args.data_dir, ds, codef)
            if ckpt is None:
                log(f"  ⚠ {name:<14} 缺 ckpt ({tag}) → 跳过")
                continue
            if not os.path.exists(code_path):
                log(f"  ⚠ {name:<14} 缺 codes ({codef}) → 跳过")
                continue
            codes = np.load(code_path)
            if codes.shape[0] != n_items:
                log(f"  ⚠ {name:<14} codes 行数 {codes.shape[0]} != {n_items} → 跳过")
                continue
            cb = load_codebook_weights(ckpt)

            spaces = {}
            if "xq[L0]" in want:
                spaces["xq[L0]"] = normalize_rows(rebuild_xq_partial(codes, cb, 1))
            if "xq[full]" in want:
                spaces["xq[full]"] = normalize_rows(rebuild_xq_partial(codes, cb, 3))
            if "z" in want:
                zpath = getattr(torch.load(ckpt, map_location="cpu", weights_only=False)["args"],
                                "data_path", None)
                try:
                    if zpath and os.path.exists(zpath):
                        spaces["z"] = encode_z(ckpt, zpath)
                    else:
                        log(f"  ⚠ {name:<14} z 输入缺失 ({zpath})")
                except Exception as e:                       # noqa: BLE001
                    log(f"  ⚠ {name:<14} z 计算失败: {e}")

            nd, nd_seeds = ndcg.get(name, (float("nan"), []))
            for space_name, emb_n in spaces.items():
                for pname, (pos_ok, neg_deg, neg_unif) in negcg.items():
                    m = eval_space(emb_n, codes, pos_ok, neg_deg, neg_unif)
                    m.update(dataset=ds, variant=name, space=space_name, pos_regime=pname,
                             n_pos=len(pos_ok), n_neg=len(neg_deg),
                             ndcg20=nd, ndcg_seeds=";".join(f"{x:.4f}" for x in nd_seeds),
                             ckpt_tag=tag, codes_file=codef)
                    rows.append(m)
                    log(f"  {name:<14} {space_name:<9} [{pname:<4}] "
                        f"AUC cos(degmatch)={m['auc_cos_degmatch']:.4f} "
                        f"cos(uniform)={m['auc_cos_uniform']:.4f} | "
                        f"Ham(degmatch)={m['auc_ham_degmatch']:.4f} "
                        f"L0match(degmatch)={m['auc_l0match_degmatch']:.4f} | NDCG@20={nd:.4f}")

    if not rows:
        log("\n无结果。")
        return 1

    # ---------------------------------------------------------------- CSV
    cols = ["dataset", "variant", "space", "pos_regime", "n_pos", "n_neg",
            "ndcg20", "ndcg_seeds",
            "auc_cos_degmatch", "auc_cos_uniform", "auc_ham_degmatch", "auc_ham_uniform",
            "auc_l0match_degmatch", "auc_l0match_uniform",
            "cos_pos", "cos_neg_degmatch", "cos_neg_uniform",
            "ham_pos", "ham_neg_degmatch", "l0match_pos", "l0match_neg_degmatch",
            "ckpt_tag", "codes_file"]
    df = pd.DataFrame(rows)
    df = df[[c for c in cols if c in df.columns]]
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    df.to_csv(args.csv, index=False, float_format="%.6f")
    print(f"\n✓ CSV → {args.csv}")

    # ---------------------------------------------------------------- 相关分析
    METRICS = ["auc_cos_degmatch", "auc_cos_uniform", "auc_ham_degmatch", "auc_l0match_degmatch"]

    def corr_block(sub, metric):
        s = sub.dropna(subset=[metric, "ndcg20"])
        if len(s) < 3:
            return None
        pr, pp = pearsonr(s[metric], s["ndcg20"])
        sr, sp = spearmanr(s[metric], s["ndcg20"])
        return dict(n=len(s), pearson=pr, p_pearson=pp, spearman=sr, p_spearman=sp)

    corr_rows = []
    for space in [s for s in ["xq[full]", "xq[L0]", "z"] if s in set(df.space)]:
        for pname in ["topk", "all"]:
            for metric in METRICS:
                for scope in ["Sport", "Toys", "Pooled"]:
                    sub = df[(df.space == space) & (df.pos_regime == pname)]
                    if scope == "Pooled":
                        sub = sub[sub.dataset.isin(["Sport", "Toys"])]
                    else:
                        sub = sub[sub.dataset == scope]
                    c = corr_block(sub, metric)
                    if c:
                        corr_rows.append(dict(space=space, pos_regime=pname, metric=metric,
                                              scope=scope, **c))
    corr_df = pd.DataFrame(corr_rows)

    # ---- 数据集内标准化后的 Pooled 相关 ----
    # 直接 Pooled 会被「数据集间整体偏移」混淆（Beauty/Sport/Toys 的 AUC 与 NDCG 量纲都不同）。
    # 故在每个数据集内先 z-score（variant 维度），再合并算 Pearson —— 度量的是**数据集内**
    # 的关联强度，去掉 between-dataset 偏移。
    def zwithin(sub, col):
        out = np.empty(len(sub))
        for ds_name, idx in sub.groupby("dataset").groups.items():
            v = sub.loc[idx, col].values.astype(float)
            sd = v.std(ddof=0)
            out[sub.index.get_indexer(idx)] = (v - v.mean()) / (sd if sd > 0 else 1.0)
        return out

    std_rows = []
    for space in [s for s in ["xq[full]", "xq[L0]", "z"] if s in set(df.space)]:
        for pname in ["topk", "all"]:
            for metric in METRICS:
                sub = df[(df.space == space) & (df.pos_regime == pname)
                         & (df.dataset.isin(["Sport", "Toys"]))].dropna(subset=[metric, "ndcg20"])
                if len(sub) < 8:
                    continue
                pr, pp = pearsonr(zwithin(sub, metric), zwithin(sub, "ndcg20"))
                std_rows.append(dict(space=space, pos_regime=pname, metric=metric,
                                     scope="Pooled(z-scored)", n=len(sub), pearson=pr,
                                     p_pearson=pp, spearman=float("nan"),
                                     p_spearman=float("nan")))
    corr_df = pd.concat([corr_df, pd.DataFrame(std_rows)], ignore_index=True)

    def md_table(frame, cols_order=None):
        frame = frame[[c for c in (cols_order or frame.columns) if c in frame.columns]]
        head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
        sep = "|" + "|".join(["---"] * len(frame.columns)) + "|"
        body = []
        for _, r in frame.iterrows():
            cells = []
            for c in frame.columns:
                v = r[c]
                if isinstance(v, float):
                    cells.append("—" if v != v else f"{v:.4f}")
                else:
                    cells.append(str(v))
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([head, sep] + body)

    # 主表：xq[full]，两个正样本口径
    mcols = ["variant", "ndcg20", "auc_cos_degmatch", "auc_cos_uniform",
             "auc_ham_degmatch", "auc_l0match_degmatch"]
    main_blocks = []
    for pname in ["topk", "all"]:
        sub = df[(df.space == "xq[full]") & (df.pos_regime == pname)]
        for ds in ["Sport", "Toys", "Beauty"]:
            b = sub[sub.dataset == ds][["dataset"] + mcols]
            if len(b):
                main_blocks.append(f"\n**xq[full] · pos={pname} · {ds}**\n\n" + md_table(b))

    log("")
    log("=" * 100)
    log("主表（xq[full]，Behavior AUC vs NDCG@20）")
    log("=" * 100)
    for b in main_blocks:
        log(b)
    log("")
    log("相关分析（Behavior AUC vs NDCG@20）")
    log("")
    log(md_table(corr_df[corr_df.space == "xq[full]"], ["pos_regime", "scope", "metric",
                                                        "n", "pearson", "p_pearson",
                                                        "spearman", "p_spearman"]))

    # ---------------------------------------------------------------- 报告
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Behavior Structure Preservation vs Recommendation Performance\n\n")
        f.write(f"**运行时间**：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
                f"**产出脚本**：`tools/behavior_structure_probe.py`\n\n")
        f.write("## 0. 口径\n\n"
                "- **E 图（正样本来源）**：`train.parquet` 的 `history+[target]` 序列 → 滑窗 "
                "`window=3`（即 ±1 相邻对）共现矩阵，与 TA-SID behavior 向量**同源**；"
                "**所有 variant 共用同一张 E**。\n"
                "- **正样本两轴**：`topk` = 频次最高 1000 条边（`count≥6`，强证据，"
                "＝ `doc/e49grid/` 口径）；`all` = 从 E 均匀采样 2×10⁴ 条（覆盖全图，含长尾弱边）。\n"
                "- **负样本两轴**：`degmatch` = 度分布受控（按 ⌊log₂deg⌋ 分桶，候选取自**与正样本"
                "端点同桶**，排除 E 内边与自环）—— **主口径**，堵住『热门商品彼此相似』的流行度捷径；"
                "`uniform` = 均匀随机（排除自环与 E 内边）—— 与 `doc/e49grid/` 的**近似**参照臂"
                "（e49grid 不排除 E 内边，故数值系统性偏低约 0.013；排序一致）。\n"
                "- **SID 表示**：`xq[L0]` = 第 1 层码本向量；`xq[full]` = 3 层码本向量之和"
                "（`e_dim=32`）；`z` = 编码器量化前 latent（容量天花板）。\n"
                "- **离散 token**：`Hamming` = 前 3 个 token 的差异位数（越小越相似）；"
                "`L0-match` = 首 token 是否相同。\n"
                "- **NDCG@20**：从 `model/logs/*_eval.log` **直接解析**的 3-seed 均值"
                "（generator seed；SID codes 固定不随其变化）—— **不抄表**。\n\n")
        f.write("## 1. 主表（xq[full]）\n")
        for b in main_blocks:
            f.write(b + "\n")
        f.write("\n## 2. 相关分析（Behavior AUC vs NDCG@20）\n\n")
        f.write(md_table(corr_df[corr_df.space == "xq[full]"],
                         ["pos_regime", "scope", "metric", "n", "pearson", "p_pearson",
                          "spearman", "p_spearman"]) + "\n\n")
        f.write("## 3. 全空间相关分析（xq[L0] / z）\n\n")
        sub = corr_df[corr_df.space != "xq[full]"]
        f.write((md_table(sub[["space", "pos_regime", "scope", "metric", "n", "pearson",
                               "spearman", "p_spearman"]]) if len(sub) else "（未测）") + "\n\n")
        f.write("## 4. 产物\n\n"
                f"- 逐空间长表：`{args.csv}`（dataset × variant × space × pos_regime）\n\n")
        f.write("<details><summary>原始日志</summary>\n\n```text\n")
        f.write("\n".join(log_lines))
        f.write("\n```\n\n</details>\n")
    print(f"✓ 报告 → {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
