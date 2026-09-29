#!/usr/bin/env python3
"""
directed_transition_stats.py — Phase 1A-1：数据层 directed transition 统计（纯只读）

目的：判断 Sport / Toys 训练序列中是否存在**足够规模的 reciprocal item-transition
      pairs**，为后续方向性诊断（Stage 1B probe / edge-disjoint split）提供数据基础。

口径（严格）：
    序列        = train.parquet 的 history + [target]（逐用户）
    只统计相邻    i -> j  （C_ij = #{(t): i_t = i, i_{t+1} = j}）
    只读 train.parquet；不读 valid/test；不触碰 z / xq / SID / RQ-VAE / checkpoint
    不计算 A_ij、不做 popularity correlation、不训练任何模型

reciprocal pair 的定义（关键）：
    无序 pair {i,j} 计数**一次**，当且仅当 C_ij > 0 **且** C_ji > 0。
    即 (i,j) 与 (j,i) 不重复计为两个 reciprocal pair。

自环 i == j 单独统计并**排除**在 directed / reciprocal 统计之外
（C_ii 无法与任何反向边构成 reciprocal 关系）。

只读约束：不写任何实验产物到 data/、rqvae/、model/。仅输出 markdown + CSV 到 doc/exp/。
"""

import argparse
import datetime
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
DATA_DIR = os.path.join(ROOT, "data")


def load_sequences(ds):
    """只读 data/{ds}/train.parquet，返回 (user_ids, sequences)。

    序列 = list(history) + [int(target)]，逐用户。
    """
    path = os.path.join(DATA_DIR, ds, "train.parquet")
    df = pd.read_parquet(path)
    users, seqs = [], []
    for u, h, t in zip(df["user"], df["history"], df["target"]):
        s = [int(x) for x in list(h)]
        s.append(int(t))
        users.append(int(u))
        seqs.append(s)
    return path, users, seqs


def count_directed(seqs):
    """统计有向相邻 transfer。

    返回 (C, n_trans, n_selfloop, n_empty)。
        C         : Counter{(i,j): count}，**不含自环**
        n_trans   : 含自环的相邻转移总次数
        n_selfloop: i == j 的相邻转移次数
        n_empty   : len(seq) < 2 的序列数（无相邻对）
    """
    C = Counter()
    n_trans = 0
    n_selfloop = 0
    n_empty = 0
    for s in seqs:
        if len(s) < 2:
            n_empty += 1
            continue
        for a, b in zip(s[:-1], s[1:]):
            n_trans += 1
            if a == b:
                n_selfloop += 1
                continue
            C[(a, b)] += 1
    return C, n_trans, n_selfloop, n_empty


def classify_pairs(C):
    """把 unique directed pair 分成 reciprocal / one-way。

    reciprocal pair 按无序 {i,j} 计数一次（i < j）。
    返回 dict。
    """
    reciprocal = {}   # (i_min, j_max) -> (C_ij_for_min, C_ij_for_max) 保持方向语义
    one_way = []      # (i, j)

    for (i, j), c in C.items():
        if (j, i) in C:
            if i < j:
                reciprocal[(i, j)] = (c, C[(j, i)])
            # i > j 时已由 (j,i) 收录，跳过 → 无序计数一次
        else:
            one_way.append((i, j))

    return reciprocal, one_way


def dist_summary(vals):
    a = np.asarray(vals, dtype=np.float64)
    return dict(
        n=int(a.size), min=float(a.min()), median=float(np.median(a)),
        mean=float(a.mean()), max=float(a.max()),
        p25=float(np.percentile(a, 25)), p75=float(np.percentile(a, 75)),
    )


def count_histogram(C, max_v=6):
    """有向 pair 的计数直方图：counts -> 该 count 的 pair 数。"""
    h = Counter(C.values())
    total = len(C)
    out = []
    cum = 0
    for v in range(1, max_v + 1):
        n = h.get(v, 0)
        cum += n
        out.append((v, n, n / total if total else 0.0, cum / total if total else 0.0))
    tail = total - cum
    return out, tail, total


def reciprocal_tiers(reciprocal):
    """按 (min(C_ij,C_ji), max(C_ij,C_ji)) 给 reciprocal pair 分层。"""
    tiers = {
        "both==1": 0,
        "min==1 & max>=2": 0,
        "min>=2": 0,
        "min>=3": 0,
    }
    for (i, j), (a, b) in reciprocal.items():
        lo, hi = min(a, b), max(a, b)
        if lo == 1 and hi == 1:
            tiers["both==1"] += 1
        if lo == 1 and hi >= 2:
            tiers["min==1 & max>=2"] += 1
        if lo >= 2:
            tiers["min>=2"] += 1
        if lo >= 3:
            tiers["min>=3"] += 1
    return tiers


def analyze(ds):
    path, users, seqs = load_sequences(ds)
    C, n_trans, n_selfloop, n_empty = count_directed(seqs)

    n_users = len(users)
    D = len(C)                                    # unique directed pair（不含自环）
    reciprocal, one_way = classify_pairs(C)
    R = len(reciprocal)                           # reciprocal pair（无序，计一次）
    recip_directed_edges = 2 * R                  # 参与 reciprocal 关系的有向边数
    OW = len(one_way)

    # ---- reciprocal pair 的 C_ij / C_ji 分布 ----
    dir_counts = []       # 全部 2R 条有向计数
    strong_side = []      # 每个 reciprocal pair 的 max(C_ij, C_ji)
    weak_side = []        # 每个 reciprocal pair 的 min(C_ij, C_ji)
    for (i, j), (c_ij, c_ji) in reciprocal.items():
        dir_counts.extend([c_ij, c_ji])
        strong_side.append(max(c_ij, c_ji))
        weak_side.append(min(c_ij, c_ji))

    # ---- reciprocal 子图的 item 参与度（判断 edge-disjoint split 可行性） ----
    item_deg = Counter()
    for (i, j) in reciprocal:
        item_deg[i] += 1
        item_deg[j] += 1
    items_in_recip = len(item_deg)

    # ---- 也给出无向 E 图（全部边）作参照，说明 reciprocal 的稀缺性 ----
    undirected_all = set()
    for (i, j) in C:
        undirected_all.add((i, j) if i < j else (j, i))

    # ---- 异常检查 ----
    n_items_seen = len({x for s in seqs for x in s})
    empty_hist = sum(1 for s in seqs if len(s) < 2)
    all_ids = {x for s in seqs for x in s}
    id_min, id_max = min(all_ids), max(all_ids)
    id_is_contiguous = (all_ids == set(range(id_min, id_max + 1)))
    # 自环是否由同一用户连续重复产生
    selfloop_items = Counter()
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            if a == b:
                selfloop_items[a] += 1

    res = dict(
        ds=ds, path=os.path.relpath(path, ROOT),
        n_users=n_users, n_trans=n_trans, n_selfloop=n_selfloop,
        n_selfloop_ratio=n_selfloop / n_trans if n_trans else 0.0,
        n_directed_seq_edges=n_trans - n_selfloop,
        D=D, R=R, recip_directed_edges=recip_directed_edges, OW=OW,
        ratio_R_over_D=R / D if D else 0.0,
        ratio_recip_edges_over_D=recip_directed_edges / D if D else 0.0,
        dir_counts=dir_counts, strong_side=strong_side, weak_side=weak_side,
        items_in_recip=items_in_recip, item_deg=item_deg,
        undirected_all=len(undirected_all),
        reciprocal=reciprocal, one_way=one_way,
        n_items_seen=n_items_seen, empty_hist=empty_hist,
        id_min=id_min, id_max=id_max, id_is_contiguous=id_is_contiguous,
        selfloop_items=selfloop_items,
        sequences=seqs,
        C=C,
        n_seqs=len(seqs),
    )
    return res


def fmt_dist(s, indent="    "):
    return (f"{indent}n={s['n']}  min={s['min']:.0f}  p25={s['p25']:.0f}  "
            f"median={s['median']:.0f}  mean={s['mean']:.2f}  p75={s['p75']:.0f}  max={s['max']:.0f}")


def main():
    ap = argparse.ArgumentParser(description="Phase 1A-1 directed transition stats (read-only)")
    ap.add_argument("--datasets", type=str, default="Sport,Toys")
    ap.add_argument("--csv", type=str, default="doc/exp/phase1a1_directed_transition_stats.csv")
    ap.add_argument("--report", type=str, default="doc/exp/phase1a1_directed_transition_stats.md")
    args = ap.parse_args()

    L = []
    def log(s=""):
        print(s)
        L.append(s)

    log("=" * 96)
    log("Phase 1A-1  directed transition 统计（只读 train.parquet）")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("序列口径: history + [target]（逐用户）  |  只统计相邻 i->j  |  自环单独统计并排除")
    log("reciprocal: 无序 {i,j} 计数一次，当且仅当 C_ij>0 且 C_ji>0")
    log("=" * 96)

    rows = []
    results = {}
    for ds in [d.strip() for d in args.datasets.split(",")]:
        r = analyze(ds)
        results[ds] = r

        log("")
        log("#" * 96)
        log(f"# {ds}   ({r['path']})")
        log("#" * 96)
        log(f"用户数                      : {r['n_users']}")
        log(f"序列数 (len>=2)             : {r['n_seqs'] - r['empty_hist']}   (len<2 跳过: {r['empty_hist']})")
        log(f"出现在 train 序列中的 item  : {r['n_items_seen']}")
        log("")
        log(f"总 directed transition 次数  : {r['n_trans']}")
        log(f"  其中自环 i==j             : {r['n_selfloop']}  ({r['n_selfloop_ratio']:.4%})")
        log(f"  非自环 directed 边次       : {r['n_directed_seq_edges']}")
        log("")
        log(f"unique directed pair  D     : {r['D']}")
        log(f"reciprocal pair       R     : {r['R']}")
        log(f"  → R / D                   : {r['ratio_R_over_D']:.4%}")
        log(f"  → 参与 reciprocal 的有向边 : {r['recip_directed_edges']}  (= 2R, 占总 D 的 {r['ratio_recip_edges_over_D']:.4%})")
        log(f"one-way unique pair         : {r['OW']}   (单向，无反向边)")
        log(f"(参照) 无向 E 图边数         : {r['undirected_all']}   ← R / E = {r['R']/r['undirected_all']:.4%}" if r['undirected_all'] else "")
        log("")
        log(f"涉及 reciprocal 的 item 数   : {r['items_in_recip']}  / {r['n_items_seen']} train items"
            f"  ({r['items_in_recip']/r['n_items_seen']:.2%})")
        deg = sorted(r['item_deg'].values())
        log(f"  每个 item 的 reciprocal 伙伴数: min={deg[0]} median={int(np.median(deg))} "
            f"mean={np.mean(deg):.2f} max={deg[-1]}")
        log("")
        log("reciprocal pair 的 C_ij / C_ji 计数分布（全部 2R 条有向计数）:")
        log(fmt_dist(dist_summary(r['dir_counts'])))
        log("  拆开看（每个 reciprocal pair 的强弱两侧）:")
        log("    强侧(max):")
        log(fmt_dist(dist_summary(r['strong_side']), "      "))
        log("    弱侧(min):")
        log(fmt_dist(dist_summary(r['weak_side']), "      "))
        log("")
        log(f"最终可用于后续 pair-level analysis 的 reciprocal pair 数量: {r['R']}")
        log("  （未做任何 margin / A_ij 过滤 —— 按 Phase 1A-1 范围要求，δ 过滤属 1B）")

        # ---- 稀疏度：计数直方图 + reciprocal 分层 ----
        hist, tail, tot = count_histogram(r['C'])
        log("")
        log("有向 pair 计数直方图（稀疏度检查）:")
        for v, n, frac, cum in hist:
            log(f"    C_ij == {v}: {n:>8d} pair  ({frac:6.2%})   累计 {cum:6.2%}")
        log(f"    C_ij >= {len(hist)+1}: {tail:>8d} pair  ({tail/tot:6.2%})")
        log(f"    → 平均每条 unique directed pair 出现 {r['n_directed_seq_edges']/r['D']:.4f} 次")

        tiers = reciprocal_tiers(r['reciprocal'])
        log("")
        log("reciprocal pair 按两侧计数分层:")
        log(f"    C_ij == C_ji == 1      : {tiers['both==1']:>6d}  ({tiers['both==1']/r['R']:6.2%})  ← 两个方向各只出现 1 次")
        log(f"    min==1, max>=2         : {tiers['min==1 & max>=2']:>6d}  ({tiers['min==1 & max>=2']/r['R']:6.2%})")
        log(f"    min >= 2（两侧都 >=2）  : {tiers['min>=2']:>6d}  ({tiers['min>=2']/r['R']:6.2%})")
        log(f"    min >= 3（两侧都 >=3）  : {tiers['min>=3']:>6d}  ({tiers['min>=3']/r['R']:6.2%})")
        log("      ↑ 两侧计数越大，方向差异越不可能是单次采样的偶然")

        _hist, _tail, _tot = count_histogram(r['C'])
        _tiers = reciprocal_tiers(r['reciprocal'])
        rows.append(dict(
            dataset=ds,
            recip_both_count_1=_tiers["both==1"],
            recip_min1_max2=_tiers["min==1 & max>=2"],
            recip_min_ge2=_tiers["min>=2"],
            recip_min_ge3=_tiers["min>=3"],
            frac_directed_pairs_count1=(_hist[0][2] if _hist else 0.0),
            mean_multiplicity_per_directed_pair=r['n_directed_seq_edges'] / r['D'] if r['D'] else 0.0,
            n_users=r['n_users'], n_trans=r['n_trans'],
            n_selfloop=r['n_selfloop'], n_directed_edges=r['n_directed_seq_edges'],
            unique_directed_pairs=r['D'], reciprocal_pairs=r['R'],
            ratio_R_over_D=r['ratio_R_over_D'], recip_directed_edges=r['recip_directed_edges'],
            one_way_unique_pairs=r['OW'], undirected_E_edges=r['undirected_all'],
            items_in_recip=r['items_in_recip'], n_items_seen=r['n_items_seen'],
            n_empty_hist=r['empty_hist'],
            id_min=r['id_min'], id_max=r['id_max'], id_is_contiguous=r['id_is_contiguous'],
            recip_C_min=int(min(r['dir_counts'])), recip_C_median=float(np.median(r['dir_counts'])),
            recip_C_mean=float(np.mean(r['dir_counts'])), recip_C_max=int(max(r['dir_counts'])),
            strong_min=int(min(r['strong_side'])), strong_median=float(np.median(r['strong_side'])),
            strong_mean=float(np.mean(r['strong_side'])), strong_max=int(max(r['strong_side'])),
            weak_min=int(min(r['weak_side'])), weak_median=float(np.median(r['weak_side'])),
            weak_mean=float(np.mean(r['weak_side'])), weak_max=int(max(r['weak_side'])),
            final_usable_reciprocal_pairs=r['R'],
        ))

    # ---------- 汇总表 ----------
    log("")
    log("=" * 96)
    log("汇总表")
    log("=" * 96)
    hdr = f"{'Dataset':8s} {'users':>7s} {'trans':>8s} {'D(dir)':>8s} {'R(recip)':>9s} {'R/D':>8s} {'2R':>8s} {'one-way':>8s} {'E_undir':>8s}"
    log(hdr)
    log("-" * len(hdr))
    for x in rows:
        log(f"{x['dataset']:8s} {x['n_users']:>7d} {x['n_trans']:>8d} {x['unique_directed_pairs']:>8d} "
            f"{x['reciprocal_pairs']:>9d} {x['ratio_R_over_D']:>7.2%} {x['recip_directed_edges']:>8d} "
            f"{x['one_way_unique_pairs']:>8d} {x['undirected_E_edges']:>8d}")

    # ---------- 70/10/20 edge-disjoint 可行性核算 ----------
    log("")
    log("=" * 96)
    log("70/10/20 edge-disjoint split 可行性核算（基于 reciprocal pair 为单位）")
    log("=" * 96)
    for x in rows:
        R = x['reciprocal_pairs']
        tr, va, te = int(round(R * .7)), int(round(R * .1)), R - int(round(R * .7)) - int(round(R * .1))
        log(f"\n── {x['dataset']}  R={R} ──")
        log(f"  train≈{tr}  val≈{va}  test≈{te}")
        log(f"  最小类(val) 样本量: {va}  → {'✅ 足够（>=1000）' if va >= 1000 else '⚠ 偏少（<1000）' if va >= 200 else '❌ 过少（<200）'}")
        log(f"  test 中正/负方向各约: {te//2} / {te - te//2}（若方向标签均衡）")
        log(f"  参与 item 数 {x['items_in_recip']} → 每 item 平均出现于 "
            f"{x['items_in_recip'] and (R*2/x['items_in_recip']):.2f} 个 reciprocal pair")

    # ---------- 异常 ----------
    log("")
    log("=" * 96)
    log("异常检查")
    log("=" * 96)
    for x in rows:
        r = results[x['dataset']]
        log(f"\n── {x['dataset']} ──")
        log(f"  自环 i==j（相邻重复 item）: {x['n_selfloop']}  ({x['n_selfloop']/x['n_trans']:.4%})"
            f"  |  涉及 {len(r['selfloop_items'])} 个不同 item")
        if r['selfloop_items']:
            top = r['selfloop_items'].most_common(3)
            log(f"    最高频自环 item: {top}")
        log(f"  len<2 的序列（无相邻对）: {x['n_empty_hist']}")
        log(f"  item id 范围: [{x['id_min']}, {x['id_max']}]  "
            f"连续无洞: {x['id_is_contiguous']}  |  train 中出现 {x['n_items_seen']} 个")
        # 每个用户序列长度分布（history + target 的总长度）
        lens = [len(s) for s in r['sequences']]
        log(f"  序列总长 (history+target): min={min(lens)} median={int(np.median(lens))} "
            f"mean={np.mean(lens):.2f} max={max(lens)}  (共 {len(lens)} 个用户)")

    # ---------- 写 CSV / 报告 ----------
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    log("")
    log(f"CSV 已写入: {args.csv}")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# Phase 1A-1 directed transition 统计（只读）\n\n")
        f.write(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　"
                f"脚本：`tools/directed_transition_stats.py`\n\n")
        f.write("输入：`data/{Sport,Toys}/train.parquet` 的 `history + [target]`。"
                "不读 valid/test，不触碰 z / xq / SID / RQ-VAE / checkpoint。\n\n")
        f.write("```text\n")
        f.write("\n".join(L))
        f.write("\n```\n")
    print(f"报告已写入: {args.report}")

    return results


if __name__ == "__main__":
    main()
