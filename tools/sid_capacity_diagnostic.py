#!/usr/bin/env python3
"""
sid_capacity_diagnostic.py — RQ-VAE 码本容量诊断 (codebook capacity diagnostic)

回答 3 个问题：
  Q1. 不同 quantization level (z, L0, L0+L1, full) 各自承担什么？
  Q2. 各数据集量化瓶颈在哪？（input / 编码器 / 量化环节）
  Q3. Codebook capacity 是不是瓶颈？（尤其 Toys）

Q3 核心实验（不重新训练，纯只读）：
    在 z 空间做【单层 k-means 重量化】，扫描 k = 256..8192，
    比较过渡一致性 AUC 是否随 k 上升并收敛到 z 水平。
      - AUC 随 k 上升到 z 水平        → 粗层分辨率是瓶颈（容量问题，可修）
      - AUC 上升但 plateau 低于 z     → 容量是部分瓶颈，还有别的东西
      - AUC 不随 k 变化               → 容量不是瓶颈（训练/结构问题）

辅助诊断（回答"容量到底卡在哪"）：
    - 真实模型各层码本利用率：256 码用了多少格、每码多少 item
    - 3-token 组合空间碰撞率：codes 第 4 列 != 0 的比例（需要碰撞消解的商品占比）
    - fresh k-means-256 vs 真实 xq[L0]：区分"容量不够" vs "码本训练质量"
    - z AUC = k→∞ 精确重建的天然上界，作为容量扫描的天花板参照

只读约束：只读 checkpoint / SID codes / parquet，仅输出一份 markdown 报告（doc/），
          不训练、不写任何实验产物。

用法：
    python3 tools/sid_capacity_diagnostic.py \
        [--datasets Toys,Beauty,Sport] [--topk 1000] [--n_random 5000] \
        [--ks 256,512,1024,2048,4096,8192] [--report doc/sid_capacity_diagnostic.md]
"""

import argparse
import os
import sys
import datetime

import numpy as np
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)

from sid_consistency_contrast import (
    CONFIG,
    find_best_ckpt,
    load_codebook_weights,
    top_transition_pairs,
    pairs_to_rows,
    pair_sims,
    random_pair_sims,
    compute_auc,
)
from sid_per_layer_diagnostic import (
    encode_z,
    normalize_rows,
    rebuild_xq_partial,
    load_input_emb,
)

DEFAULT_KS = [256, 512, 1024, 2048, 4096, 8192]


def kmeans_quantize(z, k, seed=0, n_init=5):
    """在 z 空间做单层 k-means 量化。返回 (归一化 xq, KMeans 对象)。
    z: (N, e_dim) 已归一化 → 相当于 spherical k-means（欧氏距离单调于余弦距离）。"""
    from sklearn.cluster import KMeans
    k = min(k, z.shape[0])
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed, max_iter=300)
    labels = km.fit_predict(z.numpy())
    xq = torch.from_numpy(km.cluster_centers_[labels].astype(np.float32))
    return normalize_rows(xq), km


def code_utilization(codes, n_codes_list=None):
    """codes: (N, 4)。返回 (per-layer stats, collision_ratio, n_distinct_combos)。
    n_codes_list: 每层码本容量（长度 3）；缺省时按第 4 列最大值推断（旧 3×256 兼容）。
    stats[layer] = {used, max_per_code, mean_per_code}
    collision_ratio: 需要第 4 列消解碰撞的商品比例（3-token 空间饱和度的直接度量）。"""
    if n_codes_list is None:
        n_codes_list = [int(codes[:, layer].max()) + 1 for layer in range(3)]
    stats = []
    for layer in range(3):
        col = codes[:, layer]
        counts = np.bincount(col, minlength=n_codes_list[layer])
        nz = counts[counts > 0]
        stats.append({
            "used": int(len(nz)),
            "max_per_code": int(nz.max()) if len(nz) else 0,
            "mean_per_code": float(nz.mean()) if len(nz) else 0.0,
        })
    collision_ratio = float((codes[:, 3] != 0).mean())
    n_distinct = len(np.unique(codes[:, :3], axis=0))
    return stats, collision_ratio, n_distinct


def auc_triple(xq_n, pair_ab, n_random, seed=0):
    """对一个（已归一化）空间返回 (auc, pos_mean, neg_mean, mean_diff)。"""
    pos = pair_sims(xq_n, pair_ab)
    neg = random_pair_sims(xq_n, n_pairs=n_random, seed=seed)
    auc = compute_auc(pos, neg)
    return auc, float(pos.mean()), float(neg.mean()), float(pos.mean() - neg.mean())


def main():
    parser = argparse.ArgumentParser(description="RQ-VAE codebook capacity diagnostic")
    parser.add_argument("--datasets", type=str, default="Toys,Beauty,Sport")
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--n_random", type=int, default=5000)
    parser.add_argument("--ks", type=str, default=",".join(map(str, DEFAULT_KS)))
    parser.add_argument("--report", type=str, default="doc/sid_capacity_diagnostic.md")
    parser.add_argument("--skip_z", action="store_true", help="跳过编码器前向（z 空间）")
    parser.add_argument("--skip_sweep", action="store_true",
                        help="跳过 k-means 容量扫描，只出 anchors（z/xq[L0]/xq[L0+L1]/xq[full]）")
    parser.add_argument("--ckpt-tag", type=str, default=None,
                        help="覆盖 TA-SID ckpt 目录 tag（E3 新模型，如 Toys_TA_SID_L0SK）")
    parser.add_argument("--code-file", type=str, default=None,
                        help="覆盖 TA-SID codes 文件名（E3 新模型）")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    report_lines = []
    summary_rows = []  # (ds, n_items, z_auc, l0_auc, full_auc, coll_ratio, n_distinct)
    def log(s=""):
        print(s)
        report_lines.append(s)

    log("=" * 100)
    log("RQ-VAE 码本容量诊断（codebook capacity diagnostic）")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"转移对: top-{args.topk} 高频相邻对 | 随机对: {args.n_random} (seed=0)")
    log(f"容量扫描 k: {ks}")
    log("AUC = P(转移对 cos > 随机对 cos)，0.5=随机，1=完美")
    log("只读：不训练、不写实验产物，仅本报告")
    log("=" * 100)

    for ds in [d.strip() for d in args.datasets.split(",")]:
        if ds not in CONFIG:
            log(f"⚠ 跳过未知数据集: {ds}")
            continue
        cfg = dict(CONFIG[ds])
        if args.ckpt_tag:
            cfg["ta_ckpt"] = args.ckpt_tag
        if args.code_file:
            cfg["ta_code"] = args.code_file

        pairs = top_transition_pairs(ds, top_k=args.topk)
        if not pairs:
            log(f"{ds}: 无转移对，跳过")
            continue
        pair_ab = pairs_to_rows(ds, pairs)

        ckpt = find_best_ckpt(cfg["ta_ckpt"])
        if ckpt is None:
            log(f"  ⚠ checkpoint 缺失: {cfg['ta_ckpt']}")
            continue
        codes = np.load(os.path.join("data", ds, cfg["ta_code"]))
        cb = load_codebook_weights(ckpt)
        n_items = codes.shape[0]

        log("")
        log("=" * 100)
        log(f"▌ {ds}  |  ckpt={cfg['ta_ckpt']}  |  codes={cfg['ta_code']}  |  "
            f"items={n_items} | {cb[0].shape[0]}×{cb[1].shape[0]}×{cb[2].shape[0]} 码本, e_dim={cb[0].shape[1]}")
        log("=" * 100)

        # ---- 码本利用率 / 碰撞（真实模型）----
        n_codes_list = [cb[l].shape[0] for l in range(3)]
        stats, coll_ratio, n_distinct = code_utilization(codes, n_codes_list)
        log(f"\n── 真实模型码本利用率（{n_codes_list[0]}×{n_codes_list[1]}×{n_codes_list[2]}）──")
        for layer, s in enumerate(stats):
            log(f"  L{layer}: 用 {s['used']}/{n_codes_list[layer]} 码 | 每码 item max={s['max_per_code']}, "
                f"mean={s['mean_per_code']:.1f}")
        log(f"  3-token 组合: 去重后 {n_distinct}/{n_items} 个唯一向量 | "
            f"碰撞消解率（第4列!=0）= {coll_ratio*100:.2f}%")

        # ---- 锚点：真实模型量化空间 ----
        log("\n── 锚点（真实训练码本）──")
        anchors = {}
        for n_layers, tag in [(1, "xq[L0]"), (2, "xq[L0+L1]"), (3, "xq[full]")]:
            xq = normalize_rows(rebuild_xq_partial(codes, cb, n_layers))
            auc, pm, nm, md = auc_triple(xq, pair_ab, args.n_random)
            anchors[tag] = auc
            log(f"  {tag}: AUC={auc:.4f}  转移cos={pm:.4f} 随机cos={nm:.4f} Δ={md:+.4f}")

        # ---- z 空间（容量天花板）----
        if not args.skip_z:
            a = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
            in_path = getattr(a, "data_path", None)
            try:
                z = encode_z(ckpt, in_path)
                auc, pm, nm, md = auc_triple(z, pair_ab, args.n_random)
                anchors["z"] = auc
                log(f"  z (编码器输出, 量化前): AUC={auc:.4f}  "
                    f"转移cos={pm:.4f} 随机cos={nm:.4f} Δ={md:+.4f}")
                log(f"  → z 为容量扫描天花板（k→∞ 精确重建上界）")
            except Exception as e:
                log(f"  z 计算失败: {e}")

        # ---- Q3 核心：容量扫描 ----
        rows = []
        if args.skip_sweep:
            log("\n── 容量扫描：已跳过（--skip_sweep）──")
        else:
            log("\n── 容量扫描：z 空间单层 k-means 重量化 ──")
            log(f"  {'k':>6s} | {'AUC':>7s} | {'转移cos':>8s} {'随机cos':>8s} "
                f"{'Δ':>7s} | 距离 z ({anchors.get('z', float('nan')):.4f})")
            log("  " + "-" * 66)
        if not args.skip_z and not args.skip_sweep:
            for k in ks:
                if k > n_items:
                    break
                xq_k, _ = kmeans_quantize(z, k)
                auc, pm, nm, md = auc_triple(xq_k, pair_ab, args.n_random)
                rows.append((k, auc, pm, nm, md))
                gap = auc - anchors["z"]
                log(f"  {k:>6d} | {auc:.4f} | {pm:.4f}    {nm:.4f}   "
                    f"{md:+.4f} | {gap:+.4f}")

        # ---- fresh k-means-256 vs 真实 xq[L0]：容量 vs 训练质量 ----
        if not args.skip_z and not args.skip_sweep:
            log("\n── fresh k-means-256 vs 真实 xq[L0]（区分容量 vs 码本训练质量）──")
            for k in [256]:
                if k > n_items:
                    continue
                xq_k, _ = kmeans_quantize(z, k)
                auc, pm, nm, md = auc_triple(xq_k, pair_ab, args.n_random)
                l0_real = anchors.get("xq[L0]", float("nan"))
                diff = auc - l0_real
                log(f"  k-means-{k} (fresh): AUC={auc:.4f} | 真实 xq[L0]: {l0_real:.4f}"
                    f" | Δ={diff:+.4f}")
                if not np.isnan(diff):
                    if diff > 0.02:
                        log(f"  → fresh 显著更好：训练码本未用满容量（优化问题）")
                    elif diff < -0.02:
                        log(f"  → 真实码本更好：k-means 只是粗糙参照，容量结论需谨慎")
                    else:
                        log(f"  → 两者相当：码本训练与 k-means 一致，容量结论可信")

        # ---- 本数据集判定 ----
        if not args.skip_z and "z" in anchors and len(rows) >= 2:
            log("\n── 判定 ──")
            z_auc = anchors["z"]
            k_max, auc_max, _, _, _ = max(rows, key=lambda r: r[1])
            ceiling_ratio = (auc_max - 0.5) / (z_auc - 0.5) if z_auc > 0.5 else float("nan")
            log(f"  z={z_auc:.4f} | 扫描最大 AUC={auc_max:.4f} (k={k_max}) | "
                f"达 z 的水平={ceiling_ratio*100:.0f}%")
            if auc_max >= z_auc - 0.01:
                log("  → post-hoc 单层容量即达 z 水平：z 中结构可被量化表示，容量非硬限制；"
                    "若训练码本利用正常则量化无损，瓶颈在更早（input/编码器）")
            elif ceiling_ratio >= 0.9:
                log("  → 容量大幅缓解（≥90% 补回）：粗层分辨率是主瓶颈，可修")
            elif ceiling_ratio >= 0.6:
                log("  → 容量部分缓解（60-90% 补回）：容量是重要瓶颈，但非全部")
            else:
                log("  → 容量几乎不缓解（<60%）：容量不是瓶颈，是训练/结构问题")

            # L0 利用率判定：码本塌缩检测
            l0_used = stats[0]["used"]
            l0_mean = stats[0]["mean_per_code"]
            if l0_used < 128:
                log(f"  L0 只用了 {l0_used}/256 码（每码 mean={l0_mean:.0f} item）→ 码本塌缩："
                    f"训练码本未用满容量，真实量化丢失大于必要（post-hoc 可补回）")
            elif l0_used < 192:
                log(f"  L0 用了 {l0_used}/256 码：利用率偏低，粗层偏粗")
            else:
                log(f"  L0 用了 {l0_used}/256 码：利用率正常，容量确非瓶颈")

        summary_rows.append((
            ds, cfg["ta_ckpt"], n_items,
            anchors.get("z"), anchors.get("xq[L0]"), anchors.get("xq[L0+L1]"),
            anchors.get("xq[full]"), coll_ratio, n_distinct,
        ))

    # ================= 汇总 =================
    log("")
    log("=" * 100)
    log("汇总表（AUC + 码本利用）")
    log("=" * 100)
    hdr = f"{'Dataset':8s} {'ckpt_tag':22s} {'items':>6s} {'z':>7s} {'xq[L0]':>7s} " \
          f"{'xq[L0+L1]':>9s} {'xq[full]':>9s} {'coll%':>7s} {'唯一组合%':>9s}"
    log(hdr)
    log("-" * len(hdr))
    for ds, tag, n_items, z, l0, l01, full, coll, nd in summary_rows:
        f = lambda v: "   -  " if v is None else f"{v:.4f}"
        log(f"{ds:8s} {tag:22s} {n_items:>6d} {f(z):>7s} {f(l0):>7s} {f(l01):>9s} "
            f"{f(full):>9s} {coll*100:>6.2f}% {nd/n_items*100:>8.1f}%")
    log("备注: coll% = 需要第4列消解碰撞的商品比例（3-token 组合饱和度）；")
    log("      唯一组合% = 去重 3-token 向量占全部商品比例（≈100% = 组合空间未饱和）")

    # 写报告
    report_path = args.report
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RQ-VAE 码本容量诊断\n\n")
        f.write("```text\n")
        f.write("\n".join(report_lines))
        f.write("\n```\n")
    print(f"\n报告已写入: {report_path}")


if __name__ == "__main__":
    main()
