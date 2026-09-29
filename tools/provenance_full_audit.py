#!/usr/bin/env python3
"""全量 codes provenance 审计（**只读**；不训练、不改任何产物）。

背景
----
本项目已出现 6 次"单个 provenance 读数出错 ⇒ 整条论断作废"的同型事故，其中
**孤儿 codes**（`Toys_ta_sid_768d_sid.npy` L0=62 无法由任何现存 ckpt 复现，
见 `doc/e48_toys_main_codes_provenance_fix.md`）是最贵的一次——它支撑的 Toys 塌缩
与 NDCG 0.0367 全部作废，而闸门 B 的唯一正例也随之消失。

既有的孤儿搜索（`tools/checkpoint_trajectory_probe.py` 的 `toys_audit` 阶段）有两个局限：
  1. 只查 **2 个已知孤儿**，不覆盖其余 codes 产物；
  2. 候选集只取每个 run 目录的 `best_collision_model.pth`，**未穷举全部 epoch 快照**。
本脚本补齐这两点：**全部 codes 产物 × 全部 ckpt 快照**。

设计
----
朴素做法是"每个 codes 产物扫一遍全部 ckpt"，复杂度 O(码表 × ckpt) ≈ 68×795 ≈ 5.4 万次重导。
本脚本**倒过来**：每个 ckpt 只编码一次，再与**同族**全部 codes 产物比对
（同族判定取自 `args.data_path`，比目录名可靠），复杂度 O(ckpt) ≈ 795 次重导。
实测单次 ≈ 0.25 s（load 0.02 + encode 0.21 + rederive 0.01，Toys 11,924 items）⇒ 全程约 3–5 min。

判据（关键）
------------
**决策列只有 L0**。理由是 `sf_rq.rederive_codes_sequential` 的 docstring 已写明：
给定同一码本与同一 z，**L0 一致率 = 100% ⟹ L1 必然 = 100%**（残差被完全决定），
故 L1 不构成独立证据；L2 还额外受碰撞消解循环（`use_sk=True` 那一趟）影响。
故 L1/L2 仅作"码本同一性检查"记录，**不参与 clean/orphan 判定**。
（同一判断的先例：`doc/e49grid_diag_summary.md` §1。）

`use_sk` 取 **False** —— 那是与 codes 文件匹配的**主 pass**（codegen 是两趟：初始 pass
`use_sk=False`，碰撞消解循环 `use_sk=True`）。实测 L0/L1 与 use_sk 无关，只有 L2 受影响。

复用（不重写）
--------------
`rqvae/sf_rq.py` : `load_rqvae_from_ckpt` / `encode_z` / `rederive_codes_sequential`

产物（全部落在本脚本的 --out-dir，默认 doc/）
----------
`provenance_codes_verdict.csv` : 每个 codes 产物一行 —— 判定 clean/orphan + 最佳匹配
`provenance_codes_matrix.csv`  : 长表 (codes_file × ckpt) 的逐层一致率
`provenance_ckpt_inventory.csv`: 795 个 ckpt 的清单（含 data_path / in_dim / 码本配置）
`provenance_full_audit.log`    : 全量日志
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rqvae"))
from sf_rq import load_rqvae_from_ckpt, encode_z, rederive_codes_sequential  # noqa: E402

# 已知数据集族（按长度降序匹配，避免 "Sport" 抢到 "Sports"）。
# 用途：仅当 ckpt 的 args.data_path 不可解析时的兜底，以及 codes 产物的族归属。
KNOWN_FAMILIES = [
    "Musical_Instruments", "Video_Games", "Electronics", "Sports", "Sport",
    "Beauty", "Toys", "MI",
]

# 自检锚点（必须在 --self-check 下全部通过，否则脚本输出不可采信）
ANCHORS_ORPHAN = [
    # (codes 文件相对 data/ 的路径, 期望判定, 期望最佳匹配包含的子串, 期望最佳 L0 一致率的近似值)
    ("Toys/Toys_ta_sid_768d_sid.npy", "orphan", "Toys_TA_SID_768d", 0.4897),
    ("Toys/Toys_random_sid.npy", "orphan", "Toys_TA_SID_seed2025", 0.0107),
]
ANCHORS_CLEAN = [
    ("Beauty/Beauty_ta_sid_768d_sid.npy", "Beauty_TA_SID"),
    ("Sport/Sport_ta_sid_768d_sid.npy", "Sport_TA_SID"),
]
CLEAN_TOL = 1e-9
NEAR_MISS_TOL = 0.999   # 母码本找对、仅碰撞消解重写少数行


def log(msg=""):
    print(msg, flush=True)


def family_of_name(name: str):
    """从文件名/目录名推断数据集族。按长度降序匹配以避开 Sport/Sports 前缀冲突。"""
    base = os.path.basename(name)
    for fam in sorted(KNOWN_FAMILIES, key=len, reverse=True):
        if base.startswith(fam):
            return fam
    return None


def family_of_data_path(dp: str):
    """从 args.data_path 推断数据集族（比目录名可靠）。

    注意：必须取**最后**一个 "data" 段。绝对路径形如
    /data/gtx/project/code/TIGER_minilm/data/Beauty/fused_item_emb_4096d.parquet，
    首个 "data" 是挂载点，取它会把族判成 "gtx"（2026-09-28 冒烟测试踩到的坑）。
    只认 KNOWN_FAMILIES 里的下一段，否则回退到文件名/目录名推断。
    """
    norm = dp.replace("\\", "/")
    parts = norm.split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data" and i + 1 < len(parts):
            fam = family_of_name(parts[i + 1])
            if fam:
                return fam
    return family_of_name(norm)


def enumerate_ckpts(roots):
    """roots: [(label, dir)]，期望目录结构 {dir}/{tag}/{run_ts}/*.pth。"""
    out = []
    for label, d in roots:
        for p in sorted(glob.glob(os.path.join(d, "*", "*", "*.pth"))):
            if label == "legacy":
                tag = p[len(d):].lstrip(os.sep).split(os.sep)[0]
                run = os.path.basename(os.path.dirname(p))
            else:
                rel = p[len(d):].lstrip(os.sep).split(os.sep)
                tag = rel[1] if len(rel) >= 3 else "?"
                run = rel[-2] if len(rel) >= 2 else "?"
            out.append({"root": label, "tag": tag, "run": run, "path": p})
    return out


def enumerate_codes(data_dir, traj_root, backup_root):
    """枚举全部 codes 产物：data/*/*.npy + traj/codes/*.npy + backup/*.npy。

    过滤方式：ndim==2、shape[1]==4、整数 dtype —— 借此排除 item_mapping / user_mapping /
    t5 embedding 等非码表文件（它们形状/类型不同）。
    """
    cands = []
    for pat, kind in ((os.path.join(data_dir, "*", "*.npy"), "data"),
                      (os.path.join(traj_root, "codes", "*.npy"), "traj"),
                      (os.path.join(backup_root, "*.npy"), "backup")):
        cands += [(p, kind) for p in sorted(glob.glob(pat))]
    out = []
    for p, kind in cands:
        try:
            a = np.load(p)
        except Exception:                                    # noqa: BLE001
            continue
        if a.ndim != 2 or a.shape[1] != 4 or not np.issubdtype(a.dtype, np.integer):
            continue
        if a.shape[0] < 100:                                 # 排除小 mapping 表
            continue
        if kind == "data":
            fam = family_of_name(os.path.dirname(p))
        else:
            fam = family_of_name(p)
        out.append({"path": p, "kind": kind, "family": fam, "N": int(a.shape[0])})
    return out


def main():
    ap = argparse.ArgumentParser(description="全量 codes provenance 审计（只读）")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--legacy-ckpt-root", default=os.path.join(ROOT, "rqvae/ckpt"))
    ap.add_argument("--traj-root", default=os.path.join(ROOT, "rqvae/checkpoint_trajectory"))
    ap.add_argument("--backup-root", default=os.path.join(ROOT, "backup_old_sport"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "doc"))
    ap.add_argument("--log", default=None, help="日志路径（默认 <out-dir>/provenance_full_audit.log）")
    ap.add_argument("--self-check", action="store_true", help="额外校验自检锚点，任一不符即非零退出")
    ap.add_argument("--max-ckpt", type=int, default=0, help="调试用：只跑前 N 个 ckpt（0=全部）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.dirname(args.log) if args.log else ""
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_path = args.log or os.path.join(args.out_dir, "provenance_full_audit.log")
    fh = open(log_path, "w", encoding="utf-8")

    def L(msg=""):
        log(msg)
        fh.write(msg + "\n")

    L("=" * 100)
    L("全量 codes provenance 审计   %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    L("=" * 100)

    # ---- 枚举 ----------------------------------------------------------------
    ckpts = enumerate_ckpts([("legacy", args.legacy_ckpt_root), ("traj", os.path.join(args.traj_root, "ckpt"))])
    codes = enumerate_codes(args.data_dir, args.traj_root, args.backup_root)
    if args.max_ckpt:
        ckpts = ckpts[:args.max_ckpt]
    L("\n▌ 0. 规模")
    L("  ckpt 候选      : %d" % len(ckpts))
    L("  codes 产物     : %d" % len(codes))
    fam_cnt = {}
    for c in codes:
        fam_cnt[c["family"]] = fam_cnt.get(c["family"], 0) + 1
    L("  codes 按族分布 : %s" % ", ".join("%s=%d" % kv for kv in sorted(fam_cnt.items(), key=lambda x: str(x[0]))))

    # codes 产物索引：family -> list，便于按族取
    codes_by_fam = {}
    for c in codes:
        codes_by_fam.setdefault(c["family"], []).append(c)
    # 每族内按 N 分组（N 不同不可能同源）
    for fam, lst in codes_by_fam.items():
        by_n = {}
        for c in lst:
            by_n.setdefault(c["N"], []).append(c)
        codes_by_fam[fam] = by_n

    # 逐 codes 产物的最佳匹配（只跟踪 L0）
    best = {c["path"]: {"ckpt": None, "L0": -1.0, "L1": -1.0, "L2": -1.0, "n_exact": 0} for c in codes}
    matrix_rows = []
    inv_rows = []

    # ---- 主循环：每个 ckpt 只编码一次 ----------------------------------------
    L("\n▌ 1. 逐 ckpt 重导并与同族 codes 产物比对（决策列 = L0）")
    n_ok = n_skip = n_fail = 0
    for i, ck in enumerate(ckpts, 1):
        try:
            model, a = load_rqvae_from_ckpt(ck["path"], device="cpu")
        except Exception as e:                               # noqa: BLE001
            n_fail += 1
            L("  [%3d/%d] %-56s ❌ load 失败: %s" % (i, len(ckpts), ck["tag"], e))
            continue

        dp = a.data_path
        fam = family_of_data_path(a.data_path) or family_of_name(ck["tag"])
        # 路径解析：ckpt 里记录的可能是相对路径（如 ../data/Sport/x.parquet），
        # 其含义随训练时的 CWD 变化。若按 ROOT 解析不到，回退到 data/{fam}/{basename}。
        # 2026-09-28：不加此回退会静默跳过 61/795 个 ckpt（含 Sport_Trans_* 共 33 个），
        # 而这些恰好是 Sport track_b 两份 codes 的最可能母 ckpt ⇒ 孤儿判定不可信。
        dp_note = "as_recorded"
        if not os.path.isabs(dp):
            cand = os.path.normpath(os.path.join(ROOT, dp))
            if os.path.exists(cand):
                dp = cand
            else:
                alt = os.path.join(ROOT, "data", fam, os.path.basename(dp)) if fam else None
                if alt and os.path.exists(alt):
                    dp, dp_note = alt, "basename_fallback"
                else:
                    dp = cand

        inv_rows.append({
            "root": ck["root"], "tag": ck["tag"], "run": ck["run"],
            "ckpt_file": os.path.relpath(ck["path"], ROOT),
            "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(ck["path"])).strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": a.data_path,
            "data_exists": int(os.path.exists(dp)),
            "dp_note": dp_note,
            "family": fam,
            "num_emb_list": "x".join(str(x) for x in a.num_emb_list),
            "e_dim": a.e_dim,
            "sk_epsilons": "x".join(str(x) for x in getattr(a, "sk_epsilons", [])),
        })

        if not os.path.exists(dp):
            n_skip += 1
            L("  [%3d/%d] %-56s ⚠ data_path 不存在，跳过（%s）" % (i, len(ckpts), ck["tag"], a.data_path))
            continue
        try:
            z = encode_z(model, dp, device="cpu")
            pred = rederive_codes_sequential(model, z, use_sk=False)
        except Exception as e:                               # noqa: BLE001
            n_skip += 1
            L("  [%3d/%d] %-56s ⚠ 编码/重导失败，跳过: %s" % (i, len(ckpts), ck["tag"], e))
            continue
        N = int(pred.shape[0])
        inv_rows[-1]["N_items"] = N
        inv_rows[-1]["L0_distinct"] = int(len(np.unique(pred[:, 0])))
        n_ok += 1
        if n_ok % 25 == 0 or i == len(ckpts):
            L("  [进度 %3d/%d] 成功=%d 跳过=%d 失败=%d 矩阵行=%d"
              % (i, len(ckpts), n_ok, n_skip, n_fail, len(matrix_rows)))

        for c in codes_by_fam.get(fam, {}).get(N, []):
            tgt = np.load(c["path"])
            if tgt.shape[0] != N:
                continue
            a0 = float((tgt[:, 0] == pred[:, 0]).mean())
            a1 = float((tgt[:, 1] == pred[:, 1]).mean())
            a2 = float((tgt[:, 2] == pred[:, 2]).mean())
            tag_run = "%s/%s/%s" % (ck["tag"], ck["run"], os.path.basename(ck["path"]))
            matrix_rows.append({
                "codes_file": os.path.relpath(c["path"], ROOT),
                "codes_kind": c["kind"], "dataset": fam, "N": N,
                "ckpt": os.path.relpath(ck["path"], ROOT), "ckpt_tag": ck["tag"], "ckpt_run": ck["run"],
                "agree_L0": a0, "agree_L1": a1, "agree_L2": a2,
            })
            b = best[c["path"]]
            if a0 > b["L0"]:
                b.update({"ckpt": tag_run, "L0": a0, "L1": a1, "L2": a2})
            if a0 >= 1.0 - CLEAN_TOL:
                b["n_exact"] += 1

    L("\n  完成：成功 %d / 跳过 %d / 失败 %d" % (n_ok, n_skip, n_fail))

    # ---- 判定 ----------------------------------------------------------------
    L("\n▌ 2. 逐 codes 产物判定（clean = 存在 L0 完全一致的 ckpt）")
    verdict_rows = []
    for c in sorted(codes, key=lambda x: (str(x["family"]), x["path"])):
        if c["path"] not in best:
            continue
        b = best[c["path"]]
        tgt = np.load(c["path"])
        # 三分类之外额外区分「近似命中」：best_agree_L0 ≥ NEAR_MISS_TOL 说明母码本找对了，
        # 差异只是碰撞消解循环（use_sk=True 第二趟）重写的少数行，不是 provenance 事故。
        # 2026-09-28：Sport_ta_sid_768d_seed2026_sid.npy 差 1/18357 行（L0=0.999946），
        # 若与 L0=0.005 的真孤儿同标签会严重误导。
        if b["n_exact"] > 0:
            v = "clean"
        elif b["L0"] < 0:
            v = "NO-CANDIDATE"
        elif b["L0"] >= NEAR_MISS_TOL:
            v = "near-miss"
        else:
            v = "orphan"
        verdict_rows.append({
            "codes_file": os.path.relpath(c["path"], ROOT), "codes_kind": c["kind"],
            "dataset": c["family"], "N": c["N"],
            "L0_distinct": int(len(np.unique(tgt[:, 0]))),
            "n_exact_match_ckpt": b["n_exact"],
            "best_ckpt": b["ckpt"] or "", "best_agree_L0": "" if b["L0"] < 0 else "%.6f" % b["L0"],
            "best_agree_L1": "" if b["L1"] < 0 else "%.6f" % b["L1"],
            "best_agree_L2": "" if b["L2"] < 0 else "%.6f" % b["L2"],
            "verdict": v,
        })
    for r in verdict_rows:
        mark = {"clean": "✅", "orphan": "❌", "near-miss": "🔸", "NO-CANDIDATE": "⚠"}[r["verdict"]]
        L("  %s %-58s L0dist=%4d n_exact=%3d 最佳=%-48s L0=%-9s L2=%-9s"
          % (mark, os.path.basename(r["codes_file"]), r["L0_distinct"], r["n_exact_match_ckpt"],
             r["best_ckpt"] or "—", r["best_agree_L0"] or "—", r["best_agree_L2"] or "—"))

    cnt = {}
    for r in verdict_rows:
        cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
    L("\n  ⇒ " + " / ".join("%s %d" % (k, cnt.get(k, 0))
       for k in ("clean", "near-miss", "orphan", "NO-CANDIDATE")) + "（共 %d）" % len(verdict_rows))

    # ---- 自检锚点 ------------------------------------------------------------
    ok = True
    if args.self_check:
        L("\n▌ 3. 自检锚点（任一不符 ⇒ 输出不可采信）")
        vmap = {r["codes_file"]: r for r in verdict_rows}
        # ⚠ 锚点只校验「判定 + 一致率」这两项实质条件，**不**校验 best_ckpt 的目录名子串。
        # 理由（2026-09-28 全量跑第一次踩到）：加入 137 个 checkpoint_trajectory ckpt 后，
        # 轨迹复现 ckpt 与旧 ckpt 是同一模型、一致率逐位相同，「最佳匹配」会两者皆可。
        # 用目录名子串做断言会因此误报失败，而判定本身是对的 —— 那是锚点写法脆，不是脚本错。
        for rel, want_verdict, want_sub, want_agree in ANCHORS_ORPHAN:
            key = os.path.join("data", rel)
            r = vmap.get(key)
            if r is None:
                L("  ❌ 锚点缺失: %s" % key); ok = False; continue
            good = (r["verdict"] == want_verdict
                    and abs(float(r["best_agree_L0"]) - want_agree) < 0.01)
            ok &= good
            L("  %s %s → %s, 最佳=%s, L0=%s（期望 %s / L0≈%.4f；参考来源 %s）"
              % ("✅" if good else "❌", rel, r["verdict"], r["best_ckpt"], r["best_agree_L0"],
                 want_verdict, want_agree, want_sub))
        for rel, want_sub in ANCHORS_CLEAN:
            key = os.path.join("data", rel)
            r = vmap.get(key)
            if r is None:
                L("  ❌ 锚点缺失: %s" % key); ok = False; continue
            good = (r["verdict"] == "clean")
            ok &= good
            L("  %s %s → %s, n_exact=%s, 最佳=%s（期望 clean；参考来源 %s）"
              % ("✅" if good else "❌", rel, r["verdict"], r["n_exact_match_ckpt"], r["best_ckpt"], want_sub))
        L("\n  ⇒ 自检 %s" % ("全部通过 ✅" if ok else "**未通过 ❌ —— 脚本不可信，先修脚本**"))

    # ---- 落盘 ----------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows, cols in (
        ("provenance_codes_verdict.csv", verdict_rows,
         ["codes_file", "codes_kind", "dataset", "N", "L0_distinct", "n_exact_match_ckpt",
          "best_ckpt", "best_agree_L0", "best_agree_L1", "best_agree_L2", "verdict"]),
        ("provenance_codes_matrix.csv", matrix_rows,
         ["codes_file", "codes_kind", "dataset", "N", "ckpt", "ckpt_tag", "ckpt_run",
          "agree_L0", "agree_L1", "agree_L2"]),
        ("provenance_ckpt_inventory.csv", inv_rows,
         ["root", "tag", "run", "ckpt_file", "mtime", "data_path", "data_exists", "dp_note", "family",
          "num_emb_list", "e_dim", "sk_epsilons", "N_items", "L0_distinct"]),
    ):
        p = os.path.join(args.out_dir, name)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        L("  写出 %s（%d 行）" % (p, len(rows)))

    fh.close()
    if args.self_check and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
