#!/usr/bin/env python3
"""
structure_topk_collect.py — P1.2：统一 Top-K 邻域采集器（含并列与 item universe 规则）

━━ 本文件的范围（严格限定，见 research/experiment_plan.md §P1.2）━━━━━━━━━━━━
    ① 基于 canonical ItemID 对齐（复用 P1.1 的 build_spaces / item_id_to_row）
    ② **余弦**相似度（禁用欧氏，协议 §3）
    ③ **强制排除自身**（协议 §3）
    ④ **确定性 tie-breaking**：两级键 = 余弦降序 + canonical item index 升序（协议 §4.1），
       实现用 `argsort(kind="stable")`；**禁止**共现频次等次级键
    ⑤ **K ∈ {5,10,20,50,100}，主 K = 20**（协议 §8.1）；实际只算 K_max=100 的全序，
       小 K 取前缀（全序的性质保证二者逐位相同）
    ⑥ 按协议 §5 处理 `U_beh` / `U_cont` 与零向量
    ⑦ 记录 `tie_share@K` 与 `collapse`（协议 §4.2）
    ⑧ 产出统一的 Top-K neighbor cache

    **不做**：Neighbor Recall@K / Jaccard@K / Overlap@K（P1.3）、任何 AUC、
             §4.3 的敏感性**判定**（需要 P1.3 的指标；本工具只预先产出翻转次级键的
             邻居序，供 P1.3 直接取用）、任何协议变更、任何训练。

━━ 候选集合与 universe 的取法（协议 §3 + §5）━━━━━━━━━━━━━━━━━━━━━━━━
    §3 规定"候选集合 = **全库 N 个 item**（不做 degree 限制、不做分块过滤）；
    **仅受 §5 的 item universe 约束**"。
    ⇒ 本工具实现为：**universe `U` 同时约束 query 端与候选端**。
    理由：一个空间对 (A→B) 的两侧必须共享同一候选集合，否则交集/并集无意义；
    且 §8.3 明确"抽样只作用于 query 端"、候选端是全库减去 §5 的排除项。

━━ 只读约束（协议 §12 A9）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    只读 `data/`、`rqvae/ckpt/`、P1.1 缓存；只写 `research/_p12_cache/` 与 `--report` 目标。

用法（仓库根，CPU）：
    python3 tools/structure_topk_collect.py --datasets Sport,Toys,Beauty \\
        --report research/p12_topk_report.md
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)

from structure_probe_common import build_spaces  # noqa: E402

CACHE_ROOT = os.path.join(ROOT, "research", "_p12_cache", "topk")
DEFAULT_REPORT = os.path.join(ROOT, "research", "p12_topk_report.md")

K_SET = [5, 10, 20, 50, 100]      # 协议 §8.1 冻结
K_MAIN = 20                       # 协议 §8.1 冻结
K_MAX = max(K_SET)
CHUNK = 512                       # 协议 §8.3 建议值
NEG_INF = np.float32(-np.inf)

# 协议 §7-C 预注册的空间对 → 需要的 (空间, universe) 组合
# (甲) 行为结构保持：Bh → {z_norm, q1, q12, q123}，universe = U_beh
# (乙) 连续→量化保持：z_norm → {q1, q12, q123}、X → z_norm，universe = U_cont
SPACES_UBEH = ["behavior128", "z_norm", "q1", "q12", "q123", "content768"]
SPACES_UCONT = ["z_norm", "q1", "q12", "q123", "input896"]


def verdict_of(st, k=K_MAIN):
    """按 `tie_share@K` 的**超出下界**部分给出并列严重度标签。

    ⚠ 协议 §4.2 的 `tie_share@K = mean_i #{j≠i : cos(i,j) == cos(i, n_K(i))} / K`
    中，`j = n_K(i)` 自身恒被计入 ⇒ **该量的下界是 1/K**（K=20 时为 0.05），
    "完全无并列"读数为 1/K 而非 0。故判定必须扣掉这 1 个，只看 **超出量**。
    """
    ts = st["tie_share"][str(k)]
    excess = st["tie_count_mean"][str(k)] - 1.0
    if excess < 0.02:
        return "**无并列**（读数 = 1/K 下界）"
    if ts > 0.5:
        return "**并列主导**（预期 tie-sensitive）"
    return f"有并列但不主导（均值超出 {excess:.2f} 个）"


# ------------------------------------------------------------------ 基础算子
def normalize_matrix(v):
    """按行 L2 归一化，clamp(1e-9) ⇒ 零向量保持为零（协议 §3，不产生 NaN）。"""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def build_universe_mask(spaces):
    """协议 §5.2：U_beh = {i : ‖Bh[i]‖₂ > 0}；U_cont = 全部 N。"""
    n = spaces["behavior128"].shape[0]
    u_beh = np.where(np.linalg.norm(spaces["behavior128"], axis=1) > 0.0)[0]
    u_cont = np.arange(n, dtype=np.int64)
    return {"U_beh": u_beh, "U_cont": u_cont}


def collapse_rate(v, U):
    """协议 §4.2：collapse = 1 − |unique(rows)| / |U|（在 universe 上行集上计算）。"""
    rows = v[U]
    u, cnt = np.unique(rows, axis=0, return_counts=True)
    return float(1 - len(u) / len(U)), int(len(u)), int(cnt.max())


def collect_topk(Xn, U, kmax=K_MAX, chunk=CHUNK, flip=False):
    """在 universe `U` 内采集 Top-`kmax` 邻居（自身已排除）。

    排序键（协议 §4.1，**两级，不允许第三级**）：
        第一键：余弦相似度 **降序**
        第二键：canonical item index **升序**（`flip=True` 时翻为**降序**，供 §4.3 敏感性检查）

    实现要点：`np.argsort(-S, kind="stable")` 的**稳定性**使输入列序成为隐式次级键 ——
    故只需把候选列按目标次序预排（升序 / 降序），无需 `np.lexsort`，且**逐位确定**。

    返回 (order, tie_counts)：order (M, kmax) int32 = 邻居的 **canonical 行号**；
    tie_counts (M, len(K_SET)) int32 = 与第 k 名邻居余弦**完全相等**的候选个数（不含自身）。
    """
    M = len(U)
    cand = U[::-1] if flip else U                      # 候选列序 = 隐式次级键
    XnC = np.ascontiguousarray(Xn[cand])               # (M, d)
    XnQ = Xn[U]                                        # (M, d)
    # 自身在候选列中的位置
    pos = (M - 1 - np.arange(M)) if flip else np.arange(M)

    order = np.empty((M, kmax), dtype=np.int32)
    tie_counts = np.zeros((M, len(K_SET)), dtype=np.int32)

    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        b = e - s
        S = XnQ[s:e] @ XnC.T                           # (b, M) 余弦（两侧均已 L2 归一化）
        S[np.arange(b), pos[s:e]] = NEG_INF            # 强制排除自身
        o = np.argsort(-S, axis=1, kind="stable")      # 稳定排序 ⇒ 并列由列序决定
        order[s:e] = cand[o[:, :kmax]]
        avail = min(kmax, M)                            # K 不得超出候选规模（协议 §8.1）
        for j, k in enumerate(K_SET):
            if k > avail:
                continue
            s_k = S[np.arange(b), o[:, k - 1]]         # 第 k 名邻居的相似度
            tie_counts[s:e, j] = (S == s_k[:, None]).sum(axis=1)
        del S, o
    return order, tie_counts


# -------------------------------------------------------------------- 单空间
def run_one_space(spaces, U_mask, space_key, universe, out_dir, chunk=CHUNK,
                  kmax=K_MAX, write=True):
    """对 (空间, universe) 采集两种次级键方向的邻居序，返回统计字典。"""
    U = U_mask[universe]
    Xn = normalize_matrix(spaces[space_key])
    coll, n_uniq, max_grp = collapse_rate(Xn, U)

    t = time.time()
    order, tie = collect_topk(Xn, U, kmax=kmax, chunk=chunk, flip=False)
    t_asc = time.time() - t
    t = time.time()
    order_flip, tie_flip = collect_topk(Xn, U, kmax=kmax, chunk=chunk, flip=True)
    t_flip = time.time() - t

    # A2：强制排除自身
    self_present = int(np.sum(order == U[:, None]))

    stats = {
        "space": space_key, "universe": universe, "n_query": int(len(U)),
        "collapse": coll, "n_unique_rows": n_uniq, "max_group_size": max_grp,
        "self_in_neighbors": self_present,
        "tie_share": {str(k): float(tie[:, j].mean() / k)
                      for j, k in enumerate(K_SET)},
        "tie_share_flip": {str(k): float(tie_flip[:, j].mean() / k)
                           for j, k in enumerate(K_SET)},
        "tie_count_mean": {str(k): float(tie[:, j].mean())
                           for j, k in enumerate(K_SET)},
        "tie_count_max": {str(k): int(tie[:, j].max()) for j, k in enumerate(K_SET)},
        "secs_asc": round(t_asc, 2), "secs_flip": round(t_flip, 2),
    }

    if write:
        os.makedirs(out_dir, exist_ok=True)
        stem = f"{space_key}__{universe}"
        np.save(os.path.join(out_dir, f"{stem}.idx.npy"), U.astype(np.int64))
        np.save(os.path.join(out_dir, f"{stem}.order.npy"), order)
        np.save(os.path.join(out_dir, f"{stem}.order_flip.npy"), order_flip)
        stats["files"] = [f"{stem}.idx.npy", f"{stem}.order.npy",
                          f"{stem}.order_flip.npy"]
    return stats, order, order_flip


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="P1.2 统一 Top-K 邻域采集器")
    ap.add_argument("--datasets", default="Sport")
    ap.add_argument("--variant", default="TA-SID")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--cache_root", default=CACHE_ROOT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--report_only", action="store_true",
                    help="不重算，从 cache_root 下各数据集的 stats.json 重绘报告（秒级）")
    ap.add_argument("--verify", action="store_true",
                    help="A1 决定性自检：重跑首个 (空间, universe) 并逐位比对")
    args = ap.parse_args()

    log_lines = []
    t0 = time.time()

    def log(s=""):
        print(s)
        log_lines.append(s)

    log("# P1.2 统一 Top-K 邻域采集 — 实测报告")
    log()
    log(f"- 生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"- 脚本：`tools/structure_topk_collect.py`")
    log(f"- 口径来源：`research/behavior_structure_protocol.md`（v1.0，未改动）")
    log(f"- 冻结参数：K ∈ {K_SET}（主 K={K_MAIN}）｜余弦｜self 排除｜"
        f"次级键 = canonical index（升序 / 翻转版降序）｜chunk={args.chunk}")
    if args.report_only:
        log("- 模式：`--report_only` — **从既有 `stats.json` 重绘**，未重算；"
            "Top-K 产物与上次采集逐位相同。")
    log()

    all_stats = {}
    ds_meta = {}
    det_check = None
    verify_done = False

    # ---- --report_only：从缓存 stats.json 重绘报告，不重算（800 s → 秒级）----
    if args.report_only:
        for ds in args.datasets.split(","):
            d = os.path.join(args.cache_root, f"{ds}_{args.variant}_seed{args.seed}")
            with open(os.path.join(d, "stats.json"), encoding="utf-8") as f:
                js = json.load(f)
            all_stats[ds] = js["spaces"]
            ds_meta[ds] = {
                "N": js["spaces"]["z_norm@U_cont"]["n_query"],
                "u_beh": js["spaces"]["z_norm@U_beh"]["n_query"],
                "u_cont": js["spaces"]["z_norm@U_cont"]["n_query"],
                "order": [f"{s}@{u}" for u in ("U_beh", "U_cont")
                          for s in (SPACES_UBEH if u == "U_beh" else SPACES_UCONT)],
            }
        _render(all_stats, ds_meta, args, det_check, log_lines, log, t0,
                write_report=True)
        return

    for ds in args.datasets.split(","):
        log(f"## {ds} / {args.variant} / seed{args.seed}")
        log()
        spaces, meta, fwd, cb, codes = build_spaces(ds, args.variant, args.seed)
        U_mask = build_universe_mask(spaces)
        n = spaces["behavior128"].shape[0]
        log(f"- N = {n}；`U_beh` = {len(U_mask['U_beh'])}（coverage "
            f"{len(U_mask['U_beh'])/n:.4f}），`U_cont` = {len(U_mask['U_cont'])}")
        log()

        out_dir = os.path.join(args.cache_root, f"{ds}_{args.variant}_seed{args.seed}")
        log("| 空间 | universe | \|U\| | unique 行 | collapse | 最大同值组 | "
            "self∈邻居 | " + " | ".join(f"tie_share@{k}" for k in K_SET) + " |")
        log("|:--|:--|--:|--:|--:|--:|--:|" + "---:|" * len(K_SET))
        ds_stats = {}
        for universe, key_list in (("U_beh", SPACES_UBEH),
                                   ("U_cont", SPACES_UCONT)):
            for sk in key_list:
                st, order, order_flip = run_one_space(
                    spaces, U_mask, sk, universe, out_dir, chunk=args.chunk)
                ds_stats[(sk, universe)] = st
                log(f"| `{sk}` | {universe} | {st['n_query']} | {st['n_unique_rows']} | "
                    f"{st['collapse']:.4f} | {st['max_group_size']} | "
                    f"{'❌' if st['self_in_neighbors'] else '✅ 0'} | "
                    + " | ".join(f"{st['tie_share'][str(k)]:.4f}" for k in K_SET) + " |")

                if args.verify and not verify_done:
                    _, o2, _ = run_one_space(spaces, U_mask, sk, universe, out_dir,
                                             chunk=args.chunk, write=False)
                    same = bool(np.array_equal(order, o2))
                    verify_done = True
                    det_check = (ds, sk, universe, same)
        log()

        all_stats[ds] = {f"{k[0]}@{k[1]}": v for k, v in ds_stats.items()}
        ds_meta[ds] = {
            "N": n,
            "u_beh": len(U_mask["U_beh"]),
            "u_cont": len(U_mask["U_cont"]),
            "order": [f"{s}@{u}" for u in ("U_beh", "U_cont")
                      for s in (SPACES_UBEH if u == "U_beh" else SPACES_UCONT)],
        }

    # 落盘 stats
    for ds, tab in all_stats.items():
        d = os.path.join(args.cache_root, f"{ds}_{args.variant}_seed{args.seed}")
        with open(os.path.join(d, "stats.json"), "w", encoding="utf-8") as f:
            json.dump({"dataset": ds, "variant": args.variant, "seed": args.seed,
                       "k_set": K_SET, "k_main": K_MAIN, "chunk": args.chunk,
                       "spaces": tab}, f, ensure_ascii=False, indent=2)

    _render(all_stats, ds_meta, args, det_check, log_lines, log, t0,
            write_report=bool(args.report))


def _render(all_stats, ds_meta, args, det_check, log_lines, log, t0,
            write_report):
    """汇总 + 缓存清单 + 落盘报告（采集与仅重绘两条路径共用）。"""
    log("## A1 决定性自检")
    log()
    if det_check:
        ds_, sk_, un_, same = det_check
        log(f"- `{ds_}` / `{sk_}`@{un_}：重跑同格并逐位比对 → "
            f"{'✅ 逐位一致' if same else '❌ 不一致'}。")
        log("- 覆盖面：**仅此一格**被复算（协议未要求全量复算）。")
    else:
        log("- 本次为 `--report_only` 重绘，未复算；该检查的执行记录见上次采集日志。")
    log()

    log("## 汇总")
    log()
    log("| 数据集 | 空间 | universe | collapse | " +
        " | ".join(f"tie_share@{k}" for k in K_SET) + f" | 主 K={K_MAIN} 判定 |")
    log("|:--|:--|:--|--:|" + "---:|" * len(K_SET) + ":--|")
    for ds, tab in all_stats.items():
        order = ds_meta.get(ds, {}).get("order") or list(tab.keys())
        for key in order:
            if key not in tab:
                continue
            st = tab[key]
            sk, un = key.split("@")
            log(f"| {ds} | `{sk}` | {un} | {st['collapse']:.4f} | " +
                " | ".join(f"{st['tie_share'][str(k)]:.4f}" for k in K_SET) +
                f" | {verdict_of(st)} |")
    log()
    log(f"> ⚠ **`tie_share@K` 的下界是 `1/K`，不是 0。** 协议 §4.2 的定义 "
        "`mean_i #{{j≠i : cos(i,j) == cos(i, n_K(i))}} / K` 恒把第 K 名邻居**自身**计入，"
        f"⇒ 完全无并列的格读数恰为 `1/K`（K={K_MAIN} 时 = {1/K_MAIN:.4f}）。"
        "上表的判定已扣除该项；`stats.json` 的 `tie_count_mean` 给出原始均值，"
        "超出量 = `tie_count_mean − 1`。")
    log()
    log("> ⚠ `tie_share@K` 是**并列规模**的描述量，**不是**邻域保持指标；"
        "§4.3 的 `tie-sensitive` 判定需要 P1.3 的指标，本表**不构成判定**。")
    log()

    log("## 缓存")
    log()
    for ds in all_stats:
        d = os.path.join(args.cache_root, f"{ds}_{args.variant}_seed{args.seed}")
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
            log(f"- `{os.path.relpath(d, ROOT)}`：{len(files)} 文件，{size/1e6:.1f} MB")
    log()
    log("文件命名：`{space}__{universe}.idx.npy`（query 的 canonical 行号）、"
        "`.order.npy`（次级键 = 索引**升序**）、`.order_flip.npy`（索引**降序**，供 §4.3）。"
        f"三者行数均为 |U|，列数为 K_max={K_MAX}；小 K 取前缀即可（全序性质保证逐位相同）。"
        "`stats.json` 汇总每个 (空间, universe) 的 `collapse` / `tie_share` / `tie_count`。")
    log()
    log(f"（总耗时 {time.time() - t0:.1f} s，全程 CPU、只读）")
    log()
    log("> 附注：升序 / 降序两套邻居序的**集合差异**（描述性、非协议指标、不构成判定）"
        "见 `research/p12_flip_divergence.md`。")

    if write_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"\n报告已写入: {args.report}")


if __name__ == "__main__":
    main()
