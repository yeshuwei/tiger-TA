#!/usr/bin/env python3
"""
ablation_3datasets_summary.py — 行为消融**三数据集**主表汇总（N@20 + R@20，逐 seed + mean±std）

与 `tools/e49_summary.py` 的关系：
  e49_summary.py 只覆盖 Sport/Toys（E49，9-23 收工），且 Recall 只印 mean；
  Beauty 六变体是 9-24 单独补的（Tier-1/Tier-2），从未进入任何汇总脚本。
  本脚本把三者合并为**一张逐 seed 表**，并把 R@20 的 std 一并算出（此前无任何文档有 R@20 的 ±）。

⚠️ 沿用 e49_summary.py 的「**逐 lane 显式文件名，禁 glob 拼接**」纪律，绕开两个已知陷阱：
  ① Toys TA-SID 必须取**健康 codes** 的 `tiger_ta_sid_768d_toys_rq2024_seed*_eval.log`（L0=207, 9-09 重跑）；
     按 tag 拼出的 `Toys_ta_sid_768d_seed*_eval.log` 是**孤儿 codes**（L0=62）日志，NDCG@20=0.0367 **已作废**。
  ② Toys Shuffled 实际 tag 是 `shuffled_768d`，不是 `shuffled_ablation`（后者只对 Sport 成立）。
  另：Sport Shuffled/ZeroPad 的 seed2025 是 **legacy 无 seed 后缀**文件。

⚠️ 日志格式两种并存（列数 3 或 5），**@20 都是第 3 个值**，故只匹配前 3 列。

用法：
    python tools/ablation_3datasets_summary.py [--out doc/ablation_main_table_3datasets.md]
"""

import argparse
import datetime
import os
import re
import statistics as st

SEEDS = (2025, 2026, 2027)
LOG_DIR = "model/logs"

# 变体展示顺序（按消融叙事：从"无行为"到"完整方法"）
VARIANTS = ["Content-only", "ZeroPad", "Shuffled", "ItemCF-SVD", "Raw-CoOcc-SVD", "TA-SID"]
DATASETS = ["Sport", "Toys", "Beauty"]


def _std(tag):
    return {s: f"{tag}_seed{s}_eval.log" for s in SEEDS}


def _legacy_2025(tag):
    """seed2025 用 legacy 无后缀文件名，2026/2027 用带 seed 的新名。"""
    d = {2025: f"{tag}_eval.log"}
    d.update({s: f"{tag}_seed{s}_eval.log" for s in (2026, 2027)})
    return d


# ---------------------------------------------------------------- lane 定义
LANES = {
    "Sport": {
        "Content-only":  _std("Sport_content768d_baseline"),
        "ZeroPad":       _legacy_2025("Sport_zeropad_ablation"),
        "Shuffled":      _legacy_2025("Sport_shuffled_ablation"),
        "ItemCF-SVD":    _std("Sport_itemcf_768d"),
        "Raw-CoOcc-SVD": _std("Sport_rawcooc_768d"),
        "TA-SID":        _std("Sport_ta_sid_768d"),
    },
    "Toys": {
        "Content-only":  _std("Toys_content768d_baseline"),
        "ZeroPad":       _std("Toys_zeropad_ablation"),
        "Shuffled":      _std("Toys_shuffled_768d"),          # 陷阱②
        "ItemCF-SVD":    _std("Toys_itemcf_768d"),
        "Raw-CoOcc-SVD": _std("Toys_rawcooc_768d"),
        "TA-SID":        {s: f"tiger_ta_sid_768d_toys_rq2024_seed{s}_eval.log"
                          for s in SEEDS},                     # 陷阱①：健康 codes
    },
    "Beauty": {
        "Content-only":  _std("Beauty_content768d_baseline"),
        "ZeroPad":       _std("Beauty_zeropad_768d"),
        "Shuffled":      _std("Beauty_shuffled_768d"),
        "ItemCF-SVD":    _std("Beauty_itemcf_768d"),
        "Raw-CoOcc-SVD": _std("Beauty_rawcooc_768d"),
        "TA-SID":        _std("Beauty_ta_sid_768d"),
    },
}

NDCG_RE = re.compile(r"^\s*NDCG\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M)
RCL_RE = re.compile(r"^\s*Recall\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.M)


def parse(fname):
    """读 eval 日志 → (NDCG@20, Recall@20)；取最后一次结果表。缺文件/无表返回 None。"""
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        return None
    txt = open(path, errors="ignore").read()
    n = NDCG_RE.findall(txt)
    if not n:
        return None
    r = RCL_RE.findall(txt)
    return float(n[-1][2]), (float(r[-1][2]) if r else None)

# ---------------------------------------------------------------- 对比定义
COMPARISONS = [
    ("TA-SID", "Content-only",  "TA-SID 相对无行为基线的总增益"),
    ("TA-SID", "Raw-CoOcc-SVD", "严格单变量：PPMI 是否必要"),
    ("TA-SID", "ItemCF-SVD",    "相对通用 item-level CF 的增量"),
    ("Content-only", "Shuffled", "错配行为是否比不给行为更差"),
    ("Content-only", "ZeroPad",  "零行为通道是否带来增益（预期打平）"),
    ("Raw-CoOcc-SVD", "ItemCF-SVD", "相邻窗 vs 整篮共现（预期不稳定）"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="doc/ablation_main_table_3datasets.md")
    args = ap.parse_args()

    # ---- 采集
    res, missing = {}, []
    for ds in DATASETS:
        for v in VARIANTS:
            files = LANES[ds][v]
            got = {}
            for s in SEEDS:
                r = parse(files[s]) if s in files else None
                if r is None:
                    missing.append(f"{ds}/{v}/seed{s} ← {files.get(s)}")
                else:
                    got[s] = r
            res[(ds, v)] = dict(files=files, got=got)

    n_lane = sum(len(r["got"]) for r in res.values())
    n_exp = len(DATASETS) * len(VARIANTS) * len(SEEDS)

    KIDX = {"ndcg": 0, "rcl": 1}

    def mn(ds, v, k):
        """k ∈ {"ndcg","rcl"} → (mean, sample std)，缺失返回 (None, None)。"""
        i = KIDX[k]
        vals = [res[(ds, v)]["got"][s][i] for s in SEEDS if s in res[(ds, v)]["got"]]
        vals = [x for x in vals if x is not None]
        return (st.mean(vals) if vals else None), (st.stdev(vals) if len(vals) > 1 else None)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    o = []
    A = o.append

    A("# 行为消融主表 — 三数据集（N@20 + R@20，逐 seed + mean±std）")
    A("")
    A(f"**生成时间**：{ts}　**产出脚本**：`tools/ablation_3datasets_summary.py`（自动生成，勿手改）")
    A(f"**进度**：{n_lane}/{n_exp} 路（数据集 × 变体 × seed）")
    A("")
    A("> **本表性质：纯归档。** 全部实验已于 2026-09-23（Sport/Toys，E49）与 2026-09-24（Beauty，Tier-2）")
    A("> 运行完毕；本脚本**只读解析既有 eval 日志**，不训练、不评测、不改任何超参。")
    A("")
    A("> **本表存在的理由**：此前 N@20 的权威源是 `doc/e49_results_corrected.md`（仅 Sport/Toys）+")
    A("> `logs/beauty_tier2_summary.log`（仅 Beauty，且无 Recall）；")
    A("> **本消融 54 条 lane 的 R@20 此前只有均值（且仅 Sport/Toys 有），既无逐 seed 也无 std**。")
    A("> 本表把三数据集 54 条 lane 的 N@20/R@20 逐 seed 汇总于一处，作为论文 §5.2.2 的**唯一数字源**。")
    A("")
    A("> ⚠️ **命名陷阱（沿用 `tools/e49_summary.py` 纪律）**：本脚本逐 lane 显式列文件名，")
    A("> 不使用 tag glob 拼接。两个已知陷阱见文末 §5。")
    A("")

    # ---- §1 主表
    A("---")
    A("")
    A("## 1. 主表（3 数据集 × 6 变体）")
    A("")
    for metric, name in (("ndcg", "NDCG@20"), ("rcl", "Recall@20")):
        A(f"### 1.{1 if metric == 'ndcg' else 2} {name}")
        A("")
        head = "| 变体 |" + "".join(f" {d} |" for d in DATASETS)
        A(head)
        A("|:--|" + ":--:|" * len(DATASETS))
        best = {d: max((mn(d, v, metric)[0] or -1) for v in VARIANTS) for d in DATASETS}
        for v in VARIANTS:
            cells = []
            for d in DATASETS:
                m, s = mn(d, v, metric)
                if m is None:
                    cells.append("∅")
                    continue
                txt = f"{m:.4f}±{s:.4f}" if s is not None else f"{m:.4f}"
                cells.append(f"**{txt}**" if abs(m - best[d]) < 1e-9 else txt)
            A(f"| {v} |" + "".join(f" {c} |" for c in cells))
        A("")
    A("> 加粗 = 该数据集该指标最优。3 seeds（2025/2026/2027），std 为样本标准差（n−1）。")
    A("")

    # ---- §2 逐 seed
    A("---")
    A("")
    A("## 2. 逐 seed 明细（54 路）")
    A("")
    for d in DATASETS:
        A(f"### {d}")
        A("")
        A("| 变体 | N@20 s2025 | N@20 s2026 | N@20 s2027 | **N@20 mean±std** | R@20 s2025 | R@20 s2026 | R@20 s2027 | **R@20 mean±std** |")
        A("|:--|--:|--:|--:|:--|--:|--:|--:|:--|")
        for v in VARIANTS:
            got = res[(d, v)]["got"]
            def c(k, s):
                g = got.get(s)
                if g is None or g[k] is None:
                    return "—"
                return f"{g[k]:.4f}"
            nm, ns = mn(d, v, "ndcg")
            rm, rs = mn(d, v, "rcl")
            nms = f"{nm:.4f}±{ns:.4f}" if nm is not None and ns is not None else "∅"
            rms = f"{rm:.4f}±{rs:.4f}" if rm is not None and rs is not None else "∅"
            A(f"| {v} | {c(0,2025)} | {c(0,2026)} | {c(0,2027)} | **{nms}** "
              f"| {c(1,2025)} | {c(1,2026)} | {c(1,2027)} | **{rms}** |")
        A("")

    # ---- §3 方向一致性
    A("---")
    A("")
    A("## 3. 关键对比的方向一致性")
    A("")
    A("Δ = 前者 − 后者（正 = 前者更优）。")
    A("")
    for metric, name in (("ndcg", "NDCG@20"), ("rcl", "Recall@20")):
        A(f"### 3.{1 if metric == 'ndcg' else 2} {name}")
        A("")
        A("| 对比 | 含义 | Sport Δ | Toys Δ | Beauty Δ | 同向 |")
        A("|:--|:--|--:|--:|--:|:--:|")
        for a, b, desc in COMPARISONS:
            deltas, pos = [], 0
            for d in DATASETS:
                ma = mn(d, a, metric)[0]
                mb = mn(d, b, metric)[0]
                if ma is None or mb is None:
                    deltas.append("—")
                    continue
                dv = ma - mb
                deltas.append(f"{dv:+.4f}")
                if dv > 0:
                    pos += 1
            A(f"| {a} − {b} | {desc} | {deltas[0]} | {deltas[1]} | {deltas[2]} | **{pos}/3** |")
        A("")

    # ---- §4 排序一致性
    A("---")
    A("")
    A("## 4. N@20 与 R@20 的排序一致性")
    A("")
    A("| 数据集 | NDCG@20 排序 | Recall@20 排序 | 同一排序？ |")
    A("|:--|:--|:--|:--:|")
    for d in DATASETS:
        on = sorted(VARIANTS, key=lambda v: -(mn(d, v, "ndcg")[0] or 0))
        orc = sorted(VARIANTS, key=lambda v: -(mn(d, v, "rcl")[0] or 0))
        A(f"| {d} | {' > '.join(on)} | {' > '.join(orc)} | {'✅' if on == orc else '⚠️ 有差异'} |")
    A("")
    mism = []
    for d in DATASETS:
        on = sorted(VARIANTS, key=lambda v: -(mn(d, v, "ndcg")[0] or 0))
        orc = sorted(VARIANTS, key=lambda v: -(mn(d, v, "rcl")[0] or 0))
        if on != orc:
            for i in range(len(on)):
                if on[i] != orc[i]:
                    mism.append(f"{d}: 第 {i+1} 位 N@20 为 {on[i]}，R@20 为 {orc[i]}")
                    break
    if mism:
        A("差异位置：" + "；".join(mism) + "。")
    else:
        A("三数据集两指标排序完全一致。")
    A("")

    # ---- §5 口径与核验
    A("---")
    A("")
    A("## 5. 口径、命名陷阱与核验")
    A("")
    A("### 5.1 命名陷阱（两处，均已在本脚本中固化正确文件名）")
    A("")
    A("| 陷阱 | 错误做法 | 后果 | 本脚本采用 |")
    A("|:--|:--|:--|:--|")
    A("| ① Toys TA-SID | tag 拼出 `Toys_ta_sid_768d_seed*_eval.log` | 那是**孤儿 codes**(L0=62) 日志，"
      "NDCG@20=0.0367 **已作废** | `tiger_ta_sid_768d_toys_rq2024_seed*_eval.log`（健康 codes, L0=207, 2026-09-09） |")
    A("| ② Toys Shuffled | 拼 `Toys_shuffled_ablation_seed*_eval.log` | 该 tag 只对 Sport 成立，Toys 会**取空** | "
      "`Toys_shuffled_768d_seed*_eval.log` |")
    A("")
    A("> 另：Sport 的 Shuffled / ZeroPad 的 **seed2025 为 legacy 无 seed 后缀**文件"
      "（`Sport_shuffled_ablation_eval.log` / `Sport_zeropad_ablation_eval.log`）。")
    A("")
    A("### 5.2 评测配置差异（已核实不影响 @20）")
    A("")
    A("| | topk_list | @20 取值位置 | 运行时间 |")
    A("|:--|:--|:--|:--|")
    A("| Beauty TA-SID / Content-only（既有锚点） | `5,10,20`（3 列） | 第 3 个值 | 2026-07-31 ~ 08-03 |")
    A("| 其余全部 lane | `5,10,20,50,100`（5 列） | 第 3 个值 | 2026-09-09 ~ 09-24 |")
    A("")
    A("> `topk_list` 只控制 `evaluate()` 打印哪几列，不影响 @20 取值（源码核实：`tools/eval_trained_model.py` 第 91–101 行）。")
    A("> 两种格式 @20 **均为第 3 列**，故本脚本只匹配前 3 列。")
    A("> 评测固定 `--beam_size 30`，**新旧 lane 一致**。`topk_list` 是既有锚点与新增运行之间**唯一**的配置差异")
    A("> （见 `doc/beauty_tier2_conclusion.md` §7-3 与 §8-5）。")
    A("")
    A("### 5.3 核验命令")
    A("")
    A("```bash")
    A("# 一键复现本表（只读）")
    A("python3 tools/ablation_3datasets_summary.py")
    A("")
    A("# 手工抽查任一路（示例：Toys TA-SID 健康 codes, seed2026）")
    A("grep -E '^\\s*(NDCG|Recall)\\s+' model/logs/tiger_ta_sid_768d_toys_rq2024_seed2026_eval.log | tail -2")
    A("```")
    A("")
    A("### 5.4 唯一单 seed 的行（不属于本表）")
    A("")
    A("| 数据集 | 变体 | 日志 | 说明 |")
    A("|:--|:--|:--|:--|")
    A("| Sport | Random | `model/logs/Sport_random_ablation_eval.log` | 谱匹配噪声，**单 seed 且已弃用**，"
      "不计入 54 路，不得引用 |")
    A("")

    # ---- §6 关系与修正
    A("---")
    A("")
    A("## 6. 与其他文档的关系")
    A("")
    A("| 文档 | 覆盖 | 本表如何取代/补充 |")
    A("|:--|:--|:--|")
    A("| `doc/e49_results_corrected.md` | Sport/Toys 逐 seed **N@20** + R@20 **仅均值** | 本表 §2 补全 R@20 逐 seed 与 std；N@20 数值完全一致 |")
    A("| `logs/beauty_tier2_summary.log` | Beauty 逐 seed **N@20** | 本表 §2 补全 Beauty **R@20** 逐 seed |")
    A("| `doc/behavior_structure_probe_conclusion.md` | Behavior AUC 与 NDCG 相关性 | 未涉及，本表不重复 |")
    A("")
    A("## 7. 修正记录")
    A("")
    A("1. **2026-09-28 建表**。收录三数据集 54 路逐 seed N@20/R@20。")
    A("2. **ZeroPad 与 Content-only 的表述修正**：不得写「差距符号不一致」。")
    A("   实测 N@20 为 −0.0002 ~ +0.0016、R@20 为 +0.0010 ~ +0.0030，")
    A("   **ZeroPad 偏高的格数多于偏低**，符号并不系统地翻转。")
    A("   正确论证是**量级论证**：该对差距 ≤0.0030，而 TA-SID 相对 Content-only 的增益达 0.0041–0.0130，")
    A("   **高出一个数量级**；且 LN(0)=0 使该 128 维梯度恒为 0，不承载任何可学习信号，")
    A("   小幅差异可归因于输入维度不同（768 vs 896）带来的 RQ-VAE 形状/初始化轨迹差异。")
    A("3. **核验时的一次踩坑记录**：手工 `grep Toys_ta_sid_768d_seed*_eval.log`")
    A("   曾复现作废值 N@20=0.0367 / R@20=0.0873。本表使用健康 codes，")
    A("   正确值 N@20=0.0377 / R@20=0.0911。此事再次说明 tag glob 不可用于取日志。")
    A("")

    if missing:
        A("## ⚠️ 缺失 lane")
        A("")
        for m in missing:
            A(f"- {m}")
        A("")

    txt = "\n".join(o) + "\n"
    with open(args.out, "w") as f:
        f.write(txt)

    print(f"[ok] {n_lane}/{n_exp} 路写入 {args.out}")
    if missing:
        print(f"[warn] {len(missing)} 路缺失:")
        for m in missing:
            print("   ", m)


if __name__ == "__main__":
    main()
