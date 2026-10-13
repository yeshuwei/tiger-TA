#!/usr/bin/env python3
"""
verify_paper_tables_46.py — 核对论文表 4-2 / 4-3 的两套 std（ddof=0 vs ddof=1），并复算 §2.2 百分比。

只读。不训练、不评测、不改任何文件。

用法: python3 tools/verify_paper_tables_46.py
"""
import os
import re
import sys
import statistics as st

LOG_DIR = "model/logs"
SEEDS = (2025, 2026, 2027)
KS = (5, 10, 20)


# ---------------------------------------------------------------- eval log 解析
def parse_eval(fname):
    """返回 {'NDCG': {'5':..,'10':..,'20':..}, 'Recall': {...}}；按**表头**定位，兼容 3/5 列。"""
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        return None
    lines = open(path, errors="ignore").read().splitlines()
    out = {}
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Metric") and "@" in ln:
            hdr = [t.lstrip("@") for t in ln.split() if t.startswith("@")]
            for l2 in lines[i + 1:]:
                s = l2.strip()
                if s.startswith("NDCG") or s.startswith("Recall"):
                    name = s.split()[0]
                    cells = s.split()[1:]
                    out[name] = {k: v for k, v in zip(hdr, cells)}
                elif out and not s:
                    break
            if out:
                break
    return out or None


def parse_3seed_txt(path):
    """读 sasrec/gru4rec/bert4rec 的 summary_3seed.txt → {metric: {seed: val}}"""
    if not os.path.exists(path):
        return None
    res = {}
    for ln in open(path):
        p = ln.split()
        if len(p) >= 7 and p[0] in ("2025", "2026", "2027"):
            s = int(p[0])
            for name, v in zip(("NDCG@5", "NDCG@10", "NDCG@20",
                                "Recall@5", "Recall@10", "Recall@20"), p[1:7]):
                res.setdefault(name, {})[s] = float(v)
    return res or None


# ---------------------------------------------------------------- lane 定义
def _std(tag):
    return {s: f"{tag}_seed{s}_eval.log" for s in SEEDS}


LANES = {
    "Sport": {
        "Content-only": _std("Sport_content768d_baseline"),
        "TA-SID":       _std("Sport_ta_sid_768d"),
    },
    "Toys": {
        "Content-only": _std("Toys_content768d_baseline"),
        "TA-SID":       {s: f"tiger_ta_sid_768d_toys_rq2024_seed{s}_eval.log" for s in SEEDS},
    },
    "Beauty": {
        "Content-only": _std("Beauty_content768d_baseline"),
        "TA-SID":       _std("Beauty_ta_sid_768d"),
    },
}

SEQ = {  # 经典基线：读 3seed.txt
    "SASRec":   ("sasrec_output/{D}/summary_3seed.txt"),
    "GRU4Rec":  ("gru4rec_output/{D}/summary_3seed.txt"),
    "BERT4Rec": ("bert4rec_output/{D}/summary_3seed.txt"),
}


def sd(vals, ddof):
    return st.stdev(vals) if ddof == 1 else st.pstdev(vals)


def fmt(v, s):
    return f"{v:.4f}±{s:.4f}"


def main():
    # ---------------- 采集
    data = {}   # data[(model, ds)][metric] = {seed: val}
    missing = []

    for ds, variants in LANES.items():
        for v, files in variants.items():
            got = {}
            for s in SEEDS:
                r = parse_eval(files[s])
                if r is None:
                    missing.append(f"{ds}/{v}/seed{s} ← {files[s]}")
                    continue
                for m in ("NDCG", "Recall"):
                    for k in KS:
                        got.setdefault(f"{m}@{k}", {})[s] = float(r[m][str(k)])
            data[("TIGER" if v == "Content-only" else v, ds)] = got
            data[(v, ds)] = got

    for model, tmpl in SEQ.items():
        for ds in ("Sport", "Toys", "Beauty"):
            r = parse_3seed_txt(tmpl.format(D=ds))
            if r is None:
                missing.append(f"{ds}/{model} ← {tmpl.format(D=ds)}")
                continue
            data[(model, ds)] = r

    # ---------------- 论文表 4-2 / 4-3 抄写值（用户提供，ddof=0 口径）
    T42 = {  # NDCG@k
        "SASRec":   {"Beauty": [0.0121, 0.0183, 0.0254], "Sport": [0.0064, 0.0096, 0.0134], "Toys": [0.0127, 0.0184, 0.0241]},
        "GRU4Rec":  {"Beauty": [0.0244, 0.0284, 0.0323], "Sport": [0.0086, 0.0106, 0.0129], "Toys": [0.0138, 0.0164, 0.0192]},
        "BERT4Rec": {"Beauty": [0.0144, 0.0185, 0.0233], "Sport": [0.0069, 0.0091, 0.0118], "Toys": [0.0080, 0.0105, 0.0132]},
        "TIGER":    {"Beauty": [0.0227, 0.0291, 0.0358], "Sport": [0.0161, 0.0213, 0.0275], "Toys": [0.0206, 0.0268, 0.0336]},
        "TA-SID":   {"Beauty": [0.0264, 0.0335, 0.0415], "Sport": [0.0190, 0.0252, 0.0322], "Toys": [0.0225, 0.0294, 0.0377]},
    }
    T42S = {  # 表 4-2 的 std（用户提供）
        "SASRec":   {"Beauty": [0.0005, 0.0007, 0.0007], "Sport": [0.0003, 0.0001, 0.0002], "Toys": [0.0004, 0.0001, 0.0003]},
        "GRU4Rec":  {"Beauty": [0.0014, 0.0011, 0.0013], "Sport": [0.0005, 0.0004, 0.0004], "Toys": [0.0003, 0.0003, 0.0002]},
        "BERT4Rec": {"Beauty": [0.0011, 0.0009, 0.0006], "Sport": [0.0004, 0.0007, 0.0009], "Toys": [0.0008, 0.0010, 0.0010]},
        "TIGER":    {"Beauty": [0.0008, 0.0007, 0.0007], "Sport": [0.0002, 0.0004, 0.0006], "Toys": [0.0005, 0.0005, 0.0006]},
        "TA-SID":   {"Beauty": [0.0006, 0.0008, 0.0009], "Sport": [0.0001, 0.0002, 0.0004], "Toys": [0.0008, 0.0005, 0.0007]},
    }
    T43 = {  # Recall@k
        "SASRec":   {"Beauty": [0.0234, 0.0429, 0.0711], "Sport": [0.0121, 0.0220, 0.0374], "Toys": [0.0245, 0.0423, 0.0651]},
        "GRU4Rec":  {"Beauty": [0.0333, 0.0458, 0.0612], "Sport": [0.0127, 0.0189, 0.0282], "Toys": [0.0203, 0.0286, 0.0394]},
        "BERT4Rec": {"Beauty": [0.0221, 0.0348, 0.0539], "Sport": [0.0109, 0.0178, 0.0284], "Toys": [0.0127, 0.0206, 0.0313]},
        "TIGER":    {"Beauty": [0.0347, 0.0547, 0.0811], "Sport": [0.0255, 0.0421, 0.0666], "Toys": [0.0324, 0.0516, 0.0785]},
        "TA-SID":   {"Beauty": [0.0402, 0.0620, 0.0941], "Sport": [0.0299, 0.0491, 0.0769], "Toys": [0.0365, 0.0579, 0.0911]},
    }
    T43S = {
        "SASRec":   {"Beauty": [0.0008, 0.0014, 0.0016], "Sport": [0.0006, 0.0004, 0.0014], "Toys": [0.0011, 0.0006, 0.0015]},
        "GRU4Rec":  {"Beauty": [0.0016, 0.0007, 0.0017], "Sport": [0.0007, 0.0004, 0.0004], "Toys": [0.0005, 0.0011, 0.0011]},
        "BERT4Rec": {"Beauty": [0.0013, 0.0006, 0.0007], "Sport": [0.0006, 0.0017, 0.0024], "Toys": [0.0010, 0.0017, 0.0017]},
        "TIGER":    {"Beauty": [0.0009, 0.0007, 0.0008], "Sport": [0.0004, 0.0008, 0.0015], "Toys": [0.0005, 0.0006, 0.0007]},
        "TA-SID":   {"Beauty": [0.0004, 0.0009, 0.0016], "Sport": [0.0001, 0.0005, 0.0011], "Toys": [0.0014, 0.0003, 0.0020]},
    }

    order = ["SASRec", "GRU4Rec", "BERT4Rec", "TIGER", "TA-SID"]
    dss = ["Beauty", "Sport", "Toys"]

    print("=" * 100)
    print("表 4-2 核对（NDCG@k）—— 逐格比对：论文抄写值 vs 从原始逐 seed 重算的 ddof=0 / ddof=1")
    print("=" * 100)
    n_ddof0 = n_ddof1 = n_bad = 0
    for m in order:
        for ds in dss:
            if (m, ds) not in data or not data[(m, ds)]:
                print(f"  ⚠ 缺数据 {m}/{ds}"); continue
            for i, k in enumerate(KS):
                key = f"NDCG@{k}"
                vals = [data[(m, ds)][key][s] for s in SEEDS]
                mu = sum(vals) / 3
                s0, s1 = sd(vals, 0), sd(vals, 1)
                pv, ps = T42[m][ds][i], T42S[m][ds][i]
                ok_mu = abs(mu - pv) < 5e-5
                m0 = abs(s0 - ps) < 5e-5
                m1 = abs(s1 - ps) < 5e-5
                flag = ""
                if not ok_mu:
                    flag += " ⚠均值不符"
                    n_bad += 1
                if m0: n_ddof0 += 1
                if m1: n_ddof1 += 1
                if not (m0 or m1):
                    flag += " ⚠std两者都不符"
                print(f"  {m:9s} {ds:7s} N@{k:<2d} 论文 {pv:.4f}±{ps:.4f} | "
                      f"重算 mean={mu:.4f} ddof0=±{s0:.4f}{'✓' if m0 else ' '} "
                      f"ddof1=±{s1:.4f}{'✓' if m1 else ' '}{flag}")
    print(f"\n  ⇒ 表4-2 std 命中：ddof=0 口径 {n_ddof0}/45 格，ddof=1 口径 {n_ddof1}/45 格；均值不符 {n_bad}/45")

    print()
    print("=" * 100)
    print("表 4-3 核对（Recall@k）")
    print("=" * 100)
    n0 = n1 = nbad = 0
    for m in order:
        for ds in dss:
            if (m, ds) not in data or not data[(m, ds)]:
                continue
            for i, k in enumerate(KS):
                key = f"Recall@{k}"
                vals = [data[(m, ds)][key][s] for s in SEEDS]
                mu = sum(vals) / 3
                s0, s1 = sd(vals, 0), sd(vals, 1)
                pv, ps = T43[m][ds][i], T43S[m][ds][i]
                ok_mu = abs(mu - pv) < 5e-5
                m0 = abs(s0 - ps) < 5e-5
                m1 = abs(s1 - ps) < 5e-5
                flag = ""
                if not ok_mu:
                    flag += " ⚠均值不符"
                    nbad += 1
                if m0: n0 += 1
                if m1: n1 += 1
                if not (m0 or m1):
                    flag += " ⚠std两者都不符"
                print(f"  {m:9s} {ds:7s} R@{k:<2d} 论文 {pv:.4f}±{ps:.4f} | "
                      f"重算 mean={mu:.4f} ddof0=±{s0:.4f}{'✓' if m0 else ' '} "
                      f"ddof1=±{s1:.4f}{'✓' if m1 else ' '}{flag}")
    print(f"\n  ⇒ 表4-3 std 命中：ddof=0 口径 {n0}/45，ddof=1 口径 {n1}/45；均值不符 {nbad}/45")

    # ---------------- item 7：§2.2 百分比
    print()
    print("=" * 100)
    print("§2.2 百分比复算（TA-SID vs SASRec），两种口径")
    print("=" * 100)
    print(f"  {'数据集':8s} {'指标':6s} {'论文值':>9s} {'按表4-2/4-3四位小数':>20s} {'按全精度均值':>14s}")
    for ds in ("Sport", "Beauty", "Toys"):
        for key, label, tbl in (("NDCG@20", "N@20", "NDCG"), ("Recall@20", "R@20", "Rec")):
            tv = data[("TA-SID", ds)][key]
            sv = data[("SASRec", ds)][key]
            t_mu = sum(tv[s] for s in SEEDS) / 3
            s_mu = sum(sv[s] for s in SEEDS) / 3
            full = (t_mu / s_mu - 1) * 100
            r4 = (round(t_mu, 4) / round(s_mu, 4) - 1) * 100
            doc = {"Sport": {"NDCG@20": 139.5, "Recall@20": 105.5},
                   "Beauty": {"NDCG@20": 63.2, "Recall@20": 32.3},
                   "Toys": {"NDCG@20": 56.4, "Recall@20": 39.9}}[ds][key]
            print(f"  {ds:8s} {label:6s} {doc:8.1f}% {r4:19.1f}% {full:13.1f}%")

    if missing:
        print("\n⚠ 缺失 lane：")
        for m in missing:
            print("   ", m)


if __name__ == "__main__":
    main()
