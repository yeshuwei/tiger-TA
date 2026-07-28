#!/usr/bin/env python3
"""
TA-SID 实验指标提取工具

从 T5 训练日志中提取 Test Dataset 的 Recall@K 和 NDCG@K 指标，
输出汇总表格（Markdown + LaTeX 格式）。

用法：
    conda activate tiger_t5
    cd /data/gtx/project/code/TIGER_minilm
    python3 tools/extract_metrics.py [--output TABLE.md]
"""

import re
import os
import argparse


# ============================================================
# 日志文件 → (数据集, 方法, 种子) 映射表
# 按数据集分组，同一个数据集中按方法排序（baseline 在前，TA-SID 在后）
# ============================================================
LOG_MAP = [
    # (日志文件名, 数据集, 方法名, 种子)
    # --- Beauty ---
    ("tiger.log",                        "Beauty", "TIGER",  2025),
    ("tiger_ta_sid.log",                 "Beauty", "TA-SID", 2025),
    ("tiger_ta_sid_beauty_seed42.log",   "Beauty", "TA-SID", 42),
    # --- Sports ---
    ("sports.log",                       "Sports", "TIGER",  2025),
    ("tiger_ta_sid_sport.log",           "Sports", "TA-SID", 2025),
    ("tiger_ta_sid_sport_seed42.log",    "Sports", "TA-SID", 42),
    # --- Toys ---
    ("toys.log",                         "Toys",   "TIGER",  2025),
    ("tiger_ta_sid_toys.log",            "Toys",   "TA-SID", 2025),
    ("tiger_ta_sid_toys_seed42.log",     "Toys",   "TA-SID", 42),
]


def extract_metrics(log_path: str):
    """
    从日志文件中提取最后一组 Test Dataset 指标。

    日志格式（每两行一组，先 Recall 后 NDCG）：
        Test Dataset: {'Recall@5': 0.037, ...}
        Test Dataset: {'NDCG@5': 0.023, ...}

    返回: (recall_dict, ndcg_dict) 或 (None, None) 如果未找到
    """
    if not os.path.exists(log_path):
        return None, None

    with open(log_path, 'r') as f:
        content = f.read()

    # 找到所有 Test Dataset 行
    recall_lines = re.findall(
        r"Test Dataset: \{'Recall@5': ([0-9.]+), 'Recall@10': ([0-9.]+), 'Recall@20': ([0-9.]+)\}",
        content
    )
    ndcg_lines = re.findall(
        r"Test Dataset: \{'NDCG@5': ([0-9.]+), 'NDCG@10': ([0-9.]+), 'NDCG@20': ([0-9.]+)\}",
        content
    )

    if not recall_lines or not ndcg_lines:
        return None, None

    # 取最后一次评估的结果
    last_recall = recall_lines[-1]
    last_ndcg = ndcg_lines[-1]

    recall_dict = {
        'Recall@5':  float(last_recall[0]),
        'Recall@10': float(last_recall[1]),
        'Recall@20': float(last_recall[2]),
    }
    ndcg_dict = {
        'NDCG@5':  float(last_ndcg[0]),
        'NDCG@10': float(last_ndcg[1]),
        'NDCG@20': float(last_ndcg[2]),
    }

    return recall_dict, ndcg_dict


def format_markdown(results: list) -> str:
    """生成 Markdown 格式表格"""
    lines = [
        "| Dataset | Method | Seed | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |",
        "|:--------|:-------|:----:|:--------:|:---------:|:---------:|:------:|:-------:|:-------:|",
    ]

    for r in results:
        if r is None:
            continue
        ds, method, seed, rec, ndcg = r
        lines.append(
            f"| {ds} | {method} | {seed} "
            f"| {rec['Recall@5']:.4f} | {rec['Recall@10']:.4f} | {rec['Recall@20']:.4f} "
            f"| {ndcg['NDCG@5']:.4f} | {ndcg['NDCG@10']:.4f} | {ndcg['NDCG@20']:.4f} |"
        )

    return "\n".join(lines) + "\n"


def format_summary(results: list) -> str:
    """生成简化的汇总表（基线 vs TA-SID 对比）"""
    # 只保留 seed=2025 的结果
    baseline_ta = [(ds, method, seed, rec, ndcg)
                   for ds, method, seed, rec, ndcg in results
                   if seed == 2025 and method in ('TIGER', 'TA-SID')]

    lines = [
        "| Dataset | Baseline NDCG@20 | TA-SID NDCG@20 | ΔNDCG@20 | ΔRecall@10 |",
        "|:--------|:----------------:|:---------------:|:--------:|:----------:|",
    ]

    # 按数据集分组
    baselines = {r[0]: r for r in baseline_ta if r[1] == 'TIGER'}
    ta_sids = {r[0]: r for r in baseline_ta if r[1] == 'TA-SID'}

    for ds in ['Beauty', 'Sports', 'Toys']:
        if ds in baselines and ds in ta_sids:
            b = baselines[ds]
            t = ta_sids[ds]
            delta_ndcg = t[4]['NDCG@20'] - b[4]['NDCG@20']
            delta_pct = (t[4]['NDCG@20'] / b[4]['NDCG@20'] - 1) * 100
            delta_rec10 = t[3]['Recall@10'] - b[3]['Recall@10']
            lines.append(
                f"| {ds} | {b[4]['NDCG@20']:.4f} | {t[4]['NDCG@20']:.4f} "
                f"| +{delta_ndcg:.4f} (+{delta_pct:.1f}%) | +{delta_rec10:.4f} |"
            )
        else:
            lines.append(
                f"| {ds} | — | — | — | — |"
            )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="提取 TA-SID 实验指标")
    parser.add_argument('--log_dir', default='model/logs',
                        help='T5 日志目录 (default: model/logs)')
    parser.add_argument('--output', default=None,
                        help='输出文件路径（默认输出到终端）')
    args = parser.parse_args()

    # 提取所有指标
    results = []
    errors = []
    for log_name, dataset, method, seed in LOG_MAP:
        log_path = os.path.join(args.log_dir, log_name)
        rec, ndcg = extract_metrics(log_path)
        if rec is not None:
            results.append((dataset, method, seed, rec, ndcg))
            print(f"  ✓ {log_name:45s} → {dataset} {method} seed={seed}")
        else:
            errors.append(log_name)
            print(f"  ✗ {log_name:45s} → 未找到 Test Dataset 指标")

    if errors:
        print(f"\n⚠ 以下 {len(errors)} 个日志文件未找到指标：")
        for e in errors:
            print(f"   - {e}")

    if not results:
        print("\n❌ 未提取到任何指标！")
        return

    # 生成完整表格
    print("\n" + "=" * 80)
    print("完整指标表")
    print("=" * 80)
    md_table = format_markdown(results)
    print(md_table)

    # 生成汇总表
    print("=" * 80)
    print("汇总表（Baseline vs TA-SID, seed=2025）")
    print("=" * 80)
    summary = format_summary(results)
    print(summary)

    # 写入文件
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write("# TA-SID 主实验指标汇总\n\n")
            f.write("## 完整指标表\n\n")
            f.write(md_table)
            f.write("\n## Baseline vs TA-SID 对比\n\n")
            f.write(summary)
        print(f"✅ 已保存到: {args.output}")

    # 按数据集分开展示
    print("\n" + "=" * 80)
    print("按数据集分开展示")
    print("=" * 80)
    datasets = ['Beauty', 'Sports', 'Toys']
    for ds in datasets:
        ds_results = [r for r in results if r[0] == ds]
        if not ds_results:
            continue
        print(f"\n--- {ds} ---")
        print(f"{'Method':20s} {'Seed':>5s}  {'R@5':>8s} {'R@10':>8s} {'R@20':>8s}  "
              f"{'N@5':>8s} {'N@10':>8s} {'N@20':>8s}")
        print("-" * 75)
        for ds_, method, seed, rec, ndcg in ds_results:
            print(f"{method:20s} {seed:5d}  {rec['Recall@5']:.4f}  {rec['Recall@10']:.4f}  "
                  f"{rec['Recall@20']:.4f}  {ndcg['NDCG@5']:.4f}  "
                  f"{ndcg['NDCG@10']:.4f}  {ndcg['NDCG@20']:.4f}")


if __name__ == "__main__":
    main()
