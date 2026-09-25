"""
run_multiseed.py — T5 多 seed 报告（RQ-VAE 固定，仅 T5 换 seed 重训）

目的：对已有 code 文件，用多个 seed 重训 T5 + eval，报均值±std。
设计：RQ-VAE 生成的 SID code 由数据决定（确定性强），随机性主要来自 T5 → 固定 code，T5 多 seed。

用法：
    # baseline 全部数据集补 seed 2026, 2027
    python rqvae/run_multiseed.py --method baseline --dataset all --seeds 2026 2027

    # TA-SID 指定数据集
    python rqvae/run_multiseed.py --method ta_sid --dataset Sport --seeds 2026 2027

输出（均不覆盖现有 seed2025 文件）：
    model/ckpt/tiger_{tag}_{ds_lower}_seed{seed}.pth
    model/logs/tiger_{tag}_{ds_lower}_seed{seed}.log
    model/logs/{ds}_{tag}_seed{seed}_eval.log
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(".")

# method → (命名 tag, code 文件模板)
METHODS = {
    "baseline": {
        "tag": "content768d_baseline",
        "code": "data/{ds}/{ds}_content768d_baseline_sid.npy",
    },
    "ta_sid": {
        "tag": "ta_sid_768d",
        "code": {
            "Sport": "data/Sport/Sport_ta_sid_768d_sid.npy",
            "Beauty": "data/Beauty/Beauty_ta_sid_768d_sid.npy",
            "Toys": "data/Toys/Toys_ta_sid_768d_sid.npy",
        },
    },
}


def log(msg):
    print(f"\n{'='*60}\n[{msg}]\n{'='*60}", flush=True)


def run_cmd(cmd, desc):
    log(f"Running: {desc}")
    print(f"  {cmd}", flush=True)
    result = subprocess.run(f"set -o pipefail; {cmd}", shell=True, cwd=ROOT,
                            executable='/bin/bash')
    if result.returncode != 0:
        log(f"ERROR: {desc} failed (code {result.returncode})")
        sys.exit(1)


def run_one(method, ds, seed):
    tag = METHODS[method]["tag"]
    ds_lower = ds.lower()
    code_tpl = METHODS[method]["code"]
    code_path = code_tpl.format(ds=ds) if method == "baseline" else code_tpl[ds]

    if not os.path.exists(code_path):
        log(f"ERROR: code 不存在 {code_path}")
        sys.exit(1)

    t5_ckpt = f"model/ckpt/tiger_{tag}_{ds_lower}_seed{seed}.pth"
    t5_log = f"model/logs/tiger_{tag}_{ds_lower}_seed{seed}.log"
    eval_log = f"model/logs/{ds}_{tag}_seed{seed}_eval.log"

    if os.path.exists(eval_log):
        log(f"[{method}][{ds}][seed{seed}] eval 已存在，跳过: {eval_log}")
        return

    # T5 训练（若已训过则跳过）
    if not os.path.exists(t5_ckpt):
        log(f"[{method}][{ds}] T5 training seed={seed}")
        step = (
            f"python model/main.py "
            f"--dataset_path data/{ds} "
            f"--code_path {code_path} "
            f"--save_path {t5_ckpt} "
            f"--log_path {t5_log} "
            f"--seed {seed} "
            f"2>&1 | tee model/logs/{ds}_{tag}_seed{seed}_t5_train.log"
        )
        run_cmd(step, f"[{method}][{ds}] T5 (seed={seed})")

    # eval
    log(f"[{method}][{ds}] Evaluation seed={seed}")
    step = (
        f"python3 tools/eval_trained_model.py "
        f"--dataset {ds} "
        f"--ckpt_path {t5_ckpt} "
        f"--code_path {code_path} "
        f"--beam_size 30 "
        f"--topk_list 5,10,20 "
        f"--seed {seed} "
        f"2>&1 | tee {eval_log}"
    )
    run_cmd(step, f"[{method}][{ds}] eval (seed={seed})")
    log(f"✅ [{method}][{ds}][seed{seed}] → {eval_log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "ta_sid"])
    parser.add_argument("--dataset", type=str, default="all",
                        help="Sport/Beauty/Toys/all")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    datasets = ["Sport", "Beauty", "Toys"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        for seed in args.seeds:
            run_one(args.method, ds, seed)
    log(f"✅ 全部完成: method={args.method}, datasets={datasets}, seeds={args.seeds}")


if __name__ == "__main__":
    main()
