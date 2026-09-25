"""
run_content_baseline.py — 768d content-only baseline 流水线（论文公平对比用）

目标：在同一管线上复刻 TIGER（纯 content SID），与 TA-SID(768d+128 behavior) 公平对比。
输入：data/{ds}/item_emb_768d_contentonly.parquet（= LN(768d content)，与 TA-SID content 部分同源）
流程：RQ-VAE(in_dim=768) → SID codes → T5 → eval
命名：全部带 _content768d_baseline_ 标记，不覆盖任何现有实验文件。

用法：
    python rqvae/run_content_baseline.py --dataset Sport
    python rqvae/run_content_baseline.py --dataset Beauty
    python rqvae/run_content_baseline.py --dataset all
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="768d content-only baseline pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Sport / Toys / Beauty / all")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def log(msg, flush=True):
    print(f"\n{'='*60}\n[{msg}]\n{'='*60}", flush=flush)


def run_cmd(cmd, desc, cwd):
    log(f"Running: {desc}")
    print(f"  Command: {cmd}", flush=True)
    result = subprocess.run(f"set -o pipefail; {cmd}", shell=True, cwd=cwd,
                            executable='/bin/bash')
    if result.returncode != 0:
        log(f"ERROR: {desc} failed (code {result.returncode})")
        sys.exit(1)
    log(f"✓ {desc}", flush=True)


def run_ds(ds, seed):
    ds_lower = ds.lower()
    root = os.path.abspath(".")

    content_only = f"data/{ds}/item_emb_768d_contentonly.parquet"
    rqvae_ckpt = f"rqvae/ckpt/{ds}_Content768d_Baseline"
    rqvae_log = f"rqvae/logs/{ds}_content768d_baseline_rqvae.log"
    code_output = f"data/{ds}/{ds}_content768d_baseline_sid.npy"
    gen_log = f"rqvae/logs/{ds}_content768d_baseline_generate_code.log"
    collision_log = f"rqvae/logs/{ds}_content768d_baseline_collision.txt"
    t5_ckpt = f"model/ckpt/tiger_content768d_baseline_{ds_lower}_seed{seed}.pth"
    t5_log = f"model/logs/tiger_content768d_baseline_{ds_lower}_seed{seed}.log"
    t5_train_log = f"model/logs/{ds}_content768d_baseline_seed{seed}_t5_train.log"
    eval_log = f"model/logs/{ds}_content768d_baseline_seed{seed}_eval.log"

    os.makedirs("rqvae/logs", exist_ok=True)

    # ================================================================
    # Step 1: Train RQ-VAE on LN(768d content) — in_dim=768, 与 TA-SID 同参数
    # ================================================================
    log(f"[{ds}] Step 1: RQ-VAE training (in_dim=768 content-only)")
    if not os.path.exists(content_only):
        log(f"ERROR: {content_only} 不存在（先跑构造步骤）")
        sys.exit(1)

    ckpt_base = os.path.join(root, rqvae_ckpt)
    best_path = None
    if os.path.exists(ckpt_base):
        dirs = sorted([d for d in os.listdir(ckpt_base)
                       if os.path.isdir(os.path.join(ckpt_base, d))],
                      key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
        for d in reversed(dirs):
            p = os.path.join(ckpt_base, d, "best_collision_model.pth")
            if os.path.exists(p):
                best_path = p
                break

    if best_path is not None:
        log(f"[{ds}] 检测到已有 RQ-VAE checkpoint，跳过训练: {best_path}")
    else:
        step1 = (
            f"python rqvae/main.py "
            f"--data_path {content_only} "
            f"--ckpt_dir {rqvae_ckpt} "
            f"--epochs 3000 --batch_size 1024 --eval_step 50 "
            f"--num_emb_list 256 256 256 --e_dim 32 "
            f"--layers 512 256 128 64 --loss_type mse "
            f"--quant_loss_weight 1.0 --beta 0.25 "
            f"--kmeans_init True --kmeans_iters 100 "
            f"--sk_epsilons 0.0 0.0 0.003 --sk_iters 50 "
            f"--save_limit 5 "
            f"2>&1 | tee -a {rqvae_log}"
        )
        run_cmd(step1, f"[{ds}] RQ-VAE training", root)
        # 重找 checkpoint
        dirs = sorted([d for d in os.listdir(ckpt_base)
                       if os.path.isdir(os.path.join(ckpt_base, d))],
                      key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
        for d in reversed(dirs):
            p = os.path.join(ckpt_base, d, "best_collision_model.pth")
            if os.path.exists(p):
                best_path = p
                break

    if best_path is None:
        log(f"ERROR: best_collision_model.pth not found in {ckpt_base}")
        sys.exit(1)
    log(f"  Checkpoint: {best_path}")

    # ================================================================
    # Step 2: Generate SID codes
    # ================================================================
    if os.path.exists(os.path.join(root, code_output)):
        log(f"[{ds}] SID codes 已存在，跳过生成: {code_output}")
    else:
        log(f"[{ds}] Step 2: Generating SID codes")
    gen_code_path = os.path.join(root, "rqvae", f"generate_code_{ds}_contentbaseline.py")

    gen_content = f'''"""
Auto-generated: generate SID codes for {ds} content768d baseline
"""
import collections, json, logging, numpy as np, torch, os
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import EmbDataset
from models.rqvae import RQVAE

DATASET = "{ds}"
ckpt_path = r"{best_path}"
output_file = r"{os.path.join(root, code_output)}"
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(args.data_path):
    args.data_path = os.path.normpath(os.path.join(script_dir, "..", args.data_path))
data = EmbDataset(args.data_path)
print(f"data dim = {{data.dim}} (期望 768)")

model = RQVAE(in_dim=data.dim,
              num_emb_list=args.num_emb_list,
              e_dim=args.e_dim,
              layers=args.layers,
              dropout_prob=args.dropout_prob,
              bn=args.bn,
              loss_type=args.loss_type,
              quant_loss_weight=args.quant_loss_weight,
              kmeans_init=args.kmeans_init,
              kmeans_iters=args.kmeans_iters,
              sk_epsilons=args.sk_epsilons,
              sk_iters=args.sk_iters)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

data_loader = DataLoader(data, num_workers=args.num_workers,
                         batch_size=64, shuffle=False)

all_indices = []
all_indices_str = []
prefix = ["<a_{{}}>","<b_{{}}>","<c_{{}}>","<d_{{}}>","<e_{{}}>"]

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0

tt = 0
while True:
    if tt >= 30 or len(set(all_indices_str.tolist())) == len(all_indices_str):
        break

    from collections import Counter
    indices_count = Counter(all_indices_str.tolist())
    collision_item_groups = []
    processed = set()
    for idx, s in enumerate(all_indices_str):
        if idx in processed:
            continue
        if indices_count[s] > 1:
            group = [i for i, x in enumerate(all_indices_str) if x == s]
            collision_item_groups.append(group)
            processed.update(group)

    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices[item] = code
            all_indices_str[item] = str(code)
    tt += 1

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
collision_rate = (tot_item - tot_indice) / tot_item
print(f"Collision Rate: {{collision_rate:.6f}}")

with open(r"{os.path.join(root, collision_log)}", "w") as fc:
    fc.write(f"Collision Rate: {{collision_rate:.6f}}\\n")
    fc.write(f"Total items: {{tot_item}}\\n")
    fc.write(f"Unique indices: {{tot_indice}}\\n")

all_indices_dict = {{}}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)

codes = []
for key, value in all_indices_dict.items():
    code = [int(item.split('_')[1].strip('>')) for item in value]
    codes.append(code)

codes_array = np.array(codes)
codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = unique_codes[counts > 1]
if len(duplicates) > 0:
    print("Resolving duplicates in codes...")
    for duplicate in duplicates:
        duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
        for i, idx in enumerate(duplicate_indices):
            codes_array[idx, -1] = i

print(f"Saving codes to {{output_file}}")
np.save(output_file, codes_array)
print("✓ Codes saved!")
'''

    if not os.path.exists(os.path.join(root, code_output)):
        with open(gen_code_path, "w") as f:
            f.write(gen_content)

        step2 = (
            f"cd {root}/rqvae && python generate_code_{ds}_contentbaseline.py 2>&1 | "
            f"tee {root}/rqvae/logs/{ds}_content768d_baseline_generate_code.log"
        )
        run_cmd(step2, f"[{ds}] SID code generation", root)

    # ================================================================
    # Step 3: Train T5 (seed 统一 2025)
    # ================================================================
    if os.path.exists(os.path.join(root, t5_ckpt)):
        log(f"[{ds}] T5 模型已存在，跳过训练: {t5_ckpt}")
    else:
        log(f"[{ds}] Step 3: T5 training (seed={seed})")
        step3 = (
            f"python model/main.py "
            f"--dataset_path data/{ds} "
            f"--code_path {code_output} "
            f"--save_path {t5_ckpt} "
            f"--log_path {t5_log} "
            f"--seed {seed} "
            f"2>&1 | tee {t5_train_log}"
        )
        run_cmd(step3, f"[{ds}] T5 training", root)

    # ================================================================
    # Step 4: Evaluate
    # ================================================================
    if os.path.exists(os.path.join(root, eval_log)):
        log(f"[{ds}] Eval 结果已存在，跳过评估: {eval_log}")
    else:
        log(f"[{ds}] Step 4: Evaluation")
        step4 = (
            f"python3 tools/eval_trained_model.py "
            f"--dataset {ds} "
            f"--ckpt_path {t5_ckpt} "
            f"--code_path {code_output} "
            f"--beam_size 30 "
            f"--topk_list 5,10,20 "
            f"--seed {seed} "
            f"2>&1 | tee {eval_log}"
        )
        run_cmd(step4, f"[{ds}] Evaluation", root)

    log(f"🎉 [{ds}] content768d baseline complete! → {eval_log}")


def main():
    args = parse_args()
    if args.dataset == "all":
        for ds in ["Sport", "Beauty", "Toys"]:
            run_ds(ds, args.seed)
    else:
        run_ds(args.dataset, args.seed)
    log("All content768d baselines complete!")


if __name__ == "__main__":
    main()
