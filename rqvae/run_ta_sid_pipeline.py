"""
run_ta_sid_pipeline.py — Run TA-SID pipeline (Steps 3→4→5) for one dataset.

Steps 1 (transition graph) and 2 (fuse embeddings) should be done beforehand.

Usage:
    python rqvae/run_ta_sid_pipeline.py --dataset Sport
    python rqvae/run_ta_sid_pipeline.py --dataset Toys
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def find_best_checkpoint(ckpt_base):
    """
    Find the checkpoint dir with the LOWEST collision rate across all training runs.

    遍历所有 checkpoint 子目录，解析 epoch 文件名中的 collision rate，
    返回 (最佳目录名, best_collision_model.pth 完整路径, 最佳碰撞率) 三元组。
    如果无法解析（如缺少 epoch 文件），回退到按 mtime 选最新。
    """
    if not os.path.exists(ckpt_base):
        return None, None, None

    dirs = sorted([d for d in os.listdir(ckpt_base)
                   if os.path.isdir(os.path.join(ckpt_base, d))])
    if not dirs:
        return None, None, None

    best_collision = float('inf')
    best_ckpt_path = None
    best_dir = None

    for d in dirs:
        dir_path = os.path.join(ckpt_base, d)
        best_path = os.path.join(dir_path, "best_collision_model.pth")
        if not os.path.exists(best_path):
            continue

        # 从 epoch_*_collision_*.pth 文件名解析碰撞率
        for f in os.listdir(dir_path):
            m = re.search(r'_collision_([0-9.]+)_', f)
            if m:
                collision = float(m.group(1))
                if collision < best_collision:
                    best_collision = collision
                    best_ckpt_path = best_path
                    best_dir = d

    if best_ckpt_path is not None:
        return best_dir, best_ckpt_path, best_collision

    # Fallback: 按 mtime 最新
    dirs_sorted = sorted(dirs, key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
    latest_dir = dirs_sorted[-1]
    fallback_path = os.path.join(ckpt_base, latest_dir, "best_collision_model.pth")
    if os.path.exists(fallback_path):
        return latest_dir, fallback_path, None
    # 最后尝试：该目录下任意 .pth
    ckpt_files = sorted([f for f in os.listdir(os.path.join(ckpt_base, latest_dir)) if f.endswith(".pth")])
    if ckpt_files:
        return latest_dir, os.path.join(ckpt_base, latest_dir, ckpt_files[-1]), None
    return None, None, None


def parse_args():
    parser = argparse.ArgumentParser(description="TA-SID pipeline runner")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--project_root", type=str, default=".",
                        help="Project root (default: cwd)")
    parser.add_argument("--rqvae_seed", type=int, default=2024)
    parser.add_argument("--t5_seed", type=int, default=2025)
    return parser.parse_args()


def log(msg):
    print(f"\n{'='*60}")
    print(f"[TA-SID {ds}] {msg}")
    print(f"{'='*60}")
    sys.stdout.flush()


def run_cmd(cmd, desc):
    log(f"Running: {desc}")
    log(f"Command: {cmd}")
    sys.stdout.flush()
    result = subprocess.run(cmd, shell=True, cwd=project_root)
    if result.returncode != 0:
        log(f"ERROR: {desc} failed with code {result.returncode}")
        sys.exit(1)
    log(f"✓ {desc} completed successfully")
    sys.stdout.flush()


args = parse_args()
ds = args.dataset
project_root = os.path.abspath(args.project_root)

# ============================================================
# Step 3: Train RQ-VAE on fused embeddings
# ============================================================
log(f"Step 3: Training RQ-VAE on fused embeddings ({ds})")

fused_path = f"data/{ds}/fused_item_emb.parquet"
rqvae_ckpt = f"rqvae/ckpt/{ds}_TA_SID"
rqvae_log = f"rqvae/logs/{ds}_TA_SID_train.log"

os.makedirs(f"rqvae/logs", exist_ok=True)

# The rqvae/main.py uses ckpt_dir as base, and trainer creates a timestamp subdir
# We capture the timestamp dir to find best_collision_model.pth later

step3_cmd = (
    f"python rqvae/main.py "
    f"--data_path {fused_path} "
    f"--ckpt_dir {rqvae_ckpt} "
    f"--epochs 3000 "
    f"--batch_size 1024 "
    f"--eval_step 50 "
    f"2>&1 | tee -a {rqvae_log}"
)
run_cmd(step3_cmd, f"RQ-VAE training ({ds})")

# ============================================================
# Find the best checkpoint across ALL runs (by collision rate)
# ============================================================
log(f"Selecting best RQ-VAE checkpoint across all training runs...")

ckpt_base = os.path.join(project_root, rqvae_ckpt)
best_dir, best_collision_path, best_collision_val = find_best_checkpoint(ckpt_base)

if best_dir is None:
    log(f"ERROR: Could not locate any checkpoint in {ckpt_base}")
    sys.exit(1)

log(f"Selected checkpoint dir: {best_dir}")
if best_collision_val is not None:
    log(f"Best collision rate: {best_collision_val:.6f} ({best_collision_val*100:.2f}%)")
else:
    log(f"Best collision rate: (fallback, from latest mtime)")
log(f"Checkpoint path: {best_collision_path}")

# ============================================================
# Step 4: Generate TA-SID codes
# ============================================================
log(f"Step 4: Generating TA-SID codes ({ds})")

# Create a temporary generate_code script with correct paths
code_output = f"data/{ds}/{ds}_ta_sid.npy"

gen_code_path = os.path.join(project_root, "rqvae", f"generate_code_{ds}.py")
gen_code_content = f'''"""
Auto-generated: generate TA-SID codes for {ds}
"""
import collections
import json
import logging
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import EmbDataset
from models.rqvae import RQVAE
import os

DATASET = "{ds}"
ckpt_path = r"{best_collision_path}"
output_file = r"{os.path.join(project_root, code_output)}"
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

# Fix data_path: checkpoint stores relative path (project-root relative)
# but this script runs from rqvae/ dir, so resolve to absolute
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(args.data_path):
    args.data_path = os.path.normpath(os.path.join(script_dir, "..", args.data_path))
data = EmbDataset(args.data_path)

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
              sk_iters=args.sk_iters,
              )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data, num_workers=args.num_workers,
                         batch_size=64, shuffle=False,
                         pin_memory=True)

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
    print(f"Collision groups: {{len(collision_item_groups)}}")

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

print("All indices number: ", len(all_indices))
collision_counts = collections.Counter(all_indices_str.tolist())
print("Max number of conflicts: ", max(collision_counts.values()))
tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate", (tot_item - tot_indice) / tot_item)

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

new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
duplicates = new_unique_codes[new_counts > 1]
if len(duplicates) > 0:
    print("There still have duplicates:", duplicates)
else:
    print("There are no duplicates in the codes after resolution.")

print(f"Saving codes to {{output_file}}")
print(f"the first 5 codes: {{codes_array[:5]}}")
np.save(output_file, codes_array)
print("✓ Codes saved!")
'''

with open(gen_code_path, "w") as f:
    f.write(gen_code_content)

step4_cmd = (
    f"cd {project_root}/rqvae && python generate_code_{ds}.py 2>&1 | "
    f"tee {project_root}/rqvae/logs/{ds}_generate_code.log"
)
run_cmd(step4_cmd, f"TA-SID code generation ({ds})")

# ============================================================
# Step 5: Train T5 (TIGER) with TA-SID codes
# ============================================================
log(f"Step 5: Training T5 with TA-SID codes ({ds})")

t5_save = f"model/ckpt/tiger_ta_sid_{ds.lower()}.pth"
t5_log = f"model/logs/tiger_ta_sid_{ds.lower()}.log"

step5_cmd = (
    f"python model/main.py "
    f"--dataset_path data/{ds} "
    f"--code_path data/{ds}/{ds}_ta_sid.npy "
    f"--save_path {t5_save} "
    f"--log_path {t5_log} "
    f"--seed {args.t5_seed} "
    f"2>&1 | tee model/logs/{ds}_t5_train.log"
)
run_cmd(step5_cmd, f"T5 training ({ds})")

log(f"🎉 TA-SID pipeline complete for {ds}!")
log(f"   Model: {t5_save}")
log(f"   Log:   {t5_log}")
