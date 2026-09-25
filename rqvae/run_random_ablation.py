"""
run_random_ablation.py — Full pipeline: Random Emb → RQ-VAE → Codes → T5 → Eval

Usage:
    cd /data/gtx/project/code/TIGER_minilm
    python rqvae/run_random_ablation.py --dataset Sport

This will:
    Step 1: Generate random behavior emb + fused emb
    Step 2: Train RQ-VAE on fused_item_emb_random.parquet
    Step 3: Generate SID codes
    Step 4: Train T5 on random SIDs
    Step 5: Evaluate (beam_size=30, topk=5,10,20)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Random emb ablation pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (Sport recommended)")
    parser.add_argument("--project_root", type=str, default=".",
                        help="Project root")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def find_latest_rqvae_ckpt(ckpt_base):
    """Find best_collision_model.pth from the LATEST training run."""
    if not os.path.exists(ckpt_base):
        return None, None

    dirs = sorted([d for d in os.listdir(ckpt_base)
                   if os.path.isdir(os.path.join(ckpt_base, d))],
                  key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))

    if not dirs:
        return None, None

    latest_dir = dirs[-1]
    best_path = os.path.join(ckpt_base, latest_dir, "best_collision_model.pth")
    if os.path.exists(best_path):
        return latest_dir, best_path

    # Fallback: any .pth file
    pth_files = sorted([f for f in os.listdir(os.path.join(ckpt_base, latest_dir))
                        if f.endswith(".pth")])
    if pth_files:
        return latest_dir, os.path.join(ckpt_base, latest_dir, pth_files[-1])
    return latest_dir, None


def log(msg):
    print(f"\n{'='*60}")
    print(f"[Random Ablation {ds}] {msg}")
    print(f"{'='*60}")
    sys.stdout.flush()


def run_cmd(cmd, desc):
    log(f"Running: {desc}")
    print(f"  Command: {cmd}")
    sys.stdout.flush()
    # Use pipefail so that if the main command fails, the pipeline fails
    result = subprocess.run(f"set -o pipefail; {cmd}", shell=True, cwd=project_root,
                           executable='/bin/bash')
    if result.returncode != 0:
        log(f"ERROR: {desc} failed with code {result.returncode}")
        sys.exit(1)
    log(f"✓ {desc} completed successfully")
    sys.stdout.flush()


args = parse_args()
ds = args.dataset
project_root = os.path.abspath(args.project_root)
seed = args.seed

# Create log directory
os.makedirs("rqvae/logs", exist_ok=True)

# ================================================================
# Step 1: Generate random behavior embedding + fused embedding
# ================================================================
log("Step 1: Generating random behavior embedding + fused embedding")

step1_cmd = (
    f"python rqvae/random_emb_ablation.py "
    f"--dataset {ds} --seed {seed} "
    f"2>&1 | tee -a rqvae/logs/{ds}_random_ablation_step1.log"
)
run_cmd(step1_cmd, "Generate random emb + fused emb")

# ================================================================
# Step 2: Train RQ-VAE on (Content + Random) fused embedding
# ================================================================
log("Step 2: Training RQ-VAE on content+random fused embedding")

fused_path = f"data/{ds}/fused_item_emb_random.parquet"
rqvae_ckpt = f"rqvae/ckpt/{ds}_Random_Ablation"
rqvae_log = f"rqvae/logs/{ds}_random_ablation_rqvae.log"

# Use SAME config as the real Sport 768d TA-SID experiment
step2_cmd = (
    f"python rqvae/main.py "
    f"--data_path {fused_path} "
    f"--ckpt_dir {rqvae_ckpt} "
    f"--epochs 3000 "
    f"--batch_size 1024 "
    f"--eval_step 50 "
    f"--num_emb_list 256 256 256 "
    f"--e_dim 32 "
    f"--layers 512 256 128 64 "
    f"--loss_type mse "
    f"--quant_loss_weight 1.0 "
    f"--beta 0.25 "
    f"--kmeans_init True "
    f"--kmeans_iters 100 "
    f"--sk_epsilons 0.0 0.0 0.003 "
    f"--sk_iters 50 "
    f"--save_limit 5 "
    f"2>&1 | tee -a {rqvae_log}"
)
run_cmd(step2_cmd, "RQ-VAE training (content+random)")

# ================================================================
# Find best RQ-VAE checkpoint
# ================================================================
log("Finding best RQ-VAE checkpoint...")

ckpt_base = os.path.join(project_root, rqvae_ckpt)
best_dir, best_ckpt_path = find_latest_rqvae_ckpt(ckpt_base)

if best_ckpt_path is None:
    log(f"ERROR: Could not find any checkpoint in {ckpt_base}")
    sys.exit(1)

log(f"RQ-VAE checkpoint: {best_ckpt_path}")
log(f"Checkpoint dir: {best_dir}")

# ================================================================
# Step 3: Generate SID codes
# ================================================================
log("Step 3: Generating SID codes")

code_output = f"data/{ds}/{ds}_random_ablation_sid.npy"
gen_code_path = os.path.join(project_root, "rqvae", f"generate_code_{ds}_random_ablation.py")

gen_code_content = f'''"""
Auto-generated: generate SID codes for {ds} Random Ablation
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
ckpt_path = r"{best_ckpt_path}"
output_file = r"{os.path.join(project_root, code_output)}"
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

# Resolve data_path to absolute
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
collision_rate = (tot_item - tot_indice) / tot_item
print("Collision Rate", collision_rate)

# Save collision rate for reference
with open(r"{os.path.join(project_root, f'rqvae/logs/{ds}_random_ablation_collision.txt')}", "w") as fc:
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

step3_cmd = (
    f"cd {project_root}/rqvae && python generate_code_{ds}_random_ablation.py 2>&1 | "
    f"tee {project_root}/rqvae/logs/{ds}_random_ablation_generate_code.log"
)
run_cmd(step3_cmd, f"SID code generation ({ds} Random)")

# ================================================================
# Step 4: Train T5 (TIGER) with random-ablation SID codes
# ================================================================
log("Step 4: Training T5 with random-ablation SID codes")

t5_save = f"model/ckpt/tiger_random_ablation_{ds.lower()}.pth"
t5_log = f"model/logs/tiger_random_ablation_{ds.lower()}.log"

step4_cmd = (
    f"python model/main.py "
    f"--dataset_path data/{ds} "
    f"--code_path data/{ds}/{ds}_random_ablation_sid.npy "
    f"--save_path {t5_save} "
    f"--log_path {t5_log} "
    f"--seed {seed} "
    f"2>&1 | tee model/logs/{ds}_random_ablation_t5_train.log"
)
run_cmd(step4_cmd, f"T5 training ({ds} Random)")

# ================================================================
# Step 5: Evaluate (beam_size=30, topk=5,10,20)
# ================================================================
log("Step 5: Evaluating random-ablation model")

eval_ckpt_map = f"model/ckpt/tiger_random_ablation_{ds.lower()}.pth"
eval_code_map = f"data/{ds}/{ds}_random_ablation_sid.npy"
valid_sids_file = eval_code_map  # same file for valid SID set

# ================================================================
# Step 5: Evaluate with eval_trained_model.py
# ================================================================
log("Step 5: Evaluating random-ablation model")

t5_ckpt = f"model/ckpt/tiger_random_ablation_{ds.lower()}.pth"
code_file = f"data/{ds}/{ds}_random_ablation_sid.npy"

step5_cmd = (
    f"python3 tools/eval_trained_model.py "
    f"--dataset {ds} "
    f"--ckpt_path {t5_ckpt} "
    f"--code_path {code_file} "
    f"--beam_size 30 "
    f"--topk_list 5,10,20 "
    f"--seed {seed} "
    f"2>&1 | tee model/logs/{ds}_random_ablation_eval.log"
)
run_cmd(step5_cmd, f"Evaluation ({ds} Random Ablation)")

log(f"🎉 Random Ablation pipeline complete for {ds}!")
log(f"   RQ-VAE ckpt:  {rqvae_ckpt}/{best_dir}")
log(f"   T5 ckpt:      model/ckpt/tiger_random_ablation_{ds.lower()}.pth")
log(f"   Results log:  model/logs/{ds}_random_ablation_eval.log")
log(f"")
log(f"📊 现在可以对比:")
log(f"   TIGER (content only):     NDCG@20 = 0.0207 (已有)")
log(f"   Content + Random emb:     NDCG@20 = ?   (刚跑的)")
log(f"   Content + Transition emb: NDCG@20 = 0.0327 (TA-SID)")
log(f"")
log(f"   如果: Random ≈ TIGER << TA-SID → 证明 Transition 结构是关键")
