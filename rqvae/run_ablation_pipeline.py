"""
run_ablation_pipeline.py — 运行消融对照实验流水线

用法：
    # Shuffled transition
    python rqvae/run_ablation_pipeline.py --variant shuffled

    # Zero padding
    python rqvae/run_ablation_pipeline.py --variant zeropad

    # 连续跑两个
    python rqvae/run_ablation_pipeline.py --variant all
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation pipeline runner")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["shuffled", "zeropad", "all"],
                        help="Which ablation to run")
    parser.add_argument("--dataset", type=str, default="Sport")
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


# ----- Configuration per variant -----
VARIANTS = {
    "shuffled": {
        "tag": "shuffled",
        "fused_emb": "data/{ds}/fused_item_emb_shuffled.parquet",
        "rqvae_ckpt": "rqvae/ckpt/{ds}_Shuffled_Ablation",
        "code_output": "data/{ds}/{ds}_shuffled_ablation_sid.npy",
        "t5_ckpt": "model/ckpt/tiger_shuffled_ablation_{ds_lower}.pth",
        "t5_log": "model/logs/tiger_shuffled_ablation_{ds_lower}.log",
        "t5_train_log": "model/logs/{ds}_shuffled_ablation_t5_train.log",
        "eval_log": "model/logs/{ds}_shuffled_ablation_eval.log",
        "collision_log": "rqvae/logs/{ds}_shuffled_ablation_collision.txt",
    },
    "zeropad": {
        "tag": "zeropad",
        "fused_emb": "data/{ds}/fused_item_emb_zeropad.parquet",
        "rqvae_ckpt": "rqvae/ckpt/{ds}_Zeropad_Ablation",
        "code_output": "data/{ds}/{ds}_zeropad_ablation_sid.npy",
        "t5_ckpt": "model/ckpt/tiger_zeropad_ablation_{ds_lower}.pth",
        "t5_log": "model/logs/tiger_zeropad_ablation_{ds_lower}.log",
        "t5_train_log": "model/logs/{ds}_zeropad_ablation_t5_train.log",
        "eval_log": "model/logs/{ds}_zeropad_ablation_eval.log",
        "collision_log": "rqvae/logs/{ds}_zeropad_ablation_collision.txt",
    },
}


def log(msg):
    print(f"\n{'='*60}")
    print(f"[{msg}]")
    print(f"{'='*60}")
    sys.stdout.flush()


def run_cmd(cmd, desc, cwd):
    log(f"Running: {desc}")
    print(f"  Command: {cmd}")
    sys.stdout.flush()
    result = subprocess.run(f"set -o pipefail; {cmd}", shell=True, cwd=cwd,
                           executable='/bin/bash')
    if result.returncode != 0:
        log(f"ERROR: {desc} failed (code {result.returncode})")
        sys.exit(1)
    log(f"✓ {desc}")
    sys.stdout.flush()


def run_variant(ds, seed, cfg):
    tag = cfg["tag"]
    ds_lower = ds.lower()
    project_root = os.path.abspath(".")

    def p(template):
        """Format path template with ds/ds_lower."""
        return template.format(ds=ds, ds_lower=ds_lower)

    # ================================================================
    # Step 1: Train RQ-VAE
    # ================================================================
    log(f"[{tag}] Step 1: RQ-VAE training")

    fused_path = p(cfg["fused_emb"])
    rqvae_ckpt = p(cfg["rqvae_ckpt"])
    rqvae_log = f"rqvae/logs/{ds}_{tag}_ablation_rqvae.log"

    os.makedirs("rqvae/logs", exist_ok=True)

    step1_cmd = (
        f"python rqvae/main.py "
        f"--data_path {fused_path} "
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
    run_cmd(step1_cmd, f"[{tag}] RQ-VAE training", project_root)

    # ================================================================
    # Find best checkpoint
    # ================================================================
    log(f"[{tag}] Finding best RQ-VAE checkpoint...")
    ckpt_base = os.path.join(project_root, rqvae_ckpt)
    dirs = sorted([d for d in os.listdir(ckpt_base)
                   if os.path.isdir(os.path.join(ckpt_base, d))],
                  key=lambda d: os.path.getmtime(os.path.join(ckpt_base, d)))
    if not dirs:
        log(f"ERROR: No checkpoint dirs in {ckpt_base}")
        sys.exit(1)
    latest_dir = dirs[-1]
    best_path = os.path.join(ckpt_base, latest_dir, "best_collision_model.pth")
    if not os.path.exists(best_path):
        log(f"ERROR: best_collision_model.pth not in {latest_dir}")
        sys.exit(1)
    log(f"  Checkpoint: {best_path}")

    # ================================================================
    # Step 2: Generate SID codes
    # ================================================================
    log(f"[{tag}] Step 2: Generating SID codes")

    code_output = p(cfg["code_output"])
    gen_code_path = os.path.join(project_root, "rqvae", f"generate_code_{ds}_{tag}.py")

    gen_code_content = f'''"""
Auto-generated: generate SID codes for {ds} {tag}
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
output_file = r"{os.path.join(project_root, code_output)}"
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

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

with open(r"{os.path.join(project_root, p(cfg['collision_log']))}", "w") as fc:
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

    with open(gen_code_path, "w") as f:
        f.write(gen_code_content)

    step2_cmd = (
        f"cd {project_root}/rqvae && python generate_code_{ds}_{tag}.py 2>&1 | "
        f"tee {project_root}/rqvae/logs/{ds}_{tag}_generate_code.log"
    )
    run_cmd(step2_cmd, f"[{tag}] SID code generation", project_root)

    # ================================================================
    # Step 3: Train T5
    # ================================================================
    log(f"[{tag}] Step 3: T5 training")

    t5_save = p(cfg["t5_ckpt"])
    t5_log = p(cfg["t5_log"])
    t5_train_log = p(cfg["t5_train_log"])

    step3_cmd = (
        f"python model/main.py "
        f"--dataset_path data/{ds} "
        f"--code_path {code_output} "
        f"--save_path {t5_save} "
        f"--log_path {t5_log} "
        f"--seed {seed} "
        f"2>&1 | tee {t5_train_log}"
    )
    run_cmd(step3_cmd, f"[{tag}] T5 training", project_root)

    # ================================================================
    # Step 4: Evaluate
    # ================================================================
    log(f"[{tag}] Step 4: Evaluation")

    eval_log = p(cfg["eval_log"])

    step4_cmd = (
        f"python3 tools/eval_trained_model.py "
        f"--dataset {ds} "
        f"--ckpt_path {t5_save} "
        f"--code_path {code_output} "
        f"--beam_size 30 "
        f"--topk_list 5,10,20 "
        f"--seed {seed} "
        f"2>&1 | tee {eval_log}"
    )
    run_cmd(step4_cmd, f"[{tag}] Evaluation", project_root)

    log(f"🎉 [{tag}] Pipeline complete!")


def main():
    args = parse_args()
    ds = args.dataset
    seed = args.seed

    if args.variant == "all":
        for v in ["shuffled", "zeropad"]:
            run_variant(ds, seed, VARIANTS[v])
    else:
        run_variant(ds, seed, VARIANTS[args.variant])

    log("All ablation pipelines complete!")
    print("\n📊 完整对比表（预计）:")
    print(f"   TIGER (content only):          0.0207  ← 基线")
    print(f"   Zero Padding (content + zeros): ?")
    print(f"   Shuffled Transition:            ?")
    print(f"   Random Noise (旧实验):         0.0281")
    print(f"   TA-SID (real transition):      0.0327  ← 目标")
    print()
    print(f"   预期: Zero Pad ≈ Shuffled ≈ TIGER << TA-SID")


if __name__ == "__main__":
    main()
