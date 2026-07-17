"""
Auto-generated: generate TA-SID codes for Sport
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

DATASET = "Sport"
ckpt_path = r"/data/gtx/project/code/TIGER_minilm/rqvae/ckpt/Sport_TA_SID/Jul-16-2026_14-04-01/best_collision_model.pth"
output_file = r"/data/gtx/project/code/TIGER_minilm/data/Sport/Sport_ta_sid.npy"
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]

# Fix data_path: checkpoint stores relative path (project-root relative)
# but this script runs from rqvae/ dir, so resolve to absolute
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(args.data_path):
    args.data_path = os.path.normpath(os.path.join(script_dir, "..", args.data_path))
print(f"Loading data from: {args.data_path}")
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
prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

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
    print(f"Collision groups: {len(collision_item_groups)}")

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

all_indices_dict = {}
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

print(f"Saving codes to {output_file}")
print(f"the first 5 codes: {codes_array[:5]}")
np.save(output_file, codes_array)
print("✓ Codes saved!")
