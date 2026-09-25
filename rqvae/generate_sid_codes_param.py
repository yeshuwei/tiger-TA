#!/usr/bin/env python3
"""
generate_sid_codes_param.py — 参数化的 SID codes 生成器（供 E3 新模型使用）

基于 generate_sid_codes_sport_768d.py 抽取参数，逻辑完全一致：
    初始 pass 用 use_sk=False（纯 argmin），碰撞消解循环用 use_sk=True。
输出 codes (N, 4)：前 3 列为 L0/L1/L2 码号，第 4 列为碰撞消解序号。

用法：
    python3 rqvae/generate_sid_codes_param.py \
        --ckpt_path rqvae/ckpt/Toys_TA_SID_L0SK/<ts>/best_collision_model.pth \
        --data_path data/Toys/fused_item_emb_768d.parquet \
        --output_file data/Toys/Toys_ta_sid_L0SK_sid.npy \
        [--device cuda:0]
"""

import argparse
import collections
import numpy as np
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from datasets import EmbDataset
from models.rqvae import RQVAE


def main():
    parser = argparse.ArgumentParser(description="Parameterized SID code generator")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to best_collision_model.pth")
    parser.add_argument("--data_path", type=str, required=True, help="fused embedding parquet")
    parser.add_argument("--output_file", type=str, required=True, help="output .npy path")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    a = ckpt["args"]

    data = EmbDataset(args.data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=a.num_emb_list,
                  e_dim=a.e_dim,
                  layers=a.layers,
                  dropout_prob=a.dropout_prob,
                  bn=a.bn,
                  loss_type=a.loss_type,
                  quant_loss_weight=a.quant_loss_weight,
                  kmeans_init=a.kmeans_init,
                  kmeans_iters=a.kmeans_iters,
                  sk_epsilons=a.sk_epsilons,
                  sk_iters=a.sk_iters)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    print(f"模型: num_emb_list={a.num_emb_list} e_dim={a.e_dim} sk_epsilons={a.sk_epsilons}")
    print(f"数据: {args.data_path} ({len(data)} items)")

    data_loader = DataLoader(data, num_workers=a.num_workers,
                             batch_size=64, shuffle=False, pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in data_loader:
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
        indices_count = collections.Counter(all_indices_str.tolist())
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

    collision_counts = collections.Counter(all_indices_str.tolist())
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Max conflicts:", max(collision_counts.values()))
    print("Collision Rate:", (tot_item - tot_indice) / tot_item)

    codes = []
    for item, indices in enumerate(all_indices.tolist()):
        codes.append([int(item.split('_')[1].strip('>')) for item in indices])
    codes_array = np.array(codes)
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    for dup in unique_codes[counts > 1]:
        dup_idx = np.where((codes_array == dup).all(axis=1))[0]
        for i, idx in enumerate(dup_idx):
            codes_array[idx, -1] = i

    new_unique, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    dupes = new_unique[new_counts > 1]
    print("Still have duplicates:", len(dupes) if len(dupes) else 0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    np.save(args.output_file, codes_array)
    print(f"✓ Codes saved ({codes_array.shape}): {args.output_file}")


if __name__ == "__main__":
    main()
