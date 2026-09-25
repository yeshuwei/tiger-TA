#!/usr/bin/env python3
"""验证 (ckpt_dir, codes_file) 配对：重建码本 → 重算 argmin 分配 → 与 codes 比行级一致率。
一致率高（>90%）→ pairing 正确；低 → 配错（rebuild_xq 会失效）。"""
import glob
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "rqvae"))
from sf_rq import load_rqvae_from_ckpt, encode_z

# (ds, 变体名, codes 文件, ckpt 目录候选 glob)
TARGETS = [
    ("Sport", "Content-only", "Sport_content768d_baseline_sid.npy", ["Sport_Content768d_Baseline"]),
    ("Sport", "Shuffled", "Sport_shuffled_ablation_sid.npy", ["Sport_Shuffled_Ablation", "Sport_Shuffled_768d"]),
    ("Sport", "ZeroPad", "Sport_zeropad_ablation_sid.npy", ["Sport_Zeropad_Ablation", "Sport_Zeropad_768d"]),
    ("Sport", "ItemCF", "Sport_itemcf_768d_sid.npy", ["Sport_ItemCF_768d"]),
    ("Sport", "Raw-CoOcc", "Sport_rawcooc_768d_sid.npy", ["Sport_RawCoOcc_768d"]),
    ("Sport", "TA-SID", "Sport_ta_sid_768d_sid.npy", ["Sport_TA_SID_768d"]),
    ("Toys", "Content-only", "Toys_content768d_baseline_sid.npy", ["Toys_Content768d_Baseline"]),
    ("Toys", "Shuffled", "Toys_shuffled_768d_sid.npy", ["Toys_Shuffled_768d"]),
    ("Toys", "ZeroPad", "Toys_zeropad_seed2024_sid.npy", ["Toys_Zeropad_seed2024"]),
    ("Toys", "ItemCF", "Toys_itemcf_768d_sid.npy", ["Toys_ItemCF_768d"]),
    ("Toys", "Raw-CoOcc", "Toys_rawcooc_768d_sid.npy", ["Toys_RawCoOcc_768d"]),
    ("Toys", "TA-SID(健康)", "Toys_ta_sid_768d_seed2024_sid.npy",
     ["Toys_TA_SID_seed2024", "Toys_TA_SID_768d", "Toys_TA_SID", "Toys_TA_SID_seed2025",
      "Toys_TA_SID_seed2026", "Toys_TA_SID_skl0_seed2024"]),
]


def argmin_col(model, z, level):
    w = model.rq.vq_layers[level].embedding.weight.detach().to(z.device)
    return torch.cdist(z, w).argmin(dim=1)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for ds, vname, codef, cktags in TARGETS:
        if only and only not in vname:
            continue
        cpath = os.path.join("data", ds, codef)
        if not os.path.exists(cpath):
            print(f"\n{'='*92}\n{ds} / {vname}: ❌ codes 文件不存在: {codef}")
            continue
        codes = np.load(cpath)
        print(f"\n{'='*92}\n{ds} / {vname}: {codef}  shape={codes.shape} "
              f"L0distinct={len(np.unique(codes[:,0]))}")

        found_any = False
        for tag in cktags:
            dirs = sorted(glob.glob(f"rqvae/ckpt/{tag}/*/"))
            if not dirs:
                continue
            for ckpt_dir in dirs:
                sub = os.path.basename(ckpt_dir.rstrip("/"))
                ck = os.path.join(ckpt_dir, "best_collision_model.pth")
                if not os.path.exists(ck):
                    continue
                try:
                    model, a = load_rqvae_from_ckpt(ck, device="cpu")
                    z = encode_z(model, a.data_path, device="cpu")
                except Exception as e:
                    print(f"  [{tag}/{sub}] ✗ 加载失败: {e}")
                    continue
                if z.shape[0] != codes.shape[0]:
                    print(f"  [{tag}/{sub}] ✗ 行数不符 z={z.shape[0]} vs codes={codes.shape[0]}")
                    continue
                found_any = True
                fr = []
                for lvl in range(3):
                    pred = argmin_col(model, z, lvl).cpu().numpy()
                    frac = float((codes[:, lvl] == pred).mean())
                    fr.append(frac)
                verdict = "✅ 配对正确" if fr[0] > 0.90 else ("⚠ 存疑" if fr[0] > 0.5 else "❌ 不匹配")
                print(f"  [{tag}/{sub}]  data_path={os.path.basename(str(a.data_path))}")
                print(f"      L0={fr[0]*100:6.2f}%  L1={fr[1]*100:6.2f}%  L2={fr[2]*100:6.2f}%   {verdict}")
        if not found_any:
            print("  ❌ 无任何可用 ckpt 候选")


if __name__ == "__main__":
    main()
