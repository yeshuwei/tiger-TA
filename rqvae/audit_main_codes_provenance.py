#!/usr/bin/env python3
"""audit_main_codes_provenance.py — 主表 codes provenance 审计（2026-09-09）

对每个主表 *_ta_sid_768d_sid.npy：用候选 ckpt（best_collision / best_loss）重建模型，
重算 argmin L0 分配，与 codes npy 的 L0 列比对行级一致率 → 判定该 codes 是否出自现存 ckpt、
出自哪个、是否健康。顺带测 L1/L2 列。

用法（仓库根，CPU 即可）：
    python rqvae/audit_main_codes_provenance.py
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_rq import load_rqvae_from_ckpt, encode_z

MAIN = {
    "Beauty": ("data/Beauty/Beauty_ta_sid_768d_sid.npy",
               glob.glob("rqvae/ckpt/Beauty_TA_SID/*/")),
    "Sport": ("data/Sport/Sport_ta_sid_768d_sid.npy",
              glob.glob("rqvae/ckpt/Sport_TA_SID_768d/*/")),
    "Toys": ("data/Toys/Toys_ta_sid_768d_sid.npy",
             glob.glob("rqvae/ckpt/Toys_TA_SID_768d/*/")),
}


def argmin_col(model, z, level):
    w = model.rq.vq_layers[level].embedding.weight.detach().to(z.device)
    return torch.cdist(z, w).argmin(dim=1)


def usage_and_agree(code_col, pred_col):
    n = len(code_col)
    agree = int((code_col == pred_col).sum())
    return n, agree, agree / n


def main():
    for ds, (codes_path, ckpt_dirs) in MAIN.items():
        codes = np.load(codes_path)
        print(f"\n{'='*90}\n{ds}: {codes_path}  shape={codes.shape} "
              f"L0distinct={len(np.unique(codes[:,0]))}")

        for ckpt_dir in sorted(ckpt_dirs):
            sub = os.path.basename(ckpt_dir.rstrip("/"))
            for which in ("best_collision_model.pth", "best_loss_model.pth"):
                ck = os.path.join(ckpt_dir, which)
                if not os.path.exists(ck):
                    continue
                model, a = load_rqvae_from_ckpt(ck, device="cpu")
                z = encode_z(model, a.data_path, device="cpu")
                # codes 的 parquet 应与 ckpt args.data_path 同源；用 ckpt 自己的 data_path 重算
                rows = []
                for lvl in range(3):
                    pred = argmin_col(model, z, lvl).cpu().numpy()
                    n, agree, frac = usage_and_agree(codes[:, lvl], pred)
                    rows.append(f"L{lvl}: agree={agree}/{n} ({frac*100:.2f}%) "
                                f"pred_distinct={len(np.unique(pred))}")
                print(f"  [{sub} {which}]")
                for r in rows:
                    print(f"      {r}")


if __name__ == "__main__":
    main()
