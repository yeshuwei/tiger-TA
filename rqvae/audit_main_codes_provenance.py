#!/usr/bin/env python3
"""audit_main_codes_provenance.py — 主表 codes provenance 审计（2026-09-09）

对每个主表 *_ta_sid_768d_sid.npy：用候选 ckpt（best_collision / best_loss）重建模型，
重放 RQ 逐层分配，与 codes npy 的 L0 列比对行级一致率 → 判定该 codes 是否出自现存 ckpt、
出自哪个、是否健康。

⚠ 2026-09-27 订正（L1/L2 定级）—— **判定列只有 L0，L1/L2 已降级为冗余一致性检查**
------------------------------------------------------------------------------
**旧实现的 bug**：本脚本曾用 `argmin(z, cb_l)` 对**每一层独立**求最近码，忽略了 RQ 的
**层间残差**（第 l 层的输入是 `z - 已选码本向量之和`，不是 z）。因此旧输出里"顺带测"的
L1/L2 一致率**是错的、无意义**，不能作为"配错"或"某层塌缩"的证据。
（该值曾被 `doc/e48_toys_main_codes_provenance_fix.md` 引用为 2.3% / 0.6%。）

**修正**：改用 `sf_rq.rederive_codes_sequential` 按真实前向顺序重放残差。
**为什么不把 L1/L2 当独立证据**：给定同一码本与同一 z，**L0 = 100% 蕴含 L1 = L2 = 100%**
→ 三列冗余。**判配对应只看 L0。**

用法（仓库根，CPU 即可）：
    python rqvae/audit_main_codes_provenance.py
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_rq import load_rqvae_from_ckpt, encode_z, rederive_codes_sequential

MAIN = {
    "Beauty": ("data/Beauty/Beauty_ta_sid_768d_sid.npy",
               glob.glob("rqvae/ckpt/Beauty_TA_SID/*/")),
    "Sport": ("data/Sport/Sport_ta_sid_768d_sid.npy",
              glob.glob("rqvae/ckpt/Sport_TA_SID_768d/*/")),
    "Toys": ("data/Toys/Toys_ta_sid_768d_sid.npy",
             glob.glob("rqvae/ckpt/Toys_TA_SID_768d/*/")),
}


def argmin_col(model, z, level):
    """⚠ 已废弃，仅供对照旧输出用：对第 level 层直接 argmin(z, cb_l)，
    **忽略层间残差** → L1/L2 结果无意义。新代码请用 rederive_codes_sequential。"""
    w = model.rq.vq_layers[level].embedding.weight.detach().to(z.device)
    return torch.cdist(z, w).argmin(dim=1)


def usage_and_agree(code_col, pred_col):
    n = len(code_col)
    agree = int((code_col == pred_col).sum())
    return n, agree, agree / n


def main():
    print("=" * 90)
    print("主表 codes provenance 审计　—— 判据只看 L0；L1/L2 是冗余一致性检查，不是独立证据。")
    print("L1/L2 由 RQ 真实前向（逐层残差）重放；旧版 argmin(z) 得到的 L1/L2 一致率已作废。")
    print("L2 可略低于 100%（codegen 的碰撞消解 pass 用 use_sk=True 改写了少数行），属预期。")
    print("=" * 90)
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
                # codes 的 parquet 应与 ckpt args.data_path 同源；用 ckpt 自己的 data_path 重算。
                # use_sk=False = codegen 主 pass；use_sk=True 仅用于解释 L2 的残留行。
                pred = rederive_codes_sequential(model, z, use_sk=False)
                pred_sk = rederive_codes_sequential(model, z, use_sk=True)
                print(f"  [{sub} {which}]")
                for lvl in range(3):
                    n, agree, frac = usage_and_agree(codes[:, lvl], pred[:, lvl])
                    tag = "  ← **判定列**" if lvl == 0 else "  ← 冗余（由 L0 蕴含）"
                    extra = ""
                    if lvl == 2 and frac > 0.90:   # 仅在配对成立时才解释 L2 余量
                        ns, agrees, fracs = usage_and_agree(codes[:, lvl], pred_sk[:, lvl])
                        extra = (f"　| 该行按碰撞消解 pass（use_sk=True）重放为 {fracs*100:.2f}%"
                                 f" ⇒ 残差 {frac*100-fracs*100:.2f}pp 即被消解循环改写的行")
                    print(f"      L{lvl}: agree={agree}/{n} ({frac*100:.2f}%) "
                          f"pred_distinct={len(np.unique(pred[:, lvl]))}{tag}{extra}")


if __name__ == "__main__":
    main()
