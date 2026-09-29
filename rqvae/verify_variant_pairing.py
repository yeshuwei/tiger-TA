#!/usr/bin/env python3
"""验证 (ckpt_dir, codes_file) 配对：重建码本 → 重放 RQ 逐层分配 → 与 codes 比行级一致率。
一致率高（>90%）→ pairing 正确；低 → 配错（rebuild_xq 会失效）。

⚠ 2026-09-27 订正（L1/L2 定级）—— **判定列只有 L0，L1/L2 已降级为冗余一致性检查**
------------------------------------------------------------------------------
**旧实现的 bug**：本脚本与 `audit_main_codes_provenance.py` 曾用 `argmin(z, cb_l)`
对 **每一层独立**求最近码，忽略了 RQ 的**层间残差**（第 l 层的输入是
`z - 已选码本向量之和`，不是 z）。因此旧输出里的 L1/L2 一致率（`doc/e48_toys_main_codes_provenance_fix.md`
引用的 **2.3% / 0.6%**）**是错的、无意义**，不能作为"配错"或"某层塌缩"的证据。

**修正**：改用 `sf_rq.rederive_codes_sequential`，按真实前向顺序重放残差。修正后
L0/L1 为 100.00%，L2 可略低于 100%（主表 `sk_epsilons=[0.0,0.0,0.003]` ⇒ L2 走 Sinkhorn
**全局**分配而非逐行 argmin，少数行会偏离）。

**为什么不把 L1/L2 当三条独立证据**：给定同一码本与同一 z，**L0 = 100% 蕴含 L1 = L2 = 100%**
（残差被完全决定）→ 三列高度冗余，不是三次独立验证。**判配对应只看 L0。**
"""
import glob
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "rqvae"))
from sf_rq import load_rqvae_from_ckpt, encode_z, rederive_codes_sequential

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
    """⚠ 已废弃，仅供对照旧输出用：对第 level 层直接 argmin(z, cb_l)，
    **忽略层间残差** → L1/L2 结果无意义。新代码请用 rederive_codes_sequential。"""
    w = model.rq.vq_layers[level].embedding.weight.detach().to(z.device)
    return torch.cdist(z, w).argmin(dim=1)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print("=" * 92)
    print("(ckpt, codes) 配对验证　—— 判据只看 L0；L1/L2 是冗余一致性检查，不是独立证据。")
    print("L1/L2 由 RQ 真实前向（逐层残差）重放；旧版 argmin(z) 得到的 L1/L2 一致率已作废。")
    print("L2 可略低于 100%（codegen 的碰撞消解 pass 用 use_sk=True 改写了少数行），属预期。")
    print("=" * 92)
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
                # L1/L2 由 RQ 真实前向重放得到（见模块 docstring：旧版用 argmin(z) 是错的）
                # use_sk=False = codegen 主 pass；use_sk=True 仅用于解释 L2 的残留行
                pred = rederive_codes_sequential(model, z, use_sk=False)
                pred_sk = rederive_codes_sequential(model, z, use_sk=True)
                fr = [float((codes[:, lvl] == pred[:, lvl]).mean()) for lvl in range(3)]
                l2sk = float((codes[:, 2] == pred_sk[:, 2]).mean())
                verdict = "✅ 配对正确" if fr[0] > 0.90 else ("⚠ 存疑" if fr[0] > 0.5 else "❌ 不匹配")
                print(f"  [{tag}/{sub}]  data_path={os.path.basename(str(a.data_path))}")
                print(f"      L0={fr[0]*100:6.2f}%   ← **判定列**"
                      f"    | 冗余检查（由 L0 蕴含）: L1={fr[1]*100:6.2f}%  L2={fr[2]*100:6.2f}%"
                      f"   {verdict}")
                if fr[2] > 0.90:   # 仅在配对成立时才解释 L2 余量
                    print(f"      （L2 若按碰撞消解 pass 的 use_sk=True 重放则为 {l2sk*100:6.2f}%；"
                          f"两者之差 {fr[2]*100-l2sk*100:.2f}pp 即被消解循环改写的行）")
        if not found_any:
            print("  ❌ 无任何可用 ckpt 候选")


if __name__ == "__main__":
    main()
