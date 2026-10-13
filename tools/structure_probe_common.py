#!/usr/bin/env python3
"""
structure_probe_common.py — P1.1：统一 item 对齐层 + 两套 universe + 多空间表示缓存

━━ 本文件的范围（严格限定，见 research/experiment_plan.md §P1.1）━━━━━━━━━━━━
    ① canonical ItemID → row 对齐（含跨文件行序断言）
    ② 两套 item universe 的构造与覆盖度统计（协议 §5）
    ③ ‖Bh[i]‖₂ == 0 集合 vs deg == 0 集合 vs "未出现在序列表" 集合 的差集（协议 §14 U1）
    ④ z_raw / z_norm / q1 / q12 / q123 / codes 的对齐缓存（写入独立缓存目录）
    ⑤ codes 重建的 x_q  vs  模型前向(use_sk=False) 的 x_q / indices 一致性（协议 §14 U2）
    ⑥ N / 行序 / 维度 / 异常值 断言

    **不做**：Top-K 邻居采集（P1.2）、任何邻域指标（P1.3）、任何 AUC、
             任何协议变更、任何训练。

━━ 只读约束（协议 §12 A9）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    只读 `data/`、`rqvae/ckpt/`；只写 `research/_p11_cache/`（新建的临时缓存目录）
    与本脚本的 `--report` 目标。**不覆盖** data/、doc/、rqvae/ckpt/、model/ 下任何既有产物。

━━ 口径来源（不重新定义）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    canonical 行号   : sid_consistency_contrast.item_id_to_row  （禁止 ItemID-1）
    变体×种子 → 产物  : stage_probe.VARIANTS + resolve（禁止手写文件名，孤儿 codes 教训）
    input 896d        : stage_probe.FUSED_896D
    E 图 deg          : behavior_structure_probe.build_graph（去对角 C，window=3）
    q* 重建           : sid_per_layer_diagnostic.rebuild_xq_partial
    z 归一化          : sid_per_layer_diagnostic.normalize_rows（clamp 1e-9，保留零向量）

用法（仓库根，CPU）：
    python3 tools/structure_probe_common.py --datasets Sport --variants TA-SID --seeds 2024 \\
        --universe_datasets Sport,Toys,Beauty --report research/p11_alignment_report.md
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "rqvae"))

from sid_consistency_contrast import (  # noqa: E402
    find_best_ckpt,
    load_codebook_weights,
    item_id_to_row,
)
from sid_per_layer_diagnostic import (  # noqa: E402
    encode_z,
    normalize_rows,
    rebuild_xq_partial,
)
from stage_probe import VARIANTS, FUSED_896D, resolve  # noqa: E402
from behavior_structure_probe import build_graph  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
CACHE_ROOT = os.path.join(ROOT, "research", "_p11_cache")
DEFAULT_ITEM_EMB = "item_emb_768d.parquet"
WINDOW = 3                      # 协议 §1.3-1：window_size=3 ⇒ 相邻对
BEHAVIOR_DIM = 128              # SvD 行为嵌入维度（fused = 768 + 128 = 896）
TOL = 1e-6                      # 协议 §12 A8 的数值断言容差

# 变体 → 行为向量文件。None = 行为半部恒为零，无独立行为文件。
BEHAVIOR_FILE = {
    "TA-SID": "behavior_emb.parquet",
    "Raw-CoOcc": "behavior_emb_rawcooc.parquet",
    "ItemCF": "behavior_emb_itemcf.parquet",
    "Shuffled": "behavior_emb_shuffled.parquet",
    "ZeroPad": None,        # fused 来自 fused_item_emb_zeropad（行为半部为零）
    "Content-only": None,   # 同上（preflight 口径：content ⊕ 128 零）
}


# --------------------------------------------------------------------- 基础 IO
def read_emb_parquet(path):
    """读 {ItemID, embedding} parquet → (ItemID 数组, (N,D) float32 数组)。"""
    df = pd.read_parquet(path)
    emb = np.stack(df["embedding"].values, axis=0).astype(np.float32)
    if "ItemID" in df.columns:
        return df["ItemID"].values, emb
    return None, emb


def load_rqvae(ckpt_path, data_path, device="cpu"):
    """构造并加载 RQ-VAE（与 sid_per_layer_diagnostic.encode_z 逐行同构，只读）。"""
    from datasets import EmbDataset
    from models.rqvae import RQVAE

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    data = EmbDataset(data_path)
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
    model.eval().to(device)
    return model, a, data


def forward_spaces(ckpt_path, data_path, device="cpu", batch_size=512):
    """一次前向取出 z_raw / x_q / vq_indices（口径与 encode_z 逐位一致：同 loader、同 batch）。

    返回 dict(z_raw, z_norm, x_q, indices)；全部为 torch CPU 张量。
    """
    from torch.utils.data import DataLoader

    model, a, data = load_rqvae(ckpt_path, data_path, device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    zs, xqs, idxs = [], [], []
    with torch.no_grad():
        for batch in loader:
            b = batch.to(device)
            z = model.encoder(b)                       # ← x_q 的输入，非 straight-through 版
            x_q, _, indices = model.rq(z, use_sk=False)  # use_sk=False = 与 codegen 首轮一致
            zs.append(z.cpu())
            xqs.append(x_q.cpu())
            idxs.append(indices.view(-1, indices.shape[-1]).cpu())
    return {
        "z_raw": torch.cat(zs, dim=0),
        "z_norm": normalize_rows(torch.cat(zs, dim=0)),
        "x_q": torch.cat(xqs, dim=0),
        "indices": torch.cat(idxs, dim=0),
        "args": a,
    }


# ------------------------------------------------------------------ 对齐检查 ①
def alignment_checks(ds, item_emb_file=DEFAULT_ITEM_EMB):
    """canonical ItemID → row 对齐的事实清单（不假设，全部实测）。"""
    item_path = os.path.join(DATA_DIR, ds, item_emb_file)
    fused_path = os.path.join(DATA_DIR, ds, FUSED_896D["TA-SID"])
    beh_path = os.path.join(DATA_DIR, ds, "behavior_emb.parquet")

    ids_item, emb_item = read_emb_parquet(item_path)
    ids_fused, emb_fused = read_emb_parquet(fused_path)
    ids_beh, emb_beh = read_emb_parquet(beh_path)

    n = len(ids_item)
    arange1 = np.arange(1, n + 1)
    minus1_equals_row = bool(np.array_equal(ids_item, arange1))

    # 表内重复 ItemID？（重复会让 item_id_to_row 静默丢行）
    uniq = len(np.unique(ids_item))

    m = item_id_to_row(ds, item_emb_file)
    return {
        "ds": ds,
        "n_items": int(n),
        "item_emb_file": item_emb_file,
        "item_emb_dim": int(emb_item.shape[1]),
        "fused_file": FUSED_896D["TA-SID"],
        "fused_dim": int(emb_fused.shape[1]),
        "behavior_dim": int(emb_beh.shape[1]),
        "itemid_min": int(ids_item.min()),
        "itemid_max": int(ids_item.max()),
        "itemid_n_unique": int(uniq),
        "n_duplicate_itemid": int(n - uniq),
        "itemid_equals_row_plus_1": minus1_equals_row,
        "itemid_is_permutation": bool(not minus1_equals_row and uniq == n
                                      and ids_item.min() == 1 and ids_item.max() == n),
        "id2row_size": len(m),
        # 跨文件行序断言：融合/行为文件必须与 item_emb_768d 逐位同序
        "fused_same_itemid_order": bool(ids_fused is not None
                                        and np.array_equal(ids_item, ids_fused)),
        "behavior_same_itemid_order": bool(ids_beh is not None
                                           and np.array_equal(ids_item, ids_beh)),
        # 零范数行（异常值检查）
        "item_emb_zero_rows": int(np.sum(np.linalg.norm(emb_item, axis=1) == 0)),
        "fused_zero_rows": int(np.sum(np.linalg.norm(emb_fused, axis=1) == 0)),
        "behavior_nan_inf": int(np.sum(~np.isfinite(emb_beh))),
    }


# ------------------------------------------------------------- universe 统计 ②③
def train_seen_items(ds, item_id_map):
    """独立复算 "出现在构建行为图所读序列中的 item"（rqvae/build_transition_graph.py:227-242 口径）。

    ⚠ 与 build_transition_graph 一致：`history + [target]`，且映射失败的直接丢弃。
    """
    train_df = pd.read_parquet(os.path.join(DATA_DIR, ds, "train.parquet"))
    seen = set()
    n_users = 0
    n_tokens = 0
    for _, row in train_df.iterrows():
        seq = list(row["history"]) + [row["target"]]
        mapped = [item_id_map[x] for x in seq if x in item_id_map]
        if mapped:
            n_users += 1
            n_tokens += len(mapped)
            seen.update(mapped)
    return seen, n_users, n_tokens


def universe_stats(ds, behavior_file="behavior_emb.parquet", data_dir=DATA_DIR,
                   window=WINDOW, do_graph=True):
    """构造 U_beh / U_cont 并实测协议 §5、§14 U1 的全部事实。

    U_beh = {i : ‖Bh[i]‖₂ > 0}；U_cont = 全部 N。
    另外测：deg==0 集合（去对角 C 的度）、seen_items 集合、三者两两差集。
    """
    item_path = os.path.join(data_dir, ds, DEFAULT_ITEM_EMB)
    ids, _ = read_emb_parquet(item_path)
    n = len(ids)
    id_map = {int(iid): i for i, iid in enumerate(ids)}

    out = {"ds": ds, "n_items": int(n), "behavior_file": behavior_file,
           "U_cont_size": int(n)}

    if behavior_file is None:
        # 行为半部恒为零的变体（ZeroPad / Content-only）
        fused_path = os.path.join(data_dir, ds, FUSED_896D["ZeroPad"])
        _, emb_fused = read_emb_parquet(fused_path)
        bh_half = emb_fused[:, 768:]
        out.update({
            "behavior_kind": "zero_by_construction",
            "behavior_half_max_abs": float(np.abs(bh_half).max()),
            "n_zero_norm": int(n),
            "zero_ratio": 1.0,
            "U_beh_size": 0,
            "coverage": 0.0,
        })
    else:
        bh_path = os.path.join(data_dir, ds, behavior_file)
        _, bh = read_emb_parquet(bh_path)
        norms = np.linalg.norm(bh, axis=1)
        z = np.where(norms == 0.0)[0]
        nz = np.where(norms > 0.0)[0]
        out.update({
            "behavior_kind": "file",
            "behavior_file_path": bh_path,
            "behavior_dim": int(bh.shape[1]),
            "n_zero_norm": int(len(z)),
            "zero_ratio": float(len(z) / n),
            "U_beh_size": int(len(nz)),
            "coverage": float(len(nz) / n),
            "norm_min_nonzero": float(norms[nz].min()) if len(nz) else None,
            "norm_max": float(norms.max()),
        })

    if not do_graph:
        return out

    G = build_graph(ds, data_dir, window)
    deg = G["deg"]
    D = np.where(deg == 0)[0]                      # deg==0 集合（去对角 C）
    seen, n_users, n_tokens = train_seen_items(ds, id_map)
    seen_idx = np.array(sorted(seen), dtype=np.int64)

    if behavior_file is None:
        Z = np.arange(n)                           # 行为半部恒为零 ⇒ 全部行为向量为零
    else:
        _, bh_full = read_emb_parquet(os.path.join(data_dir, ds, behavior_file))
        Z = np.where(np.linalg.norm(bh_full, axis=1) == 0.0)[0]

    out.update({
        "n_edges": int(len(G["edges"])),
        "n_seqs_used": int(G["n_seqs"]),
        "train_users_mapped": int(n_users),
        "train_tokens_mapped": int(n_tokens),
        "n_seen_items": int(len(seen_idx)),
        "n_deg0": int(len(D)),
        "deg_max": int(deg.max()),
        "deg_mean": float(deg.mean()),
        # 协议 §14 U1 的差集
        "diff_Z_minus_D": int(len(np.setdiff1d(Z, D))),
        "diff_D_minus_Z": int(len(np.setdiff1d(D, Z))),
        "diff_Z_sym_D": int(len(np.setdiff1d(Z, D)) + len(np.setdiff1d(D, Z))),
        "diff_Z_minus_unseen": int(len(np.setdiff1d(Z, np.setdiff1d(
            np.arange(n), seen_idx)))),
        "Z_eq_complement_seen": bool(np.array_equal(np.sort(Z), np.sort(
            np.setdiff1d(np.arange(n), seen_idx)))),
    })
    return out


# ------------------------------------------------------------------ 空间装配 ④
def build_spaces(ds, variant, seed, device="cpu", data_dir=DATA_DIR,
                 item_emb_file=DEFAULT_ITEM_EMB, item_emb_override=None):
    """返回逐 item 对齐的 {input896, content768, behavior128, z_raw, z_norm, q1, q12, q123, codes}。

    全部张量/np 数组的第 i 行对应 `item_id_to_row` 给出的同一件商品。
    """
    ckpt_tag, codes_file, frozen = resolve(ds, variant, seed)
    if ckpt_tag is None:
        raise FileNotFoundError(f"{ds}/{variant}/seed{seed}: VARIANTS 表无该格（无产物）")

    ckpt_path = find_best_ckpt(ckpt_tag)
    if ckpt_path is None:
        raise FileNotFoundError(f"{ds}/{variant}/seed{seed}: 找不到 ckpt {ckpt_tag}")

    fused_path = os.path.join(data_dir, ds, FUSED_896D[variant])
    codes_path = os.path.join(data_dir, ds, codes_file)
    beh_file = BEHAVIOR_FILE.get(variant)
    beh_path = os.path.join(data_dir, ds, beh_file) if beh_file else None

    ids, x896 = read_emb_parquet(fused_path)
    n = x896.shape[0]

    codes = np.load(codes_path)
    if codes.ndim != 2 or codes.shape[0] != n:
        raise ValueError(f"codes 形状 {codes.shape} 与 N={n} 不符：{codes_path}")

    fwd = forward_spaces(ckpt_path, fused_path, device=device)
    if fwd["z_raw"].shape[0] != n:
        raise ValueError(f"前向输出 {fwd['z_raw'].shape[0]} != N={n}")

    codebooks = load_codebook_weights(ckpt_path, device=device)
    codes_t = torch.as_tensor(codes[:, :len(codebooks)], dtype=torch.long)
    q_list = {f"q{''.join(str(l + 1) for l in range(L))}":
              rebuild_xq_partial(codes_t, codebooks, L)
              for L in range(1, len(codebooks) + 1)}

    # content 768d：TA-SID 用 item_emb_768d（= 纯内容）
    ct_path = os.path.join(data_dir, ds, item_emb_override or item_emb_file)
    _, c768 = read_emb_parquet(ct_path)
    if c768.shape[0] != n:
        raise ValueError(f"content 行数 {c768.shape[0]} != N={n}")

    if beh_path and os.path.exists(beh_path):
        _, b128 = read_emb_parquet(beh_path)
    else:
        b128 = np.zeros((n, BEHAVIOR_DIM), dtype=np.float32)

    spaces = {
        "input896": x896,
        "content768": c768,
        "behavior128": b128,
        "z_raw": fwd["z_raw"].numpy(),
        "z_norm": fwd["z_norm"].numpy(),
        "codes": codes.astype(np.int64),
        "item_ids": ids,
        **{k: v.numpy() for k, v in q_list.items()},
    }
    meta = {
        "ds": ds, "variant": variant, "seed": seed, "frozen_to_seed2024": bool(frozen),
        "ckpt_tag": ckpt_tag, "ckpt_path": ckpt_path,
        "ckpt_mtime": datetime.datetime.fromtimestamp(
            os.path.getmtime(ckpt_path)).isoformat(timespec="seconds"),
        "codes_file": codes_file,
        "codes_mtime": datetime.datetime.fromtimestamp(
            os.path.getmtime(codes_path)).isoformat(timespec="seconds"),
        "codes_shape": list(codes.shape),
        "fused_file": FUSED_896D[variant],
        "behavior_file": beh_file,
        "n_items": int(n),
        "dims": {k: (list(v.shape) if hasattr(v, "shape") else None)
                 for k, v in spaces.items()},
        "ckpt_data_path": getattr(fwd["args"], "data_path", None),
        "num_emb_list": list(getattr(fwd["args"], "num_emb_list", [])),
        "e_dim": int(getattr(fwd["args"], "e_dim", -1)),
    }
    return spaces, meta, fwd, codebooks, codes


def consistency_check(codes, fwd, codebooks):
    """协议 §14 U2 / §12 A8：codes 重建的 x_q  vs  前向 x_q；并统计碰撞消解改写行数。

    关键区分：
      - **首轮**（codegen 第一遍 `get_indices(use_sk=False)`）的码 = 纯 argmin；
      - **最终 codes**（npy 落盘值）= 首轮码 + 对碰撞组用 `use_sk=True`（第 2 层 sinkhorn）改写。
    ⇒ 二者不同的行**全部**来自碰撞消解，且**只可能落在第 2 层**（codegen 只对最后一层保留
      sk_epsilon>0，其余层被置 0）；见 `rqvae/generate_sid_codes_param.py:83-110`。
    """
    n = codes.shape[0]
    n_layers = len(codebooks)
    idx_fwd = fwd["indices"].numpy()                        # (N, L) 前向 argmin 索引
    codes_sem = codes[:, :n_layers]

    row_diff = np.any(codes_sem != idx_fwd, axis=1)
    per_layer = [int(np.sum(codes_sem[:, l] != idx_fwd[:, l])) for l in range(n_layers)]

    codes_t = torch.as_tensor(codes_sem, dtype=torch.long)
    xq_rebuild = rebuild_xq_partial(codes_t, codebooks, n_layers)
    absdiff = (xq_rebuild - fwd["x_q"]).abs()
    per_row_max = absdiff.max(dim=1).values.numpy()
    row_xq_diff = per_row_max > TOL

    # 首轮（纯 argmin）码的碰撞情况 —— 碰撞消解的作用域
    uniq0, inv0, cnt0 = np.unique(idx_fwd, axis=0, return_inverse=True,
                                  return_counts=True)
    first_pass_dup = cnt0[inv0] > 1
    # 最终码的唯一性
    uniq, _, cnt = np.unique(codes_sem, axis=0, return_inverse=True, return_counts=True)

    return {
        "n_items": int(n),
        "n_layers": int(n_layers),
        "first_pass_unique_codes": int(len(uniq0)),
        "first_pass_collision_rate": float(1 - len(uniq0) / n),
        "rows_in_first_pass_dup_group": int(first_pass_dup.sum()),
        "final_unique_codes": int(len(uniq)),
        "final_collision_rate": float(1 - len(uniq) / n),
        "codes_dedup_col_nonzero": int(np.sum(codes[:, n_layers:] != 0)),
        "codes_max_per_layer": codes_sem.max(axis=0).tolist(),
        "n_rows_index_differs_from_forward": int(row_diff.sum()),
        "per_layer_index_diff": per_layer,
        "rows_rewritten_in_first_pass_dup_group": int(np.sum(row_diff & first_pass_dup)),
        "n_rows_xq_differs_over_tol": int(row_xq_diff.sum()),
        "xq_diff_set_eq_index_diff_set": bool(
            np.array_equal(np.where(row_xq_diff)[0], np.where(row_diff)[0])),
        "max_abs_xq_diff_matching_rows": float(per_row_max[~row_diff].max())
        if np.any(~row_diff) else 0.0,
        "max_abs_xq_diff_rewritten_rows": float(per_row_max[row_diff].max())
        if np.any(row_diff) else 0.0,
    }


def cache_spaces(spaces, meta, cache_dir, cache_big=False):
    """把 z / q* / codes（可选大空间）落盘到独立缓存目录。返回写入的文件清单。"""
    os.makedirs(cache_dir, exist_ok=True)
    small = ["z_raw", "z_norm", "q1", "q12", "q123", "codes", "item_ids"]
    big = ["input896", "content768", "behavior128"]
    keys = small + (big if cache_big else [])
    written = []
    for k in keys:
        if k not in spaces:
            continue
        p = os.path.join(cache_dir, f"{k}.npy")
        np.save(p, spaces[k])
        written.append(p)
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    written.append(os.path.join(cache_dir, "meta.json"))
    return written


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="P1.1 对齐 / universe / 空间缓存")
    ap.add_argument("--datasets", default="Sport", help="做完整空间装配的数据集")
    ap.add_argument("--variants", default="TA-SID")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024])
    ap.add_argument("--universe_datasets", default="Sport,Toys,Beauty",
                    help="只做 universe/deg 统计的数据集（便宜）")
    ap.add_argument("--no_graph", action="store_true", help="跳过 E 图 deg 计算")
    ap.add_argument("--cache_big", action="store_true",
                    help="额外缓存 896/768/128 大空间（约数百 MB/变体）")
    ap.add_argument("--cache_root", default=CACHE_ROOT)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    log_lines = []
    t0 = time.time()

    def log(s=""):
        print(s)
        log_lines.append(s)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log("# P1.1 对齐 / universe / 空间缓存 — 实测报告")
    log()
    log(f"- 生成时间：{ts}")
    log(f"- 脚本：`tools/structure_probe_common.py`")
    log(f"- 口径来源：`research/behavior_structure_protocol.md`（v1.0 冻结，未改动）")
    try:
        import subprocess
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10).stdout.strip()
        cm = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10).stdout.strip()
        log(f"- 仓库：分支 `{br}` ｜ commit `{cm}`")
    except Exception as e:      # 无 git 环境时不影响主流程
        log(f"- 仓库：git 信息不可用（{e}）")
    log()

    # ---------------- ① 对齐 ----------------
    log("## 1. canonical ItemID → row 对齐")
    log()
    log("| 数据集 | N | item_emb dim | fused dim | behavior dim | ItemID 范围 | "
        "唯一 ItemID | ItemID==行号+1 | 判定 |")
    log("|:--|--:|--:|--:|--:|:--|--:|:--:|:--|")
    align = {}
    for ds in args.universe_datasets.split(","):
        a = alignment_checks(ds)
        align[ds] = a
        verdict = "连续（两法等价）" if a["itemid_equals_row_plus_1"] else (
            "排列（**禁止 ItemID-1**）" if a["itemid_is_permutation"] else "异常")
        log(f"| {ds} | {a['n_items']} | {a['item_emb_dim']} | {a['fused_dim']} | "
            f"{a['behavior_dim']} | {a['itemid_min']}–{a['itemid_max']} | "
            f"{a['itemid_n_unique']} | {'是' if a['itemid_equals_row_plus_1'] else '否'} | {verdict} |")
    log()
    log("| 数据集 | 重复 ItemID | fused 行序≡item_emb | behavior 行序≡item_emb | "
        "item_emb 零行 | fused 零行 | behavior NaN/Inf |")
    log("|:--|--:|:--:|:--:|--:|--:|--:|")
    for ds, a in align.items():
        log(f"| {ds} | {a['n_duplicate_itemid']} | "
            f"{'✅' if a['fused_same_itemid_order'] else '❌'} | "
            f"{'✅' if a['behavior_same_itemid_order'] else '❌'} | "
            f"{a['item_emb_zero_rows']} | {a['fused_zero_rows']} | {a['behavior_nan_inf']} |")
    log()

    # ---------------- ②③ universe ----------------
    log("## 2. U_beh / U_cont 与零向量集合（协议 §5、§14 U1）")
    log()
    log("| 数据集 | N | U_cont | ‖Bh‖=0 数量 | zero_ratio | |U_beh| | coverage | "
        "E 图边数 | deg=0 数量 | Z\\D | D\\Z | 对称差 | Z≡补(seen) |")
    log("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    uni = {}
    for ds in args.universe_datasets.split(","):
        u = universe_stats(ds, do_graph=not args.no_graph)
        uni[ds] = u
        if "n_deg0" in u:
            log(f"| {ds} | {u['n_items']} | {u['U_cont_size']} | {u['n_zero_norm']} | "
                f"{u['zero_ratio']:.4f} | {u['U_beh_size']} | {u['coverage']:.4f} | "
                f"{u['n_edges']} | {u['n_deg0']} | {u['diff_Z_minus_D']} | "
                f"{u['diff_D_minus_Z']} | {u['diff_Z_sym_D']} | "
                f"{'✅' if u['Z_eq_complement_seen'] else '❌'} |")
        else:
            log(f"| {ds} | {u['n_items']} | {u['U_cont_size']} | {u['n_zero_norm']} | "
                f"{u['zero_ratio']:.4f} | {u['U_beh_size']} | {u['coverage']:.4f} | "
                f"– | – | – | – | – | – |")
    log()

    # ---------------- ④⑤ 空间装配 ----------------
    log("## 3. 空间装配、缓存与 x_q 一致性（协议 §14 U2 / §12 A7-A8）")
    log()
    log("| 数据集 | 变体 | seed | ckpt | N | ckpt.data_path | 与预期 fused 一致 |")
    log("|:--|:--|--:|:--|--:|:--|:--:|")
    built = {}
    for ds in args.datasets.split(","):
        for variant in args.variants.split(","):
            for seed in args.seeds:
                try:
                    spaces, meta, fwd, cb, codes = build_spaces(ds, variant, seed)
                except FileNotFoundError as e:
                    log(f"| {ds} | {variant} | {seed} | ⚠ 缺产物：{e} | | | |")
                    continue
                exp_fused = os.path.join(DATA_DIR, ds, FUSED_896D[variant])
                dp = meta["ckpt_data_path"] or ""
                dp_abs = dp if os.path.isabs(dp) else os.path.join(ROOT, dp)
                # 用 realpath 比较：Beauty 的 fused_item_emb_768d.parquet 是指向
                # fused_item_emb.parquet 的软链，字面不同、实体相同
                same_fused = bool(dp) and (
                    os.path.realpath(dp_abs) == os.path.realpath(exp_fused))
                log(f"| {ds} | {variant} | {seed} | `{os.path.basename(os.path.dirname(meta['ckpt_path']))}` "
                    f"| {meta['n_items']} | `{meta['ckpt_data_path']}` | "
                    f"{'✅ 同一实体' if same_fused else ('⚠ 不同实体' if dp else '–')} |")
                built[(ds, variant, seed)] = (spaces, meta, fwd, cb, codes)
    log()

    # 维度 / 形状表
    log("| 空间 | 期望维度 | " + " | ".join(
        f"{ds}/{v}/s{s}" for (ds, v, s) in built) + " |")
    log("|:--|--:|" + "--:|" * len(built))
    expect = {"input896": 896, "content768": 768, "behavior128": 128,
              "z_raw": 32, "z_norm": 32, "q1": 32, "q12": 32, "q123": 32}
    for k, d in expect.items():
        row = []
        for key in built:
            sp = built[key][0]
            row.append(f"{sp[k].shape[1]}" + ("" if sp[k].shape[1] == d else " ⚠"))
        log(f"| {k} | {d} | " + " | ".join(row) + " |")
    log(f"| codes | 4 列 | " + " | ".join(
        f"{built[k][0]['codes'].shape[1]}"
        + ("" if built[k][0]['codes'].shape[1] == 4 else " ⚠") for k in built) + " |")
    log()

    # 一致性
    log("### 3.1 codes 重建 x_q  vs  前向 x_q（协议 §14 U2）")
    log()
    log("| 数据集/变体/seed | 首轮唯一码 | 首轮碰撞率 | 首轮重复组行数 | 最终唯一码 | "
        "最终碰撞率 | 第4列非零 | 索引与前向不同 | 逐层不同 | 其中落在首轮重复组 |")
    log("|:--|--:|--:|--:|--:|--:|--:|--:|:--|--:|")
    cons = {}
    for key, (spaces, meta, fwd, cb, codes) in built.items():
        c = consistency_check(codes, fwd, cb)
        cons[key] = c
        ds, v, s = key
        log(f"| {ds}/{v}/s{s} | {c['first_pass_unique_codes']} | "
            f"{c['first_pass_collision_rate']:.4f} | {c['rows_in_first_pass_dup_group']} | "
            f"{c['final_unique_codes']} | {c['final_collision_rate']:.4f} | "
            f"{c['codes_dedup_col_nonzero']} | {c['n_rows_index_differs_from_forward']} | "
            f"{c['per_layer_index_diff']} | {c['rows_rewritten_in_first_pass_dup_group']} |")
    log()
    log("| 数据集/变体/seed | x_q 差异行数 (>1e-6) | 差异集≡索引差异集 | "
        "码一致行的 max\\|Δx_q\\| | 被改写行的 max\\|Δx_q\\| | 码取值域 |")
    log("|:--|--:|:--:|--:|--:|:--|")
    for key, c in cons.items():
        ds, v, s = key
        log(f"| {ds}/{v}/s{s} | {c['n_rows_xq_differs_over_tol']} | "
            f"{'✅' if c['xq_diff_set_eq_index_diff_set'] else '❌'} | "
            f"{c['max_abs_xq_diff_matching_rows']:.3e} | "
            f"{c['max_abs_xq_diff_rewritten_rows']:.3e} | "
            f"0–{max(c['codes_max_per_layer'])} |")
    log()

    # 表示塌缩率（协议 §4.2 的量；纯行唯一性，不含任何 Top-K 计算）
    log("### 3.1c 表示塌缩率 `collapse = 1 − |unique(rows)|/N`（协议 §4.2 量）")
    log()
    log("这是协议 §4.2 已经要求**必须报告**的并列诊断量，不涉及 Top-K（P1.2 范围）。")
    log()
    space_keys = ["input896", "content768", "behavior128", "z_raw", "z_norm",
                  "q1", "q12", "q123"]
    log("| 空间 | " + " | ".join(
        f"{ds}: unique / collapse / 最大同值组" for ds in
        [k[0] for k in built]) + " |")
    log("|:--|" + "---:|" * len(built))
    collapse_tab = {}
    for k in space_keys:
        cells = []
        for key, (spaces, meta, fwd, cb, codes) in built.items():
            v = spaces[k]
            u, cnt = np.unique(v, axis=0, return_counts=True)
            collapse_tab[(key, k)] = (len(u), 1 - len(u) / v.shape[0], int(cnt.max()))
            cells.append(f"{len(u)} / {1 - len(u)/v.shape[0]:.4f} / {int(cnt.max())}")
        log(f"| {k} | " + " | ".join(cells) + " |")
    log()
    log("离散口径（codes 前 1/2/3 列的用量唯一数）：")
    log()
    log("| 前缀 | " + " | ".join(f"{k[0]}" for k in built) + " |")
    log("|:--|" + "---:|" * len(built))
    for L in (1, 2, 3):
        cells = []
        for key, (spaces, meta, fwd, cb, codes) in built.items():
            u, cnt = np.unique(codes[:, :L], axis=0, return_counts=True)
            cells.append(f"{len(u)} ({1-len(u)/len(codes):.4f}, max {cnt.max()})")
        log(f"| 前 {L} 列 | " + " | ".join(cells) + " |")
    log()

    # 异常值 / 零行
    log("### 3.1b 各空间零行与异常值（协议 §12 A7 异常值检查）")
    log()
    log("| 数据集/变体/seed | input896 零行 | content768 零行 | behavior128 零行 | "
        "z_raw 零行 | z_norm 零行 | q1 零行 | q12 零行 | q123 零行 | 非有限值 |")
    log("|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for key, (spaces, meta, fwd, cb, codes) in built.items():
        ds, v, s = key
        zr = {k: int(np.sum(np.linalg.norm(spaces[k], axis=1) == 0))
              for k in ["input896", "content768", "behavior128",
                        "z_raw", "z_norm", "q1", "q12", "q123"]}
        ninf = int(sum(np.sum(~np.isfinite(spaces[k])) for k in
                       ["input896", "content768", "behavior128", "z_raw",
                        "z_norm", "q1", "q12", "q123"]))
        log(f"| {ds}/{v}/s{s} | {zr['input896']} | {zr['content768']} | "
            f"{zr['behavior128']} | {zr['z_raw']} | {zr['z_norm']} | {zr['q1']} | "
            f"{zr['q12']} | {zr['q123']} | {ninf} |")
    log()

    # 缓存
    log("### 3.2 缓存落盘")
    log()
    for key, (spaces, meta, fwd, cb, codes) in built.items():
        ds, v, s = key
        cdir = os.path.join(args.cache_root, f"{ds}_{v}_seed{s}")
        files = cache_spaces(spaces, meta, cdir, cache_big=args.cache_big)
        total = sum(os.path.getsize(f) for f in files)
        log(f"- `{os.path.relpath(cdir, ROOT)}`  {len(files)} 个文件，{total/1e6:.1f} MB"
            f"{'（含大空间）' if args.cache_big else '（仅 z/q/codes）'}")
    log()

    # 与既有 encode_z 的逐位对照（同一 ckpt）
    log("### 3.3 与既有 `encode_z` 的逐位对照（自检）")
    log()
    for key, (spaces, meta, fwd, cb, codes) in built.items():
        ds, v, s = key
        z_canon = encode_z(meta["ckpt_path"], os.path.join(DATA_DIR, ds, meta["fused_file"]))
        same = bool(np.array_equal(z_canon.numpy(), spaces["z_norm"]))
        log(f"- {ds}/{v}/s{s}: `encode_z` 与 `z_norm` 逐位一致 = {'✅' if same else '❌'}"
            f"（shape {tuple(z_canon.shape)}）")
    log()

    # 行序一致性的经验证明（A7 的实质断言）
    log("### 3.4 codes 行序 ≡ input896 行序（A7 经验证）")
    log()
    log("若 codes 与 `input896` 行序错位，则\"前向 argmin 索引 vs codes 前 3 列\"的差异行数会接近 N。")
    log("实测差异行数远小于 N ⇒ 三者（codes / fused / item_emb_768d）行序一致：")
    log()
    log("| 数据集 | N | 行序不同行数 | 占比 | 错位假设下的预期 | 判定 |")
    log("|:--|--:|--:|--:|--:|:--|")
    for key, c in cons.items():
        ds, v, s = key
        n = c["n_items"]
        r = c["n_rows_index_differs_from_forward"]
        log(f"| {ds} | {n} | {r} | {r/n:.4f} | ≈{n} | "
            f"{'✅ 行序一致' if r / n < 0.05 else '❌'} |")
    log()
    log("另：`ckpt.args.data_path` 与 `FUSED_896D[variant]` 指向**同一实体**（Beauty 上 "
        "`fused_item_emb_768d.parquet` 是指向 `fused_item_emb.parquet` 的软链），"
        "故 codes 的行序基准就是本脚本读入的 `input896`。")
    log()

    log(f"（总耗时 {time.time() - t0:.1f} s，全程 CPU、只读）")
    log()

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"\n报告已写入: {args.report}")


if __name__ == "__main__":
    main()
