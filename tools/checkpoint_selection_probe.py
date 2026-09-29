#!/usr/bin/env python3
"""checkpoint_selection_probe.py — RQ-VAE checkpoint selection 对行为结构保持的影响

对应实验方案：`doc/exp/RQ-VAE checkpoint selection实验设计.md`

**唯一自变量**：TA-SID 的 RQ-VAE checkpoint 取点策略
        best_collision_model.pth  ← 现役口径（三层离散码 collision rate 最小）
    vs  best_loss_model.pth       ← 对照（train loss 最小）
其余全部冻结：A 输入 / encoder 架构 / latent / codebook 配置 / quantization 方法 /
pair sampling / 正负样本定义 / AUC 定义 / 数据预处理。**不重新训练。**

三个空间（口径逐位复用 `tools/stage_probe.py`，未新定义 Behavior AUC）：
    B = z        编码器输出（量化前 latent，32d）
    C = xq[full] 码本前三层重建（32d）

复用（不复制）：
    tools/stage_probe.py               : build_samples / VARIANTS（9 格 (tag, codes) 声明）
    tools/behavior_structure_probe.py  : cos_sim / auc_of / E 图口径常量
    tools/sid_per_layer_diagnostic.py  : encode_z / normalize_rows / rebuild_xq_partial
    tools/sid_consistency_contrast.py  : find_best_ckpt / load_codebook_weights
    tools/sid_quality.py               : compute_utilization / compute_collision_rate
    rqvae/sf_rq.py                     : load_rqvae_from_ckpt / rederive_codes_sequential
    rqvae/generate_sid_codes_param.py  : C_loss 的 codes 生成（与既有 codes 同一脚本）

用法（仓库根）：
    python3 tools/checkpoint_selection_probe.py --stage main          # B→C AUC / gap / 码结构 / recon loss
    python3 tools/checkpoint_selection_probe.py --stage prov          # E49 Toys 各 variant provenance
"""

import argparse
import datetime
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
os.chdir(ROOT)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "rqvae"))

from behavior_structure_probe import (  # noqa: E402
    cos_sim, auc_of, DATA_DIR, N_POS, N_NEG, SEED,
)
from stage_probe import build_samples, VARIANTS  # noqa: E402
from sid_per_layer_diagnostic import (  # noqa: E402
    encode_z, normalize_rows, rebuild_xq_partial,
)
from sid_consistency_contrast import (  # noqa: E402
    find_best_ckpt, load_codebook_weights,
)
from sf_rq import load_rqvae_from_ckpt, rederive_codes_sequential  # noqa: E402
from sid_quality import compute_utilization, compute_collision_rate  # noqa: E402

DATASETS = ["Beauty", "Sport", "Toys"]
SEEDS = [2024, 2025, 2026]
PRIMARY_POS, PRIMARY_NEG = "topk", "degmatch"     # 与 g2_stage_probe 主口径一致


# --------------------------------------------------------------------------- utils
def find_ckpt_pair(tag):
    """返回同一 run 目录下的 (best_collision_path, best_loss_path, run_dir)。

    `sid_consistency_contrast.find_best_ckpt` 按 mtime 取最后一个 run 目录的
    best_collision；配对的 best_loss 必须取自**同一 run 目录**，否则两个臂来自不同训练。
    """
    ckpt_dir = os.path.join("rqvae", "ckpt", tag)
    if not os.path.exists(ckpt_dir):
        return None, None, None
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*/best_collision_model.pth")),
                   key=lambda p: os.path.getmtime(p))
    if not files:
        return None, None, None
    coll = files[-1]
    run = os.path.dirname(coll)
    loss = os.path.join(run, "best_loss_model.pth")
    return coll, (loss if os.path.exists(loss) else None), run


def resolve_data_path(ckpt_path):
    """ckpt args.data_path（仓库相对路径）→ 绝对路径；不存在返回 None。"""
    a = torch.load(ckpt_path, map_location="cpu", weights_only=False)["args"]
    p = getattr(a, "data_path", None)
    if not p:
        return None
    if not os.path.isabs(p):
        p = os.path.normpath(os.path.join(ROOT, p))
    return p if os.path.exists(p) else None


def z_and_losses(ckpt_path, data_path, device="cpu"):
    """一次 forward 同时取 z（编码器输出）与三类损失（trainer 口径，eval 模式全量）。

    复用 `RQVAE.forward`（`use_sk=True` 为其默认值，与 `trainer._train_epoch` 的
    `self.model(data, return_z=True)` 同口径）与 `RQVAE.compute_loss` 的定义：
        recon_loss = F.mse_loss(decoder(rq(encoder(x))), x)
        total_loss = recon_loss + quant_loss_weight * rq_loss
    `total_loss` 即 trainer 里用来挑 `best_loss_model.pth` 的那个量（`train_loss`），
    本列用于检验「best_loss 是否真是 total loss 的更小者」。

    ⚠ ckpt 自带的 `best_loss` 字段是 **total** loss（recon+quant）的历史最小值，
    不是重建损失；两臂直接比该字段是错的（见 summary 的口径说明）。
    """
    from torch.utils.data import DataLoader
    from datasets import EmbDataset

    model, a = load_rqvae_from_ckpt(ckpt_path, device=device)
    data = EmbDataset(data_path)
    loader = DataLoader(data, batch_size=1024, shuffle=False, num_workers=0)
    zs = []
    s_recon = s_quant = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            z = model.encoder(x)
            out, rq_loss, _ = model(x, return_z=False)
            # ⚠ 2026-09-28 订正：必须逐 batch 用 reduction='mean' 再求和，
            # 与 trainer._train_epoch 的 total_recon_loss / total_quant 完全同口径。
            # 旧写法 `F.mse_loss(..., reduction='sum')` 除以 item 数 = 逐元素均值的
            # **768 倍**（e_dim），而 quant 列是逐 item 均值（~1 倍），两列量纲不一致，
            # 导致 `total_loss` 列被 768 倍膨胀的 recon 主导、完全失真。
            # 该错误曾让 summary §4 得出「best_loss 的 total loss 反而更大」的**错误**
            # 结论；实测两者同口径为 2.8132 (ep49) vs 5.1901 (ep2049)，ep49 确为最小值。
            s_recon += float(F.mse_loss(out, x, reduction="mean"))
            s_quant += float(rq_loss)
            zs.append(z.cpu())
    z = torch.cat(zs, dim=0)
    return normalize_rows(z), s_recon, s_quant, s_recon + a.quant_loss_weight * s_quant


def code_structure(codes):
    """离散码结构：逐深度 prefix 碰撞率 + 逐层用码数。"""
    n = codes.shape[0]
    out = {}
    for k in (1, 2, 3):
        out[f"prefix{k}_distinct"] = int(len(np.unique(codes[:, :k], axis=0)))
        out[f"prefix{k}_collision_rate"] = 1.0 - out[f"prefix{k}_distinct"] / n
    out["L0_distinct"] = int(len(np.unique(codes[:, 0])))
    out["L1_distinct"] = int(len(np.unique(codes[:, 1])))
    out["L2_distinct"] = int(len(np.unique(codes[:, 2])))
    out["full_sid_collision_rate"] = 1.0 - len(np.unique(codes, axis=0)) / n
    return out


def maybe_generate_loss_codes(tag, loss_ckpt, data_path, out_path, device, log):
    """用 best_loss ckpt 生成 C_loss 的 codes —— 调既有 `rqvae/generate_sid_codes_param.py`。"""
    if os.path.exists(out_path):
        log(f"      · C_loss codes 已存在，复用: {out_path}")
        return out_path
    cmd = [sys.executable, "rqvae/generate_sid_codes_param.py",
           "--ckpt_path", loss_ckpt, "--data_path", data_path,
           "--output_file", out_path, "--device", device]
    log(f"      · 生成 C_loss codes: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    for ln in (r.stdout or "").strip().splitlines():
        log(f"          | {ln}")
    if r.returncode != 0:
        log(f"          | ✗ 失败 (rc={r.returncode})\n{r.stderr}")
        return None
    return out_path


# --------------------------------------------------------------------------- main
def stage_main(args, log):
    csv_rows = []
    struct_rows = []
    device = args.device

    for ds in args.datasets:
        log(f"\n{'#' * 100}\n# {ds}\n{'#' * 100}")
        n_items, n_edges, negcg = build_samples(ds, args.data_dir, args.window_size,
                                                args.n_pos, args.n_neg, args.sample_seed)
        log(f"  E 图: {n_items} items | {n_edges} 边 | 正样本 topk={len(negcg['topk'][0])} "
            f"all={len(negcg['all'][0])}（sample_seed={args.sample_seed}，与 g2 同一路径）")

        for seed in args.seeds:
            table = VARIANTS[ds]["TA-SID"]
            if seed not in table:
                log(f"\n  ⚠ [{ds}/{seed}] VARIANTS 无声明 → 跳过")
                continue
            tag, codes_file = table[seed]

            log(f"\n  ▸ {ds} / seed{seed}　tag={tag}　codes={codes_file}")
            coll_ckpt, loss_ckpt, run_dir = find_ckpt_pair(tag)
            if coll_ckpt is None:
                log("    ✗ 缺 ckpt → 跳过")
                continue
            if loss_ckpt is None:
                log(f"    ✗ 缺 best_loss_model.pth（run={run_dir}）→ 停止该格")
                continue
            data_path = resolve_data_path(coll_ckpt)
            if data_path is None:
                log("    ✗ ckpt 的 args.data_path 不存在 → 停止该格")
                continue
            codes_path = os.path.join(args.data_dir, ds, codes_file)
            if not os.path.exists(codes_path):
                log(f"    ✗ 缺 codes（{codes_file}）→ 停止该格")
                continue

            codes_coll = np.load(codes_path)
            if codes_coll.shape[0] != n_items:
                log(f"    ✗ codes 行数 {codes_coll.shape[0]} != n_items {n_items} → 停止该格")
                continue

            # ---- provenance：codes 必须出自 best_collision（判据 = L0 行级一致率）----
            # 在 CPU 上做：`rederive_codes_sequential` 要求 z 与模型同设备，且 g2 的 B 空间
            # 就是 CPU 算的（`stage_probe` 未传 device → 默认 cpu）→ 逐位可比。
            m_coll, _ = load_rqvae_from_ckpt(coll_ckpt, device="cpu")
            z_coll_raw = encode_z_raw(m_coll, data_path, "cpu")
            pred = rederive_codes_sequential(m_coll, z_coll_raw, use_sk=False)
            agree0 = float((codes_coll[:, 0] == pred[:, 0]).mean())
            log(f"    provenance: {codes_file} ← {os.path.basename(run_dir)}/best_collision "
                f"L0 一致率 = {agree0 * 100:.2f}%")
            if agree0 < 0.90:
                log("    ✗✗ codes 与 best_collision 不匹配（<90%）→ **停止该格，不猜测**")
                csv_rows.append({"dataset": ds, "seed": seed, "status": "PROVENANCE_FAIL",
                                 "ckpt_tag": tag, "codes_file": codes_file,
                                 "agree_L0": agree0})
                continue

            # ---- C_loss codes ----
            out_name = f"{tag}_bestloss_sid.npy"
            loss_codes_path = maybe_generate_loss_codes(
                tag, loss_ckpt, data_path, os.path.join(args.data_dir, ds, out_name),
                device, log)
            if loss_codes_path is None:
                log("    ✗ C_loss codes 生成失败 → 停止该格")
                continue
            codes_loss = np.load(loss_codes_path)

            # ---- 三个空间 ----
            z_coll = normalize_rows(z_coll_raw)
            z_loss = encode_z(loss_ckpt, data_path, device="cpu")   # 与 g2 的 B 同一实现/设备
            cb_coll = load_codebook_weights(coll_ckpt)
            cb_loss = load_codebook_weights(loss_ckpt)
            spaces = {
                "C_collision": normalize_rows(rebuild_xq_partial(codes_coll, cb_coll, 3)),
                "C_loss":      normalize_rows(rebuild_xq_partial(codes_loss, cb_loss, 3)),
            }
            z_spaces = {"z_collision": z_coll, "z_loss": z_loss}

            # ---- AUC（逐正/负样本口径）----
            auc_tab = {}
            for pname, (pos_ok, neg_deg, neg_unif) in negcg.items():
                for sname, emb in {**z_spaces, **spaces}.items():
                    sp = cos_sim(emb, pos_ok)
                    auc_tab[(pname, "degmatch", sname)] = auc_of(sp, cos_sim(emb, neg_deg))
                    auc_tab[(pname, "uniform", sname)] = auc_of(sp, cos_sim(emb, neg_unif))

            def A(space, pname=PRIMARY_POS, neg=PRIMARY_NEG):
                return auc_tab[(pname, neg, space)]

            gap_coll = A("z_collision") - A("C_collision")
            gap_loss = A("z_loss") - A("C_loss")
            gap_loss_common_z = A("z_collision") - A("C_loss")
            rec = {
                "dataset": ds, "seed": seed, "status": "OK",
                "ckpt_tag": tag, "run_dir": os.path.basename(run_dir),
                "codes_file": codes_file, "loss_codes_file": out_name,
                "auc_z": A("z_collision"),
                "auc_z_best_loss": A("z_loss"),
                "auc_best_collision": A("C_collision"),
                "auc_best_loss": A("C_loss"),
                "gap_best_collision": gap_coll,
                "gap_best_loss": gap_loss,
                "delta_gap": gap_loss - gap_coll,
                # 敏感性：C_loss 相对**同一个** z_collision 的 gap（设计只给了一列 auc_z）
                "gap_best_loss_common_z": gap_loss_common_z,
                "delta_gap_common_z": gap_loss_common_z - gap_coll,
                # 全部四轴（topk/all × degmatch/uniform）
                "auc_collision_topk_uniform": A("C_collision", "topk", "uniform"),
                "auc_loss_topk_uniform": A("C_loss", "topk", "uniform"),
                "auc_collision_all_degmatch": A("C_collision", "all", "degmatch"),
                "auc_loss_all_degmatch": A("C_loss", "all", "degmatch"),
                "auc_z_all_degmatch": A("z_collision", "all", "degmatch"),
                "auc_z_loss_all_degmatch": A("z_loss", "all", "degmatch"),
                "agree_L0": agree0,
                "n_pos": len(negcg[PRIMARY_POS][0]), "n_neg": len(negcg[PRIMARY_POS][1]),
            }
            csv_rows.append(rec)
            log(f"    AUC(z_collision)={A('z_collision'):.4f}  AUC(C_collision)={A('C_collision'):.4f}"
                f"  gap_collision={gap_coll:+.4f}")
            log(f"    AUC(z_loss)     ={A('z_loss'):.4f}  AUC(C_loss)     ={A('C_loss'):.4f}"
                f"  gap_loss     ={gap_loss:+.4f}")
            log(f"    delta_gap = gap_loss − gap_collision = {rec['delta_gap']:+.4f}"
                f"　（同 z 口径：{rec['delta_gap_common_z']:+.4f}）")

            # ---- 离散码结构 + utilization ----（两个 ckpt 各一份）
            util_coll = compute_utilization(coll_ckpt, data_path, device)
            util_loss = compute_utilization(loss_ckpt, data_path, device)
            for which, ck, codes, util in (("best_collision", coll_ckpt, codes_coll, util_coll),
                                           ("best_loss", loss_ckpt, codes_loss, util_loss)):
                meta = torch.load(ck, map_location="cpu", weights_only=False)
                row = {"dataset": ds, "seed": seed, "which": which, "ckpt_tag": tag,
                       "epoch": meta["epoch"], "ckpt_best_loss": meta["best_loss"],
                       "ckpt_best_collision_rate": meta["best_collision_rate"],
                       "codes_file": codes_file if which == "best_collision" else out_name,
                       **code_structure(codes)}
                for i in (1, 2, 3):
                    k = f"Layer_{i}"
                    row[f"util_L{i-1}_used"] = util[k]["used"] if util else np.nan
                    row[f"util_L{i-1}"] = util[k]["utilization"] if util else np.nan
                struct_rows.append(row)

            # ---- 重建损失 / total loss（eval 全量，trainer 口径）----
            _, r_coll, q_coll, t_coll = z_and_losses(coll_ckpt, data_path, device)
            _, r_loss, q_loss, t_loss = z_and_losses(loss_ckpt, data_path, device)
            csv_rows[-1].update({"recon_loss_best_collision": r_coll,
                                 "recon_loss_best_loss": r_loss,
                                 "quant_loss_best_collision": q_coll,
                                 "quant_loss_best_loss": q_loss,
                                 "total_loss_best_collision": t_coll,
                                 "total_loss_best_loss": t_loss})
            log(f"    recon loss（eval 全量 MSE）: best_collision={r_coll:.4f}  "
                f"best_loss={r_loss:.4f}　（best_loss 相对 Δ={r_loss - r_coll:+.4f}）")
            log(f"    total loss（recon+quant，eval）: best_collision={t_coll:.4f}  "
                f"best_loss={t_loss:.4f}　→ "
                f"{'best_loss 确实是 total loss 更小者（该量≈其 ckpt 字段）' if t_loss < t_coll else '⚠ best_loss 的 total loss 反而更大（字段不反映该 ckpt 实况）'}")
            log(f"    ckpt 元数据: best_collision ep={struct_rows[-2]['epoch']} "
                f"best_loss_field={struct_rows[-2]['ckpt_best_loss']:.6f} | "
                f"best_loss ep={struct_rows[-1]['epoch']} "
                f"best_loss_field={struct_rows[-1]['ckpt_best_loss']:.6f}")

    # ---- 写出 ----
    df = pd.DataFrame(csv_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    df.to_csv(args.csv, index=False, float_format="%.6f")
    log(f"\n→ CSV: {args.csv}  ({df.shape[0]} 行)")
    ds_df = pd.DataFrame(struct_rows)
    ds_df.to_csv(args.struct_csv, index=False, float_format="%.6f")
    log(f"→ 码结构 CSV: {args.struct_csv}  ({ds_df.shape[0]} 行)")
    return df, ds_df


def encode_z_raw(model, data_path, device):
    """z（未归一化）——复用 sf_rq.encode_z（eval 全量，encoder 输出）。"""
    from sf_rq import encode_z as _ez
    return _ez(model, data_path, device=device)


def stage_prov(args, log):
    """E49 Toys 各 variant 的 (ckpt, codes) provenance —— 全交叉矩阵 + orphan 判定。"""
    ds = "Toys"
    variants = [
        ("Content-only", "Toys_content768d_baseline_sid.npy", ["Toys_Content768d_Baseline"]),
        ("Content-only", "Toys_content768d_seed2025_sid.npy", ["Toys_Content768d_seed2025"]),
        ("Content-only", "Toys_content768d_seed2026_sid.npy", ["Toys_Content768d_seed2026"]),
        ("ZeroPad", "Toys_zeropad_seed2024_sid.npy", ["Toys_Zeropad_seed2024"]),
        ("ZeroPad", "Toys_zeropad_seed2025_sid.npy", ["Toys_Zeropad_seed2025"]),
        ("Shuffled", "Toys_shuffled_768d_sid.npy", ["Toys_Shuffled_768d"]),
        ("ItemCF", "Toys_itemcf_768d_sid.npy", ["Toys_ItemCF_768d"]),
        ("Raw-CoOcc", "Toys_rawcooc_768d_sid.npy", ["Toys_RawCoOcc_768d"]),
        ("TA-SID(健康)", "Toys_ta_sid_768d_seed2024_sid.npy",
         ["Toys_TA_SID_768d", "Toys_TA_SID_seed2025", "Toys_TA_SID_seed2026",
          "Toys_TA_SID_skl0_seed2024", "Toys_TA_SID"]),
        ("TA-SID(孤儿·作废)", "Toys_ta_sid_768d_sid.npy",
         ["Toys_TA_SID_768d", "Toys_TA_SID_seed2025", "Toys_TA_SID_seed2026",
          "Toys_TA_SID_skl0_seed2024", "Toys_TA_SID"]),
        ("Random", "Toys_random_sid.npy", ["Toys_Random_Ablation", "Toys_TA_SID_768d"]),
    ]

    # 候选 ckpt（tag → 全部 run × best_collision/best_loss）
    cands = []
    for tag in sorted({t for _, _, tags in variants for t in tags}):
        for d in sorted(glob.glob(f"rqvae/ckpt/{tag}/*/")):
            for which in ("best_collision_model.pth", "best_loss_model.pth"):
                p = os.path.join(d, which)
                if os.path.exists(p):
                    cands.append((tag, os.path.basename(d.rstrip("/")), which, p))
    log(f"候选 ckpt: {len(cands)} 个")

    codes_cache = {}
    for _, f, _ in variants:
        p = os.path.join(args.data_dir, ds, f)
        if os.path.exists(p):
            c = np.load(p)
            codes_cache[f] = c
            log(f"  codes {f:44s} N={c.shape[0]:6d} L0d={len(np.unique(c[:, 0])):3d} "
                f"L1d={len(np.unique(c[:, 1])):3d} L2d={len(np.unique(c[:, 2])):3d}")

    log(f"\n▌ 交叉矩阵（单元 = L0 行级一致率 %；判配对的唯一判据是 L0）")
    hdr = f"  {'ckpt':<58s}" + "".join(f"{f[:20]:>22s}" for f in codes_cache)
    log(hdr)
    rows = []
    agree = {}
    for tag, run, which, p in cands:
        try:
            # 一律 CPU：与 rqvae/verify_variant_pairing.py、audit_main_codes_provenance.py
            # 的既有审计口径一致（sf_rq.encode_z 恒返回 CPU tensor，且 argmin 平局在
            # 不同设备上可能翻转 → 审计读数必须与既有 L0 一致率可直接对照）。
            model, a = load_rqvae_from_ckpt(p, device="cpu")
            dp = a.data_path
            if not os.path.isabs(dp):
                dp = os.path.normpath(os.path.join(ROOT, dp))
            z = encode_z_raw(model, dp, "cpu")
            pred = rederive_codes_sequential(model, z, use_sk=False)
        except Exception as e:                                    # noqa: BLE001
            log(f"  [{tag}/{run}/{which[:13]}] ✗ {e}")
            continue
        cells = []
        for f, c in codes_cache.items():
            if c.shape[0] != z.shape[0]:
                agree[(tag, run, which, f)] = None
                cells.append("        n/a(行数)")
                continue
            fr = float((c[:, 0] == pred[:, 0]).mean())
            agree[(tag, run, which, f)] = fr
            cells.append(f"{fr * 100:20.2f}")
            rows.append({"variant_codes": f, "codes_N": c.shape[0],
                         "codes_L0distinct": int(len(np.unique(c[:, 0]))),
                         "ckpt_tag": tag, "ckpt_run": run, "ckpt_file": which,
                         "ckpt_data_path": str(a.data_path), "agree_L0": fr})
        log(f"  {tag + '/' + run + '/' + which[:13]:<58s}" + "".join(f"{x:>22s}" for x in cells))

    log(f"\n▌ 逐 variant 判定（L0 ≥ 90% = 配对成立；无任何 ckpt ≥ 90% = **orphan**）")
    verdicts = []
    for name, f, tags in variants:
        if f not in codes_cache:
            log(f"  {name:<18s} {f:44s} ❌ codes 文件不存在（该变体未跑）")
            verdicts.append({"variant": name, "codes_file": f, "status": "NO_CODES",
                             "codes_L0distinct": np.nan, "best_ckpt": "", "agree_L0": np.nan})
            continue
        hits = [(k, v) for k, v in agree.items() if k[3] == f and v is not None]
        hits.sort(key=lambda kv: -kv[1])
        best = hits[0] if hits else None
        declared = [t for t in tags]
        if best is None:
            st, bt, bv = "ORPHAN", "", np.nan
        elif best[1] > 0.90:
            bt, bv = f"{best[0][0]}/{best[0][1]}/{best[0][2][:13]}", best[1]
            st = "OK" if best[0][0] in declared else "OK(来源≠声明)"
        else:
            bt, bv = f"{best[0][0]}/{best[0][1]}/{best[0][2][:13]}", best[1]
            st = "ORPHAN"
        log(f"  {name:<18s} {f:44s} L0d={len(np.unique(codes_cache[f][:, 0])):3d} "
            f"{st:<14s} 最佳={bt or '—'}  agree={'' if np.isnan(bv) else f'{bv*100:.2f}%'}")
        verdicts.append({"variant": name, "codes_file": f, "status": st,
                         "codes_L0distinct": int(len(np.unique(codes_cache[f][:, 0]))),
                         "best_ckpt": bt, "agree_L0": bv})

    vdf = pd.DataFrame(verdicts)
    os.makedirs(os.path.dirname(os.path.abspath(args.prov_csv)), exist_ok=True)
    vdf.to_csv(args.prov_csv, index=False, float_format="%.6f")
    pd.DataFrame(rows).to_csv(args.prov_csv.replace(".csv", "_matrix.csv"),
                              index=False, float_format="%.6f")
    log(f"\n→ provenance CSV: {args.prov_csv} / {args.prov_csv.replace('.csv', '_matrix.csv')}")
    return vdf


def main():
    ap = argparse.ArgumentParser(description="RQ-VAE checkpoint selection probe")
    ap.add_argument("--stage", choices=["main", "prov", "both"], default="both")
    ap.add_argument("--datasets", type=str, default="Beauty,Sport,Toys")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sample_seed", type=int, default=SEED)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv", type=str, default="doc/checkpoint_selection_behavior_auc.csv")
    ap.add_argument("--struct_csv", type=str,
                    default="doc/checkpoint_selection_code_structure.csv")
    ap.add_argument("--prov_csv", type=str, default="doc/checkpoint_selection_toys_provenance.csv")
    ap.add_argument("--log", type=str, default="doc/checkpoint_selection_probe.log")
    args = ap.parse_args()
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    log("=" * 100)
    log("RQ-VAE checkpoint selection 对行为结构保持的影响")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　device={args.device}")
    log("自变量: best_collision_model.pth（现役） vs best_loss_model.pth（对照）")
    log(f"口径: E 图 window={args.window_size} / 正 topk(=1000) 主 + all 副 / "
        f"负 degmatch 主 + uniform 副 / cosine AUC（逐位复用 g2_stage_probe，未新定义）")
    log("空间: B=z（编码器输出，两臂各自的 encoder） | C=xq[full]（三层码本重建）")
    log("=" * 100)

    if args.stage in ("main", "both"):
        stage_main(args, log)
    if args.stage in ("prov", "both"):
        log(f"\n{'=' * 100}\nE49 Toys 各 variant 的 checkpoint provenance\n{'=' * 100}")
        stage_prov(args, log)

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n→ log: {args.log}")


if __name__ == "__main__":
    main()
