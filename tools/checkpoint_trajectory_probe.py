#!/usr/bin/env python3
"""checkpoint_trajectory_probe.py — Checkpoint Trajectory 实验的指标计算（只读）

对应实验方案：`doc/exp/checkpoint_trajectory/experiment_plan.md`

**唯一自变量**：同一次 training run 内的 training epoch（50 / 500 / 1000 / 2000）。
其余全部冻结 —— 输入 / 架构 / latent / codebook / 训练超参 / **pair sampling** /
**AUC 定义** / **quantization+codegen 路径** 一律复用既有实现，未新定义任何量。

三个空间（口径逐位复用 g2 与 checkpoint_selection，未新定义 Behavior AUC）：
    B = z         编码器输出（量化前 latent，32d）  → AUC(z)
    C = xq[full]  三层码本重建（用 codegen 产出的 codes 索引）→ AUC(xq)
    C_native      原生（未做 Sinkhorn 缓解）码重建  → **审计列**，用于拆开
                  「量化误差」与「碰撞缓解代价」两个混淆项

复用（不复制）：
    tools/stage_probe.py                 : build_samples（pair sampling，每数据集只调用一次）
    tools/behavior_structure_probe.py    : cos_sim / auc_of / DATA_DIR / N_POS / N_NEG / SEED
    tools/sid_per_layer_diagnostic.py    : normalize_rows / rebuild_xq_partial
    tools/sid_consistency_contrast.py    : load_codebook_weights / find_best_ckpt
    rqvae/sf_rq.py                       : load_rqvae_from_ckpt / encode_z / rederive_codes_sequential
    tools/sid_quality.py                 : compute_utilization
    rqvae/models/rqvae.py                : compute_loss（loss 定义，与 trainer 完全同源）
    rqvae/trainer.py                     : _train_epoch 的累加口径（total_loss / total_recon_loss）

用法（仓库根）：
    python3 tools/checkpoint_trajectory_probe.py                     # 主实验
    python3 tools/checkpoint_trajectory_probe.py --stage toys_audit  # Toys orphan/污染审计
"""

import argparse
import datetime
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
os.chdir(ROOT)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "rqvae"))

from behavior_structure_probe import (  # noqa: E402
    cos_sim, auc_of, DATA_DIR, N_POS, N_NEG, SEED,
)
from stage_probe import build_samples  # noqa: E402
from sid_per_layer_diagnostic import normalize_rows, rebuild_xq_partial  # noqa: E402
from sid_consistency_contrast import load_codebook_weights  # noqa: E402
from sf_rq import load_rqvae_from_ckpt, encode_z, rederive_codes_sequential  # noqa: E402
from sid_quality import compute_utilization  # noqa: E402

TRAJ_ROOT = "rqvae/checkpoint_trajectory"
TRAJ_CKPT_ROOT = os.path.join(TRAJ_ROOT, "ckpt")
TRAJ_CODE_ROOT = os.path.join(TRAJ_ROOT, "codes")
MANIFEST_DIR = os.path.join(TRAJ_ROOT, "manifest.d")
LEGACY_CKPT_ROOT = os.path.join("rqvae", "ckpt")

EPOCHS = [50, 500, 1000, 2000]
PRIMARY_POS, PRIMARY_NEG = "topk", "degmatch"     # 与 g2 / checkpoint_selection 主口径一致
DATASETS = ["Beauty", "Sport", "Toys"]
# Toys 已知的两个历史孤儿 codes（e48 / checkpoint_selection 已裁定作废）
TOYS_KNOWN_ORPHANS = ["Toys_ta_sid_768d_sid.npy", "Toys_random_sid.npy"]


# --------------------------------------------------------------------------- utils
def ep_ckpt_name(ep):
    return "traj_epoch%05d.pth" % ep


def ep_code_name(tag, ep):
    return os.path.join(TRAJ_CODE_ROOT, "%s_ep%d_sid.npy" % (tag, ep))


def load_manifest():
    """读 manifest.d/*.tsv（每 tag 一个片段，由 run_checkpoint_trajectory.sh 写出）。"""
    rows = []
    for p in sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.tsv"))):
        tag = os.path.basename(p)[:-4]
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                parts = ln.split("\t")
                if len(parts) != 5:
                    continue
                t, ds, seed, data_path, run_dir = parts
                rows.append({"tag": t, "dataset": ds, "seed": int(seed),
                             "data_path": data_path, "run_dir": run_dir.rstrip("/")})
    return rows


def code_structure(codes):
    """离散码结构：逐深度 prefix 碰撞率 + 逐层用码数。"""
    n = codes.shape[0]
    out = {}
    for k in (1, 2, 3):
        out["prefix%d_distinct" % k] = int(len(np.unique(codes[:, :k], axis=0)))
        out["prefix%d_collision_rate" % k] = 1.0 - out["prefix%d_distinct" % k] / n
    out["L0_distinct"] = int(len(np.unique(codes[:, 0])))
    out["L1_distinct"] = int(len(np.unique(codes[:, 1])))
    out["L2_distinct"] = int(len(np.unique(codes[:, 2])))
    out["full_sid_collision_rate"] = 1.0 - len(np.unique(codes, axis=0)) / n
    return out


@torch.no_grad()
def eval_losses(ckpt_path, data_path, device="cpu", batch_size=1024):
    """recon / quant / total loss —— **与 trainer._train_epoch 完全同口径**。

    trainer 用 `out, rq_loss, _ = self.model(data, return_z=False)`（use_sk 默认 True）
    再 `self.model.compute_loss(out, rq_loss, xs=data)`，逐 batch 取 `.item()` **累加**。
    本函数逐字复刻该调用序列与累加方式（reduction='mean' 逐 batch，再求和），
    不重新定义 loss。qlw = args.quant_loss_weight（本项目为 1.0）。
    """
    from torch.utils.data import DataLoader
    from datasets import EmbDataset

    model, a = load_rqvae_from_ckpt(ckpt_path, device=device)
    data = EmbDataset(data_path)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    s_total = s_recon = s_quant = 0.0
    for batch in loader:
        x = batch.to(device)
        out, rq_loss, _ = model(x, return_z=False)
        loss, loss_recon = model.compute_loss(out, rq_loss, xs=x)
        s_total += float(loss.item())
        s_recon += float(loss_recon.item())
        s_quant += float(rq_loss.item())
    return s_recon, s_quant, s_total, float(a.quant_loss_weight)


def auc_axes(emb_n, negcg):
    """4 轴 AUC：{topk,all} × {degmatch,uniform}。返回 {(pos,neg): auc}。"""
    out = {}
    for pname, (pos_ok, neg_deg, neg_unif) in negcg.items():
        sp = cos_sim(emb_n, pos_ok)
        out[(pname, "degmatch")] = auc_of(sp, cos_sim(emb_n, neg_deg))
        out[(pname, "uniform")] = auc_of(sp, cos_sim(emb_n, neg_unif))
    return out


# --------------------------------------------------------------------------- main
def stage_main(args, log):
    ckpt = args.traj_ckpt_root
    manifest = load_manifest()
    if not manifest:
        log("✗ manifest 为空（先跑 rqvae/run_checkpoint_trajectory.sh）→ 退出")
        return None, None

    log(f"manifest: {len(manifest)} 个 run")
    for m in manifest:
        log("   %-24s ds=%-6s seed=%s  run_dir=%s" % (m["tag"], m["dataset"], m["seed"], m["run_dir"]))

    rows, repro_rows = [], []
    sample_cache = {}

    for ds in DATASETS:
        runs = [m for m in manifest if m["dataset"] == ds]
        if not runs:
            log(f"\n⚠ {ds}: manifest 无该数据集的 run → 跳过")
            continue

        # ---- pair sampling：**每数据集只调用一次**，4 个 epoch × 3 个 seed 共用 ----
        if ds not in sample_cache:
            n_items, n_edges, negcg = build_samples(
                ds, args.data_dir, args.window_size, args.n_pos, args.n_neg, args.sample_seed)
            sample_cache[ds] = (n_items, n_edges, negcg)
        n_items, n_edges, negcg = sample_cache[ds]

        log(f"\n{'#' * 100}\n# {ds}　n_items={n_items}　E 边={n_edges}　"
            f"n_pos(topk)={len(negcg['topk'][0])}　n_neg(degmatch)={len(negcg['topk'][1])}"
            f"　（sample_seed={args.sample_seed}，全部 epoch/seed 共用同一组样本）\n{'#' * 100}")

        for m in sorted(runs, key=lambda r: r["seed"]):
            tag, seed, data_path = m["tag"], m["seed"], m["data_path"]
            run_dir = m["run_dir"]
            log(f"\n  ▸ {ds} / seed{seed}　tag={tag}")

            for ep in EPOCHS:
                ck = os.path.join(run_dir, ep_ckpt_name(ep))
                cd = ep_code_name(tag, ep)
                if not os.path.exists(ck) or not os.path.exists(cd):
                    log(f"    ✗ ep{ep}: 缺 ckpt 或 codes（{ck} / {cd}）→ 跳过该点")
                    continue

                codes = np.load(cd)
                if codes.shape[0] != n_items:
                    log(f"    ✗ ep{ep}: codes 行数 {codes.shape[0]} != n_items {n_items} → 跳过")
                    continue

                meta = torch.load(ck, map_location="cpu", weights_only=False)
                model, a = load_rqvae_from_ckpt(ck, device="cpu")
                cb = load_codebook_weights(ck)

                z_raw = encode_z(model, data_path, device="cpu")
                z_n = normalize_rows(z_raw)
                xq_n = normalize_rows(rebuild_xq_partial(codes, cb, 3))

                # 原生（未缓解）码：审计列
                native = rederive_codes_sequential(model, z_raw, use_sk=False)
                xq_nat_n = normalize_rows(rebuild_xq_partial(native, cb, 3))
                agree0 = float((codes[:, 0] == native[:, 0]).mean())

                az = auc_axes(z_n, negcg)
                ax = auc_axes(xq_n, negcg)
                an = auc_axes(xq_nat_n, negcg)

                def A(tab, p=PRIMARY_POS, n=PRIMARY_NEG):
                    return tab[(p, n)]

                r_loss, q_loss, t_loss, qlw = eval_losses(ck, data_path, device="cpu")
                util = compute_utilization(ck, data_path, "cpu")
                st = code_structure(codes)

                row = {
                    "dataset": ds, "seed": seed, "tag": tag,
                    "epoch": ep, "ckpt_epoch_idx": meta.get("epoch", ep - 1),
                    "run_dir": run_dir, "ckpt_path": ck, "codes_file": cd,
                    "data_path": data_path,
                    # ---- Structure ----
                    "auc_z": A(az), "auc_xq": A(ax), "z_gap": A(az) - A(ax),
                    # 审计：原生（未 Sinkhorn 缓解）码口径
                    "auc_xq_native": A(an), "z_gap_native": A(az) - A(an),
                    "codes_L0_agree_native": agree0,
                    # 其余 AUC 轴
                    "auc_z_topk_uniform": az[("topk", "uniform")],
                    "auc_xq_topk_uniform": ax[("topk", "uniform")],
                    "auc_z_all_degmatch": az[("all", "degmatch")],
                    "auc_xq_all_degmatch": ax[("all", "degmatch")],
                    # ---- Training（trainer 同口径）----
                    "recon_loss": r_loss, "quant_loss": q_loss, "total_loss": t_loss,
                    "quant_loss_weight": qlw,
                    "train_loss_logged": meta.get("train_loss", np.nan),
                    "train_recon_loss_logged": meta.get("train_recon_loss", np.nan),
                    "valid_collision_rate": meta.get("valid_collision_rate", np.nan),
                    # ---- Codebook ----
                    "n_pos": len(negcg[PRIMARY_POS][0]), "n_neg": len(negcg[PRIMARY_POS][1]),
                }
                for i in (1, 2, 3):
                    k = "Layer_%d" % i
                    row["util_L%d_used" % (i - 1)] = util[k]["used"] if util else np.nan
                    row["util_L%d" % (i - 1)] = util[k]["utilization"] if util else np.nan
                row.update(st)
                rows.append(row)

                log("    ep%-5d AUC(z)=%.4f  AUC(xq)=%.4f  z-gap=%+.4f　|　原生 z-gap=%+.4f"
                    "　recon=%.4f  quant=%.4f  total=%.4f　L0util=%.3f  prefix3碰撞=%.4f"
                    % (ep, row["auc_z"], row["auc_xq"], row["z_gap"], row["z_gap_native"],
                       r_loss, q_loss, t_loss,
                       row["util_L0"], row["prefix3_collision_rate"]))

            # ---- 逐位复现校验：与既有已发表 ckpt 对照 ----
            repro_rows.extend(reproduction_check(tag, run_dir, log))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "seed", "epoch"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    df.to_csv(args.csv, index=False, float_format="%.6f")
    log(f"\n→ CSV: {args.csv}  ({df.shape[0]} 行)")

    rdf = pd.DataFrame(repro_rows)
    if not rdf.empty:
        rdf.to_csv(args.repro_csv, index=False, float_format="%.6f")
        log(f"→ 复现校验 CSV: {args.repro_csv}  ({rdf.shape[0]} 行)")

    # ---- 逐格 trend（Spearman(epoch, metric)，n=4）----
    log(f"\n{'=' * 100}\n逐格 trend：Spearman(epoch, metric)，n=4（仅描述方向，不做显著性宣称）\n{'=' * 100}")
    log("  %-8s %-6s | %-26s %-26s %-22s" % ("dataset", "seed", "recon↓ 与 z_gap↑",
                                              "recon↓ 与 z_gap_native↑", "recon 与 auc_z"))
    for (ds, seed), g in df.groupby(["dataset", "seed"]):
        g = g.sort_values("epoch")
        if len(g) < 3:
            continue
        # r_recon<0 表示 recon 随 epoch 下降；r_gap>0 表示 z-gap 随 epoch 上升
        # 「recon↓ 伴随 z-gap↑」⇔ r_recon<0 且 r_gap>0
        r_recon = spearmanr(g["epoch"], g["recon_loss"]).statistic
        r_gap = spearmanr(g["epoch"], g["z_gap"]).statistic
        r_gapn = spearmanr(g["epoch"], g["z_gap_native"]).statistic
        r_z = spearmanr(g["epoch"], g["auc_z"]).statistic

        def verdict(rg):
            if r_recon < 0 and rg > 0:
                return "✅ 是 (ρ_gap=%+.2f)" % rg
            if r_recon > 0 and rg < 0:
                return "逆反 (ρ_gap=%+.2f)" % rg
            return "否/无趋势 (ρ_gap=%+.2f)" % rg

        log("  %-8s %-6d | %-26s %-26s %s (ρ=%+.2f)"
            % (ds, seed, verdict(r_gap), verdict(r_gapn),
               "同向" if r_z > 0 else "反向", r_z))
    return df, pd.DataFrame(repro_rows)


def reproduction_check(tag, traj_run_dir, log):
    """逐位复现校验：traj_epoch{N} 与既有已发表 ckpt（同 epoch_idx）做 state_dict 逐位比较。

    既有 run 目录里凡是 `epoch` 字段 == N-1 的 .pth 都是候选（同一次评估点落盘，状态相同）。
    9/9 逐位相同 ⇒ 本次重训就是既有已发表 run 的同一条轨迹。
    """
    legacy_dirs = sorted(glob.glob(os.path.join(LEGACY_CKPT_ROOT, tag, "*/")),
                         key=lambda p: os.path.getmtime(p))
    if not legacy_dirs:
        log(f"    ⚠ 复现校验：找不到既有 run 目录 rqvae/ckpt/{tag}/*/ → 无法校验")
        return []

    rows = []
    for ep in EPOCHS:
        tck = os.path.join(traj_run_dir, ep_ckpt_name(ep))
        if not os.path.exists(tck):
            continue
        tstate = torch.load(tck, map_location="cpu", weights_only=False)["state_dict"]
        target_epoch = ep - 1
        for ld in legacy_dirs:
            for cand in sorted(glob.glob(os.path.join(ld, "*.pth"))):
                try:
                    c = torch.load(cand, map_location="cpu", weights_only=False)
                except Exception:                     # noqa: BLE001
                    continue
                if c.get("epoch") != target_epoch:
                    continue
                cstate = c["state_dict"]
                same = (set(tstate) == set(cstate)) and all(
                    tstate[k].shape == cstate[k].shape and torch.equal(tstate[k], cstate[k])
                    for k in tstate)
                maxdiff = 0.0
                if not same:
                    for k in tstate:
                        if k in cstate and tstate[k].shape == cstate[k].shape:
                            maxdiff = max(maxdiff,
                                          float((tstate[k].float() - cstate[k].float()).abs().max()))
                rows.append({
                    "tag": tag, "epoch": ep, "epoch_idx": target_epoch,
                    "legacy_ckpt": cand, "bit_identical": bool(same),
                    "max_abs_diff": maxdiff,
                })
                log("    复现校验 ep%-5d vs %s : %s"
                    % (ep, os.path.relpath(cand, ROOT),
                       "✅ 逐位相同" if same else "❌ 不一致 (max|Δ|=%.3e)" % maxdiff))
    if not rows:
        log(f"    ⚠ 复现校验：既有 {tag} 目录下无与 ep{EPOCHS} 同 epoch_idx 的 ckpt → 该 tag 无法校验")
    return rows


# --------------------------------------------------------------------------- Toys 审计
def stage_toys_audit(args, log):
    """要求 #7：Toys 是否存在历史 orphan SID / checkpoint 污染。"""
    log("=" * 100)
    log("Toys orphan SID / checkpoint 污染审计")
    log("=" * 100)

    # (1) 既有 Toys ckpt 目录的重复 run 目录
    log("\n▌ 1. Toys ckpt 目录结构（run 目录 >1 即为重复训练残留）")
    dup_tags = []
    for d in sorted(glob.glob(os.path.join(LEGACY_CKPT_ROOT, "Toys*"))):
        subs = sorted(glob.glob(os.path.join(d, "*/")), key=lambda p: os.path.getmtime(p))
        flag = "  ⚠ 重复" if len(subs) > 1 else ""
        if len(subs) > 1:
            dup_tags.append(os.path.basename(d))
        log("  %-32s runs=%d%s" % (os.path.basename(d), len(subs), flag))
        for s in subs:
            log("      %s  (mtime %s)" % (os.path.basename(s.rstrip("/")),
                                          datetime.datetime.fromtimestamp(os.path.getmtime(s))
                                          .strftime("%Y-%m-%d %H:%M")))
    log("  ⇒ find_best_ckpt / find_ckpt_pair 按 mtime 取**最后一个** run 目录。"
        " 上列 mtime 最大者即既有审计口径实际使用的目录。重复 tag: %s" % (dup_tags or "无"))

    # (2) 已知孤儿：用**本次新训的** checkpoints 再当一次候选，确认它们仍是孤儿
    log("\n▌ 2. 已知孤儿 codes 复核（候选集 = 本次新训的 36 个 traj ckpt + 既有 Toys ckpt）")
    cands = []
    for m in load_manifest():
        if m["dataset"] != "Toys":
            continue
        for ep in EPOCHS:
            p = os.path.join(m["run_dir"], ep_ckpt_name(ep))
            if os.path.exists(p):
                cands.append(("%s/ep%d" % (m["tag"], ep), p, m["data_path"]))
    for d in sorted(glob.glob(os.path.join(LEGACY_CKPT_ROOT, "Toys*"))):
        for s in sorted(glob.glob(os.path.join(d, "*/"))):
            p = os.path.join(s, "best_collision_model.pth")
            if os.path.exists(p):
                cands.append(("%s/%s" % (os.path.basename(d), os.path.basename(s.rstrip("/"))),
                              p, None))
    log("  候选 ckpt 数: %d" % len(cands))

    cache = {}
    for f in TOYS_KNOWN_ORPHANS:
        p = os.path.join(args.data_dir, "Toys", f)
        if not os.path.exists(p):
            log("  %-30s ❌ codes 文件不存在" % f)
            continue
        c = np.load(p)
        best = (None, -1.0)
        for name, ck, dp in cands:
            try:
                if dp is None:
                    mm, a = load_rqvae_from_ckpt(ck, device="cpu")
                    dp2 = a.data_path
                    if not os.path.isabs(dp2):
                        dp2 = os.path.normpath(os.path.join(ROOT, dp2))
                else:
                    dp2 = dp
                    if (ck, dp2) in cache:
                        pred = cache[(ck, dp2)]
                        if pred.shape[0] == c.shape[0]:
                            fr = float((c[:, 0] == pred[:, 0]).mean())
                            if fr > best[1]:
                                best = (name, fr)
                        continue
                    mm, _ = load_rqvae_from_ckpt(ck, device="cpu")
                if not os.path.exists(dp2):
                    continue
                z = encode_z(mm, dp2, device="cpu")
                pred = rederive_codes_sequential(mm, z, use_sk=False)
                cache[(ck, dp2)] = pred
                if pred.shape[0] != c.shape[0]:
                    continue
                fr = float((c[:, 0] == pred[:, 0]).mean())
                if fr > best[1]:
                    best = (name, fr)
            except Exception:                          # noqa: BLE001
                continue
        verdict = "ORPHAN（仍无匹配）" if best[1] < 0.90 else "⚠ 找到匹配，需复核"
        log("  %-30s L0distinct=%3d  最佳匹配=%-52s L0一致率=%.2f%%  → %s"
            % (f, len(np.unique(c[:, 0])), best[0] or "—", best[1] * 100, verdict))

    # (3) 本次产物不构成新污染：既有路径 mtime 快照
    log("\n▌ 3. 本次实验是否污染既有审计路径")
    log("  本次全部产物根目录: %s" % TRAJ_ROOT)
    log("  ⇒ 不在 rqvae/ckpt/ 之下 ⇒ 既有 find_best_ckpt/find_ckpt_pair 的")
    log("     `rqvae/ckpt/{tag}/*/` glob 不会命中本次重训的 timestamp 目录。")
    log("\n  证据 A：实验开始后 rqvae/ckpt/ 与 data/ 下被改动过的文件（应为空）")
    marker = os.path.join(TRAJ_ROOT, ".experiment_start")
    if os.path.exists(marker):
        import subprocess
        for root_dir in (LEGACY_CKPT_ROOT, args.data_dir):
            r = subprocess.run(["find", root_dir, "-newer", marker, "-type", "f"],
                               capture_output=True, text=True)
            hits = [x for x in r.stdout.strip().splitlines() if x]
            log("    %-14s : %d 个文件被改动 %s" % (root_dir, len(hits),
                                                    ("→ " + str(hits[:5])) if hits else "✅"))
    else:
        log("    ⚠ 缺 %s（无法做 mtime 判据；该 marker 由 run_checkpoint_trajectory.sh 创建）" % marker)

    log("\n  证据 B：本实验 codes 文件名与既有 data/{DS}/*_sid.npy 命名空间不相交")
    new_codes = sorted(os.path.basename(p) for p in glob.glob(os.path.join(TRAJ_CODE_ROOT, "*.npy")))
    old_toys = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data_dir, "Toys", "*_sid.npy")))
    log("    本实验 codes 目录: %s（%d 个文件）" % (TRAJ_CODE_ROOT, len(new_codes)))
    log("    既有 Toys codes 目录: %s/Toys/（%d 个文件）" % (args.data_dir, len(old_toys)))
    log("    交集: %s" % (sorted(set(new_codes) & set(old_toys)) or "空 ✅"))
    log("    示例 本实验: %s" % (new_codes[:4] or "—"))
    log("    示例 既有  : %s" % (old_toys[:4] or "—"))


# --------------------------------------------------------------------------- io
def main():
    ap = argparse.ArgumentParser(description="Checkpoint Trajectory 实验指标计算")
    ap.add_argument("--stage", choices=["main", "toys_audit", "both"], default="both")
    ap.add_argument("--sample_seed", type=int, default=SEED,
                    help="E 图正负样本采样 RNG 种子（固定，勿随训练种子变）")
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--data_dir", type=str, default=DATA_DIR)
    ap.add_argument("--n_pos", type=int, default=N_POS)
    ap.add_argument("--n_neg", type=int, default=N_NEG)
    ap.add_argument("--traj_ckpt_root", type=str, default=TRAJ_CKPT_ROOT)
    ap.add_argument("--csv", type=str, default="doc/checkpoint_trajectory_results.csv")
    ap.add_argument("--repro_csv", type=str, default="doc/checkpoint_trajectory_repro.csv")
    ap.add_argument("--log", type=str, default="doc/checkpoint_trajectory_probe.log")
    args = ap.parse_args()

    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    log("=" * 100)
    log("Checkpoint Trajectory 实验 —— 指标计算（只读）")
    log(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("自变量: 同一次 training run 内的 training epoch ∈ {50, 500, 1000, 2000}")
    log(f"口径: E 图 window={args.window_size} / 正 topk(=1000) 主 + all 副 / "
        f"负 degmatch 主 + uniform 副 / cosine AUC（逐位复用 g2_stage_probe，未新定义）")
    log("pair sampling: 每数据集只调用一次 build_samples，全部 epoch × seed 共用同一组样本")
    log("空间: B=z（编码器输出）| C=xq[full]（三层码本重建，用 codegen codes）"
        " | C_native（审计：未做 Sinkhorn 缓解的原生码）")
    log("=" * 100)

    if args.stage in ("main", "both"):
        stage_main(args, log)
    if args.stage in ("toys_audit", "both"):
        log("")
        stage_toys_audit(args, log)

    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n→ log: {args.log}")


if __name__ == "__main__":
    main()
