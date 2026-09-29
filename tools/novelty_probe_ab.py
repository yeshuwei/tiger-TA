"""
novelty_probe_ab.py — 零 GPU 决策探针：臂 A vs 臂 B

用途
----
在投入 GPU 之前，用**已有产物**判断两条候选路线哪条更可能产出可发表的结论：
  臂 A：用学习式行为嵌入（SASRec/LightGCN item embedding）替换 PPMI+SVD，走同一 LN+Concat
  臂 B：损失级注入（LETTER 式对比损失对齐 ẑ 与行为嵌入），输入端回内容-only

设计见 doc/prior_art_novelty_audit.md §4。两个探针：

P1（判臂 A）—— 三嵌入互补性
    SASRec item embedding 能线性重建多少「行为信号」和多少「内容信号」？
    - 若 R²(E→H) 高 ⇒ SASRec emb 已包含 PPMI 行为通道的信息 ⇒ 臂 A 大概率打平（≈ TA-SID）
      ⇒ D1（"手工转移图特殊"）死 ⇒ 应选臂 B
    - 若 R²(E→H) 低 ⇒ 两者信息不同 ⇒ 臂 A 有区分度

P2（判臂 B）—— 量化瓶颈对行为信息的选择性保留率
    s_H = R²(ẑ→H) / R²(z→H)   行为信息在量化前后的存活率
    s_C = R²(ẑ→C) / R²(z→C)   内容信息在量化前后的存活率
    - 若 s_H ≪ s_C ⇒ 量化瓶颈**不成比例地**破坏行为信息 ⇒ 损失级对齐正是对症解法
      ⇒ 臂 B 有真实 headroom
    - 若 s_H ≈ s_C ⇒ 瓶颈对两类信号一视同仁 ⇒ 输入级 concat 已足够 ⇒ 臂 B 无增量

两者都是**描述性诊断**，不预测 NDCG（E49 已证 AUC 类指标不能作下游筛选，见
doc/e49grid_diag_summary.md）。判据只用于**二选一的方向决策**，不替代正式实验。

用法
----
    python3 tools/novelty_probe_ab.py --dataset Sport
    python3 tools/novelty_probe_ab.py --dataset Sport --ta_sid_ckpt <path>
    python3 tools/novelty_probe_ab.py --dataset all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rqvae.models.rqvae import RQVAE  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认 ckpt：与 3-seed 主实验同源的 seed2025 TA-SID RQ-VAE（768d 口径）
DEFAULT_CKPT = {
    "Sport": "rqvae/ckpt/Sport_TA_SID_seed2025/Sep-04-2026_16-46-49/best_collision_model.pth",
    "Toys": "rqvae/ckpt/Toys_TA_SID_seed2025/Sep-03-2026_22-07-02/best_collision_model.pth",
}
SASREC_CKPT = {
    "Sport": "sasrec_output/Sport/seed2025_best.pth",
    "Toys": "sasrec_output/Toys/seed2025_best.pth",
    "Beauty": "sasrec_output/Beauty/seed2025_best.pth",
}

# ⚠️ 行为嵌入的正确来源 —— 三数据集文件名不一致，且「_ppmi」后缀的那个**不是** TA-SID 用的那份。
#   - behavior_emb.parquet      = TA-SID 实际产物（build_transition_graph.py:210
#                                 "Original Track A: PPMI + SVD → behavior_emb.parquet"）
#   - behavior_emb_ppmi.parquet = behavior_variants.py --variant ppmi 产出的**自检副本**，
#                                 与上一份的 128d 原始对象逐元素 cos≈0（仅 top-32 PCA
#                                 子空间相同 ⇒ 秩匹配后的 R² 才会一致）。
#   - Beauty 目录**没有** _ppmi 文件；旧代码写死 "_ppmi" 会在 Beauty 上直接崩溃。
# 详见 doc/exp/ab_decision_and_design.md §4.2。
BEHAVIOR_FILE = {
    "Sport": "behavior_emb.parquet",
    "Toys": "behavior_emb.parquet",
    "Beauty": "behavior_emb.parquet",
}


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def load_emb(dataset, filename, col="embedding"):
    """读 parquet → (ItemID 升序的 ndarray, ItemID 数组)"""
    df = pd.read_parquet(PROJECT_ROOT / "data" / dataset / filename)
    df = df.sort_values("ItemID").reset_index(drop=True)
    mat = np.stack(df[col].values).astype(np.float64)
    return mat, df["ItemID"].values


def cv_r2(X, Y, n_folds=5, seed=0):
    """5 折交叉验证的 OLS R²（X 预测 Y，两折内标准化，返回 pooled R²）。

    Y 的维度需与 X 可比：调用方负责先把 Y 降到与 X 同秩（见 pca_reduce）。
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)

    ss_res = 0.0
    ss_tot = 0.0
    for k in range(n_folds):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != k])

        mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-8
        mu_y, sd_y = Y[tr].mean(0), Y[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu_x) / sd_x
        Xte = (X[te] - mu_x) / sd_x
        Ytr = (Y[tr] - mu_y) / sd_y
        Yte = (Y[te] - mu_y) / sd_y

        A = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
        B, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
        pred = np.hstack([Xte, np.ones((Xte.shape[0], 1))]) @ B

        ss_res += float(((Yte - pred) ** 2).sum())
        ss_tot += float((Yte**2).sum())  # 已中心化 ⇒ 以 0 为基线
    return 1.0 - ss_res / ss_tot


def pca_reduce(X, k, seed=0):
    """按秩 k 降维（fit 在全量上，仅作目标空间的秩匹配用；见报告 caveat）"""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def build_rqvae_from_ckpt(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    a = vars(a) if not isinstance(a, dict) else a
    in_dim = ck["state_dict"]["encoder.mlp_layers.1.weight"].shape[1]
    model = RQVAE(
        in_dim=in_dim,
        num_emb_list=a["num_emb_list"],
        e_dim=a["e_dim"],
        layers=a["layers"],
        dropout_prob=0.0,
        bn=False,
        loss_type=a.get("loss_type", "mse"),
        kmeans_init=True,
        kmeans_iters=1,
        sk_epsilons=a.get("sk_epsilons"),
        sk_iters=1,
    )
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, a


@torch.no_grad()
def encode(model, X, device="cpu", batch=4096):
    """返回 (z 潜变量, z_hat 量化后表示, indices)"""
    zs, qs, ids = [], [], []
    Xt = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch):
        xb = Xt[i:i + batch].to(device)
        z = model.encoder(xb)
        x_q, _, indices = model.rq(z)
        zs.append(z.cpu().numpy().astype(np.float64))
        qs.append(x_q.cpu().numpy().astype(np.float64))
        ids.append(indices.cpu().numpy())
    return np.vstack(zs), np.vstack(qs), np.vstack(ids)


# --------------------------------------------------------------------------- #
# P1：三嵌入互补性（判臂 A）
# --------------------------------------------------------------------------- #
def probe_p1(dataset, rank=32):
    print(f"\n{'='*78}\nP1 三嵌入互补性（臂 A 判据） — {dataset}\n{'='*78}")
    C, _ = load_emb(dataset, "item_emb_768d.parquet")
    H, _ = load_emb(dataset, BEHAVIOR_FILE[dataset])
    sas_path = PROJECT_ROOT / SASREC_CKPT[dataset]
    sd = torch.load(sas_path, map_location="cpu", weights_only=False)
    E = sd["item_emb.weight"].numpy()[1:].astype(np.float64)  # index 0 = padding
    print(f"  content {C.shape} | behavior {H.shape} | SASRec emb {E.shape}")

    n = min(len(C), len(H), len(E))
    C, H, E = C[:n], H[:n], E[:n]

    k = min(rank, E.shape[1])
    H_k = pca_reduce(H, rank)
    C_k = pca_reduce(C, rank)
    E_k = pca_reduce(E, k)

    # 行为/内容自身可分性上界（同一目标秩下的参照）
    r2_hh = cv_r2(pca_reduce(H, k), H_k)
    r2_cc = cv_r2(pca_reduce(C, k), C_k)

    r_e_h = cv_r2(E, H_k)
    r_e_c = cv_r2(E, C_k)
    r_h_e = cv_r2(pca_reduce(H, k), E_k)
    r_c_e = cv_r2(pca_reduce(C, k), E_k)
    r_c_h = cv_r2(C, H_k)

    print(f"\n  [同一目标秩 {rank} 下的交叉重建 R²]")
    print(f"    R²(SASRec → 行为 H)      = {r_e_h:+.4f}")
    print(f"    R²(SASRec → 内容 C)      = {r_e_c:+.4f}")
    print(f"    R²(行为 H → SASRec)      = {r_h_e:+.4f}")
    print(f"    R²(内容 C → SASRec)      = {r_c_e:+.4f}")
    print(f"    R²(内容 C → 行为 H)      = {r_c_h:+.4f}   ← 内容含多少行为（冗余基准）")
    print(f"    [参照] 自重建 R²(H→H)={r2_hh:+.4f}  R²(C→C)={r2_cc:+.4f}")

    ratio = r_e_h / max(r_e_c, 1e-8)
    print(f"\n  行为/内容可还原比 R²(E→H)/R²(E→C) = {ratio:.3f}")
    print("  判定：比值 ≥0.8 ⇒ SASRec emb 已含行为通道信息 ⇒ 臂 A 大概率打平 ⇒ 倾向臂 B")
    print("        比值 ≤0.4 ⇒ 两类信息可区分 ⇒ 臂 A 有区分度")
    return dict(dataset=dataset, r_e_h=r_e_h, r_e_c=r_e_c, r_c_h=r_c_h, ratio=ratio)


# --------------------------------------------------------------------------- #
# P2：量化瓶颈的行为保留率（判臂 B）
# --------------------------------------------------------------------------- #
def probe_p2(dataset, ckpt_path, rank=32):
    print(f"\n{'='*78}\nP2 量化瓶颈保留率（臂 B 判据） — {dataset}\n{'='*78}")
    model, a = build_rqvae_from_ckpt(ckpt_path)
    print(f"  ckpt: {Path(ckpt_path).parent.name}  in_dim={model.in_dim} "
          f"e_dim={a['e_dim']} K={a['num_emb_list']} epoch={torch.load(ckpt_path, map_location='cpu', weights_only=False)['epoch']}")

    C, ids_c = load_emb(dataset, "item_emb_768d.parquet")
    H, ids_h = load_emb(dataset, BEHAVIOR_FILE[dataset])
    X, _ = load_emb(dataset, "fused_item_emb_768d.parquet")
    assert np.array_equal(ids_c, ids_h) and len(C) == len(X)

    z, zq, idx = encode(model, X)
    lat_rank = z.shape[1]
    print(f"  潜空间秩={lat_rank} | 码本命中唯一组合数={len(np.unique(idx, axis=0))}/{len(idx)}")

    H_k = pca_reduce(H, rank)
    C_k = pca_reduce(C, rank)

    # 全 128/768 维目标（不加 PCA，检查是否受降维影响）
    rho_h = cv_r2(z, H_k)
    q_h = cv_r2(zq, H_k)
    rho_c = cv_r2(z, C_k)
    q_c = cv_r2(zq, C_k)

    s_h = q_h / max(rho_h, 1e-8)
    s_c = q_c / max(rho_c, 1e-8)

    print(f"\n  [目标降到秩 {rank}]")
    print(f"    行为 H : R²(z→H) = {rho_h:+.4f}  →  R²(ẑ→H) = {q_h:+.4f}   存活率 s_H = {s_h:.3f}")
    print(f"    内容 C : R²(z→C) = {rho_c:+.4f}  →  R²(ẑ→C) = {q_c:+.4f}   存活率 s_C = {s_c:.3f}")
    print(f"\n  s_H / s_C = {s_h / max(s_c, 1e-8):.3f}")
    print("  判定：比值 ≤0.7 ⇒ 量化对行为信息有选择性破坏 ⇒ 损失级对齐有 headroom ⇒ 倾向臂 B")
    print("        比值 ≈1.0 ⇒ 瓶颈一视同仁 ⇒ 输入级 concat 已够 ⇒ 臂 B 无增量")
    return dict(dataset=dataset, s_h=s_h, s_c=s_c, rho_h=rho_h, q_h=q_h,
                rho_c=rho_c, q_c=q_c, ratio=s_h / max(s_c, 1e-8))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Sport", help="Sport / Toys / Beauty / all")
    ap.add_argument("--ta_sid_ckpt", default=None)
    ap.add_argument("--skip_p1", action="store_true")
    ap.add_argument("--skip_p2", action="store_true")
    ap.add_argument("--rank", type=int, default=32)
    args = ap.parse_args()

    datasets = ["Sport", "Toys"] if args.dataset == "all" else [args.dataset]
    res1, res2 = [], []
    for d in datasets:
        if not args.skip_p1 and d in SASREC_CKPT:
            try:
                res1.append(probe_p1(d, args.rank))
            except Exception as e:
                print(f"  P1 {d} 失败: {e}")
        if not args.skip_p2:
            ck = args.ta_sid_ckpt or DEFAULT_CKPT.get(d)
            if ck is None:
                print(f"  P2 {d}: 未提供 ckpt，跳过")
                continue
            try:
                res2.append(probe_p2(d, PROJECT_ROOT / ck if not Path(ck).is_absolute() else ck, args.rank))
            except Exception as e:
                print(f"  P2 {d} 失败: {e}")

    print(f"\n{'='*78}\n汇总\n{'='*78}")
    for r in res1:
        print(f"  P1 {r['dataset']}: R²(E→H)={r['r_e_h']:+.4f} R²(E→C)={r['r_e_c']:+.4f} "
              f"比={r['ratio']:.3f}")
    for r in res2:
        print(f"  P2 {r['dataset']}: s_H={r['s_h']:.3f} s_C={r['s_c']:.3f} 比={r['ratio']:.3f}")


if __name__ == "__main__":
    main()
