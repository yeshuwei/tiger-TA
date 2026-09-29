#!/usr/bin/env python3
"""
sf_rq.py — Semantic-First Residual Quantization (SF-RQ) 库模块（E48 方向 1）

核心思想：RQ-VAE 的粗层码本 L0 未被任何目标约束、训练中可静默塌缩（recon 不可见，
仅 z-gap 检出，见 E47）。SF-RQ 把 L0 改为「在学到语义的 z 上显式构造的分区」并冻结：
    暖机(Stage A, 现役) → 在健康 z 上 k-means 切分 L0 → 冻结 L0 → 细调 encoder+L1/L2。

模块分解（每函数独立可测，对应 rqvae/test_sf_rq.py 的 T1..T5）：
    M1  load_rqvae_from_ckpt : 从 ckpt 重建模型 + 恢复 vq.initted（防懒 k-means 覆盖码本）
    M2  encode_z             : eval 全量 z（编码器输出，量化前）
    M3  relock_l0            : 在 z 上做确定性 k-means(256) 并写回 L0（语义分区锁定）
    M4  freeze_l0 / unfreeze : L0 码本 requires_grad_(False)
    M5  repair               : 编排 load→relock→freeze→Trainer 续训（见 __main__）

与第一创新的隔离（E48 §6）：不改输入构造 / RQ-VAE 架构 / T5；只改 RQ-VAE 训练规程。
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import EmbDataset
from models.rqvae import RQVAE


# ----------------------------------------------------------------------------
# M1. 加载器
# ----------------------------------------------------------------------------
def _infer_in_dim(state_dict):
    """从 encoder 第一个 2D Linear 的 in_features 推断输入维度，避免加载时读 parquet。"""
    for k, v in state_dict.items():
        if k.startswith("encoder.") and k.endswith(".weight") and v.ndim == 2:
            return v.shape[1]
    raise ValueError("cannot infer in_dim from state_dict")


def load_rqvae_from_ckpt(ckpt_path, device="cpu"):
    """从保存的 .pth 重建 RQVAE + 训练 args。

    Returns (model, args). 关键点：
      - state_dict 恢复码本与 encoder/decoder 权重；
      - 逐层把 vq.initted 置 True —— 该 flag 是普通 python 属性、不进 state_dict，
        载入后默认为 False；若直接 train()，首次 forward 会懒触发 init_emb()
        （kmeans on 随机/首 batch z）**覆盖已加载码本**（vq.py init_emb / forward）。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    in_dim = _infer_in_dim(ckpt["state_dict"])
    model = RQVAE(in_dim=in_dim,
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
    for vq in model.rq.vq_layers:      # ← 关键陷阱修复
        vq.initted = True
    model.to(device)
    model.eval()
    return model, a


# ----------------------------------------------------------------------------
# M2. 全量 z 提取
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_z(model, data_path, batch_size=1024, device="cpu"):
    """eval 模式下过 encoder 提取全部 item 的 z（N, e_dim），返回 CPU tensor。"""
    data = EmbDataset(data_path)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device).eval()
    zs = []
    for batch in loader:
        zs.append(model.encoder(batch.to(device)).cpu())
    return torch.cat(zs, dim=0)


# ----------------------------------------------------------------------------
# M2b. 逐层 code 的真实重放（2026-09-27 新增）—— 供 provenance 审计使用
# ----------------------------------------------------------------------------
@torch.no_grad()
def rederive_codes_sequential(model, z, use_sk=False):
    """按 RQ 的真实前向顺序重算每层 code：residual 逐层减去已选中的码本向量。

    ⚠ 为什么不能对每层直接 `argmin(z, cb_l)`：
        `ResidualVectorQuantizer.forward` 里是 `residual = residual - x_res`，
        第 l 层的输入是**残差**而不是 z。用 z 对每一层独立 argmin，得到的 L1/L2
        与 codes 文件里的列**不可比** —— 会报出近乎随机的 ~0.6%–2.3% 一致率
        （`doc/e48_toys_main_codes_provenance_fix.md` 曾把该数字当真引用）。
    ⚠ `use_sk` 的语义与**实测哪个匹配 codegen**：
        `VectorQuantizer` 在 `sk_epsilon > 0` 时改用 Sinkhorn 做**全局**分配（非逐行
        argmin）。主表 `args.sk_epsilons = [0.0, 0.0, 0.003]` ⇒ L0/L1 恒为纯 argmin，
        只有 **L2** 受 `use_sk` 影响。codegen 是**两趟**：初始 pass `use_sk=False`，
        碰撞消解循环 `use_sk=True`（见 `generate_sid_codes_param.py` docstring）。
        2026-09-27 实测（Beauty/Sport best_collision，L0 均为 100.00%）：

            use_sk=False →  L2 = 98.06% / 98.86%
            use_sk=True  →  L2 = 86.03% / 89.52%

        ⇒ **`use_sk=False` 才是与 codes 文件匹配的主 pass**（故设为默认值）；
        L2 残留的 ~1–2% 正是被碰撞消解循环（`use_sk=True`）改写过的那些行。
        **L0/L1 与 use_sk 无关，两者都精确 100.00%。**
    ⚠ 结构冗余：给定同一码本与同一 z，**L0 一致率 = 100% ⟹ L1 = 100%**（残差被完全
        决定）。故 L1/L2 **不构成独立证据**，只能当"码本同一性检查"。
        （同一判断的先例见 `doc/e49grid_diag_summary.md` §1："L0 是 z 上的直接 argmin，
        判据有效；L1/L2 在残差上量化，直接 argmin 无判据性"。）

    Returns: (N, num_quantizers) int64 numpy array。
    """
    residual = z
    out = []
    for q in model.rq.vq_layers:
        x_res, _, idx = q(residual, use_sk=use_sk)
        out.append(idx.view(-1))
        residual = residual - x_res
    return torch.stack(out, dim=1).cpu().numpy()


# ----------------------------------------------------------------------------
# M3. 语义切分：在 z 上确定性 k-means 并写回 L0
# ----------------------------------------------------------------------------
def relock_l0(model, z, k=None, seed=2024, n_init=10, max_iter=300):
    """在 (全量) z 上做确定性 k-means(k) → 用中心覆盖 L0 码本。

    确定性：sklearn KMeans(random_state=seed)，同 z 同 seed 产出同分区。
    Returns (vq0, usage)：vq0 为 L0 VectorQuantizer；usage = 新码本下 argmin 用码数。
    """
    from sklearn.cluster import KMeans

    vq0 = model.rq.vq_layers[0]
    k = k or vq0.n_e
    X = z.cpu().numpy().astype(np.float64)
    assert k <= X.shape[0], f"k({k}) > n_samples({X.shape[0]})"
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter).fit(X)
    centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    with torch.no_grad():
        vq0.embedding.weight.copy_(centers)
    vq0.initted = True
    # argmin 用码统计（与 codegen use_sk=False 一致）
    d = torch.cdist(z, vq0.embedding.weight.detach().to(z.device))
    usage = int(d.argmin(dim=1).unique().numel())
    return vq0, usage


# ----------------------------------------------------------------------------
# M4. 冻结 / 解冻 L0
# ----------------------------------------------------------------------------
def freeze_l0(model):
    """冻结 L0 码本。优化器建在 model.parameters() 上，requires_grad=False 参数
    grad=None → 自动被跳过，无需改 optimizer。"""
    w = model.rq.vq_layers[0].embedding.weight
    w.requires_grad_(False)
    return w


def unfreeze_l0(model):
    model.rq.vq_layers[0].embedding.weight.requires_grad_(True)
