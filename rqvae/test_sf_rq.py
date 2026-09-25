#!/usr/bin/env python3
"""test_sf_rq.py — SF-RQ 库逐模块测试（E48 方向 1）

用法（在仓库根目录）：
    python rqvae/test_sf_rq.py [cpu|cuda:0]     # 默认 cpu（避免与训练抢 GPU）

模块 → 测试：
    M1 load_rqvae_from_ckpt : T1 复现 62 塌缩 + initted 陷阱
    M2 encode_z             : T2 shape/有限性/确定性
    M3 relock_l0            : T3 用码提升 + 确定性
    M4 freeze_l0            : T4 冻结后梯度不泄漏
"""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sf_rq import (load_rqvae_from_ckpt, encode_z, relock_l0, freeze_l0)


def latest_best_ckpt(tag):
    subs = sorted(glob.glob(f"rqvae/ckpt/{tag}/*/"))
    assert subs, f"no ckpt dir for {tag}"
    ck = os.path.join(subs[-1], "best_collision_model.pth")
    assert os.path.exists(ck), ck
    return ck


def _l0_usage(model, z):
    w = model.rq.vq_layers[0].embedding.weight.detach()
    return int(torch.cdist(z, w).argmin(dim=1).unique().numel())


def _input_batch(data_path, n=256, device="cpu"):
    """取 n 行真实输入（与训练同维度）作前向测试。"""
    from datasets import EmbDataset
    emb = EmbDataset(data_path).embeddings[:n]
    return torch.from_numpy(emb).to(device)


def t1_loader(device):
    """M1：load 复现 62/256 塌缩；train() 前向不覆盖码本（initted 陷阱）。"""
    ck = latest_best_ckpt("Toys_TA_SID_768d")
    model, a = load_rqvae_from_ckpt(ck, device=device)
    w0 = model.rq.vq_layers[0].embedding.weight.data
    enc0 = model.encoder.mlp_layers[1].weight.data  # 第一个 Linear
    assert enc0.shape == (512, 896), f"in_dim 异常: {enc0.shape}（应为 896d 主模型）"
    assert all(vq.initted for vq in model.rq.vq_layers), "initted 未恢复"

    z = encode_z(model, a.data_path, device=device)
    assert z.shape == (11924, model.e_dim)
    usage = _l0_usage(model, z)
    assert usage == 62, f"L0 用码 {usage} ≠ 62（load 未复现塌缩锚点）"

    # initted 陷阱：若 initted 丢失，train() 首个 forward 会重跑 kmeans 覆盖 L0
    model.train()
    xb = _input_batch(a.data_path, n=256, device=device)
    out, rq_loss, indices = model(xb, return_z=False)
    loss, _ = model.compute_loss(out, rq_loss, xs=xb)
    loss.backward()
    w1 = model.rq.vq_layers[0].embedding.weight.data
    assert torch.equal(w0.cpu(), w1.cpu()), "train() 前向覆盖了加载的 L0 码本（initted 陷阱未修复）"
    print("  T1 PASS — load 复现 L0=62/256 塌缩；initted 恢复，train() 不覆盖码本")


def t2_encode_z(device):
    ck = latest_best_ckpt("Toys_TA_SID_768d")
    model, a = load_rqvae_from_ckpt(ck, device=device)
    z1 = encode_z(model, a.data_path, device=device)
    z2 = encode_z(model, a.data_path, device=device)
    assert z1.shape == (11924, model.e_dim)
    assert torch.isfinite(z1).all()
    assert torch.equal(z1, z2), "encode_z 不确定"
    print(f"  T2 PASS — z shape={tuple(z1.shape)}, finite, 确定性")


def t3_relock(device):
    ck = latest_best_ckpt("Toys_TA_SID_768d")
    model, a = load_rqvae_from_ckpt(ck, device=device)
    z = encode_z(model, a.data_path, device=device)
    vq0, usage = relock_l0(model, z, seed=2024)
    assert usage >= 200, f"relock 后 L0 用码 {usage} < 200（k-means 未铺开）"
    w_a = vq0.embedding.weight.data.clone()
    # 确定性：同 z 同 seed → 同中心
    model2, _ = load_rqvae_from_ckpt(ck, device=device)
    vq0b, usage_b = relock_l0(model2, z, seed=2024)
    assert torch.equal(w_a.cpu(), vq0b.embedding.weight.data.cpu()), "relock 不确定（random_state 未固定）"
    assert usage_b == usage
    # 语义提升参照：健康 seed2025 用码 194；此处应远高于塌缩的 62
    print(f"  T3 PASS — relock L0 用码 62 → {usage}/256, 确定性")


def t4_freeze_grad(device):
    ck = latest_best_ckpt("Toys_TA_SID_768d")
    model, a = load_rqvae_from_ckpt(ck, device=device)
    w = freeze_l0(model)
    enc = model.encoder.mlp_layers[1].weight
    assert w.requires_grad is False

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    batch = _input_batch(a.data_path, n=128, device=device)
    out, rq_loss, _ = model(batch, return_z=False)
    loss, _ = model.compute_loss(out, rq_loss, xs=batch)
    opt.zero_grad(); loss.backward(); opt.step()

    assert w.grad is None, "L0 冻结后仍有 grad"
    assert enc.grad is not None, "encoder 应有 grad"
    print("  T4 PASS — L0 冻结后 grad=None 且权重不变；encoder/L1/L2 仍可训")


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(f"== SF-RQ 模块测试 (device={device}) ==")
    torch.manual_seed(0)
    t1_loader(device)
    t2_encode_z(device)
    t3_relock(device)
    t4_freeze_grad(device)
    print("== ALL PASS (M1–M4) ==")


if __name__ == "__main__":
    main()
