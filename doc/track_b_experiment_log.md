# Track B 实验日志
> ⚠️ **口径声明（2026-08-03）：** 本文件数字为**历史口径**（旧 TIGER 4096d/64d baseline + 单 seed）。**当前正式实验结果为 3-seed 768d 口径**：主实验见 [experiment_results_summary.md §1.5](experiment_results_summary.md#L53)、消融见 §4.3、转移一致性见 §5.3。本文件正文数字仅存档参考，勿直接引用。

> 维护目的：记录每轮线B训练的完整超参配置、碰撞率、NDCG 结果
> 创建日期：2026-07-29

## 实验编号规则

`B-{轮次}-{数据集}-{lambda}-{tau}`

---

## 实验记录

### B-1: Sport 768d, λ=0.1, τ=0.1

| 项目 | 值 |
|:----|:----|
| 数据集 | Sport 768d |
| 输入维度 | 896 (fused: 768 content + 128 behavior) |
| 输入文件 | `data/Sport/fused_item_emb_768d.parquet` |
| 过渡对 | `data/Sport/transition_pairs_768d.npy` (189,543 pairs) |
| **RQ-VAE 超参** | |
| λ_trans | 0.1 |
| τ | 0.1 |
| λ_balance | 0.0 |
| epochs | 2000 |
| eval_step | 50 |
| 其他 | 与 Track A 一致 (3×256 codebook, e_dim=32, [512,256,128,64] 隐层) |
| **RQ-VAE 结果** | |
| 最佳碰撞率 | 0.34% |
| 最终碰撞率 | 0.36% (epoch 1999) |
| 对比 Track A | 10.66% → **0.34%** ↓97% |
| **SID codes** | |
| 输出文件 | `data/Sport/Sport_track_b_sid_codes_768d.npy` |
| 码碰撞率 | 0.02% |
| **T5 超参** | |
| epochs | 200 (early stop at 43) |
| 早停耐心 | 10 |
| beam size | 30 |
| **T5 结果** | |
| Best Valid NDCG@20 | 0.03455 (epoch 34) |
| **Best Test NDCG@20** | **0.03046** |
| vs Track A TA-SID (0.0327) | **-6.8%** ❌ |
| vs Baseline (0.0207) | +47.1% |
| 日志路径 | `model/logs/tiger_ta_sid_sport_track_b_768d.log` |
| 训练日志 | `model/logs/tiger_ta_sid_sport_track_b_768d_train.log` |
| 状态 | ✅ 完成 |

### B-2: Sport 768d, λ=0.01, τ=0.1

| 项目 | 值 |
|:----|:----|
| 数据集 | Sport 768d |
| 输入维度 | 896 (fused) |
| 输入文件 | `data/Sport/fused_item_emb_768d.parquet` |
| 过渡对 | `data/Sport/transition_pairs_768d.npy` |
| **RQ-VAE 超参** | |
| λ_trans | 0.01 |
| τ | 0.1 |
| λ_balance | 0.0 |
| epochs | 2000 |
| eval_step | 50 |
| checkpoint | `rqvae/ckpt/Sport_Trans_768d_l001/Jul-29-2026_09-40-50/` |
| **RQ-VAE 结果** | |
| 最佳碰撞率 | 0.43% |
| 最终碰撞率 | 0.50% (epoch 1999) |
| 对比 Track A | 10.66% → **0.43%** ↓96% |
| 对比 B-1 | 0.34% → 0.43% (略高，但仍在低位) |
| **SID codes** | |
| 输出文件 | `data/Sport/Sport_track_b_sid_codes_768d_l001.npy` |
| 码碰撞率 | 0.00% |
| **T5 超参** | |
| epochs | 200 |
| 早停耐心 | 10 |
| beam size | 30 |
| **T5 结果** | |
| Best Valid NDCG@20 | 0.03697 (epoch 65) |
| **Best Test NDCG@20** | **0.03246** |
| vs Track A TA-SID (0.0327) | **-0.7%** 🤏 (基本打平) |
| vs Baseline (0.0207) | +56.8% |
| 日志路径 | `model/logs/tiger_ta_sid_sport_track_b_768d_l001.log` |
| 训练日志 | `model/logs/tiger_ta_sid_sport_track_b_768d_l001_train.log` |
| 状态 | ✅ 完成 |

> **B-2 总结：** λ=0.01 将 L_trans 对 NDCG 的损害降到最低（-0.7%），
> 碰撞率仍保持在 0.43% 的低位。但 L_trans on z 的**根本矛盾**未解决——
> 编码器和量化器在对抗。B-3 用 v2（L_trans on x_q + L_balance）尝试突破。

### B-3: Sport 768d, λ=0.01, λ_balance=0.01, τ=0.1 (v2 — L_trans on x_q)

| 项目 | 值 |
|:----|:----|
| 数据集 | Sport 768d |
| 输入维度 | 896 (fused) |
| 输入文件 | `data/Sport/fused_item_emb_768d.parquet` |
| 过渡对 | `data/Sport/transition_pairs_768d.npy` |
| **L_trans 版本** | **v2: 作用在 x_q (量化向量) 上，不再是 z (编码器输出)** |
| **RQ-VAE 超参** | |
| λ_trans | 0.01 |
| λ_balance | 0.01 |
| τ | 0.1 |
| epochs | 2000 |
| eval_step | 50 |
| checkpoint | `rqvae/ckpt/Sport_Trans_768d_v2_xq/Jul-29-2026_11-03-03/` |
| **RQ-VAE 结果** | |
| 最佳碰撞率 | 0.81% |
| 最终碰撞率 | 0.97% (epoch 1999) |
| 对比 Track A | 10.66% → **0.81%** ↓92% |
| L_recon 终值 | 0.151 |
| L_trans 终值 | ~6.2（远高于 L_recon，x_q 上对比损失大） |
| L_balance 终值 | ~0.254（λ_balance=0.01 → 有效贡献仅 0.0025） |
| **SID codes** | |
| 输出文件 | `data/Sport/Sport_track_b_sid_codes_768d_v2_xq.npy` |
| 码碰撞率 | 0.01% |
| L0/L1/L2 利用率 | 73% / 100% / 100% |
| L0 码本 norm std | **1.91**（Track A 为 0.89 → 码本严重失衡） |
| **T5 超参** | |
| epochs | 200 (early stop at 66) |
| 早停耐心 | 10 |
| beam size | 30 |
| **T5 结果** | |
| Best Valid NDCG@20 | 0.03551 (epoch ~55) |
| **Best Test NDCG@20** | **0.03122** |
| vs Track A (0.03270) | **-4.5%** ❌ |
| vs B-2 (0.03246) | **-3.8%** ❌ |
| vs Baseline (0.0207) | +50.8% |
| 日志路径 | `model/logs/tiger_ta_sid_sport_track_b_768d_v2_xq.log` |
| 训练日志 | `model/logs/tiger_ta_sid_sport_track_b_768d_v2_xq_train.log` |
| 状态 | ✅ 完成 |
| **结论** | **v2 (L_trans on x_q) 比 v1 (on z) 效果更差。码本 L0 被 L_trans 扭曲（std=1.91 vs 0.89），L_balance 不足以抵消。Track B 全部结果均未超过 Track A。** |

---

## 版本变更记录

### v1 (B-1, B-2) — L_trans on z (encoder output)

```
L_trans = ContrastiveLoss(z_anchor, z_pos)
         z = encoder(x)    # 编码器输出, (B, 32)
```

梯度流向: 编码器权重
问题: L_trans 拉近 z → 量化器被迫分散 → 碰撞率↓但码本质量↓ (量化器在对抗 L_trans)

### v2 (B-3+) — L_trans on x_q (quantized vector)

```
L_trans = ContrastiveLoss(x_q_anchor, x_q_pos)
         x_q = Σ(codebook[vq_indices])   # 码本查表求和, (B, 32)
```

梯度流向: 码本嵌入
问题: 梯度直接作用于码本，创建了 L_recon 和 L_trans 的内部冲突。
      L0 码本 norm 标准差从 0.89 (Track A) 飙升至 1.91 → 码本结构扭曲。
      实验证明 v2 比 v1 效果更差 (B-3 0.03122 vs B-2 0.03246)。

### λ_trans (contrastive loss weight)
- 控制 L_trans 对总损失的贡献比例
- 越大 → 行为连续性对编码的约束越强
- 越小 → 越接近 Track A（λ=0 即 Track A）

### τ (temperature)
- InfoNCE 损失的温度系数
- 越小 → 对比越"硬"（只关心最近的正/负样本）
- 越大 → 越"软"（所有样本都考虑）

### 损失组成
```
L_total = L_recon + 1.0·L_vq + λ_trans·L_trans + λ_balance·L_balance
```
- L_recon: 重建损失 (MSE)，保 Embedding 质量
- L_vq: 码本损失，保量化质量
- L_trans: 对比损失，拉近行为连续的商品
- L_balance: 均衡损失，防码本崩塌

---

## Track B 实验总结

### 全部结果一览

| 实验 | λ_trans | 作用位置 | λ_balance | Best Test NDCG@20 | vs Track A |
|:----|:------:|:--------:|:---------:|:----------------:|:----------:|
| B-1 | 0.1 | z (encoder) | 0.0 | 0.03046 | -6.8% |
| B-2 | 0.01 | z (encoder) | 0.0 | **0.03246** | -0.7% |
| B-3 | 0.01 | x_q (量化) | 0.01 | 0.03122 | -4.5% |

### 核心结论

1. **Track A (no L_trans) 是所有实验中最优的 (0.03270)**
2. **L_trans 不论放哪 (z 或 x_q)、不论多大 λ，都无法超越 Track A**
3. **B-2 (λ=0.01 on z) 最接近 Track A (-0.7%)，趋势明确：λ→0 越接近 Track A**
4. **L_trans 效果不行的根本原因：**
   - Behavior embedding (PPMI+SVD) 已经通过融合输入传递了 transition 信号
   - T5 自身的自回归训练已经学到了 transition 模式
   - L_trans 是对同一信号的冗余约束，且与 L_recon 存在冲突
5. **建议：不再投入资源到 Track B，将精力用于分析实验（Longtail/Coldstart/t-SNE）**
