# E48 Step-0 sinkhorn-L0 机制探针 gate0 结果（2026-09-09）

> **属于 [e48_sfrq_direction_route.md](e48_sfrq_direction_route.md) Step 0 的 gate0 收口。**
> 一句话：**L0 塌缩 62→256 可修（机制确认，z-gap 0.0070 ≤ ε），但 sinkhorn 式"强制均匀"以伤 encoder
> 为代价（z 0.9407 vs 同 seed 标准训练 0.9656）→ 印证路线预案"需结构式分配"，SF-RQ 必要性上升。**
> 本步是机制诊断，不是方法承诺，也未烧 T5。
>
> 关联：路线 [e48_sfrq_direction_route.md](e48_sfrq_direction_route.md) §7 Step 0 /
> §4.3 兜底梯；塌缩实锤 [e47_seed_health_isolation.md](e47_seed_health_isolation.md)；
> 裁定 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)。

---

## 1. 实验与配置

| 项 | 值 |
|---|---|
| 目的 | 验证"塌缩 = L0 用量分配崩坏、可用均衡修复"（对 E47 因果链的机制确认） |
| 输入 | `data/Toys/fused_item_emb_768d.parquet`（896d = content768 + behavior128） |
| 改动 | 仅 `--sk_epsilons 0.003 0 0.003`（L0/L2 开 sinkhorn 均衡用量），余与标准 seed2024 逐项一致 |
| 训练 | seed=2024，256³/e32，3000ep，batch1024，lr1e-3（`rqvae/ckpt/Toys_TA_SID_skl0_seed2024/`） |
| codegen | `best_collision_model.pth` → `data/Toys/Toys_ta_sid_768d_skl0_seed2024_sid.npy`（CPU） |
| 诊断 | `tools/sid_capacity_diagnostic.py` top-1000 转移对 / 3000 随机对 / ks=256（CPU） |

**可比性说明**：诊断的"真实训练码本"锚点行（xq[L0]/xq[full]/z）只依赖 top-1000 转移对 + 3000 随机对，
与 E45/E46 各 e4x 报告同设置（容量扫描 ks 不影响锚点行）→ 下表三组数字直接可比。

---

## 2. gate0 结果（同 seed2024 / 同数据 / 同配置，仅 sinkhorn 一项不同）

| 模型 | L0 用码 | 碰撞率 | xq[L0] AUC | xq[full] AUC | z AUC | **z-gap** | 判定 |
|---|---|---|---|---|---|---|---|
| 标准 seed2024（E45/E46 塌缩锚点） | 62/256 | — | — | 0.8724 | 0.9656 | **0.0931** | 塌缩 |
| 标准 seed2025（E45 健康参照） | 194/256 | 0.06% | 0.9334 | 0.9672 | 0.9688 | **0.0016** | ✅ 健康 |
| **sinkhorn seed2024（本次）** | **256/256** | **0.00%** | 0.8951 | 0.9337 | 0.9407 | **0.0070** | 形式过 / 带代价 |

sinkhorn codes：`(11924,4)`，L0 distinct=256，3-token 去重后 11924/11924 全唯一（无碰撞）。

---

## 3. gate0 判定（对照路线 §7 预注册判据）

**形式判据：PASS**
- L0 用码 62 → **256**（≥130）✓ —— sinkhorn 把粗层用量完全均摊（mean 46.6 = 11924/256）。
- **z-gap = 0.9407 − 0.9337 = 0.0070 ≤ ε=0.01** ✓ —— 塌缩的转移一致性损失（0.093）被抹平到健康侧裕度内。

**但触发代价分支：PASS-with-caveat**
路线预案原句："若强制均匀伤重构（xq[full] AUC 掉）→ 说明需结构式分配，SF-RQ 必要性上升，仍前进。"
- xq[full] 0.9337 **低于健康参照 0.9672（−0.034）**，只回收了塌缩缺口（0.8724→0.9672）的 ~60%。
- **更关键、也是同 seed 干净归因**：seed2024 标准训练 z=0.9656（健康），同 seed 换 sinkhorn 后
  **z 掉到 0.9407（−0.025）**。z 是 encoder 输出（量化前），故这是 sinkhorn 的均衡压力**侵蚀了 encoder
  的语义对齐**，而非残余的 L0 分配问题——塌缩模型的 z 本是健康的，暴力均衡去动它反而伤。

---

## 4. 对方向 1（SF-RQ）的含义

| # | 含义 |
|---|---|
| 1 | **"塌缩可修复"机制确认**：L0 用量崩坏不是不可逆的结构缺陷，均衡即铺满 256/256、z-gap 归零。 |
| 2 | **sinkhorn 不是对的修复工具**：它用全局均匀性换语义，付出 encoder z −0.025 的隐蔽代价（且这代价
      同样静默——常规 recon 也看不出）。这正面支持路线的核心设计判断。 |
| 3 | **SF-RQ 必要性上升**：正确的修复应**只动 L0 分配、不动 encoder 训练动力学**。SF-RQ 的
      "健康 z 上 k-means(256) 切分 + 冻结 L0" 恰好如此——z 在塌缩里是健康的（0.9656），
      fresh k-means-256 在健康 z 上近无损（E45：0.9344 ≈ 真实 xq[L0] 0.9334）→ 结构式分区可保住 z 达健康水平。 |
| 4 | **方向可押（仍前进到 Step 1 Repair）**，但叙事不变：SF-RQ 是"确定性/稳健性保险"，不是精度卖点
      （E48 定位 §3）。此探针并未、也不会把 sinkhorn 或扩容包装成效果加强。 |

---

## 5. 已产文件 / 复现

- 模型：`rqvae/ckpt/Toys_TA_SID_skl0_seed2024/Sep-09-2026_10-26-43/best_collision_model.pth`
- codes：`data/Toys/Toys_ta_sid_768d_skl0_seed2024_sid.npy`（L0=256，仅供诊断，**非主表 codes**）
- 诊断日志：`rqvae/logs/e48_toys_skl0_seed2024_gate0.log`；训练日志：`rqvae/logs/e48_toys_skl0_seed2024_train.log`
- 复现（纯 CPU，不占 GPU）：
  ```bash
  python rqvae/generate_sid_codes_param.py --ckpt_path rqvae/ckpt/Toys_TA_SID_skl0_seed2024/*/best_collision_model.pth \
    --data_path data/Toys/fused_item_emb_768d.parquet \
    --output_file data/Toys/Toys_ta_sid_768d_skl0_seed2024_sid.npy --device cpu
  python tools/sid_capacity_diagnostic.py --datasets Toys --topk 1000 --n_random 3000 \
    --ckpt-tag Toys_TA_SID_skl0_seed2024 --code-file Toys_ta_sid_768d_skl0_seed2024_sid.npy \
    --ks 256 --report doc/e48_step0_sinkhorn_gate0.md
  ```

## 相关
[[session-2026-09-04-seed-health-screening]] | [e48_sfrq_direction_route.md](e48_sfrq_direction_route.md) | [e47_seed_health_isolation.md](e47_seed_health_isolation.md) | [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md) | [e45_toys_multiseed_collapse.md](e45_toys_multiseed_collapse.md)
