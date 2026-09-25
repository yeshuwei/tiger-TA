# E45 Toys 多种子塌缩可复现性（2026-09-03）★ 重要发现

> 目的：检验 Toys 62/256 L0 塌缩 + z-gap 0.093 是数据集的系统性质，还是 seed=2024 单次训练的偶然。
> 背景：既有 RQ-VAE 训练种子**硬编码 2024**（main.py 曾写死；现暴露为 `--seed`，默认 2024 不变），
> 故此前全部 RQ-VAE / codes / T5 输入都来自**同一个确定性训练**；"3-seed"仅指 T5。
> 脚本：`rqvae/run_multiseed_repro.sh`（量化端 gate：A 训练 → B codegen → C z-gap 诊断，无 T5）。
> 配置与 {DS}_TA_SID_768d 逐项一致（256³/3000ep/batch1024/lr1e-3/896d fused），仅 --seed 不同。

## 1. 结果：Toys 塌缩是 seed=2024 的偶然，不是系统性质 ★

> ⚠️ **2026-09-09 provenance 订正**：本表 0.0367 / L0=62 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**；本处仅作"塌缩实例"历史对照保留。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020（健康 codes，L0=207，z-gap 0.005）。权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

| RQ-VAE seed | L0 用码 | 每码 mean | 碰撞率 | xq[full] AUC | z AUC | **z-gap** | 判定 |
|---|---|---|---|---|---|---|---|
| **2024**（既有锚点） | **62/256 (24%)** ⚠ | ~? | 0.29% | 0.8724 | 0.9656 | **0.0931** | 塌缩 → K0\*=512 |
| **2025**（本实验） | 194/256 (76%) | 61.5 | 0.06% | 0.9672 | 0.9688 | **0.0016** | ✅ 健康 |
| **2026**（本实验） | 151/256 (59%) | 79.0 | 0.12% | 0.9654 | 0.9691 | **0.0037** | ✅ 健康 |

- 换种子后 Toys L0 完全不塌缩（59-76% 用码），xq[full] 从 0.872 恢复到 0.965-0.967，
  z-gap 掉到 0.0016/0.0037（≤ ε=0.01）→ **256 上近无损**。
- 诊断报告：`doc/e4x_toys_seed2025.md`、`doc/e4x_toys_seed2026.md`
- codes：`data/Toys/Toys_ta_sid_768d_seed{2025,2026}_sid.npy`
- ckpt：`rqvae/ckpt/Toys_TA_SID_seed{2025,2026}/`

## 2. 解释：分配崩坏是初始化敏感事件（结果 B）

- 同一数据、同一配置、仅换 RQ-VAE 种子 → 码本质量在"塌缩(24%用码, gap 0.093)"与
  "健康(59-76%用码, gap≤0.004)"之间**剧烈摆动**。
- ⇒ Toys 不是"系统性需要 512"；是 seed=2024 恰好踩中坏的码本分配。
- **对叙事的改写（更强、更普遍）**：分配质量是不可事前预测的初始化事件——
  固定 K0（无论 256 还是 512）都只保证"平均上"够用，无法保证**这一次**训练不塌。
  唯一可靠的保护 = 训练后测 z-gap + 按需扩容。这从"Toys 需要更大容量"升级为
  "任何数据集、任何一次训练都可能塌，事后判据是唯一兜底"。

> ⚠️ **2026-09-09 provenance 订正**：本表 0.0367 / L0=62 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**；本处仅作"塌缩实例"历史对照保留。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020（健康 codes，L0=207，z-gap 0.005）。权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

## 3. ⚠ 连锁问题判定：+3.8% 主要来自"修坏种子"（T5 已完成，判定 (a)）

- 主实验 Q3（NDCG 256=0.0367 / 512=0.0381 **+3.8%**）建立在 seed2024 塌缩 codes 上。
- **判定实验（2026-09-04 完成）**：seed2025 健康 256 codes → T5 3-seed（tag `rq2025`）：

  | codes 来源 | NDCG@20 | R@20 |
  |---|---|---|
  | 256 塌缩（seed2024）| 0.0367 ± 0.0006 | 0.0873 |
  | 256 健康（seed2025）| **0.0372 ± 0.0012** | **0.0906** |
  | 512（Adaptive，seed2024）| 0.0381 ± 0.0023 | 0.0914 |

  **判定 (a)**：健康 256（0.0372）≈ 512（0.0381），std 重叠无统计差异 →
  扩容在推荐层无稳健额外收益；Toys 的瓶颈是**码本健康度**而非容量。
  +3.8% 主要是"修好一个坏种子"，不是"Toys 需要 512"。
- **影响**：
  - Q3 表 `Adaptive512 +3.8%` 的卖点在 Toys 上不成立，需从 §7.8 撤下或改写。
  - 判据价值仍在：坏码本 recon loss 不可见、只有 z-gap 揪出（0.093）。
  - 修复动作应从"扩容"改为更便宜的 **reseed**（RQ-VAE ~8 min 重训）。
  - 坏码本真实代价量化：NDCG@20 掉 1.4%，R@20 掉 3.8%（相对）→ 有 real 风险需要门。
- ckpt：`model/ckpt/tiger_ta_sid_768d_toys_rq2025_seed{S}.pth`；
  eval logs：`model/logs/tiger_ta_sid_768d_toys_rq2025_seed{S}_eval.log`

## 4. 复现命令

```bash
# 量化端多种子（~10 min/种子）
bash rqvae/run_multiseed_repro.sh Toys 256 2025 2026
# 健康 codes → T5 3-seed（~30-50 min/种子，后台）
bash model/run_t5_multiseed.sh Toys Toys_ta_sid_768d_seed2025_sid.npy rq2025 2025 2026 2027
```

## 5. 附带产物
- `rqvae/main.py`：新增 `--seed`（默认 2024，完全保复现）。
- `rqvae/run_multiseed_repro.sh`、`model/run_t5_multiseed.sh`：多种子量化端 / T5 通用脚本。

## 相关
[[session-2026-09-03-sport-stress-negative]] [[session-2026-08-13-perlayer-diagnostic]] [[session-2026-08-20-multidataset-screening]]
