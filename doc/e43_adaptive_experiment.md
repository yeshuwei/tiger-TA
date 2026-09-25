# 4.3 Adaptive Capacity — 实验设计与结果汇总

> 生成: 2026-08-13 | 4.3 回答 Q1 结构恢复 / Q2 是否优于固定大容量 / Q3 是否转成推荐提升
> 范围裁定（用户决定）: Fixed-large 只在 Toys 论证,Beauty/Sport 的 1024 列不补训,靠 Toys 结论外推。

> ⚠ **2026-09-04 推翻裁定（阅读前必读）**：本文件是"自适应扩容"的**历史实验记录**，其
> Q3 结论（+3.8% 扩容卖点）与"Toys K0\*=512"已被 E45 推翻——Toys 塌缩是 seed2024 单次训练的
> 分配崩坏（换种子 256 健康），健康 256 的 T5 NDCG≈512（无统计差异）→ **扩容无稳健推荐层收益**，
> 正确动作是 reseed 而非扩容。下表量化观测仍可作判据证据，但**不得再当"扩容有效"卖点引用**。
> 权威裁定与改写版算法见 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)。

## 1. 固化: Adaptive Coarse-Level Capacity Allocation 算法（第二创新方法）

```
输入: 数据集 D 行为数据、item embeddings E、量化容差 ε=0.01
输出: 粗层码本容量 K0*(D)
──────────────────────────────────────────────────────────────
1  K0 ← K_min (=256)                       # TIGER 默认粗层容量
2  repeat
3    训练 RQ-VAE(D; K0, K1=256, K2=256)     # 仅粗层可变,细层固定
4    生成 SID codes → 重建 xq[full]          # 3-token 残差量化
5    z-gap ← AUC(z) − AUC(xq[full])         # 转移结构量化损失
6    if z-gap ≤ ε: break                    # 达到 lossless,容量充足
7    K0 ← 2·K0                              # 按需扩容重训
8  return K0* = K0
──────────────────────────────────────────────────────────────
```

**设计动机**（对应创新点 2 的目标）:
- 保留 transition structure ↔ 终止条件 `z-gap ≤ ε`（量化近无损,与 z 天花板差距 <1%）。
- 避免无意义扩容 ↔ 取满足条件的最小 K0（Beauty/Sport 停在 256,Toys 升到 512）。
- 非 grid search ↔ 有明确终止准则、按需触发（每数据集 0–1 次额外训练）、结果因数据而异。

**执行结果（三个数据集）**:
| Dataset | K0* | z-gap@256 | z-gap(K0*) | 扩容次数 |
|---|---|---|---|---|
| Beauty | 256 | 0.0001 | 0.0001 | 0 |
| Sport | 256 | 0.0032 | 0.0032 | 0 |
| Toys | **512** | 0.0931 | 0.0045 | 1 |

## 2. 4.3.1 Quantization Quality（Q1 + Q2, 已有数据直接汇总）

AUC = P(转移对 cos > 随机对 cos); top-1000 转移对 + 3000 随机对 (seed=0, E44 统一口径) | 数据: Toys 768d sweep + E3 + 主实验诊断

| Method | Dataset | K0 | z-gap | xq[full] AUC | L0 用码(util) | recon MSE | collision |
|---|---|---|---|---|---|---|---|
| Fixed-small (256) | Beauty | 256 | 0.0001 | 0.8676 | 204/256 (80%) | — | 0.05% |
| Fixed-small (256) | Sport | 256 | 0.0032 | 0.9614 | 249/256 (97%) | — | 0.00% |
| Fixed-small (256) | Toys | 256 | **0.0931** | 0.8724 | 62/256 (24%) | 0.1452 | 0.29% |
| Fixed-large (1024) | Toys | 1024 | 0.0003 | 0.963 | 275/1024 (27%) | 0.1530 | 0.03% |
| **Adaptive** | Beauty | 256 | 0.0001 | 0.8676 | 204/256 | — | 0.05% |
| **Adaptive** | Sport | 256 | 0.0032 | 0.9614 | 249/256 | — | 0.00% |
| **Adaptive** | Toys | **512** | **0.0045** | **0.968** | 305/512 (**60%**) | **0.1414** | 0.03% |

**Q1（Adaptive 恢复 transition structure）** ✓:
Toys: Fixed-small(256) gap=0.0931 → Adaptive(512) gap=0.0045（vs z 天花板 0.965）;xq[full] AUC 0.872→0.968。
Beauty/Sport 本就 lossless(gap≈0),无需扩容。→ **固定 256 的缺陷仅体现在 Toys,Adaptive 恰好修复。**

**Q2（Adaptive 优于固定大容量）** ✓:
Toys 上 Adaptive(512): gap 0.0045、AUC **0.968**、util 60%、recon 0.1414;Fixed-large(1024): gap 0.0003、AUC 0.963、util 27%、recon 0.1530。
→ 512 与 1024 均近无损,但 **512 更小、AUC 反而更高、util 更高效**。"直接把 K0 调大"不能替代 Adaptive:
  (i) 对 Beauty/Sport,256 已 lossless → 扩到 1024 是纯浪费(Fixed-large 无法在 B/S 给出任何收益);
  (ii) 对 Toys,只需扩到 512,扩到 1024 无额外收益还降 util。

## 3. 4.3.2 Recommendation（Q3, Toys T5, 3-seed）✅ 完成（2026-08-14）

> ⚠️ **2026-09-09 provenance 订正**：本表 0.0367 / L0=62 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**；本处仅作"塌缩实例"历史对照保留。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020（健康 codes，L0=207，z-gap 0.005）。权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

| K0 (Method) | NDCG@20 | R@20 | 状态 |
|---|---|---|---|
| 256 (Fixed-small) | 0.0367±0.0006 | 0.0873±0.0016 | ✅ 主实验已有 |
| **512 (Adaptive)** | **0.0381±0.0023** | **0.0914±0.0060** | ✅ 新训 3-seed |
| 1024 (Fixed-large) | 0.0384±0.0005 | 0.0916±0.0005 | ✅ 新训 3-seed |

**逐 seed（NDCG@20 / R@20）**:
| seed | 512 | 1024 |
|---|---|---|
| 2025 | 0.0400 / 0.0956 | 0.0379 / 0.0920 |
| 2026 | 0.0356 / 0.0845 | 0.0389 / 0.0918 |
| 2027 | 0.0387 / 0.0941 | 0.0385 / 0.0910 |

**Q3 结论（⚠ 2026-09-04 已推翻，判 (a)）**:
- ~~**Adaptive(512) > Fixed-small(256)**: NDCG@20 +3.8%, R@20 +4.7% → 结构提升转成推荐提升。~~
  **作废**：此对比的 256 行是 seed2024 塌缩 codes。E45 对健康 256（seed2025 codes）重跑 T5 3-seed →
  NDCG@20 **0.0372±0.0012 ≈ 512 的 0.0381±0.0023**（std 重叠无统计差异）→ **+3.8% 来自修坏种子，
  扩容无稳健推荐层收益**。真实信息 = 坏码本代价 NDCG −1.4% / R@20 −3.8%，修复动作为 reseed。
- ~~**Adaptive(512) ≈ Fixed-large(1024)**: 扩到 1024 无额外收益。~~ 前半仍成立（512≈1024），但已不是
  "Adaptive 优于固定大容量"的证据——健康 256 也与二者无差异（判 a），即**根本不必扩容**。
- 权威裁定与改写版算法见 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)。

Beauty/Sport 推荐层不重跑（Adaptive=256=主实验 baseline）。

### ⚠ 口径与可复现性注记（2026-08-14 复核确认）

1. **命名重命名关系（非数据问题）**：
   - 256 baseline 的 code 文件在**训练日志**中记录为 `Toys_sid_codes_768d.npy`，后统一命名重命名为现存的 `Toys_ta_sid_768d_sid.npy`（时间戳同为 7/28 23:33，同一文件）。
   - seed2025 的 ckpt 在**旧 eval 日志**中引用 `tiger_ta_sid_Toys_768d.pth`，后统一命名重命名为现存的 `tiger_ta_sid_768d_toys_seed2025.pth`。
   - **验证方法**：用现存 ckpt + 现存 code 重新运行 `eval_trained_model.py`，9 个数字全部精确重现原日志（256: 0.0360/0.0366/0.0375；512: 0.0400；1024: 0.0379），证明命名变化不影响任何数字。
2. **T5 训练参数逐参数一致**：512/1024 与主实验 256 baseline 全部相同（200 epochs / lr 1e-4 / batch 256 / d_model 128 / vocab 1025 / beam 30 / dataset data/Toys / seed 2025/2026/2027）——从 6 个训练日志 Configuration 逐字段核对。
3. **code 文件 index 范围正确**：256→max 255、512→max 510、1024→max 1021，shape 全为 (11924, 4)，无 NaN。
4. **RQ-VAE num_emb_list 确认**：L0512=[512,256,256]、L0Big=[1024,256,256]（自训练日志 args）。

## 4. 训练配置（与主实验逐参数一致）

```
python model/main.py --dataset_path data/Toys --code_path data/Toys/Toys_ta_sid_L0{512,Big}_sid.npy \
  --save_path model/ckpt/tiger_ta_sid_768d_toys_L0{512,1024}_seed{S}.pth \
  --log_path  model/logs/tiger_ta_sid_768d_toys_L0{512,1024}_seed{S}.log --seed {S}
```
seed ∈ {2025,2026,2027};默认参数(200 epochs, lr 1e-4, T5 from-scratch 4 dec layers, d_model 128, vocab 1025)。
评估: `tools/eval_trained_model.py --dataset Toys --code_path <codes> --ckpt_path <ckpt> --topk_list 5,10,20 --seed {S}`

## 5. 待办 / 文件
- [x] 跑 T5 Toys@512 / @1024 × 3 seed → 填 §3 表（2026-08-14 完成）
- [x] 汇总第二创新结论 → `doc/e43_summary`（并入论文实验表）
- [x] 数字复核：9/9 重评估精确重现 + 命名重命名注记（见 §3 注记）
- 数据文件: `data/Toys/Toys_ta_sid_L0512_sid.npy`、`Toys_ta_sid_L0Big_sid.npy`
- 训练脚本: `rqvae/run_e43_toys_t5.sh`
