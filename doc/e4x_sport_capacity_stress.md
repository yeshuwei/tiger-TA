# E4x Sport 容量压测（负对照实验，2026-09-03）

> ⚠ **2026-09-04 补注**：本文档的判据精度/负对照结论**保留**（四方向只有 z-gap 全判对）；
> 但 §3 表末行"Toys → 正确扩容 → K0\*=512"的**动作表述已随 E45 推翻**——Toys 是 seed2024 单例分配
> 崩坏，正确动作是 **reseed（K0 不变）**而非扩容。权威裁定见 [e46_second_innovation_overturn.md](e46_second_innovation_overturn.md)。

> 目的：尝试在健康数据集 Sport 上"人为压小粗层容量"复现 Toys 式容量不足，
> 以补足第二创新的第二正例。
> **结论：压不出来 —— Sport 对粗层容量高度不敏感（K0∈{64,128,256} 全部近无损），
> 该负结果恰好支撑"判据精度叙事"：容量饿死 ≠ Toys 式训练期分配崩坏；z-gap 判据不会误触发。**
>
> 口径：top-1000 转移对 / 3000 随机对 (seed=0)，ε=0.01，xq[full]=三层全量重建。
> 训练参数与 Sport_TA_SID_768d 逐项一致（3000ep / batch1024 / lr1e-3 / sk=[0,0,0.003] /
> e_dim32 / layers[512,256,128,64]），仅 num_emb_list 的 L0 不同。
> 脚本：`rqvae/run_k0_stress.sh`（SKIP_T5=1 量化端 gate）。

## 1. 结果表（K0 越低 = 压得越狠）

| K0 | L0 用码 | 每码 mean | xq[L0] AUC | xq[full] AUC | z AUC | **z-gap** | 3-token 碰撞 | 判定 |
|---|---|---|---|---|---|---|---|---|
| 256 | 249/256 (97%) | ~74 | 0.9448 | 0.9614 | 0.9646 | **0.0032** | 0.00% | 无损（E44 锚点）|
| **128** | **128/128 (100%)** | ~143 | 0.9468 | 0.9635 | 0.9646 | **0.0011** | 0.02% | 无损 |
| **64** | **64/64 (100%)** | 286.8 | 0.9229 | 0.9526 | 0.9558 | **0.0032** | 0.11% | 无损 |

- K0=128/64 均无塌缩（全部码被用上），xq[full] 与 256 打平、gap 全部 ≤ ε。
- 即便 K0=64 使粗层 AUC 明显下降（0.945→0.923），**L1/L2 残差量化把结构几乎全补回**
  （xq[full] 0.9526 vs z 0.9558）→ 粗层砍半甚至砍 3/4 对 Sport 无害。

## 2. 解读：为什么压不出 Toys 的故障

- Toys 的容量不足态 = **训练中 L0 分配崩坏**（只用 62/256 码且摆放次优，E1/E2/E3 已证），
  不是"256 个码不够"。压小 K0 只会制造"容量饿死"，而健康数据上残差层可补偿 → 无法复现。
- 因此想靠"在健康数据上把 K0 调小"制造第二个正例，是错误路径；本实验以负结果确认了这点，
  也再次印证 E2 结论（问题是 allocation 而非容量数值）。

## 3. 论文用途：负对照（判据不误触发）

> ⚠️ **2026-09-09 provenance 订正**：本表 0.0367 / L0=62 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**；本处仅作"塌缩实例"历史对照保留。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020（健康 codes，L0=207，z-gap 0.005）。权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

| 方向 | 数据集/实验 | z-gap 判据行为 |
|---|---|---|
| 容量缩到 64 仍无损 | **Sport K0=64/128**（本实验）| ✅ 正确不扩容（无 false positive）|
| L0 塌缩但小规模无损 | MI（900 商品，L0 35/256）| ✅ 正确不扩容 |
| 容量饱和/粒度受限 | Electronics（L0 100%，z-gap 0.0116，扩容无效）| ✅ 触发后复测拒绝扩容 |
| 唯一真实分配崩坏 | Toys（L0 62/256，z-gap 0.093）| ✅ 正确扩容 → K0\*=512 |

→ 四个方向只有 z-gap 判据全部判对；裸看 L0 占用率在 MI/Electronics 两个方向都判错。

## 4. 文件

- RQ-VAE: `rqvae/ckpt/Sport_TA_SID_L0128/`、`rqvae/ckpt/Sport_TA_SID_L064/`
- codes: `data/Sport/Sport_ta_sid_L0128_sid.npy`、`data/Sport/Sport_ta_sid_L064_sid.npy`
- 诊断: `doc/e4x_sport_K0128.md`、`doc/e4x_sport_K064.md`
- 训练日志: `rqvae/logs/e4x_sport_K0128_train.log`、`rqvae/logs/e4x_sport_K064_train.log`
- 探针（多数据集 z 剖面）: 见记忆 session-2026-09-03（复用 `tools/intrinsic_capacity.py` 函数）

## 5. 复现命令

```bash
# 量化端 gate（~15 分钟/档）
SKIP_T5=1 bash rqvae/run_k0_stress.sh Sport 128
SKIP_T5=1 bash rqvae/run_k0_stress.sh Sport 64
# 若某档 gate 通过(z-gap>ε)需要补推荐层闭环，去掉 SKIP_T5=1 即可
```
