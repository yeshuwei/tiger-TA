# E49 创新点 1 消融扩展：Raw-CoOcc-SVD / ItemCF-SVD + 对照行 3-seed 对齐

**状态**：🔄 进行中（P2 决策门运行中）　**计划**：`~/.claude/plans/tidy-painting-church.md`
**配套文档**：[e49_behavior_space_gate.md](e49_behavior_space_gate.md)（自动生成表）· [e49_behavior_space_analysis.md](e49_behavior_space_analysis.md)（解读）· [e49_toys_zeropad_md5_provenance.md](e49_toys_zeropad_md5_provenance.md)（provenance 裁定）

---

## 1. 要回答的两个审稿人问题

现有消融链只能证明"多加一个向量 / 任意统计噪声都不涨"。本轮新增两个 behavior 变体回答更尖锐的两个问题：

1. **PPMI 这一步是不是必要的？** —— 会不会"任何共现矩阵做 SVD"都行？
2. **TA-SID 是不是只是普通的 item-level 协同过滤表示？**

> **术语约定（本轮起统一）**：不使用 "directional transition" / "转移概率"。Transition 一律定义为
> **"sequential behavioral association patterns extracted from user interaction sequences"**，
> 强调 TA-SID 建模的是**序列行为关联结构**，而非显式的 P(j|i) 转移概率。

## 2. 变体定义

共用：同一 train.parquet 序列、同一 canonical ItemID 顺序、同一 `svd_dim=128`、同一 `U·√S` 变换、同一冷启动物品置零、同一 `LN(content 768d) ⊕ LN(behavior 128d) → 896d`、同一 RQ-VAE 配置、同一 T5 配置。

| # | 变体 | 共现作用域 | 权重变换 | vs TA-SID 的改动 | 定位 |
|:-:|:--|:--|:--|:--|:--|
| 1 | Content-only | — | — | 无 behavior | 基线 |
| 2 | ZeroPad | — | 128 个零 | 维度 | 是否只是加维度 |
| 3 | Random | — | i.i.d. 高斯 | 任意向量 | 辅助消融（**seed 数=1**，不与其他行比显著性） |
| 4 | Shuffled | 窗口=3 相邻 | PPMI（行级打乱） | item↔向量对应 | 是否只需统计特性 |
| 5 | **Raw-CoOcc-SVD** | **窗口=3 相邻（与 TA-SID 逐位相同）** | **无（直接 SVD 原始计数 C）** | **仅去掉 PPMI** | **PPMI 是否必要（严格单变量）** |
| 6 | **ItemCF-SVD** | **整篮（同一序列内所有商品对）** | **cosine 归一** | **作用域 + 归一化（两项）** | **局部序列上下文 vs 全局 item-level 协同信号** |
| 7 | TA-SID (PPMI-SVD) | 窗口=3 相邻 | PPMI | 主方法 | — |

**设计声明（写进论文）**

- **Raw-CoOcc 是严格单变量对照**：调用与 TA-SID 完全相同的 `build_cooccurrence_matrix`（window=3, min_cooccur=1），得到**同一个 C**（已用 nnz + 逐元素校验和断言，见 §4），仅跳过 `compute_ppmi`。
- **ItemCF 不是严格单变量实验**：同时改变 ①共现作用域 ②归一化方式 ③关联方向性。因此**不用于论证"TA-SID 优于 CF"或"排除 CF 解释"**，其定位是回答："局部序列上下文共现结构是否比全局 item-level collaborative signal 更适合作为 SID augmentation？"两处同时变更在论文中显式声明为局限。
- **不引入 LightGCN / PureSVD**：本轮目标是验证 SID augmentation 的**行为表示来源**，不是比较推荐模型强弱。

## 3. 结果解读措辞（先定稿，避免事后找补）

**Raw-CoOcc vs TA-SID**

| 结果 | 措辞 |
|:--|:--|
| Raw-CoOcc ≪ TA-SID | PPMI 有效重塑行为空间（以 §5 的方向结构指标佐证，**不以"去流行度偏差"为由**） |
| Raw-CoOcc ≈ TA-SID | PPMI 在此数据上非必要，主张收缩为"共现结构 + SVD"，论文措辞须改 |

**ItemCF vs TA-SID**

| 结果 | 措辞 |
|:--|:--|
| ItemCF < TA-SID | 说明**局部序列行为上下文结构相比全局协同共现具有更强的 SID 表征能力** |
| ItemCF ≈ TA-SID | 说明**收益主要来自泛行为关联结构，需要收缩创新点表述** |

## 4. 已完成的离线自检（P1，全部通过）

| 检查 | 结果 |
|:--|:--|
| `C` 一致性（rawcooc vs PPMI 路径） | ✅ 校验和完全相同：Sport `nnz=360722, sum=758172.0, md5=b025cbfff9ccf94c`；Toys `nnz=204302, sum=437444.0, md5=2ac580d7ab964bea` |
| 融合链路等价 | ✅ 用新增 `--behavior_file` 重算主表 fused，与既有产物 **max abs diff = 0.000e+00** |
| PPMI 可复现 | ✅ `behavior_emb_ppmi.parquet` 与 legacy `behavior_emb.parquet` 诊断指标一致（范数 0.5667/0.4743 vs 0.5658/0.4744） |
| 行序对齐 | ✅ 全部新 parquet 的 `ItemID` 与 `item_emb_768d.parquet` 逐位相等 |
| 冷启动物品置零 | ✅ rawcooc / itemcf / ppmi：冷启动物品（Sport 28 / Toys 24）behavior 全零，覆盖品全非零。（`shuffled` 按设计**不满足**——打乱即破坏 item↔向量对应） |
| 内容段逐位相同 | ✅ `fused_item_emb_{rawcooc,itemcf}` 前 768 维与主表 fused **maxdiff=0.0e+00**（唯一变量确实是 behavior 侧） |

## 5. 行为空间诊断（GPU-free，RQ-VAE 之前）

完整表：[e49_behavior_space_gate.md](e49_behavior_space_gate.md)；解读：[e49_behavior_space_analysis.md](e49_behavior_space_analysis.md)

**关键判据 `cons_auc_local`** = P(cos(相邻对) > cos(同篮非相邻对))，0.5 = 不区分：

| 数据集 | TA-SID | Raw-CoOcc | ItemCF | Random/Shuffled |
|:--|:--|:--|:--|:--|
| Sport | **0.6237** | 0.5730 | 0.4093 | ~0.4995 |
| Toys | **0.7157** | 0.6297 | 0.3494 | ~0.4995 |

→ 单调序两数据集一致：**PPMI > Raw-CoOcc > ItemCF**，随机/打乱 ≈ 0.50。

⚠️ **负向结果须如实报告**：`freq_leak`（‖emb‖ vs 流行度）TA-SID **高于** Raw-CoOcc（0.8069/0.8288 vs 0.7430/0.7525），与预期相反；`pc1_cos_freq` 仅 Sport 支持、Toys 反向。故**论文不以"PPMI 去流行度偏差"作卖点**，改用 `cons_auc_local` + `top1_energy` 支持"PPMI 重塑方差方向分配"。详见 analysis 文档 §3。

## 6. 执行状态

| Phase | 内容 | 状态 |
|:--|:--|:--|
| P1 | 脚本 + 离线自检 + 行为空间诊断 | ✅ 完成 |
| **P2（决策门）** | Sport：Raw-CoOcc + ItemCF，seed 2025 单 seed smoke test | 🔄 **运行中**（`rqvae/logs/e49_p2_sport_bg.log`） |
| P3 | Sport 3-seed 补齐（2026/2027） | ⏳ |
| P4 | Toys：Raw-CoOcc + ItemCF 全 3-seed | ⏳ |
| P5 | backfill：Sport Shuffled/ZeroPad T5×2 + Toys Shuffled 全链路 + Toys ZeroPad T5×3 | ⏳ |
| P6 | 汇总表 + 本文档补全 + 一致性 AUC + `extract_metrics.py` LOG_MAP | ⏳ |

**P2 决策门参照值**（Sport seed2025，均由 `tools/eval_trained_model.py` 同口径产出）

| 行 | N@5 | N@10 | **N@20** | R@20 | seed 数 |
|:--|:--|:--|:--|:--|:--|
| Content-only | 0.0160 | 0.0211 | **0.0274** | 0.0664 | 1（本表内同口径） |
| Shuffled（legacy 单 seed） | 0.0149 | 0.0204 | **0.0262** | 0.0643 | 1 |
| ZeroPad（legacy 单 seed） | 0.0167 | 0.0223 | **0.0286** | 0.0688 | 1 |
| TA-SID（主方法） | 0.0192 | 0.0254 | **0.0327** | 0.0781 | 1（本表内同口径） |

**P2 决策门判据**（跑完直接看两项）：
1. Raw-CoOcc / ItemCF 的 Sport seed2025 NDCG@20 是否落在 [content-only **0.0274**, TA-SID **0.0327**] 邻域内（而非 0.001 或 0.06 量级）；
2. 两变体是否**互相可分**（若几乎重合 → 作用域/归一化在该数据上不起作用，需重审设计）。
任一不满足 → 停下排查，不直接进 P3。

### 6.1 P2 实测（Sport 两变体 seed2025）与排期修正

**P2 全量结果（Sport seed2025，同口径）**

| 行 | N@5 | N@10 | **N@20** | R@20 | 相对 content-only |
|:--|:--|:--|:--|:--|:--|
| content-only | 0.0160 | 0.0211 | 0.0274 | 0.0664 | — |
| Shuffled | 0.0149 | 0.0204 | 0.0262 | 0.0643 | −4.4% |
| ZeroPad | 0.0167 | 0.0223 | 0.0286 | 0.0688 | +4.4% |
| **ItemCF（新）** | 0.0173 | 0.0231 | **0.0298** | 0.0724 | **+8.8%** |
| **Raw-CoOcc（新）** | 0.0186 | 0.0242 | **0.0311** | 0.0736 | **+13.5%** |
| TA-SID | 0.0192 | 0.0254 | **0.0327** | 0.0781 | +19.3% |

**决策门：✅ 通过**（判定见 [e49_p2_gate_verdict.md](e49_p2_gate_verdict.md)）——两值均在合理带内，且可分（Δ=0.0013）。

⚠️ **单 seed 仍不足以判定 `≪` 还是 `≈`**：TA-SID 相对 Raw-CoOcc 仅 +5.1%、相对 ItemCF +9.7%。
§3 预注册的两种措辞指向相反的论文主张，**必须等 3-seed 均值±std 才能定夺**，此处不作结论。

📌 **值得记录的信号**：NDCG 排序 `TA-SID > Raw-CoOcc > ItemCF` **与行为空间诊断 `cons_auc_local` 的
前置预测完全同序**（§5：0.6237 > 0.5730 > 0.4093）。§5 提出的"行为空间可分性 → 下游收益"闭环
在 Sport seed2025 上成立；但**单 seed 单数据集不足以支撑闭环主张**，须待 P3/P4 齐备后复核。
另注意 Raw-CoOcc 与 ItemCF 均已明显高于 ZeroPad/Shuffled，说明「共现结构 + SVD」自身携带信息。

**⚠️ 排期与工程修正（实测，用于后续估算）**

- **`model/main.py` 日志语义**：`Test Dataset` 行**仅在验证集 NDCG@20 创新高时才打印**
  （[main.py:285-291](../model/main.py#L285-L291)），故 `grep -c "Test Dataset"` 数的是
  **改进次数，不是 epoch 数**；真实 epoch 须数 `Epoch N/200` 行。**本次曾据此误判进度，记录备查。**
- **实测单次 T5（Sport，含训练+评测）≈ 75 min**（约 83 epoch 触发早停，~54 s/epoch）；
  **单个新变体全链路 ≈ 86 min**（RQ-VAE 8 + codegen/诊断 3 + T5 75）。
- **⚠️ 竞态教训（工程）**：`tee` 在评测**开始**时即创建 eval 日志，而 NDCG 表格在评测**结束**才写入。
  故判"评测就绪"**必须用「能否解析出 NDCG 表格」而非「文件是否存在」**——否则会读到空文件并误判
  "训练失败"。2026-09-22 18:33 实际发生过一次，导致决策门假阴性、链路白停。
  `run_e49_p3_p6.sh` 已改为 `metric()` 判可解析 + 门内 15s×20 重试双层防护。
- 据此重估：P3–P5 剩余 21 次 T5 + 3 次量化 ≈ **27h**，预计 **2026-09-23 21:00 前后**全部结束。

### 6.0 附：首条实测（Sport Raw-CoOcc seed2025）原始记录

**结果**（`model/logs/Sport_rawcooc_768d_seed2025_eval.log`，同口径）

| 行 | N@5 | N@10 | **N@20** | R@20 | 相对 content-only |
|:--|:--|:--|:--|:--|:--|
| Content-only | 0.0160 | 0.0211 | **0.0274** | 0.0664 | — |
| Shuffled | 0.0149 | 0.0204 | 0.0262 | 0.0643 | −4.4% |
| ZeroPad | 0.0167 | 0.0223 | 0.0286 | 0.0688 | +4.4% |
| **Raw-CoOcc（新）** | 0.0186 | 0.0242 | **0.0311** | 0.0736 | **+13.5%** |
| TA-SID | 0.0192 | 0.0254 | **0.0327** | 0.0781 | +19.3% |

- 决策门判据①：**通过**（0.0311 落在合理带 [0.015, 0.045] 内）。
- ⚠️ **单 seed 不足以判定 `≪` 还是 `≈`**：TA-SID 仅领先 +5.1%（相对）。§3 预注册的两种措辞
  指向完全不同的论文主张，**必须等 3-seed 均值±std 才能定夺**，此处不作结论。
- 一个可留意的信号：Raw-CoOcc（去掉 PPMI）已明显高于 ZeroPad/Shuffled 且高于 content-only，
  说明**「相邻共现 + SVD」本身携带信息**；PPMI 是否在其上还有增量，是本轮的核心待答问题。

**排期修正（实测，用于后续估算）**

- **`model/main.py` 的日志语义**：`Test Dataset` 行**仅在验证集 NDCG@20 创新高时才打印**
  （`main.py:285-291`），因此 `grep -c "Test Dataset"` 数的是**改进次数，不是 epoch 数**。
  真实 epoch 数须数 `Epoch N/200` 行。**本次曾据此误判进度，记录备查。**
- **实测单次 T5（Sport，含训练+评测）≈ 75 min**（约 83 epoch 触发早停，~54 s/epoch）。
  比计划假设的 45–55 min 慢，但远好于按错误 epoch 计数推出的 130 min。
- **实测单个新变体全链路 ≈ 86 min**（RQ-VAE 8 min + codegen/诊断 3 min + T5 75 min）。
- 据此重估：P3–P5 剩余 21 次 T5 + 3 次量化 ≈ **27h**，预计 **2026-09-23 21:00 前后**全部结束。

**与行为空间诊断的对照预期**：`cons_auc_local` 排序为 TA-SID > Raw-CoOcc > ItemCF（§5）。若 T5 端 NDCG 呈同序，则形成"行为空间可分性 → 下游收益"闭环；若不同序，该判据降级为诊断性观察（见 [e49_behavior_space_analysis.md](e49_behavior_space_analysis.md) §5）。

## 7. 命名与产物（严格遵守 naming_convention）

| 产物 | 新变体路径 |
|:--|:--|
| behavior emb | `data/{Ds}/behavior_emb_{rawcooc,itemcf}.parquet` |
| fused | `data/{Ds}/fused_item_emb_{rawcooc,itemcf}.parquet` |
| RQ-VAE ckpt | `rqvae/ckpt/{Ds}_{RawCoOcc,ItemCF}_768d/` |
| SID codes | `data/{Ds}/{Ds}_{rawcooc,itemcf}_768d_sid.npy` |
| T5 ckpt | `model/ckpt/tiger_{tag}_{dsl}_seed{S}.pth` |
| T5 训练日志 | `model/logs/tiger_{tag}_{dsl}_seed{S}.log` |
| 评测日志 | `model/logs/{Ds}_{tag}_seed{S}_eval.log` |
| RQ-VAE 日志 | `rqvae/logs/e49_{dsl}_{variant}_rqvae.log` |
| 容量诊断 gate | `doc/e49_{dsl}_{tag}.md` |

**不做任何重命名/复制**（e48 orphan 事故教训）。既有 legacy 产物按原名复用，等价关系在 `run_behavior_ablation.sh` 输出中记录：

| 数据集 | 变体 | legacy codes | legacy tag |
|:--|:--|:--|:--|
| Sport | Shuffled | `Sport_shuffled_ablation_sid.npy` | `shuffled_ablation` |
| Sport | ZeroPad | `Sport_zeropad_ablation_sid.npy` | `zeropad_ablation` |
| Toys | ZeroPad | `Toys_zeropad_seed2024_sid.npy` | `zeropad_ablation` |
| Toys | Shuffled | 无 → 需全链路（behavior/fused 已于 2026-09-22 生成） | `shuffled_ablation` |

## 8. 环境与写入器

- **RQ-VAE / 量化端**：bare `python` = base env（`/home/media601/anaconda3/bin/python`，torch 2.6.0+cu124，pyarrow 22.0.0）——与全部既有 RQ-VAE 日志一致。
- **T5 端**：模型初始化与前向在 base(tf 4.57.6) 与 `tiger_t5`(tf 4.30.2) 下**逐字节相同**（已验：logits md5 `9ae0c796c0ef2699`，init md5 `f018cd56a6c986c8`）→ env 选择不影响结果。
- ⚠️ **md5 不是数据指纹**：跨 env 会让同数据产生不同 md5（arrow 22 vs 24）。比对须用数据级指纹或下游产物复现。见 [e49_toys_zeropad_md5_provenance.md](e49_toys_zeropad_md5_provenance.md)。

### 8.1 ⚠️ 发现：主表 Sport TA-SID 行的 codes 文件当前不在盘上

审计 P2 对照口径时发现（2026-09-22）：

- 主表 Sport TA-SID 的三个 T5 seed **共用同一个 codes 文件**——三份训练日志
  `tiger_ta_sid_sport_768d.log`(seed2025) / `tiger_ta_sid_768d_sport_seed2026.log` / `..._seed2027.log`
  内部均记录 `data/Sport/Sport_sid_codes_768d.npy`。
  → **与 E49 协议（一个变体 = 一次 RQ-VAE → 一个 codes → T5×3）逐位一致，P2 对照无口径混淆。**
- 但该文件 **当前不存在**（`data/Sport/*sid*.npy` 列表中无此项），且 `.npy` 被 `.gitignore` 忽略
  （`git log -- data/Sport/Sport_sid_codes_768d.npy` 为空），故无版本历史可查。
- 同名的 `Sport_ta_sid_768d_seed{2025,2026}_sid.npy` **不属于本行**：它们由
  [run_multiseed_repro.sh](../rqvae/run_multiseed_repro.sh) 产生，属 E4x「RQ-VAE 种子稳健性」研究
  （`--seed $S` 重训量化端，只做 gate、不进 T5）。**不要把两者混为一谈。**

**影响与处置**：
- **不影响 P2 决策门**：判据用的是既有 eval 日志里的数字，与本文件是否在盘无关。
- **可确定性重建**：`rqvae/ckpt/Sport_TA_SID_768d/` 与其 fused 输入
  `data/Sport/fused_item_emb_768d.parquet` 均在盘，`generate_sid_codes_param.py` 对
  (ckpt, data) 是确定性的 → 可在 GPU 空闲时重算并**逐位比对** `Sport_ta_sid_768d_sid.npy`。
  - 若 `np.array_equal` 为真 → 该文件只是被改名，provenance 完整，`run_behavior_ablation.sh` 的
    B2 legacy 回退逻辑已按原名记录等价关系。
  - 若为假 → 说明主行用的是另一套 codes，须补记 provenance 并核对其 eval 数字对应关系。
- **时机**：**不在 P2 期间做**（避免与 T5 抢 GPU）。排在 P2 结束、P3 启动前的空档执行。

## 9. 复现

```bash
# 全链路（幂等，可中断续跑）
bash rqvae/run_behavior_ablation.sh Sport rawcooc 2025 2026 2027
bash rqvae/run_behavior_ablation.sh Toys  itemcf  2025 2026 2027
# 只补 T5
bash rqvae/run_behavior_ablation.sh Toys t5-only Toys_zeropad_seed2024_sid.npy zeropad_ablation 2025 2026 2027
# 行为空间诊断
python tools/behavior_space_diagnostic.py --datasets Sport,Toys --report doc/e49_behavior_space_gate.md
```
