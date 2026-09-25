# TA-SID 消融实验：完整设计、结果与结论（三轮汇总）

**成文**：2026-09-23　**状态**：三轮消融已全部执行完毕（E49 36/36 路于 2026-09-23 14:40 完成）
**本文档定位**：**整个消融实验program的汇总结论**，跨三轮（2026-07-30~31 基础消融 / E49 2026-09-22~23 行为变体扩展 / E47 2026-09-04 量化端隔离）。
**权威数据源**：本轮 3-seed 数字以 [e49_results_corrected.md](e49_results_corrected.md) 为准；主表以 [paper_main_results.md](paper_main_results.md) 为准。

> ⚠️ **勿引**：[e49_behavior_ablation_results.md](e49_behavior_ablation_results.md)（P6 脚本产物，Toys TA-SID 行是作废的 0.0367、Toys Shuffled 行写"无产物"）；
> [e49_interim_report.md](e49_interim_report.md) §5（成文时 Toys 未跑，结论已被推翻）。

---

## 0. 一句话结论

**TA-SID 的增益来自"序列行为关联结构"本身**——不是维度（ZeroPad ≈ Content-only）、不是任意统计特性（Shuffled 两数据集同向掉点）、也不是"任何共现矩阵做 SVD"（Raw-CoOcc 增益跨数据集不一致）。
但**"行为空间可分性 → 下游收益"这个闭环不成立**，且 **PPMI 的增量跨数据集不一致**，论文须按数据集分别陈述。

---

## 1. 设计：三层递进，每层回答更尖锐的质疑

整个消融不是一次做成的，是**三轮**围绕"增益到底从哪来"逐步收窄。

### 1.1 第一层（2026-07-30~31）：排除"只是加了一路向量"

| 变体 | 构造 | 回答什么 |
|:--|:--|:--|
| Content-only 768d | 纯 `LN(content 768d)`，无 behavior 通道 | 公平基线 |
| **Zero Transition** | `LN(content 768d) ⊕ 0¹²⁸` → 896d | 是否只是加维度 |
| **Random Transition** | `LN(content) ⊕ LN(打乱的真实 behavior)` → 896d | 是否只需统计特性 |

**关键构造细节**（决定了论证强度）：

- **ZeroPad 垫的 128 维零，梯度恒为 0**（`∂L/∂w = ∂L/∂output · input`，input=0）→ 是"死参数"，**有效容量不变**。这比"零没信息"的说法更硬。
- **Random Transition 打乱的是 behavior embedding 的行**（`shuffled_beh = real_beh[perm]`），即每个商品拿到**另一个商品的真实行为向量**；脚本断言打乱前后 norm 均值不变 → 保留完整统计分布，只破坏 item↔向量对应。

> ⚠️ **术语注意（跨轮命名不一致）**：早期论文稿把"行级打乱"这个变体叫 **"Random Transition"**（见 `paper_draft_zh_sec53.md` 表 5-4），**E49 把它叫 "Shuffled"**。是同一个东西。
> 另有早期真正的 **"Random noise"**（谱匹配噪声）与其并存，已弃用，见 §4.2。

### 1.2 第二层（E49）：打到 PPMI 与 CF 两个要害

第一层只能证明"不是随便加向量就行"，回答不了两个更尖锐的问题（[e49_behavior_ablation.md](e49_behavior_ablation.md) §1）：

1. **PPMI 这一步是否必要？** 会不会"任何共现矩阵做 SVD"都行？
2. **TA-SID 是不是只是普通的 item-level 协同过滤表示？**

新增两个变体，**刻意设计成两种不同的对照强度**：

- **Raw-CoOcc-SVD —— 严格单变量对照**。调用与 TA-SID 完全相同的 `build_cooccurrence_matrix(window=3, min_cooccur=1)`，得到**同一个矩阵 C**（已用 nnz + 逐元素校验和断言：Sport `nnz=360722, sum=758172.0`；Toys `nnz=204302, sum=437444.0`），仅跳过 `compute_ppmi`。
  → "TA-SID − Raw-CoOcc"的差值可干净归因给 PPMI。
- **ItemCF-SVD —— 非单变量**。同时改变 ①共现作用域（整篮 vs 窗口=3）②归一化（cosine vs PPMI）③关联方向性。
  → **不得用于论证"TA-SID 优于 CF"或"排除 CF 解释"**。唯一定位："局部序列上下文结构是否比全局 item-level 协同信号更适合作为 SID augmentation？"两处同时变更须在论文中显式声明为局限。

**两个设计讲究**：

- **预注册措辞**（§3 在跑之前定稿）：`Raw-CoOcc ≪ TA-SID` → "PPMI 有效重塑行为空间"；`≈` → "PPMI 非必要，创新点表述须收缩"。**堵死事后找补空间。**
- **P1 离线自检**：`fused_item_emb_{rawcooc,itemcf}` 前 768 维与主表**逐位相同（maxdiff = 0.0e+00）** → 唯一变量确实在 behavior 侧。
- **术语统一**：不使用 "directional transition"/转移概率，一律 **"sequential behavioral association patterns"**。

### 1.3 第三层（E47，2026-09-04）：量化端健康度隔离

不比 NDCG，而是问：**L0 码本塌缩是 TA-SID 行为融合特有的，还是 content-only 也会塌？** 用"维度 × seed"矩阵隔离（Toys，L0 用码/256）：

| 输入 | seed2024 | seed2025 | seed2026 | 塌缩率 |
|:--|:--:|:--:|:--:|:--:|
| TA-SID 896d | **62 ⚠️** | 194 | 151 | 1/3 |
| ZeroPad 896d | **163 ✅** | 185 ✅ | — | 0/2 |
| Content-only 768d | 148 ✅ | 169 ✅ | 150 ✅ | 0/3 |

同一 seed、同一 896d 维度，**把行为通道换成 128 个零就健康** → 诱因是**行为内容**改变了量化难度，不是维度、不是 RQ-VAE 普遍问题。
（E48 后续修正：62 来自已被覆盖的 epoch 快照，幸存 seed2024 best_collision 实际健康 207/256。详见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md) §1 补充。）

### 1.4 七个变体总表

| # | 变体 | 共现作用域 | 权重变换 | vs TA-SID 改了什么 |
|:-:|:--|:--|:--|:--|
| 1 | Content-only | — | — | 无 behavior |
| 2 | ZeroPad | — | 128 个零 | 维度 |
| 3 | Random | — | 见 §4.2（描述待订正） | 任意向量（**单 seed 辅助，不参与显著性比较**） |
| 4 | Shuffled | 窗口=3 相邻 | PPMI（行级打乱） | item↔向量对应 |
| 5 | **Raw-CoOcc-SVD** | **窗口=3（与 TA-SID 逐位相同）** | **无（直接 SVD 原始计数）** | **仅去掉 PPMI** |
| 6 | **ItemCF-SVD** | **整篮（同一序列内所有商品对）** | **cosine 归一** | **作用域 + 归一化（两项）** |
| 7 | TA-SID (PPMI-SVD) | 窗口=3 相邻 | PPMI | 主方法 |

**共用**：同一 train.parquet、同一 canonical ItemID 顺序、同一 `svd_dim=128`、同一 `U·√S` 变换、同一冷启动物品置零、同一 `LN(content 768d) ⊕ LN(behavior 128d) → 896d`、同一 RQ-VAE（`256×256×256`）、同一 T5。

---

## 2. 结果

### 2.1 第一层（Sport，seed2025 单值，匹配口径）

| 变体 | N@5 | N@10 | **N@20** | R@5 | R@10 | R@20 |
|:--|:--|:--|:--|:--|:--|:--|
| w/o Transition（content-only 768d） | 0.0160 | 0.0211 | 0.0274 | 0.0256 | 0.0417 | 0.0664 |
| w/ Zero Transition（896d） | 0.0167 | 0.0223 | 0.0286 | 0.0261 | 0.0437 | 0.0688 |
| w/ Random Transition（896d，行级打乱） | 0.0149 | 0.0204 | **0.0262** | 0.0238 | 0.0409 | 0.0643 |
| **TA-SID**（896d） | 0.0192 | 0.0254 | **0.0327** | 0.0298 | 0.0490 | 0.0781 |

相对数（NDCG@20 / R@20）：TA-SID vs Random Transition **+24.8% / +21.5%**；vs Zero Transition **+14.3% / +13.5%**；vs content-only **+19.3%**。

### 2.2 第二层（E49，3-seed mean±std，36/36 路全齐）

| 变体 | Sport | vs content | Toys | vs content |
|:--|:--|:--:|:--|:--:|
| Content-only | 0.0275±0.0007 | — | 0.0336±0.0007 | — |
| Shuffled | 0.0264±0.0003 | −4.2% | 0.0284±0.0008 | **−15.3%** |
| ZeroPad | 0.0280±0.0006 | +1.7% | 0.0334±0.0012 | −0.4% |
| ItemCF | 0.0295±0.0004 | +7.0% | 0.0355±0.0002 | +5.9% |
| Raw-CoOcc | 0.0305±0.0006 | +10.9% | 0.0340±0.0015 | +1.3% |
| **TA-SID** | **0.0322±0.0005** | **+16.9%** | **0.0377±0.0007** | **+12.4%** |
| *Random（单 seed 辅助）* | *0.0281* | *+2.2%* | *未跑* | *—* |

```
Sport: TA-SID > Raw-CoOcc > ItemCF > ZeroPad > Content > Shuffled
Toys : TA-SID > ItemCF > Raw-CoOcc > Content ≈ ZeroPad > Shuffled
```

> 百分比由**逐 seed 原值**算均值得出。主表 [paper_main_results.md](paper_main_results.md) 的 +17.1% / +12.2% 用**四舍五入后的均值**，两者差 ≤0.2pp，引用时全文保持一致即可。

### 2.3 行为空间诊断（GPU-free，RQ-VAE 之前，独立于下游）

`cons_auc_local` = P(cos(序列相邻对) > cos(同篮非相邻对))，0.5 = 不区分：

| 数据集 | TA-SID | Raw-CoOcc | ItemCF | Random/Shuffled |
|:--|:--|:--|:--|:--|
| Sport | **0.6237** | 0.5730 | 0.4093 | ~0.4995 |
| Toys | **0.7157** | 0.6297 | 0.3494 | ~0.4995 |

其余判据：

| 判据 | 排序 | 支持"PPMI 有用"? |
|:--|:--|:--|
| `top1_energy` | TA-SID < Raw-CoOcc（两数据集一致） | ✅ |
| `pc1_cos_freq` | Sport: TA-SID ≪ Raw-CoOcc ✅；**Toys 反向** ❌ | ⚠️ 仅 Sport |
| `freq_leak` | **TA-SID > Raw-CoOcc（两数据集一致）** | ❌ **与预期相反** |
| `eff_rank` | 四者相近（119.9–125.7），无区分度 | — |

### 2.4 码本健康度（E49 四变体，`doc/e49_{ds}_{tag}.md`）

**没有任何变体塌缩。** 但 ItemCF 的 L0 利用率一致最低：

| 变体 | Sport L0 | Toys L0 | 每码 item max |
|:--|:--:|:--:|:--:|
| Raw-CoOcc | 251/256 | 222/256 | 260 / 174 |
| ItemCF | **195/256** | **161/256** | **500 / 277** |

---

## 3. 结论

### 3.1 站得住的（跨数据集一致）

1. **维度本身无功劳**。ZeroPad ≈ Content-only（+1.7% / −0.4%），两 seed 区间完全重叠。第一层质疑彻底关闭。
2. **行为信号本身携带信息**。Shuffled 两数据集**同向掉点**（−4.2% / −15.3%）。机制解释：RQ-VAE 被迫重建无意义的 128d 分量，浪费码本容量——**破坏对应关系不仅无益还有害**。
3. **PPMI 从不伤害**。`TA-SID > Raw-CoOcc` 两数据集都成立（+5.5% / +11.0%）。**这是本轮唯一跨数据集稳健的排序结论。**
4. **共现结构 + SVD 自身有信息**。Raw-CoOcc 在 Sport 上比 content-only 高 +10.9%，明显高于 ZeroPad/Shuffled。
5. **TA-SID > content-only** 在三数据集 9/9 个 seed 上成立，零 seed 交叉。

### 3.2 被推翻的

- **"行为空间可分性 → 下游收益"闭环不成立**。`cons_auc_local` 预测 TA-SID > Raw-CoOcc > ItemCF；Sport 完全同序，但 **Toys 上 ItemCF 反超 Raw-CoOcc**，而 ItemCF 的 `cons_auc_local` 仅 0.3494（**主动反向**）。按 [e49_behavior_space_analysis.md](e49_behavior_space_analysis.md) §5 **事先写好的规则**，该判据必须从"因果解释"**降级为"诊断性观察"**。
- **PPMI 增量跨数据集不一致**。TA-SID vs Raw-CoOcc = +5.5% / +11.0%，但 Raw-CoOcc vs content = **+10.9% / +1.3%**。按 §3 预注册二分，**Sport 指向 `≈`、Toys 指向 `≪`** → **论文必须按数据集分别陈述**，不能用统一说法。
- **Toys 上 Raw-CoOcc 的 +1.3% 不稳健**：seed 区间 [0.0324, 0.0354] 与 content-only [0.0328, 0.0341] 重叠。ItemCF [0.0354, 0.0357] 与 content 不重叠。

### 3.3 必须如实报告的负向结果

- **不能拿"PPMI 去流行度偏差"当卖点**（原计划预期 PPMI 降低 `freq_leak`，实测**两数据集一致相反**：Sport 0.8069 vs 0.7430；Toys 0.8288 vs 0.7525。且 `pc1_cos_freq` 仅 Sport 支持）。
- 能站住的是**方向**而非**范数**：原始计数矩阵的第一主方向几乎就是流行度方向（Sport `|cos|=0.6947`、`top1_energy` 0.0298），PPMI 把这部分方差摊开（0.3639 / 0.0192）。
  → 准确表述：**PPMI 重塑的是方差在方向上的分配（把流行度从"第一主方向"降级为众多方向之一），而非"范数与流行度的耦合强弱"。**
- `cons_auc`（相邻对 vs 随机对）**不可单独引用**——ItemCF 在该列高达 0.9055，看似最好，但那是**定义优势**（相邻对本身就是整篮对的子集）。只有 `cons_auc_local` 扣掉了泛共现。

### 3.4 可用作补充叙事

Toys 上 ItemCF > Raw-CoOcc 本身有解释力：**局部共现结构只有经 PPMI 重加权才压得过全局协同信号**；原始计数版的"局部上下文"打不过 ItemCF。码本旁证：ItemCF 的 L0 利用率最低（195/256、161/256），每码承载 item 最多 → 其行为空间按"泛热门共购"而非"序列邻接"组织。

---

## 4. 已知问题与待订正项

### 4.1 ✅ per-seed 明细的抄录错误（已定位，2026-09-23 核查完毕）

**结论：正确来源是 [experiment_results_summary.md](experiment_results_summary.md) 的「3 seed 明细（NDCG@20）」表——该表 9/9 行与 eval 日志逐位一致。[paper_main_results.md](paper_main_results.md) §3 是从它转录时抄错的副本。**

[paper_main_results.md](paper_main_results.md) §3 写 Sport TA-SID per-seed = `0.0327 / 0.0320 / 0.0320`，但全 `model/logs/` 库中**没有任何一行 NDCG@20 = 0.0320**。

**逐行核对（2026-09-23，脚本比对 eval 日志）**

| 行 | eval 日志实测 | experiment_results_summary | paper_main_results §3 |
|:--|:--|:--:|:--:|
| Sport TA-SID | 0.0327 / 0.0317 / 0.0322 | ✅ 一致 | ❌ `0.0327 / 0.0320 / 0.0320` |
| Sport Content-only | 0.0274 / 0.0283 / 0.0269 | ✅ 一致 | ✅ 一致 |
| Beauty TA-SID | 0.0418 / 0.0425 / 0.0403 | ✅ 一致 | ❌ `0.0418 / 0.0424 / 0.0403` |
| Beauty Content-only | 0.0368 / 0.0352 / 0.0354 | ✅ 一致 | ❌ `0.0357 / 0.0366 / 0.0351` |
| Toys TA-SID | 0.0369 / 0.0383 / 0.0380 | ✅ 一致 | ✅ 一致 |
| Toys Content-only | 0.0341 / 0.0338 / 0.0328 | ✅ 一致 | ❌ `0.0339 / 0.0331 / 0.0338` |
| SASRec ×3（源自 `sasrec_output/{ds}/summary_3seed.txt`） | — | ✅ 一致 | ✅ 一致 |

→ **`experiment_results_summary.md` 9/9 行正确；`paper_main_results.md` §3 三数据集 9 行中 4 行错**（Sport TA-SID 的 2026/2027、Beauty TA-SID 的 2026、Beauty Content 全 3 个、Toys Content 全 3 个）。
[paper_draft_zh_sec54_56.md](paper_draft_zh_sec54_56.md):43 是该错误表的副本，同样须订正。

**共同特征**：**均值都对得上（4 位小数），per-seed 对不上** —— 说明转录时以已知均值为准回填了明细列。

**影响面**：**均值不受影响** → 所有 headline 数字与"9/9 零 seed 交叉"的稳健性主张**用真实日志值重算依然成立**（已用日志值重算验证）。仅"审稿可复现用"的 per-seed 明细列需订正。

**处置**：以 [experiment_results_summary.md](experiment_results_summary.md) 的明细列为准，订正 `paper_main_results.md` §3 与 `paper_draft_zh_sec54_56.md` 的同名表（4 行）。
**状态**：✅ 已定位来源，🟡 待落地订正（尚未修改任何文件）。

### 4.2 ⚠️ E49 的 Random 辅助行描述与实际产物不一致

[e49_behavior_ablation.md](e49_behavior_ablation.md) §2 把 Random 描述为 **"i.i.d. 高斯"**。但该行数据取自 `Sport_random_ablation_eval.log` / `Sport_random_ablation_sid.npy`（值 0.0281），而 [naming_convention.md](naming_convention.md):26 记载 tag `random_ablation` = **"768d + 谱匹配噪声"**，状态 **⚠️ 已弃用**——弃用理由正是"谱匹配保留了 behavior 协方差结构，'随机'不纯粹"。

→ **文档描述与实际使用的产物不一致**。Random 是单 seed 辅助行、不进显著性比较，影响有限，但描述必须改对。
**状态**：🔴 待订正。

### 4.3 早期遗留：碰撞率两套口径矛盾（未裁定）

Sport 768d 碰撞率：一处记 TA-SID 10.66% / ZeroPad 1.63% / Shuffled 0.22%，另一处记 content-only 0.05%。[paper_draft_zh_sec53.md](paper_draft_zh_sec53.md):4 已注明"两套矛盾来源，不放入正文"。
**状态**：⚪ 保持不动即可，勿误引。

### 4.4 已弃用/作废清单（写作时避开）

| 项 | 状态 | 原因 |
|:--|:--|:--|
| 主实验 TIGER baseline（4096d/64d，+58%/+18%/+49.6%） | ❌ 全部作废 | 维度 + 数据管线混淆（TIGER 原始管线 `item_emb.parquet` 是 (259204, 64) 全量未 5-core） |
| Random noise 消融（谱匹配，0.0281，+35.7%） | ⚠️ 已弃用 | "随机"不纯粹（保留 behavior 协方差谱） |
| Toys TA-SID 0.0367±0.0006 / R@20 0.0873 | ❌ 作废 | 孤儿 codes（L0=62），无法从现存 ckpt 复现 → 订正为 **0.0377±0.0007 / 0.0911±0.0020** |
| Beauty TA-SID R@20 0.0933±0.0014 | ❌ 勘误 | 实测 3-seed → **0.0941±0.0016** |
| SASRec v1（2026-07-30） | ❌ 弃用 | off-by-one（实为 valid 任务）+ 10k 子集 + 单 seed，整体高估 ~16–21% |

**干净的 4096d 对照**（维度一致，可引用）：4096d TA-SID 0.0276 vs 4096d TIGER 0.0207 = **+33.3%**。
**Random SID 码质量消融**（可引用，单 seed）：Sport 0.0093 / Toys 0.0308 / Beauty 0.0263 → 证明 SID 语义结构本身是增益来源。

---

## 5. 待办

1. 🟡 按 §4.1 结论订正 `paper_main_results.md` §3 + `paper_draft_zh_sec54_56.md` 的 per-seed 表（4 行），以 [experiment_results_summary.md](experiment_results_summary.md) 为准。
2. 🔴 订正 4.2 的 Random 行描述。
3. 更新 [e49_interim_report.md](e49_interim_report.md) §4/§5（该 §5 整段过时）。
4. 定夺 Raw-CoOcc vs TA-SID 的**按数据集分述**措辞（Sport `≈` / Toys `≪`）。
5. `tools/extract_metrics.py` 的 LOG_MAP 补 E49 行（已确认目前**一行都没有**，`auto_metrics_table.md` 也不含 rawcooc/itemcf）。

---

## 6. 复现

```bash
# E49 全链路（幂等，可中断续跑）
bash rqvae/run_behavior_ablation.sh Sport rawcooc 2025 2026 2027
bash rqvae/run_behavior_ablation.sh Toys  itemcf  2025 2026 2027
# 只补 T5
bash rqvae/run_behavior_ablation.sh Toys t5-only Toys_zeropad_seed2024_sid.npy zeropad_ablation 2025 2026 2027
# 结果表（命名陷阱安全版）
python tools/e49_summary.py --out doc/e49_results_corrected.md
# 行为空间诊断
python tools/behavior_space_diagnostic.py --datasets Sport,Toys --report doc/e49_behavior_space_gate.md
```

## 相关

[e49_results_corrected.md](e49_results_corrected.md)（权威结果表）· [e49_behavior_ablation.md](e49_behavior_ablation.md)（设计与执行）· [e49_behavior_space_analysis.md](e49_behavior_space_analysis.md)（行为空间解读）· [e47_seed_health_isolation.md](e47_seed_health_isolation.md)（seed 隔离）· [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)（Toys codes 裁定）· [paper_main_results.md](paper_main_results.md)（主表）· [experiment_results_summary.md](experiment_results_summary.md)
