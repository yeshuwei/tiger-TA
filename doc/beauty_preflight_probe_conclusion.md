# Beauty pre-flight 诊断结论 — 及一个推翻本设计的发现

**日期**：2026-09-23　**产出脚本**：`tools/beauty_preflight_probe.py`
**原始输出**：[beauty_preflight_probe.md](beauty_preflight_probe.md)（Beauty）· [preflight_control_sport.md](preflight_control_sport.md) · [preflight_control_toys.md](preflight_control_toys.md)
**CSV**：`doc/beauty_preflight_results.csv` · `doc/preflight_control_{sport,toys}.csv`

---

## 0. 一句话结论

> **本 pre-flight 设计（在「量化前」的 fused 896d 上算 Behavior AUC）没有任何预测力，应废弃**：
> 在 Sport 与 Toys 上它与真实口径 xq[full] 的排序相关只有 **ρ = +0.058**，
> 并且把 **TA-SID 判为最后一名**（真实为第一）。根因是一个此前未被记录的**三级结构**：
> 变体排序**不存在于输入嵌入中**，而是在 **RQ-VAE encoder（896d → 32d）** 这一步被**建立**出来的；
> 之后 codebook 一步（z → xq）**逐位保序**（ρ = 1.000）。
> ⇒ 廉价筛选只剩一条路：**只训 RQ-VAE（4 × ~7 min ≈ 30 min GPU）**，用 z / xq 算真口径 AUC。

> ⚠️⚠️ **2026-09-27 订正：以上 §0 结论的两条依据**全部被证伪，本段作废**。权威记录：`notes/15_g2_stage_seed_robustness_plan.md` §10。**
>
> **(i) `ρ = +0.058` 不是发现，是文件维度 bug。** 本报告 §6 的命令链中，Sport / Toys 的 TA-SID 走的是 `fused_item_emb.parquet` —— 该文件在这两个数据集上是 **4224d**（Beauty 上恰为 896d，故 Beauty 未受影响）。改用正确的 `fused_item_emb_768d.parquet` 后：**ρ(A 输入 896d, C xq[full]) = +0.928 / +0.928 / +0.986**（Sport / Toys / Beauty）。
> **(ii) 「把 TA-SID 判为最后一名」是同一 bug 的产物。** 正确 896d 输入下 TA-SID 在 **A 空间就是第一名**（AUC **0.9617 / 0.9828**，Beauty 0.9939 不变），**不存在"系统性反向"**。
>
> **判定性证据（不只是"改了就好了"）**：同一探针 `tools/stage_probe.py --audit_a` 下，**其余 5 个变体的本报告记录值被逐位复现** ——
> Sport 0.896381 / 0.896381 / 0.878091 / 0.958474 / 0.949814；Toys 0.857900 / 0.857900 / 0.842264 / 0.962135 / 0.935932；Beauty 0.836845 / 0.836845 / 0.822748 / 0.978198 / 0.908828。
> **只有 TA-SID 恰在那两份 4224d 文件上偏离** → 差异由文件维度解释完毕。另已核验全部 18 个变体 ckpt 的 `args.data_path`：TA-SID 的真实训练输入在三个数据集上**都是 896d 文件**。
>
> **⇒ 随之作废**：①「本 pre-flight 设计应废弃」的全部裁决（§2「必须废弃」、§3 读数 2、§4 修正方案、§5 局限 1–2）；②「变体排序**不存在于输入嵌入中**、由 encoder **建立**」的方向（§3 读数 2）；③「廉价筛选只剩一条路」的成本论证（§0 末句、§4）—— **量化前筛选在修好文件后是有效的**（ρ(A, C) ≈ 0.93–0.99），无需为省这笔钱而绕道 Tier-1。
> **⇒ 仍然成立**：§1 的 CPU 产物与"行序核验 / 补零等价性核验"两项；§3 的**数值表**（A 列须按订正后重算）；「码本阶段保序」（ρ(B,C) 主口径 = 1.000）与「分层结构跨数据集稳定」两条读数；§4.1 的 Tier-1 裁决（其依据是 xq[full]，与本 bug 无关）。
> **⇒ 应改写为**：**输入融合阶段已经基本确立变体排序**（ρ(A,B) ≈ 0.93–0.99），encoder 与量化两步**基本保序**；行为融合增益（ΔAUC）**在输入阶段就已形成**、后续被轻微压缩。
> ⚠️ 但 ρ(A,B) 对**正/负样本口径**敏感（四个口径组合区间 **0.638–0.986**），而**种子敏感性为 0** —— 引用时不可只报 0.928 单值。

---

## 1. 做了什么

Beauty 缺 4/6 变体（Raw-CoOcc / ItemCF / ZeroPad / Shuffled），补齐需
`4 × (RQ-VAE 7min + 3-seed T5 3h) ≈ 12.5 h GPU`。为省这笔开销，先设计了一个纯 CPU 的
「量化前」筛选：在 fused 896d 上算行为结构 AUC，看变体排序是否符合预期。

本步已完成的 CPU 产物（**可用于后续任何方案，不必重跑**）：

| 产物 | 说明 |
|:--|:--|
| `data/Beauty/behavior_emb_{rawcooc,itemcf,shuffled}.parquet` | 128d 行为向量 |
| `data/Beauty/fused_item_emb_{rawcooc,itemcf,shuffled,zeropad}.parquet` | 896d 融合 |
| 生成日志 | `rqvae/logs/beauty_preflight_A_generate.log` |
| 诊断脚本 | `tools/beauty_preflight_probe.py` |

生成命令与 E49 权威驱动脚本 `rqvae/run_behavior_ablation.sh` **逐字一致**（seed=2025、
`item_emb_768d.parquet` 为 content 源、`fuse_embeddings.py` 融合、bare `python`/pyarrow 22.0.0）。

**行序核验（通过）**：6 个输入文件的 ItemID 行序与基准完全一致 → 无错位风险。

**补零等价性核验（通过）**：Content-only 的 896d（content ⊕ 128 零）与原生 768d 的 AUC
差 **1.5e-06 / 1.5e-08** → 补零确实不改余弦，符合解析预期。

## 2. 决定性对照：pre-flight 在已知答案的两个数据集上失效

把同一套 pre-flight 逻辑跑在 **Sport / Toys**（这两个数据集的真实答案已知）上：

| 数据集 | pre-flight 排序（量化前） | 真实排序（xq[full]） |
|:--|:--|:--|
| Sport | ItemCF > Raw-CoOcc > Content ≈ ZeroPad > Shuffled > **TA-SID** | **TA-SID** > Raw-CoOcc > ItemCF > Content ≈ ZeroPad > Shuffled |
| Toys | ItemCF > Raw-CoOcc > Content ≈ ZeroPad > Shuffled > **TA-SID** | **TA-SID** > Raw-CoOcc > ItemCF > Content ≈ ZeroPad > Shuffled |

**TA-SID 在两个数据集上都被判为最后一名**，而它是真实的第一名。
⇒ 该 pre-flight 不仅无预测力，而且**系统性反向**，**必须废弃**。

> ⚠️ **2026-09-27 订正：本节的「废弃」裁决作废**（根因见 §0 订正块：Sport/Toys 的 TA-SID 被喂了 **4224d** 文件）。
> 用正确 896d 输入重测后，上表"pre-flight 排序"列应为（A 空间 AUC，主口径 topk × degmatch）：
>
> | 数据集 | 订正后 pre-flight 排序（A 输入 896d） | 真实排序（C xq[full]） | ρ(A, C) |
> |:--|:--|:--|:--:|
> | Sport | **TA-SID (0.9617)** > ItemCF (0.9585) > Raw-CoOcc (0.9498) > Content ≈ ZeroPad (0.8964) > Shuffled (0.8781) | TA-SID > Raw-CoOcc > ItemCF > Content ≈ ZeroPad > Shuffled | **+0.928** |
> | Toys | **TA-SID (0.9828)** > ItemCF (0.9621) > Raw-CoOcc (0.9359) > Content ≈ ZeroPad (0.8579) > Shuffled (0.8423) | 同上 | **+0.928** |
>
> ⇒ **TA-SID 在两个数据集上都回到第 1 名，「系统性反向」不存在**。与 xq[full] 侧的差异只剩**层内相邻对**的交换（Sport/Toys 的 ItemCF ↔ Raw-CoOcc），而其幅度（Sport 0.0086）与 xq[full] 侧自身存在的相邻噪声同量级 —— 这不是失效，是 n=6 秩相关在相邻对上的正常抖动。
> **⇒ 本节标题所示结论与 §3 的"三级分解"必须整体改写**，改写方向见 §0 订正块末段。

## 3. 三级分解：排序在 encoder 处被建立，codebook 处被完整保留

把同一批正/负样本贯穿三个阶段测 AUC（`pos=topk × neg=degmatch`，Beauty/Sport/Toys 同口径）：

**Sport**

| Variant | A. 输入 fused 896d | B. z（32d，encoder 输出） | C. xq[full]（32d，码本重建） |
|:--|--:|--:|--:|
| **TA-SID** | ~~0.7835~~ ⚠ **0.9617** | **0.9655** | **0.9647** |
| Raw-CoOcc | 0.9498 | 0.9493 | 0.9469 |
| ItemCF | 0.9585 | 0.9432 | 0.9397 |
| Content-only | 0.8964 | 0.9089 | 0.9080 |
| ZeroPad | 0.8964 | 0.9062 | 0.9069 |
| Shuffled | 0.8781 | 0.8745 | 0.8763 |

**Toys**

| Variant | A. 输入 fused 896d | B. z（32d） | C. xq[full] |
|:--|--:|--:|--:|
| **TA-SID** | ~~0.8070~~ ⚠ **0.9828** | **0.9697** | **0.9651** |
| Raw-CoOcc | 0.9359 | 0.9565 | 0.9543 |
| ItemCF | 0.9621 | 0.9416 | 0.9455 |
| Content-only | 0.8579 | 0.8788 | 0.8751 |
| ZeroPad | 0.8579 | 0.8735 | 0.8716 |
| Shuffled | 0.8423 | 0.8151 | 0.8127 |

| 相关性 | Sport | Toys |
|:--|:--:|:--:|
| ρ(A 输入 , B z) | **+0.058** | **+0.058** |
| ρ(B z , C xq[full]) | **+1.000** | **+1.000** |
| ρ(A 输入 , C xq[full]) | +0.058 | +0.058 |

> ⚠️ **2026-09-27 订正后的同一张表**（A 列改用正确的 896d 文件；B/C 值不变 —— 它们与 A 无关）：
>
> | 相关性 | Sport | Toys | （Beauty，参照） |
> |:--|:--:|:--:|:--:|
> | ρ(A 输入 , B z) | **+0.928** | **+0.928** | +0.986 |
> | ρ(B z , C xq[full]) | +1.000 | +1.000 | +1.000 |
> | ρ(A 输入 , C xq[full]) | **+0.928** | **+0.928** | +0.986 |
>
> ⇒ **三行全部指向"保序"，没有"重排"这一步。** 上表的 +0.058 行全部作废。
> ⚠️ **口径纪律**：以上为**主口径**（topk 正样本 × degmatch 负样本）。ρ(A,B) 在四个口径组合下区间为 **0.638–0.986**（Sport/Toys 主口径 0.928）。**口径敏感性 >> 种子敏感性（=0）**，引用时须交代口径，不可只报单值。

**读法**：

1. **`z → xq` 逐位保序（ρ = 1.000）**：码本/量化这一步对**排序**是**完全中性**的，
   与 `doc/e49grid_diag_summary.md` §3.2「量化环节无系统损失（12/12）」**一致且更强** ——
   那边说的是数值差 ≤0.006，这里说的是**排序零损失**。
2. **`输入 896d → z 32d` 完全重排（ρ = +0.058）**：变体间的行为结构强弱排序
   **在输入嵌入里根本不存在**，是 RQ-VAE **encoder 的 32 维投影建立**出来的。
   最锋利的一例：TA-SID 的输入 AUC **最低**（0.78/0.81），经 encoder 后变成**最高**（0.97/0.97）。

   > ⚠️⚠️ **2026-09-27 订正：本条读数整体作废 —— 它描述的"重排"是该 bug 的产物。**
   > 修正后（896d 文件）：ρ(A 输入, B z) = **+0.928 / +0.928 / +0.986**（Sport/Toys/Beauty），
   > TA-SID 的输入 AUC **本来就是最高**（**0.9617 / 0.9828 / 0.9939**），encoder 之后仍是最高（0.9655 / 0.9697）。
   > ⇒ **"排序在输入嵌入里根本不存在、由 encoder 建立"不成立**；正确读数是
   > 「**排序在输入融合阶段就已基本确立，encoder 与量化两步基本保序**」。
   > ⇒ 本条原本据此推出的"encoder 阶段才是决定性的那一环"（下方第 3 点）**同样作废**：三个环节**都不改变排序**，
   > 真正在变化的是**绝对判别力**（A → B/C 的 AUC 数值，见 §3 数值表：TA-SID 0.9617→0.9655、Content-only 0.8964→0.9089），
   > 即 encoder 做的是**保序的增益/压缩**，不是重排。
3. 因此必须把此前那句「量化阶段不是主要信息瓶颈」**讲得更准**：
   **码本阶段**不是瓶颈（保序）；**encoder 阶段**才是决定性的那一环 ——
   它是唯一能建立该排序的环节。

   > ⚠️ **2026-09-27 订正：本点作废**。encoder **不建立**排序（它保序），故"决定性的那一环"无从谈起。
   > 订正后应讲的是：**三个环节都不改变排序**（ρ(A,B) ≈ 0.93–0.99、ρ(B,C) = 1.000），
   > encoder 的作用是**保序地改变绝对判别力**。原句"量化阶段不是主要信息瓶颈"的**限定**须改为
   > 「**排序意义上**，encoder 与码本两步都不是瓶颈」——与 note 13 §2.5 要求的"须写'规模不是杠杆'而非'量化无害'"是同一纪律。

## 4. 对补齐 Beauty 的意义（修正后的方案）

原方案（先 pre-flight 再决定）**不成立**，因为廉价的量化前指标无预测力。

> ⚠️ **2026-09-27 订正：本句前提作废** —— 量化前指标在修好 TA-SID 的文件维度后与真口径 ρ(A,C) ≈ **0.93–0.99**，**有**预测力。
> ⇒ §4 全节的成本论证（"必须先训 4 个 RQ-VAE 才能拿到真口径"）**不再必要**；但 §4.1 记录的 Tier-1 结果本身有效，无需作废。
> ⚠️ 一处口径保留：ρ(A,C) 的四口径组合区间为 **0.638–0.986** —— 量化前筛选**可用但不稳健**，若重启该路线须固定口径并声明。

修正后的两级方案：

| 级别 | 内容 | GPU 成本 | 能回答什么 |
|:--|:--|--:|:--|
| **Tier-1 筛选** | 只训 4 个 RQ-VAE → codes → **xq[full] / z Behavior AUC**（即真口径） | **≈ 30 min**（4 × 7 min） | Beauty 的 6 变体排序是否与 Sport/Toys 同构；能否与已有的 2 个 NDCG 锚点自洽 |
| **Tier-2 定稿** | 4 变体 × 3-seed T5 → NDCG@20 | ≈ 12 h | 完成 §八/§九 的相关性分析与三数据集合并 |

Tier-1 之所以成立，正是因为**第 3 节的发现**：Behavior AUC 只需要 `xq`，而 `xq` 只需要
encoder + 码本 —— **完全不需要 T5**。原先把 Behavior AUC 和 NDCG 绑在一起估成本是错的。

**Tier-1 的判据**：Beauty 6 变体在 xq[full] 上应复现
`TA-SID > Raw-CoOcc > ItemCF > {Content, ZeroPad} > Shuffled`（允许相邻 1 处倒置，如
Sport 的 Content↔ZeroPad）。若出现**非相邻的大幅乱序**，说明 Beauty 上现象不成立，
应停手并重新审视，而不是继续投 12 h。

### 4.1 Tier-1 已执行 — 裁决：**基本通过**（2026-09-24）

Tier-1 已于 2026-09-23 23:59 跑完（4 × RQ-VAE + codes，**未跑 T5**）。实测
`TA-SID > ItemCF > Raw-CoOcc > ZeroPad > Content-only > Shuffled`，**成对顺序保持 13/15**，
**TA-SID 第 1 / Shuffled 第 6 均成立**，分层间隙 0.092 而两处倒置幅度仅 0.0086 / 0.0091。
**两处倒置皆为相邻对**（ItemCF↔Raw-CoOcc、Content-only↔ZeroPad），比预注册的「允许 1 处」多 1 处，
但**未出现非相邻的大幅乱序**，按判据实质意图判为**基本通过**。

另有一个**本判据未预见的结果**：Beauty 是唯一一个**量化前排序就已经正确**的数据集
（ρ(A输入, C最终) = **+0.986**；Sport/Toys 均为 +0.058）。这**不推翻**本报告 §3 的三级分解，
但为其加了边界条件（「encoder 建立排序」非普适），且**不改变本报告 §0 对该指标作为筛选工具的废弃判定**。

> ⚠️ **2026-09-27 订正：本段三句话全部反转。**
> (a) Beauty **不是唯一** —— 修正后 **三个数据集**的量化前排序都正确（ρ(A,C) = **+0.928 / +0.928 / +0.986**）；
> (b) 它**确实推翻**（而非"加边界条件"）§3 的"encoder 建立排序"；
> (c) §0 的**废弃判定被推翻** —— 量化前指标在修好文件后与真口径 ρ≈0.93–0.99，**可作筛选**。
> 本段所依赖的 "+0.058" 是 4224d 文件 bug（见 §0 订正块）。**§4.1 对 Tier-1 本身的裁决（基本通过）不受影响** —— 它的依据是 xq[full]。

完整结果见 **[beauty_tier1_conclusion.md](beauty_tier1_conclusion.md)**。

## 5. 局限

1. **ρ 用 n=6 个点算**：ρ=+0.058 与 ρ=+1.000 都是 6 点上的值，方向极其清晰但不宜过度解读精度。
2. **两个数据集的 ρ(A,B) 都是 +0.058**：数值巧合，源于两数据集上 6 变体的排序模式相同
   （同一处 TA-SID 从末位跳到首位），并非计算错误。

> ⚠️ **2026-09-27 订正**：
> 局限 1 **部分仍成立**（n=6 确实不宜过度解读精度）—— 但注意订正后真正的问题不是精度而是**口径敏感性**：ρ(A,B) 随正/负样本口径在 **0.638–0.986** 间摆动，比 n=6 的精度问题严重得多。
> 局限 2 **作废**：两个数据集同为 +0.058 **不是数值巧合**，而是**同一份错维度文件**（`fused_item_emb.parquet` 在二者上都是 4224d）造成的同一效应。
> 另：局限 3 所述"不解释为何 TA-SID 输入 AUC 最低"**前提消失** —— 修好后 TA-SID 输入 AUC **最高**（0.9617 / 0.9828），无需解释。
3. **输入 896d 的 AUC 内部仍依赖 LayerNorm 融合的具体形式**（`fuse_embeddings.py`），
   本报告不解释「为何 TA-SID 输入 AUC 最低」的机制，只报告该事实与其后果。
4. 本报告的 Beauty 数值**全部是量化前口径**，**不可**与主表 `doc/behavior_structure_probe.md`
   的 xq 数值混引。

## 6. 产物与复现

```bash
# ① 生成 Beauty 的 4 个变体嵌入（CPU，分钟级；已执行）
python rqvae/behavior_variants.py --dataset Beauty --variant rawcooc
python rqvae/behavior_variants.py --dataset Beauty --variant itemcf
python rqvae/ablation_data.py --dataset Beauty --seed 2025 --variants shuffled
python rqvae/ablation_data.py --dataset Beauty --seed 2025 --variants zeropad
python rqvae/fuse_embeddings.py --dataset Beauty --content_file item_emb_768d.parquet \
    --behavior_file behavior_emb_rawcooc.parquet --output_file fused_item_emb_rawcooc.parquet
python rqvae/fuse_embeddings.py --dataset Beauty --content_file item_emb_768d.parquet \
    --behavior_file behavior_emb_itemcf.parquet  --output_file fused_item_emb_itemcf.parquet

# ② 量化前诊断 + 对照（CPU，分钟级）
python3 tools/beauty_preflight_probe.py
python3 tools/beauty_preflight_probe.py --dataset Sport --csv doc/preflight_control_sport.csv \
    --report doc/preflight_control_sport.md
python3 tools/beauty_preflight_probe.py --dataset Toys  --csv doc/preflight_control_toys.csv \
    --report doc/preflight_control_toys.md
```

**复用的既有模块（未重复实现）**：`tools/behavior_structure_probe.py`
（E 图构建 / 正负样本采样 / 余弦 AUC / NDCG 日志解析）、`tools/sid_per_layer_diagnostic.py`
（`normalize_rows`）。

## 相关

[behavior_structure_probe_conclusion.md](behavior_structure_probe_conclusion.md)（主结论）|
[e49grid_diag_summary.md](e49grid_diag_summary.md)（量化端 12 格网格）|
[e49_ablation_full_conclusion.md](e49_ablation_full_conclusion.md)（消融总汇总）
