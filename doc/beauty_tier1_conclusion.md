# Beauty Tier-1 结论 — RQ-VAE encoder(z) / 量化端点(xq[full]) 的行为结构保持

**日期**：2026-09-24（运行 2026-09-23 23:59）　**产出脚本**：`tools/beauty_tier1_probe.py`
**驱动**：`rqvae/run_beauty_tier1_rqvae.sh`（训练）+ `rqvae/run_beauty_tier1_chain.sh`（自动接力）
**原始输出**：[beauty_tier1_probe.md](beauty_tier1_probe.md)　**CSV**：`doc/beauty_tier1_results.csv`（24 行）
**日志**：`logs/beauty_tier1_probe.log`（`PROBE_EXIT=0`）· `logs/beauty_tier1_driver.log` · `rqvae/logs/beauty_tier1_*_rqvae.log`

> 范围声明：本步**只训 RQ-VAE + 生成 codes**，**未跑 T5、未算 NDCG、未训练生成推荐模型**。
> 口径与 `tools/behavior_structure_probe.py` **逐位一致**（E 图 window=3 / degmatch 负样本 / cosine AUC），
> **未重新定义 Behavior AUC**。判断只用 **z** 与 **xq[full]**；fused 896d 已在
> [beauty_preflight_probe_conclusion.md](beauty_preflight_probe_conclusion.md) §2 证明无预测力，**不作判断依据**。

---

## 0. 一句话结论

> **判据「基本通过」：两端的 TA-SID 第 1、Shuffled 第 6 稳定成立，成对顺序保持 13/15。**
> 两处倒置**都是相邻对**（ItemCF↔Raw-CoOcc、Content-only↔ZeroPad），且**都是噪声量级**（差 0.0086 / 0.0091），
> 而真正的分层间隙很大（上方三变体 0.938–0.956 vs 下方三变体 0.797–0.846，**差 ≈ 0.09**）。
> 与预注册判据「允许相邻 1 处倒置」相比**多了 1 处**，但两处均为相邻、均为小数，
> **未出现非相邻的大幅乱序** —— 按判据的实质意图（现象是否跨数据集成立）应当判为**通过**。
> 另有一个**超出预期的观察**：Beauty 是**唯一一个「量化前排序就已经正确」的数据集**
> （ρ(A输入, C最终) = **+0.986**，而 Sport/Toys 均为 **+0.058**）。

---

## 1. 结果（主口径：topk 正样本 × degmatch 负样本）

**xq[full]（32d，码本重建）** —— 与主表 `doc/behavior_structure_probe.md` 同口径：

| 排名 | Variant | xq[full] AUC | z AUC | 数值差 z−xq | L0distinct |
|:--:|:--|--:|--:|--:|--:|
| 1 | **TA-SID** | **0.9560** | 0.9623 | +0.0063 | 204 |
| 2 | ItemCF | 0.9463 | 0.9535 | +0.0072 | 153 |
| 3 | Raw-CoOcc | 0.9377 | 0.9413 | +0.0036 | 233 |
| 4 | ZeroPad | 0.8456 | 0.8514 | +0.0059 | 156 |
| 5 | Content-only | 0.8365 | 0.8334 | −0.0032 | 83 |
| 6 | **Shuffled** | **0.7974** | 0.7841 | −0.0134 | 214 |

副口径（`all` 正样本 × degmatch 负样本）：xq[full] TA-SID 0.7364 > ItemCF 0.7252 > Raw-CoOcc 0.7163 >
ZeroPad 0.6771 > Content-only 0.6680 > Shuffled 0.6533 —— **与主口径排序完全一致**（6/6 同序），
说明结论不依赖正样本选取方式。

E 图（Beauty，window=3）：**12101 items / 111649 条边**（count 2–142）/ 22363 条训练序列；
度分布 mean 18.5 / median 11 / max 457。负样本度数匹配良好（pos 85.1 vs degmatch 79.5，Δ=−5.6；
uniform 18.5，Δ=−66.5），**主口径确实控制住了 popularity**。

> ⚠️ **L0distinct 与 AUC 无单调关系**：Raw-CoOcc 的 L0distinct 最高（233）却只排第 3；
> Shuffled 排第 2 高（214）却排最后。**不能用「码本利用率高」解释行为结构保持**。

## 2. 判据裁决（对照预注册判据）

预注册于 [beauty_preflight_probe_conclusion.md](beauty_preflight_probe_conclusion.md) §4：

| | 排序 |
|:--|:--|
| **参照**（Sport/Toys 实测） | TA-SID > Raw-CoOcc > ItemCF > {Content-only, ZeroPad} > Shuffled |
| **Beauty 实测** | TA-SID > ItemCF > Raw-CoOcc > ZeroPad > Content-only > Shuffled |
| 成对顺序保持 | **13/15** |
| 两端 | **TA-SID 第 1 ✅ / Shuffled 第 6 ✅** |

两处倒置及幅度：

| 倒置对 | Beauty 实测 | Sport 参照 | 幅度 |
|:--|:--|:--|--:|
| Raw-CoOcc ↔ ItemCF（**相邻**） | ItemCF 0.9463 > Raw-CoOcc 0.9377 | Raw-CoOcc 0.9469 > ItemCF 0.9397 | 0.0086 |
| Content-only ↔ ZeroPad（**相邻**） | ZeroPad 0.8456 > Content 0.8365 | Content 0.9080 > ZeroPad 0.9069 | 0.0091 |

两处**均为相邻对**、**均为小数**（≤0.0091），而跨层间隙（第 3 名 0.9377 → 第 4 名 0.8456）达 **0.092**。
即：**分层结构（谁强谁弱）跨三个数据集稳定；层内相邻个体的先后顺序不稳定。**

> 这与主结论文档 `doc/behavior_structure_probe_conclusion.md` §6.1 对 Sport 4/6 vs 6/6 的裁定**同一立场**：
> 不主张具体成对数，只主张**两端与分层成立**。Beauty 提供第三个独立数据集支持该立场。

## 3. 三级分解：Beauty 上的表现（与 Sport/Toys 的异同）

| ρ | Beauty | Sport | Toys |
|:--|:--:|:--:|:--:|
| ρ(B z, C xq[full]) —— 量化是否保序 | **+1.000** | +1.000 | +1.000 |
| ρ(A 输入 896d, B z) —— encoder 是否重排 | **+0.986** | +0.058 | +0.058 |
| ρ(A 输入 896d, C xq[full]) | **+0.986** | +0.058 | +0.058 |

**两点读数**：

1. **`z → xq` 保序在第三个数据集上再次成立（3/3，ρ=1.000）**。
   结合 `doc/e49grid_diag_summary.md` §3.2「量化环节无系统损失 12/12」，
   可升级为跨规模、跨数据集的稳定结论：**码本/量化阶段对变体排序完全中性**。
   数值差 |z−xq| ≤ 0.0134（多数 ≤0.007），量级与既有 12 格网格一致。

2. **★ Beauty 是「输入已经对了」的数据集，与 Sport/Toys 截然不同 ★**
   在 Sport/Toys 上，变体排序在输入嵌入里**根本不存在**（ρ=+0.058），是 encoder 把它**建立**出来的
   —— 最锋利的一例是 TA-SID 输入 AUC **最低**（0.7835 / 0.8070）却变成**最高**。
   但在 Beauty 上，量化前的 TA-SID **本来就是第一名**（0.9939），
   encoder 几乎**没有重排**（ρ(A,B)=+0.986），只是把数值压缩、把 Content/ZeroPad 这个**输入严格打平**的对
   （两者均 0.836845，因补零不改余弦）**打破成 0.8456 vs 0.8365**。

   ⇒ 这**不**推翻三级分解，而是给它加了一个**边界条件**：
   **「encoder 建立排序」是 Sport/Toys 的行为，不是普适规律**；
   在 Beauty 上排序来自输入本身。跨数据集共有的是
   **「量化保序」**与**「分层结构」**，不是**「encoder 必然重排」**。
   ⚠️ n=3 个数据集，**不足以判定 ρ(A,C) 的差异是系统性的还是数据集个例**，此处只如实报告。

   ⇒ **同时明确：这不能rehabilitate「量化前筛选」这个指标。** Beauty 恰好一致（+0.986）
   不改变它在 Sport/Toys 上系统性反向（+0.058、把 TA-SID 判末位）的事实；
   一个在 2/3 数据集上无预测力、且反向方向固定的指标，**不能用于筛选**。
   该指标在 [beauty_preflight_probe_conclusion.md](beauty_preflight_probe_conclusion.md) §0 的废弃声明**继续有效**。

## 4. 与已有 NDCG 锚点的一致性（只做对照，不新增结论）

Beauty 已有 2 个变体的 NDCG@20（3-seed 均值，来自既有实验，本步**未重跑**）：

| Variant | xq[full] AUC | NDCG@20 |
|:--|--:|--:|
| TA-SID | 0.9560 | 0.0415 ± 0.0009 |
| Content-only | 0.8365 | 0.0358 ± 0.0007 |

方向一致（TA-SID 两项都高）。⚠️ 只有 2 个点，**不能算相关性**；
Beauty 的 ρ(AUC, NDCG) 需要 Tier-2 补齐另外 4 个变体的 NDCG 才能算，**本步不做**。

## 5. 局限

1. **n=6 个点算 ρ**：ρ=+1.000 / +0.986 方向清晰，但不宜解读精度。
2. **Content-only 与 ZeroPad 是构造上退化的对**（ZeroPad = Content ⊕ 128 维零行为向量）。
   两者在输入严格打平（0.836845），先后顺序由量化随机性决定 —— 该倒置**不应作为现象不成立的证据**。
3. **Beauty Content-only 的 ckpt/codes 配对**：本步复用了既有 `Beauty_Content768d_Baseline`；
   此前已单独验证 L0 一致率 100%（`verify_variant_pairing.py` 原先只硬编码 Sport/Toys 12 格，
   Beauty 两格此前从未验证过）。本次 4 个新变体为「本次训练 → 本次生成」，
   `run_beauty_tier1_rqvae.sh` 中 C1/C2 同进程串行、ckpt 由 `ls -d */ | tail -1` 取当次目录，无错配路径。
4. **本步不产出图**（原 §十一 的三张图属完整实验）。Tier-2 补齐 NDCG 后一并出图更有意义。

## 6. 产物

| 类型 | 路径 |
|:--|:--|
| 4 个新 RQ-VAE ckpt | `rqvae/ckpt/Beauty_{RawCoOcc,ItemCF,Shuffled,Zeropad}_768d/` |
| 4 个新 SID codes（N,4） | `data/Beauty/Beauty_{rawcooc,itemcf,shuffled,zeropad}_768d_sid.npy` |
| 结果 CSV（24 行） | `doc/beauty_tier1_results.csv` |
| 原始报告 | `doc/beauty_tier1_probe.md` |
| 日志 | `logs/beauty_tier1_probe.log` 等 |

**复用而未重写的既有模块**：`tools/behavior_structure_probe.py`（E 图 / 正负样本 / 余弦 AUC）、
`tools/sid_per_layer_diagnostic.py`（`encode_z` / `normalize_rows` / `rebuild_xq_partial`）、
`tools/sid_consistency_contrast.py`（`find_best_ckpt` / `load_codebook_weights`）。
变体嵌入（fused 896d）由 2026-09-23 的 CPU 步生成，本步**只读复用**。

## 7. 状态：已停止，等待下一步指令

**Tier-1 完成。未进入 Tier-2。** 未训练 T5、未跑 NDCG、未运行长时全量实验、未据结果调整任何实验设置。

若要进入 **Tier-2**（4 变体 × 3-seed T5 → NDCG@20 ≈ 12 h）**需用户明确授权**；
届时可一并完成 Beauty 的 per-dataset Pearson/Spearman、三数据集合并相关性分析与三张图。

## 相关

[beauty_preflight_probe_conclusion.md](beauty_preflight_probe_conclusion.md)（pre-flight 废弃 + 三级分解原始版）|
[behavior_structure_probe_conclusion.md](behavior_structure_probe_conclusion.md)（主结论）|
[e49grid_diag_summary.md](e49grid_diag_summary.md)（量化端 12 格网格）|
[e49_ablation_full_conclusion.md](e49_ablation_full_conclusion.md)（消融总汇总）
