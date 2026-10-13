# Semantic ID × Behavior Structure —— 代码可行性审计

> **审计性质**：code feasibility audit（只读）。回答的唯一问题：**当前代码距离回答 RQ1–RQ6 还差什么。**
> **审计日期**：2026-10-08 ｜ **分支** `experiments-2026-09-25`
> **范围约束**：未修改任何代码、未运行任何实验。所有结论均来自对现有源码/产物的阅读与核对。
> **术语纪律**：下文"直接支持 / 少量修改 / 较大修改 / 不支持"严格按第 5 节矩阵定义，不把"理论上可做"写成"已支持"。

---

## 0. 审计摘要（先读这段）

| RQ | 一句话结论 | 主要缺口 |
|:--|:--|:--|
| **RQ1** | 连续 vs 量化 的核心对比**已经跑过**（3 数据集 × 6 变体 × 3 空间），但**"逐 item 对齐导出 5 类表示"没有现成脚本** | 缺一个把 input/fusion/z/xq/SID 对齐到同一 item 索引并统一导出的探针 |
| **RQ2** | **逐层累积表示直接可得**（`xq[L0] / xq[L0+L1] / xq[full]`，已有脚本产出）；但**单层 residual vector 与单层 loss 不直接返回** | `rq.forward` 只返回**累积** `x_q` 与**平均** loss，不给逐层 |
| **RQ3** | 三类结构（强/弱边、行为近邻、item 频率）**全部有现成计算**，但**没有"结构类型 × variant"的连接脚本** | 需组装一个按 边强/弱、head/mid/tail、近邻 分层的 AUC 探针 |
| **RQ4** | **完全直接支持**，`tools/per_level_alignment_diagnostic.py` 就是"行为相似度 vs SID 前缀相似度"的分析 | 无（仅需明确 collision bit 已被排除） |
| **RQ5** | 重训扫码本的**基建齐备**（现成脚本已扫 K0∈{64,128,512,768,1024,1536,2048}）；**全层 K=64/128/256/512 的系统扫未做** | 缺 semantic/behavior **双轴**在同一 capacity 下的对照 |
| **RQ6** | **完整闭环基建齐备**（SID→T5→eval→多 seed→AUC↔NDCG 相关），且已跑过 | 无端到端一键脚本；T5 token embedding 层未测 |

**最关键的四个"表述陷阱"**（详见第 4 节）：
1. 项目里存在**多个都叫 "quantized representation" 的张量**（`x_q` 累加和 / 解码器输出 `out` / straight-through 版），不能混为一谈。
2. **collision bit（SID 第 4 列）不是 RVQ 语义前缀**，它是碰撞消解序号，已有多处代码显式排除。
3. 判据图 E 与 TA-SID 的行为向量**同源**，是已知的定义优势（项目自己已记录）。
4. **T5 输入里没有文本 prefix** —— 只有整数 token（`c + layer*256 + 1`）。任何"SID 是文本序列"的表述都与代码不符。

**最终回答（当前代码距离回答 RQ1–RQ6 还差什么）：**
RQ1–RQ6 **全部落在"直接支持"或"少量修改"区间，没有任何一项需要改动 RQ-VAE / T5 模型代码**。缺的都是**探针/编排层面**的组装：
- **RQ1**：缺一个把 input/fusion/z/xq/SID 五类表示**逐 item 对齐导出**的探针（核心对比本身已跑过）。
- **RQ2**：缺**单层** residual / x_res / 单层 loss 的只读重放（累积层已有）。
- **RQ3**：缺"结构类型（强/弱边、head/mid/tail、近邻）× variant"的**分层 AUC** 连接脚本。
- **RQ4**：**无缺口**（`per_level_alignment_diagnostic.py` 即该分析）。
- **RQ5**：缺**全层** K=64/128/256/512 的 `K K K` 系统扫 + semantic/behavior **双轴**对照（L0-only 扫已有）。
- **RQ6**：缺**端到端一键编排**与 T5 embedding 层探针（闭环基建齐备且已跑过）。

**两条 T5 侧核实结论（本次审计新增，详见 RQ6）：**
- `set_seed` 在模型创建**之后**调用 → **3-seed 不控 T5 初始化**，只衡量数据/采样方差（既有代码现状）。
- [dataset.py:71](../model/dataset.py#L71) `item2code` 的 `codebook_size` **硬编码 256** → 码本 K>256 的 variant（如 `Toys_ta_sid_L0Big`，col0 max≈1021）直接喂 T5 会**静默错位**，是 RQ5 容量实验接 T5 的前置检查项。

---

## 1. 项目结构（只列与 RQ1–RQ6 真正相关的文件）

### 1.1 RQ-VAE / SID 相关

| 文件 | 作用 | 关键符号 |
|:--|:--|:--|
| [rqvae/models/rqvae.py](../rqvae/models/rqvae.py) | RQ-VAE 主干（encoder → rq → decoder） | `RQVAE.forward` (L61)，`get_indices` (L82) |
| [rqvae/models/rq.py](../rqvae/models/rq.py) | **残差向量量化**（逐层残差） | `ResidualVectorQuantizer.forward` (L39) |
| [rqvae/models/vq.py](../rqvae/models/vq.py) | 单层量化器（argmin / Sinkhorn / straight-through） | `VectorQuantizer.forward` (L63) |
| [rqvae/models/layers.py](../rqvae/models/layers.py) | MLP 定义、kmeans、sinkhorn | `MLPLayers` (L8) |
| [rqvae/main.py](../rqvae/main.py) | RQ-VAE 训练入口（所有超参） | `--num_emb_list` (L39)、`--e_dim` (L40) |
| [rqvae/trainer.py](../rqvae/trainer.py) | 训练循环 + collision rate 验证 | `_valid_epoch` (L384) |
| [rqvae/generate_code.py](../rqvae/generate_code.py) | SID 生成（含碰撞消解） | L84 prefix、L154 hstack 碰撞位列 |
| [rqvae/generate_sid_codes_param.py](../rqvae/generate_sid_codes_param.py) | 参数化 SID 生成器（E3 用） | 文件头明确第 4 列语义 |
| [rqvae/sf_rq.py](../rqvae/sf_rq.py) | 逐层残差重放 / z 提取 / L0 重锁 | `rederive_codes_sequential` (L92)、`encode_z` (L77) |

### 1.2 Behavior representation 相关

| 文件 | 作用 | 关键符号 |
|:--|:--|:--|
| [rqvae/build_transition_graph.py](../rqvae/build_transition_graph.py) | **行为向量的唯一构造源**：共现矩阵 → PPMI → SVD(128d) | `build_cooccurrence_matrix` (L82)、`compute_ppmi` (L126) |
| [rqvae/behavior_variants.py](../rqvae/behavior_variants.py) | 行为向量消融变体（rawcooc / itemcf / ppmi） | `build_basket_matrix` (L79)、`svd_embed` (L149) |
| [rqvae/fuse_embeddings.py](../rqvae/fuse_embeddings.py) | **融合表示构造**：`LN(content) ⊕ LN(behavior)` → 896d | L96–L110 |

### 1.3 TA-SID 相关

| 文件 | 作用 |
|:--|:--|
| [rqvae/run_ta_sid_pipeline.py](../rqvae/run_ta_sid_pipeline.py) | Step1→5 编排（图→融合→RQ-VAE→codes→T5） |
| [rqvae/run_ta_sid_step45.py](../rqvae/run_ta_sid_step45.py) | Step4/5 续跑 |
| [rqvae/run_e41_k0_sweep.sh](../rqvae/run_e41_k0_sweep.sh)、[run_k0_stress.sh](../rqvae/run_k0_stress.sh) | 码本容量扫（RQ5） |

### 1.4 T5 / 生成式推荐相关

| 文件 | 作用 | 关键符号 |
|:--|:--|:--|
| [model/main.py](../model/main.py) | TIGER 模型定义 + 训练 + `evaluate/recall_at_k/ndcg_at_k` | `TIGER.__init__` (L23)、`T5Config` (L25) |
| [model/dataset.py](../model/dataset.py) | SID → token 编码 | `item2code` (L71)、`GenRecDataset` (L91) |
| [model/dataloader.py](../model/dataloader.py) | 拼接 + attention mask | `GenRecDataLoader.collate_fn` |
| [model/run_t5_multiseed.sh](../model/run_t5_multiseed.sh) | 3-seed 训练 + 评估一键 | — |

### 1.5 Evaluation 相关

| 文件 | 作用 |
|:--|:--|
| [tools/eval_trained_model.py](../tools/eval_trained_model.py) | 加载 ckpt → 算 Recall/NDCG@K（beam 30） |
| [tools/longtail_analysis.py](../tools/longtail_analysis.py) | Head/Mid/Tail 分群 Recall/NDCG（RQ3/RQ6） |
| [tools/e49_summary.py](../tools/e49_summary.py) | NDCG@20 权威解析（`LANES`/`parse`/`SEEDS`） |

### 1.6 结构探针（本审计的核心资产）

| 文件 | 测什么 | 对应用 RQ |
|:--|:--|:--|
| [tools/sid_consistency_contrast.py](../tools/sid_consistency_contrast.py) | 量化空间 x_q 上 转移对 vs 随机对 AUC | RQ1/RQ6 |
| [tools/sid_per_layer_diagnostic.py](../tools/sid_per_layer_diagnostic.py) | input / z / xq[L0]/xq[L0+L1]/xq[full] 的 AUC | **RQ1/RQ2** |
| [tools/per_level_alignment_diagnostic.py](../tools/per_level_alignment_diagnostic.py) | 每层码字的 δ_content / δ_beh / gap | **RQ4** |
| [tools/behavior_structure_probe.py](../tools/behavior_structure_probe.py) | 行为结构保持 AUC + 与 NDCG 的相关 | **RQ1/RQ3/RQ6** |
| [tools/behavior_space_diagnostic.py](../tools/behavior_space_diagnostic.py) | 行为空间（量化前）频率泄漏 / 谱 / 相邻对区分 | RQ3 |
| [tools/stage_probe.py](../tools/stage_probe.py) | A(input)/B(z)/C(xq) 三空间 × 种子 | RQ1/RQ2/RQ6 |
| [tools/sid_capacity_diagnostic.py](../tools/sid_capacity_diagnostic.py) | 码本利用率 + z 空间容量扫描 | **RQ5** |
| [tools/codebook_budget_control.py](../tools/codebook_budget_control.py) | 截断 RQ 层数（不重训）看代价 | RQ2/RQ5 |

### 1.7 experiment / config 文件

- 数据产物：`data/{D}/item_emb_768d.parquet`(768d 内容)、`behavior_emb.parquet`(128d 行为)、`fused_item_emb_768d.parquet`(896d 融合)、`{D}_*_sid.npy`(SID codes)。
- checkpoint：`rqvae/ckpt/{tag}/{ts}/best_collision_model.pth`（RQ-VAE）、`model/ckpt/tiger_*.pth`（T5）。
- 评测日志：`model/logs/*_eval.log`（实测 78 份）。

---

## 2. 逐个审计 RQ1–RQ6

> 统一格式：**目标 / 当前代码支持 / 已有相关实现 / 已可获得 / 缺失 / 需修改 / 修改规模 / 主要风险**

### RQ1

**目标：** 量化「行为结构从连续表示 → SID 的损失」，需要能取到 5 类表示：
behavior representation、fusion representation、encoder continuous latent、final quantized latent、SID/code indices。

**当前代码支持：** **△ 少量修改**（核心对比**已直接支持**并已跑过；缺口只在"对齐导出"）

**已有相关实现：**

| 表示 | 文件 · 符号 | 获取方式 |
|:--|:--|:--|
| behavior representation (128d) | `data/{D}/behavior_emb.parquet`，构造于 [build_transition_graph.py:82](../rqvae/build_transition_graph.py#L82) | `pd.read_parquet` |
| fusion representation (896d) | `data/{D}/fused_item_emb_768d.parquet`，构造于 [fuse_embeddings.py:110](../rqvae/fuse_embeddings.py#L110) | `pd.read_parquet` |
| encoder continuous latent `z` (32d) | [sf_rq.py:77 `encode_z`](../rqvae/sf_rq.py#L77)、[sid_per_layer_diagnostic.py:86 `encode_z`](../tools/sid_per_layer_diagnostic.py#L86) | `model.encoder(x)` |
| final quantized latent `x_q` (32d) | [sid_per_layer_diagnostic.py:64 `rebuild_xq_partial`](../tools/sid_per_layer_diagnostic.py#L64)、[rqvae.py:70](../rqvae/models/rqvae.py#L70) | `Σ_l codebook_l[idx_l]` |
| SID / code indices (N,4) | `data/{D}/*_sid.npy` | `np.load` |

**已经可以获得的数据：**
- `input`(896d) / `z`(32d) / `xq[L0]` / `xq[L0+L1]` / `xq[full]` 五个空间的**行为结构 AUC** —— 由 [sid_per_layer_diagnostic.py](../tools/sid_per_layer_diagnostic.py) 一次产出（Beauty/Sport/Toys × TA-SID/Content-only）。
- A(input)/B(z)/C(xq) 三空间 × 6 变体 × 3 seed 的 AUC 与 ρ 表 —— [stage_probe.py](../tools/stage_probe.py)。
- "量化是否保序"：ρ(z, xq[full]) = 1.000（三数据集，已记录）。

**缺失的数据：**
1. **没有把 5 类表示逐 item 对齐到同一索引、同一次运行统一导出**的脚本 —— 现有脚本各自把 AUC 汇总成表，不落 item 级矩阵。
2. 没有把 behavior(128d) 与 fusion(896d) 单独作为"连续表示"列进同一个探针（`behavior_structure_probe` 测 xq/z，`stage_probe` 测 A/B/C，**没有单列 behavior 128d**）。
3. 缺一个**统一的"损失"标量定义**（现用 ΔAUC；项目口径已固定在 `behavior_structure_probe` 的 topk × degmatch）。

**需要修改的代码：** 把上述 5 个空间的加载/编码集中到一个探针脚本，共用同一 item 顺序与同一正负样本集，输出 (N, space) 表示矩阵 + 每空间 AUC。**不需要改任何模型代码。**

**修改规模：** **Small**（组装为主，无模型改动）

**主要风险：**
- **representation definition ambiguity（高）**：见第 4 节 —— `x_q` 有 ≥2 个定义，`input` 有 768d/896d/4096d 三种历史版本。
- **fairness issue（中）**：Behavior 128d 与 Fusion 896d 维度不同，不能直接比余弦绝对值，只能用 AUC 这类排序指标。
- computational（低）：z 需要一次 encoder 前向（CPU 可做）。

---

### RQ2

**目标：** 判断行为结构的损失发生在哪个 RVQ 层（z → L1 → L1+L2 → L1+L2+L3）。

**当前代码支持：** **✓ 直接支持（累积层）＋ △ 少量修改（单层残差/单层 loss）**

**已有相关实现：**
- [rq.py:39 `ResidualVectorQuantizer.forward`](../rqvae/models/rq.py#L39)：循环 `for quantizer in self.vq_layers`，逐层 `x_res, loss, indices = quantizer(residual)`，`residual = residual - x_res`，`x_q = x_q + x_res`。
- [rq.py:53](../rqvae/models/rq.py#L53)：`mean_losses = torch.stack(all_losses).mean()` —— **只返回逐层 loss 的均值**。
- [sid_per_layer_diagnostic.py:212–222](../tools/sid_per_layer_diagnostic.py#L212)：`for n_layers in 1..3: rebuild_xq_partial(codes, cb, n_layers)` → 输出 `xq[L0] / xq[L0+L1] / xq[full]` 的 AUC。
- [sf_rq.py:92 `rederive_codes_sequential`](../rqvae/sf_rq.py#L92)：按真实残差顺序重放每层 code（已用于 provenance 审计）。
- [sid_capacity_diagnostic.py](../tools/sid_capacity_diagnostic.py)：同样的 anchors 口径（xq[L0]/xq[L0+L1]/xq[full]）。
- [run_checkpoint_trajectory.py](../rqvae/run_checkpoint_trajectory.py)、[tools/checkpoint_trajectory_probe.py](../tools/checkpoint_trajectory_probe.py)：按训练轮次追踪。

**已经可以获得的数据：**
- **每层累积量化表示** `xq[L0]`、`xq[L0+L1]`、`xq[full]` 及其 AUC（**直接可得**）。
- **每层 code index** `indices` (N,3)（`rq.forward` 返回 `all_indices`，shape (B, num_quantizers)）。
- **每层码本向量**：`model.rq.vq_layers[l].embedding.weight`（`load_codebook_weights` 已封装）。
- 逐层残差重放（`rederive_codes_sequential`）。

**缺失的数据：**
1. **单层 residual vector**（第 l 层的输入 `residual`）—— `rq.forward` 内部计算了但**不返回**。
2. **单层 quantized vector `x_res`** —— 同上，只在循环内累加进 `x_q`，不逐层暴露。
3. **逐层 loss** —— 被 `.mean()` 压成一个标量后才返回。
4. 现有逐层 AUC 是**累积口径**（L0, L0+L1, full），**没有单层增量口径**（"只看第 2 层贡献了什么"）。

**需要修改的代码：** 若要单层 residual / x_res / 单层 loss，需要一个**只读的旁路重放函数**（不改 `rq.forward`）：模仿 `rederive_codes_sequential` 的循环，逐层记录 `residual` 与 `x_res = codebook_l[idx_l]`，并单独算 `F.mse_loss`。可与 RQ1/RQ2 探针合并。

**修改规模：** **Small**（累积层零改动；单层需 ~30 行只读重放）

**主要风险：**
- **RVQ layer access problem（中）**：`rq.forward` 不返回逐层张量，必须重放；重放时**必须**用真实残差顺序，否则 L1/L2 无意义 —— 项目已踩过此坑（[verify_variant_pairing.py:5–17](../rqvae/verify_variant_pairing.py#L5) 记录了 `argmin(z, cb_l)` 旧 bug）。
- SID decoding ambiguity（中）：`codes` 第 2 层受 `use_sk` 影响（L2 走 Sinkhorn），重放默认 `use_sk=False`，与 codegen 主 pass 一致（L2 有 1–2% 行被碰撞消解循环改写）。
- representation ambiguity（中）：见第 4 节。

---

### RQ3

**目标：** 判断"什么行为结构最容易损失"，需要：(1) 强/弱 behavior association；(2) behavior nearest neighbors；(3) item frequency / head-middle-tail。

**当前代码支持：** **△ 少量修改**（三类结构的**计算全部已有**，缺"结构类型 × variant 结构保持度"的连接脚本）

**已有相关实现：**

| 需求 | 文件 · 符号 |
|:--|:--|
| 强/弱 behavior association | [behavior_structure_probe.py:334](../tools/behavior_structure_probe.py#L334) `pos_regimes = {"topk": 高频边, "all": 均匀采样边}`；[behavior_space_diagnostic.py:88 `build_pair_sets`](../tools/behavior_space_diagnostic.py#L88) `adj` / `far` / `rand` 三档 |
| 共现强度（co-occurrence strength） | [build_transition_graph.py:82 `build_cooccurrence_matrix`](../rqvae/build_transition_graph.py#L82) 计数；[behavior_structure_probe.py:148 `build_graph`](../tools/behavior_structure_probe.py#L148) 返回 `counts`；E 图边集 |
| behavior similarity | [behavior_structure_probe.py:244 `cos_sim`](../tools/behavior_structure_probe.py#L244)；[behavior_space_diagnostic.py:73 `pair_auc`](../tools/behavior_space_diagnostic.py#L73) |
| behavior nearest neighbors | [analyze_emb_orthogonality.py:31 `topk_sims`](../tools/analyze_emb_orthogonality.py#L31)（通用 KNN）；[build_transition_graph.py:312 `_validate`](../rqvae/build_transition_graph.py#L312) 打印 top-5 最相似对 |
| item frequency | [behavior_space_diagnostic.py:65 `item_freq`](../tools/behavior_space_diagnostic.py#L65)；[longtail_analysis.py:36 `assign_groups`](../tools/longtail_analysis.py#L36)（Head/Mid/Tail 三分） |
| 频率泄漏度量 | [behavior_space_diagnostic.py:132](../tools/behavior_space_diagnostic.py#L132) `freq_leak`=Spearman(‖emb‖, log1p(freq))、`pc1_cos_freq` |

**已经可以获得的数据：**
- 每条边的共现计数 `counts`，可任意分"强（topk, count≥6）/ 弱（all 包含长尾）"。
- 每个 item 的出现频率，可按 head/mid/tail 分群。
- 任意空间中 item 的 KNN 近邻。
- 每个 variant 的**总体**行为结构 AUC。

**缺失的数据：**
1. **没有"按结构类型分层的 AUC"**：即"强边 vs 弱边"、"head vs mid vs tail"、"近邻对 vs 非近邻对"**分别**在 6 个 variant 上的结构保持度 —— 现有 `behavior_structure_probe` 只对 `topk`/`all` 两档**整体**报 AUC，不交叉到 variant × 结构类型。
2. 没有"哪种结构最容易在量化中丢失"的**逐类型 ΔAUC**（input→xq 的损失，按边强/弱分解）。

**需要修改的代码：** 复用 `build_graph`（得边+计数+度）与 `rebuild_xq_partial`（得 xq），按 count 分位数 / deg 分桶切分正样本集，逐桶算 AUC 与 Δ(A input→xq)。**全部是组装，无新方法、无模型改动。**

**修改规模：** **Small–Medium**（取决于是否要引入双向近邻对；现有函数可直接复用）

**主要风险：**
- **information leakage（中）**：E 图（判据）与 TA-SID 行为向量**同源**（同一 window=3 共现）。按结构分层时，TA-SID 在"强边"桶上的优势会被人为放大 —— 项目已记录此为已知定义优势（[behavior_structure_study_plan.md](behavior_structure_study_plan.md) §3.3 边界 3）。
- **流行度捷径（中）**：强边天然连高频 item，按桶比较时度分布不匹配会让分类器凭流行度得分 —— 现有 `sample_neg_degmatch`（[behavior_structure_probe.py:210](../tools/behavior_structure_probe.py#L210)）已提供度受控负采样缓解。
- train/test contamination（低）：E 只从 `train.parquet` 构造，无泄漏。

---

### RQ4

**目标：** 判断 behavior similarity 是否对应 SID prefix similarity（shared z1 / z1-z2 / z1-z2-z3），并正确处理 collision bit。

**当前代码支持：** **✓ 直接支持**（已有专门脚本）

**已有相关实现：**
- **SID prefix 分组**：[per_level_alignment_diagnostic.py:111 `prefix_groups(codes, l)`](../tools/per_level_alignment_diagnostic.py#L111)（累积前 l 列）、[:117 `level_groups`](../tools/per_level_alignment_diagnostic.py#L117)（单层码字）。
- **δ_X(l) 定义**：[:121 `within_group_mean_cos`](../tools/per_level_alignment_diagnostic.py#L121) = 前缀组内平均成对余弦 − 随机对基线；`gap = δ_beh − δ_content`。
- **behavior similarity**：同一脚本用 `behavior_emb.parquet`（128d）算 `δ_beh`；content 用 `item_emb_768d_contentonly.parquet` 算 `δ_content`。
- **SID token 编码/解码**：[dataset.py:71 `item2code`](../model/dataset.py#L71) `offsets = c + i*256 + 1`；`code_to_item` 反向映射。
- **离散 token 相似度**：[behavior_structure_probe.py:248 `hamming`](../tools/behavior_structure_probe.py#L248)（前 3 token 的 Hamming）、L273 `auc_l0match`（首 token 是否相同）。
- **collision bit 处理**：脚本头 [per_level_alignment_diagnostic.py:23](../tools/per_level_alignment_diagnostic.py#L23) 明确"**第 4 列 (L3) 是碰撞消解序号，不是语义层，不参与分组**"，代码 `n_sem = min(3, codes.shape[1]-1)`（L213）。

**已经可以获得的数据：**
- 每层（L1/L2/L3）的 prefix 分组内行为相似度增量 δ_beh、内容相似度增量 δ_content、二者的 gap。
- 前 3 token 的 Hamming 距离与首 token 匹配率（含 AUC）。
- SID token 序列（经 `item2code` 转成 4 个 vocab id）。

**缺失的数据：** 基本无。可选：现成的"共享 z1 ↔ behavior 相似度"的**逐 item 对散点/相关系数**（现在只报组内均值），但覆盖率（ρ=... 级别的相关）可从 `corr_block` 口径自行加。

**需要修改的代码：** 无（RQ4 若要更细，仅需在 `per_level_alignment_diagnostic` 上加一个"按 prefix 长度算 ρ(行为相似, 前缀命中)"的输出，Small）。

**修改规模：** **None / Small**

**主要风险：**
- **SID decoding ambiguity（中）**：collision bit 若被当成第 4 个语义 token，会把"前缀"错误定义成含消解位 —— 现有代码已正确排除，但**T5 训练时它确实是第 4 个 token**（[dataset.py:85](../model/dataset.py#L85)），两个语境不同，勿混。
- information leakage（中）：同 RQ3 的定义优势。
- 定义优势（中）：δ_content 用 768d content-only，δ_beh 用 128d behavior，两者的基线不可比，只能看 **gap** 与**跨层趋势**。

---

### RQ5

**目标：** 检查 codebook capacity 是否造成 semantic / behavioral trade-off，能否控制 K = 64/128/256/512，并在其余设置一致下重训。

**当前代码支持：** **△ 少量修改**（重训配置**直接支持**；"semantic/behavior 双轴 trade-off"缺现成对照）

**已有相关实现：**
- **K 配置位置**：[rqvae/main.py:39](../rqvae/main.py#L39) `--num_emb_list`（nargs+，逐层容量）→ 传入 `RQVAE(num_emb_list=...)` → `ResidualVectorQuantizer` → 每层 `VectorQuantizer(n_e)`。
- **RVQ 层数**：`len(num_emb_list)`（如 `256 256 256` = 3 层）。
- **embedding dim**：[rqvae/main.py:40](../rqvae/main.py#L40) `--e_dim`（默认 32）。
- **encoder 隐藏层**：[rqvae/main.py:43](../rqvae/main.py#L43) `--layers 512 256 128 64`。
- **已知容量扫描脚本**（**仅改 L0，其余逐位一致**）：
  - [run_e41_k0_sweep.sh](../rqvae/run_e41_k0_sweep.sh)：K0 = 512 / 768 / 2048。
  - [run_k0_stress.sh](../rqvae/run_k0_stress.sh)：K0 = 128 / 64（压测）。
  - `run_e41_k0_1536.sh`、`run_e41_k0_sweep.sh`（含 1024）。
  - 现存 ckpt 目录：`Sport_TA_SID_L064 / L0128 / L0512 / L0Big(1024) / L01536 / L02048` 等。
- **不重训的容量诊断**：[sid_capacity_diagnostic.py:62 `kmeans_quantize`](../tools/sid_capacity_diagnostic.py#L62)（z 空间 k-means 扫 k=256…8192）、[intrinsic_capacity.py](../tools/intrinsic_capacity.py)（内在簇数）、[codebook_budget_control.py](../tools/codebook_budget_control.py)（截断 RQ 层数）。
- **码本健康度**：[sid_capacity_diagnostic.py:73 `code_utilization`](../tools/sid_capacity_diagnostic.py#L73)（利用率 / 碰撞率 / 唯一组合数）。

**已经可以获得的数据：**
- 任意 `num_emb_list` 的 RQ-VAE 重训（脚本已证明"只改 L0、其余逐位一致"）。
- 各 K0 下的：码本利用率、z-gap、行为结构 AUC。
- 不重训条件下 z 空间的容量曲线（AUC vs k）。

**缺失的数据：**
1. **K = 64/128/256/512 的"全层一致"系统扫**：现有脚本只扫 **L0**（`num_emb_list K0 256 256`），**没有** `K K K`（三层同时变）的对照。
2. **semantic 侧的度量**：现有容量诊断只报 **behavior AUC**；RQ5 要的 "semantic/behavior trade-off" 需要**同时**报 **semantic**（content δ 或 content-only AUC）与 **behavior** 两个轴 —— [per_level_alignment_diagnostic.py](../tools/per_level_alignment_diagnostic.py) 有 δ_content/δ_beh 的现成口径，但**未在容量变化下跑过**。
3. capacity 变化后 codes 会**重新编号**，要建立与 T5 的对应必须**重新生成 codes + 重训 T5**（基建现成，成本另计）。

**需要修改的代码：** 写一个把 `per_level_alignment_diagnostic` 的双轴度量挂到"每个 (数据集 × K) 跑一遍"的驱动；重训沿用 `main.py --num_emb_list K K K`（或维持 L0-only 并在报告里明确口径）。

**修改规模：** **Small（脚本）｜ Large（若含 T5 端到端重训的算力）**

**主要风险：**
- **fairness issue（高）**：容量一变，码本重训 → codes 变 → 必须**同步重训 T5** 才能谈"对推荐的影响"；若只比 RQ-VAE 端 AUC 则要在报告中限定"仅量化端"。
- computational（中）：每个 K × 数据集 × 3 seed 的 T5 约 150–240 min/数据集（已知成本口径）。
- representation ambiguity（中）：K0-only 与 K-K-K 两种"容量"含义不同，勿混。
- 已知过度泛化风险：[Session 2026-09-03/04](../doc/e46_second_innovation_overturn.md) 已裁定"扩容卖点"被推翻，本 RQ 只作**受控对照**、不作"扩容有效"的结论来源。

---

### RQ6

**目标：** 检查能否把 behavior structure preservation 与 T5/TIGER 生成式推荐性能对应起来（SID→T5→eval，Recall/NDCG，多 seed，多 variant 进同一流程）。

**当前代码支持：** **✓ 直接支持**（闭环基建齐备且已跑过）

**已有相关实现：**

| 环节 | 文件 · 符号 |
|:--|:--|
| SID → T5 训练 | [model/main.py](../model/main.py)（`--code_path` 切换 SID 文件）；[dataset.py:91 `GenRecDataset`](../model/dataset.py#L91) |
| TIGER 模型（**from-scratch**） | [model/main.py:25](../model/main.py#L25) `T5Config(...)` + [L40](../model/main.py#L40) `T5ForConditionalGeneration(t5config)`（**无 `from_pretrained`**）；vocab 1025 |
| 多 seed | [model/run_t5_multiseed.sh](../model/run_t5_multiseed.sh)（默认 seed 2025/2026/2027） |
| 评估 Recall@K / NDCG@K | [tools/eval_trained_model.py:91](../tools/eval_trained_model.py#L91) `evaluate(...)` → `recall_at_k`/`ndcg_at_k`（[model/main.py](../model/main.py)） |
| 不同 variant 进同一流程 | `--code_path data/{D}/{variant}_sid.npy`（`model/ckpt/` 下 `tiger_{variant}_{ds}_seed{S}.pth` 命名规律） |
| **结构保持 ↔ NDCG 的关联脚本** | [behavior_structure_probe.py:423 `corr_block`](../tools/behavior_structure_probe.py#L423) Pearson/Spearman(Behavior AUC, NDCG@20)；[plot_behavior_auc_vs_ndcg.py](../tools/plot_behavior_auc_vs_ndcg.py) |
| NDCG 权威解析 | [e49_summary.py](../tools/e49_summary.py) `LANES`/`parse`/`SEEDS` |

**已经可以获得的数据：**
- 每个 variant 的 NDCG@20 / Recall@20（3-seed 均值，日志在 `model/logs/*_eval.log`，实测 78 份）。
- Behavior AUC（`xq[full]`）与 NDCG@20 的数据集内相关：ρ = 0.943 / 0.943 / 1.000（已记录）。
- 分群（Head/Mid/Tail）推荐指标：[longtail_analysis.py](../tools/longtail_analysis.py)。

**评测口径（T5 侧核实）：**
- 生成用 **beam search**（`main.py:165` `model.generate(..., num_beams=beam_size)`，`beam_size=30`，`max_length=5`），**非** teacher forcing；teacher forcing 只在训练 loss 用。
- 命中判定：`calculate_pos_index`（[main.py:99](../model/main.py#L99)）把**整条 4-token 预测序列**与 label 做**完全相等**比较 —— 即 RQ4 的"前缀"在推荐侧是**整串精确匹配**，不是前缀匹配。`K ∈ {5,10,20,50,100}`。
- 训练 seed 语义：`set_seed`（[main.py:180](../model/main.py#L180)）在**模型创建之后**调用（`TIGER(config)` 在 `main.py:227`，`set_seed` 在 `main.py:232`）→ **T5 初始权重不受 `--seed` 控制**，seed 只影响 data shuffle 与 beam 采样。这意味着"3-seed"衡量的是数据/采样方差，不是初始化方差。**注：这是既有代码现状，非本次审计改动。**

**缺失的数据：**
1. **端到端一键管线**：从"新 codes 文件" → RQ-VAE→codes→T5→eval→AUC→相关，各段脚本独立，**没有串成一个命令**。
2. **T5 token embedding 层**的结构保持未测（项目自记为缺口 G4）；链条现止于 `xq`（e_dim=32），**未测编码器 embed 层的推荐侧表示**。
3. 6 变体 × 3 数据集的 **NDCG 与 AUC 同表**未脚本化统一（Sport/Toys 在 probe，Beauty 在 Tier-1/Tier-2 文件）。

**需要修改的代码：** 编排脚本（调用既有 `run_t5_multiseed.sh` + `eval_trained_model.py` + `behavior_structure_probe` 的 `load_ndcg`），以及（可选）挂 T5 embedding 层的探针。**不改推荐模型。**

**修改规模：** **Small（编排）｜ Small–Medium（T5 embedding 探针）**

**主要风险：**
- **fairness issue（中）**：不同 variant 的 T5 训练 seed 必须一致才可配对；Beauty 曾有 seed 不一致历史（项目已订正）。且 seed 不控初始化（见上），配对解释须限定为"数据/采样方差"。
- **SID decoding ambiguity（中）**：[dataset.py:71 `item2code`](../model/dataset.py#L71) 的 `codebook_size` **硬编码 256**，`GenRecDataset` 调用时不传参（[dataset.py:108](../model/dataset.py#L108)）。若某 variant 的某层码本 >256（实测 `Toys_ta_sid_L0Big_sid.npy` 的 col0 max ≈ 1021），token 会**跨层重叠/越界**（vocab=1025）。主流 variant（值域 0..255）不受影响，但 RQ5 若扫出 K>256 的 variant，直接喂 T5 会**静默错位** —— 这是容量实验接 T5 前必须检查的前置条件。
- computational（中）：每 variant × 数据集 × 3 seed 是主要成本。
- 口径不统一（中）：项目存在多套"类 AUC"（mean|Δ|≈0.013），必须统一到 `behavior_structure_probe` 的 topk × degmatch。

---

## 3. 特别检查：RQ-VAE 真实数据流

代码为准。实际实现与任务描述**基本一致**，但有 5 处必须指出的差异。

```
content embedding  (item_emb_768d.parquet, 768d)          ← 或 legacy item_emb.parquet (4096d)
        │
        ├─ fuse_embeddings.py:  LN(content) ⊕ LN(behavior)
behavior embedding  (behavior_emb.parquet, 128d)            构造于 build_transition_graph.py
        │                     (共现矩阵 C[window=3] → PPMI → svds(128))
        ▼
fused_item_emb_768d.parquet  (896d)   ← 【融合表示 fusion】
        │
        ▼
RQ-VAE encoder  MLPLayers(896 → 512 → 256 → 128 → 64 → 32)
        │        models/rqvae.py:69  z = self.encoder(x)          ← 【z = encoder 连续 latent, 32d】
        ▼
ResidualVectorQuantizer.forward(z)   models/rq.py:45-48
        │   residual = z
        ├─ VQ layer 0: x_res0 = codebook0[idx0];  residual -= x_res0     ← 【L0】
        ├─ VQ layer 1: x_res1 = codebook1[idx1];  residual -= x_res1     ← 【L1】
        └─ VQ layer 2: x_res2 = codebook2[idx2];  residual -= x_res2     ← 【L2】
        │   x_q = x_res0 + x_res1 + x_res2   ← 【final quantized latent, 32d】  models/rq.py:48
        ▼
decoder  MLPLayers(32 → 64 → 128 → 256 → 512 → 896)   ← out = self.decoder(x_q) 【重建输出 896d】
        │
        ▼
generate_code*.py:  indices (N,3)  → hstack 零列 → (N,4)
        │   col0/1/2 = 语义层码号;  col3 = 0
        │   碰撞消解循环 (use_sk=True 重跑冲突组) + 末列递增去重
        ▼
{D}_*_sid.npy   (N,4)   ← 【SID / code indices】
        │
        ▼
T5 (TIGER)  item2code: offsets = c + i*256 + 1  → 4 tokens (vocab 1025)
        │   model/dataset.py:85
        ▼
generative recommendation  →  eval: Recall@K / NDCG@K
```

**与任务描述的差异（以代码为准）：**
1. **有额外的 decoder 输出 `out`（896d）**，任务描述未提。它是重建目标空间，不是 SID 表示 —— 易被误当最终表示（见第 4 节）。
2. **"collision handling" 是两步**：先 `use_sk=True` 重跑冲突组（Sinkhorn 全局分配），再对仍重复的组合**递增最后一列**。两者都在 `generate_code.py` 内。
3. **量化器的 `x_q` 是 straight-through 的**（`vq.py:95` `x_q = x + (x_q-x).detach()`），而 `rebuild_xq_partial` 是**纯 gather**（无梯度）—— 数值在 eval 下相同，但**梯度语义不同**；`trainer._get_transition_loss` 用 gather、`_get_align_loss` 用 ST。
4. **`rq.forward` 返回的是累积 `x_q` 与平均 loss**，不逐层返回（RQ2 缺口根源）。
5. **`indices` 是 `torch.stack(..., dim=-1)`**（`rq.py:54`），即 (B, num_quantizers)，与 codes 文件的列顺序一致（col l ↔ layer l）。

---

## 4. 特别检查："连续表示" 与 "离散表示" 的定义

> 本审计的关键。以下均为**代码中的具体张量**，不新增定义。

**Q1. 当前代码里的 "continuous representation" 到底是哪一个 tensor？**
**有多个候选，取决于比较目的：**
- **模型输入 `x`**：`fused_item_emb_768d.parquet` 的 896d 向量（`EmbDataset[idx]`），是"连续表示"的**输入侧**。
- **模态分解**：content 768d（`item_emb_768d.parquet`）与 behavior 128d（`behavior_emb.parquet`）是其两个分量。
- **encoder 输出 `z`**：32d，是"量化前连续表示"的**规范含义**（`sid_per_layer_diagnostic` 称 input→z 为"编码器"环节）。
> ⚠️ 三者维度不同（896 / 768 / 128 / 32），**不可互换**，比较时必须声明是哪一个。

**Q2. encoder latent 是什么？**
`model.encoder(x)` 的输出 `z`，shape (N, e_dim=32)（[rqvae.py:69](../rqvae/models/rqvae.py#L69)）。提取函数：`encode_z`（[sf_rq.py:77](../rqvae/sf_rq.py#L77) / [sid_per_layer_diagnostic.py:86](../tools/sid_per_layer_diagnostic.py#L86)）。

**Q3. quantized representation 是什么？**
**规范定义是 `x_q = Σ_l codebook_l[idx_l]`，shape (N, 32)**（[rqvae.py:70](../rqvae/models/rqvae.py#L70)，重建见 [sid_consistency_contrast.py:107 `rebuild_xq`](../tools/sid_consistency_contrast.py#L107) / [sid_per_layer_diagnostic.py:64 `rebuild_xq_partial`](../tools/sid_per_layer_diagnostic.py#L64)）。**部分和** `xq[L0]`、`xq[L0+L1]` 是它的前缀版本。

**Q4. SID code indices 是什么？**
`{D}_*_sid.npy`，shape (N, **4**) int。前 3 列 = 三层 RVQ 码号（0–255）；第 4 列 = 碰撞消解序号。经 `item2code`（[dataset.py:85](../model/dataset.py#L85)）偏移成 4 个 vocab token。

**Q5. 是否存在多个不同的 quantized representation 会被误认为 final representation？**
**是，至少 4 个，必须区分：**

| 张量 | 维度 | 定义位置 | 语义 |
|:--|:--|:--|:--|
| `x_q`（累加和） | 32d | [rq.py:48](../rqvae/models/rq.py#L48) | ✅ **规范 final quantized** |
| `x_q`（straight-through） | 32d | [vq.py:95](../rqvae/models/vq.py#L95) | 同数值，带梯度 |
| `x_res`（单层） | 32d | [vq.py:87](../rqvae/models/vq.py#L87) | 单层量化向量（**不逐层返回**） |
| `out`（解码器输出） | **896d** | [rqvae.py:71](../rqvae/models/rqvae.py#L71) | 🔴 **最易被误当 final** —— 它是重建空间，不是 SID |
| `rebuild_xq`（gather 重建） | 32d | [sid_consistency_contrast.py:107](../tools/sid_consistency_contrast.py#L107) | 从 codes 反推，标准探针入口 |

> 🔴 **最大陷阱**：`out`（896d 重建）与 `x_q`（32d 量化）是完全不同的东西，**项目所有探针都用 `rebuild_xq_partial`（32d）**，从未用 `out`。若新脚本误用 `out`，口径立刻与既有数字不一致。

**Q6. collision bit 在整个流程中位于什么位置？**
- **产生**：`generate_code*.py` 末尾 `np.hstack((codes, zeros((N,1))))`，再对重复组合 `codes_array[idx, -1] = i`（递增）。
- **存储**：`{D}_*_sid.npy` 的**第 4 列**。
- **入 T5**：作为**第 4 个 token**（offset `+769`，[dataset.py:85](../model/dataset.py#L85)），vocab 1025 = 4×256 + 1(pad)。
- **探针中**：**被显式排除**于语义前缀之外（[per_level_alignment_diagnostic.py:23](../tools/per_level_alignment_diagnostic.py#L23)、`n_sem = codes.shape[1]-1`）。
> ⚠️ 因此："SID 前缀分析"只含前 3 token；而"T5 输入序列"含 4 个 token。**两者语境不同，不可混用。**
> ⚠️ **T5 输入里没有文本 prefix**（如 `<a_x><b_y>...`）：`<a_{}>…` 只出现在 RQ-VAE 的 code 生成脚本中，用途是**拼字符串 key 做碰撞检测**，随后又解析回整数。T5 侧只吃**整数 token**（`c + layer*256 + 1`）。任何"SID 是文本序列"的表述都与代码不符。

---

## 5. 最终可行性矩阵

> 定义：**直接支持** = 现成脚本/函数可产出该数据；**少量修改** = 组装/拼装既有组件（无模型改动）；**大修改** = 需改模型或新增算法；**当前不可行** = 缺关键前提。

| RQ | 直接支持 | 少量修改 | 大修改 | 当前不可行 | 关键缺口 |
|:--|:--:|:--:|:--:|:--:|:--|
| **RQ1** | ✅ 核心对比（input/z/xq 的 AUC 已有） | ⬜ 需**组装**：5 类表示逐 item 对齐导出 | — | — | 无单脚本同时导出 behavior/fusion/z/xq/SID；behavior 128d 未单列为对照 |
| **RQ2** | ✅ 累积层 `xq[L0]/xq[L0+L1]/xq[full]` | ⬜ 单层 residual / x_res / 单层 loss 需只读重放 | — | — | `rq.forward` 只返回累积 `x_q` + **平均** loss，不逐层暴露 |
| **RQ3** | ✅ 三类结构的计算（边强/弱、近邻、频率）全有 | ⬜ 需**组装**"结构类型 × variant"分层 AUC | — | — | 无按 边强/弱、head/mid/tail 分层的结构保持探针 |
| **RQ4** | ✅ `per_level_alignment_diagnostic.py` 即该分析（含 collision bit 排除） | — | — | — | 无（collision bit 已正确排除） |
| **RQ5** | ✅ 重训扫码本基建（K0∈{64…2048} 已有脚本/ckpt） | ⬜ 需**组装**：全层 K=64/128/256/512 + semantic/behavior 双轴 | ⬜ 若含 T5 端到端 | — | 缺 semantic 侧度量在容量变化下的对照；K0-only vs K-K-K 口径需明确 |
| **RQ6** | ✅ SID→T5→eval→多 seed→AUC↔NDCG 相关全部齐备 | ⬜ 需**编排**端到端一键；T5 embedding 层探针 | — | — | 无端到端管线；T5 token embedding 层结构保持未测（G4） |

**总体判定：** RQ1–RQ6 **全部落在"直接支持"或"少量修改"区间**，**没有任何一项需要大修改当前 RQ-VAE/T5 代码**。唯一可能升级为"较大算力"的是 RQ5 若要做**端到端容量 × 推荐**对照（需重训 T5），但那是**算力**而非**代码**问题。

---

## 6. 给下一阶段的接口（最小诊断实验的落点）

> 本节只标"下一阶段可直接用的现成入口"，不设计实验方案。

| 需要的东西 | 现成入口 | 备注 |
|:--|:--|:--|
| 取任意空间的行为结构 AUC | `tools/behavior_structure_probe.py` 的 `cos_sim`/`auc_of`/`build_graph`/`sample_neg_degmatch` | 口径 = topk × degmatch（**项目唯一权威口径**） |
| 取 z / xq 逐层 | `tools/sid_per_layer_diagnostic.py` 的 `encode_z`/`rebuild_xq_partial` | 只读，CPU 可跑 |
| 取单层残差重放 | `rqvae/sf_rq.py` 的 `rederive_codes_sequential` | 已有，需扩为"返回逐层 x_res" |
| 取 SID 前缀分组 | `tools/per_level_alignment_diagnostic.py` 的 `prefix_groups`/`level_groups` | 已排除 collision bit |
| 取 variant 配对 | `rqvae/verify_variant_pairing.py` + `tools/stage_probe.py` 的 `VARIANTS` 表 | 判据**只看 L0** |
| 取 NDCG@20 | `tools/e49_summary.py` 的 `parse`/`LANES`/`SEEDS` | **勿用 `$NF`**（列数陷阱） |
| 取分群推荐指标 | `tools/longtail_analysis.py` | Head/Mid/Tail |

**未在本次审计中做的事**（符合任务限制）：未设计实验、未提方法、未改代码、未跑实验、未评价 RQ 创新性、未扩展 RQ。
