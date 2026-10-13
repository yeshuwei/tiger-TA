# Semantic ID 行为结构保留研究 — P0 只读代码审计清单

> 审计日期：2026-10-13　|　模式：**只读**（未修改任何源码 / 配置 / checkpoint / 数据）
> 代码基线：branch `experiments-2026-09-25`
> 方法：逐文件读源码 + 读 checkpoint 目录清单 + 读日志；对可离线验证的量（SID code 列结构、ItemID 顺序、token 映射）用 **只读 numpy 复核**（未训练、未写入任何产物）。
> 规则：每条判断给出 `文件:行号`；无法确认的标 **待验证**；不把历史实验记录当作已复现证据。

---

## 0. 结论摘要

1. **数据流是完整且可离线重建的**：`train.parquet` →（window=3 共现 C）→ PPMI → SVD(128d) → LayerNorm 拼接(896d) → RQ-VAE(3×256, e_dim=32) → codes(N,4) → T5(vocab=1025)。每一环都有可复用代码，且**中间表示可在不重训的前提下重新导出**（见 §2.4）。
2. **原始共现矩阵 C 与 PPMI 矩阵没有落盘**。仓库只保存 SVD 之后的 `behavior_emb*.parquet`。E1（原始行为关系 vs SVD）必须**用 `build_transition_graph` 的函数现场重建**，而不是读现成产物。
3. **SID 是 4 列**：前 3 列是 RVQ 语义码，第 4 列是**碰撞消解序号**（不是学习到的码字）。诊断脚本已正确排除第 4 列；但 T5 侧把它当作**第 4 个 token**（vocab=1025=4×256+1）——这是设计，不是 bug，但两者口径必须分清。
4. **量化端"逐层累计向量"有现成实现**（`rebuild_xq_partial`）；**"逐层 residual"没有现成实现**，但由 `residual = residual − x_res` 的定义可一句推出（见 §2.3）。
5. **容量扫描只改 L0 的 K0**（`--num_emb_list ${K0} 256 256`），其余参数固定；**K0∈{512,768,1024,1536,2048} 有 codes 与诊断结果，但只有 K0∈{512,1024} 重训过 T5**。**关键风险**：T5 侧的 `item2code` 把 layer 步长**硬编码为 256**，K0≠256 时各层 token 区间不再互斥（见 §3 风险 1）。
6. **行为图仅用训练集**（`train.parquet` 的 history+target = `seq[:-2]`），**未触及 val/test target**，无目标泄漏。但存在三处"行为关系"定义彼此不一致的隐患（见 §3 风险 3）。
7. **既有指标实现存在若干可复现的缺陷**：随机负样本含自环/重复且不控流行度（`random_pair_sims`）、`analyze_emb_orthogonality.py` 的 Analysis-3 把**行号当 ItemID** 二次映射（§3 风险 4）。

---

## 1. 审计范围盘点（A 的 11 个环节）

**复用等级**：直接复用 / 小幅修改后复用 / 需要新增 / 暂无法判断

### ① 数据划分、训练序列口径及潜在数据泄漏

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | `rqvae/preprocess_data.py`；`data/{Ds}/{train,valid,test}.parquet` |
| 代码证据 | `preprocess_data.py:105-120`（leave-one-out）；`train_data=seq[:-2]`、`val_data=seq[:-1]`、`test_data=seq`；`make_df` 再切 `history=seq[:-1], target=seq[-1]` |
| 输入与输出 | 原始 JSON 评论流 → 三个 parquet；映射表 `item_mapping.npy` |
| 复用等级 | **直接复用** |
| 缺失项 | 无显式 seed（切分是确定性的，无需 seed）；无 `min_len` 之外的过滤说明 |
| 风险 | 训练序列口径 = `seq[:-2]`（即 `history+target` 展开后）；见 §3 风险 3(b) |
| 建议 | 冻结为"train 序列 = 每个用户除末 2 件外的全部商品（按时间升序）"并写进协议 |

### ② 原始共现矩阵 / PPMI / SVD 行为向量的构建与保存

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | 构建：`rqvae/build_transition_graph.py`（C/PPMI/SVD），变体：`rqvae/behavior_variants.py`；产物：`data/{Ds}/behavior_emb*.parquet`（128d），`behavior_emb_ppmi.parquet`（自检复现） |
| 代码证据 | C：`build_transition_graph.py:82-123`（滑窗 `half_window=window_size//2=1`，无向对称，**保留对角**(a==b 时只写一次)）；PPMI：`:126-145`；SVD：`:273-297`（`k=min(128, min(shape)-1)`，`U@diag(√S)`，冷启动物品零行）；变体：`behavior_variants.py:79`(整篮)、`:114`(itemcf cos)、`:149`(svd_embed，显式 v0 seed) |
| 输入与输出 | 输入 `train.parquet` + `item_emb_768d.parquet`（canonical 行序）；输出仅 **SVD 后的向量**（parquet） |
| 复用等级 | 构建函数 **直接复用**；`behavior_variants.py` 的 `_save_output(path, ids, emb)` 可复用来**新增落盘** C/PPMI |
| 缺失项 | ⚠ **C 与 PPMI 矩阵均未落盘**；`min_cooccur`/`window_size` 未写入文件名；无 manifest 记录 `svds` 的随机 v0（`build_transition_graph.py:278` 用 `svds(PPMI,k)` 默认随机 v0 → **不可复现**；只有 `behavior_variants.svd_embed` 传了确定性 v0） |
| 风险 | 直接在 C/PPMI 上做 E1 时必须自己重建，且必须显式固定 v0 |
| 建议 | E1 的 PPMI 参照统一走 `behavior_variants.py --variant ppmi`（同一条池化路径 + 确定性 v0），并把它作为"原始行为关系"的可复现代理 |

### ③ 内容向量、融合向量及 item_id 对齐

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | `tools/generate_content_embeddings.py` → `item_emb_768d.parquet`(768d)；`rqvae/fuse_embeddings.py` → `fused_item_emb_768d.parquet`(896d)；`rqvae/ablation_data.py` → shuffled/zeropad 融合 |
| 代码证据 | content: `generate_content_embeddings.py:149-156`（`ItemID=1..N`）；融合: `fuse_embeddings.py:79-80`（**断言两模态 ItemID 顺序一致**）、`:96-100`（各自独立 LayerNorm）、`:110`（concat）；消融: `ablation_data.py:76-77`(行级 shuffle)、`:141-144`(zeros 128) |
| 输入与输出 | content parquet + behavior parquet → 896d fused parquet（行序 = content 行序） |
| 复用等级 | **直接复用**（对齐有 assert 保护） |
| 缺失项 | 融合脚本**不校验 dim**（只校验 ItemID 与行数）；无 manifest 记录 fused 的输入两个文件 |
| 风险 | ⚠ **ItemID 不保证 = 行号+1**。实测 `item_emb_768d.parquet`：Sport/Toys 是 `1..N`，**Beauty 是 1..N 的一个排列**（`is_1..N=False`）→ 任何用 `ItemID-1` 当行号的代码在 Beauty 上会静默错位（`sid_consistency_contrast.py:135-147` 有正式告警与量化证据） |
| 建议 | 所有跨表对齐一律走 `sid_consistency_contrast.item_id_to_row()` / `pairs_to_rows()`，禁止 `ItemID-1` |

### ④ RQ-VAE Encoder latent（z）的提取

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | `tools/sid_per_layer_diagnostic.py:87-120 encode_z()`；`rqvae/models/rqvae.py:69` `z = self.encoder(x)` |
| 代码证据 | `encode_z` 复刻 codegen 的模型构造口径（从 ckpt 的 `args` 重建 RQVAE），`DataLoader(shuffle=False)` 保证行序，返回**已 L2 归一化**的 z（`:120`） |
| 输入与输出 | ckpt + 输入 parquet → (N, e_dim=32) 张量（内存，不落盘） |
| 复用等级 | **直接复用** |
| 缺失项 | 无"把 z 落盘"的脚本（每次现算，CPU 前向）；**归一化后返回**——若 E2 需要未归一化 z，需另写（一句话改动） |
| 风险 | `encode_z` 用了**已归一化**的 z；与 `rebuild_xq_partial` 的累计向量做 cosine 时两侧都归一化，自洽；但若要与"未归一化 residual"比对需注意口径 |
| 建议 | E2 复用 `encode_z`，并在协议里注明"z 按行 L2 归一化后使用" |

### ⑤ RVQ 各层累计量化向量 / 各层 code / residual 的定义与提取

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | 累计向量：`tools/sid_per_layer_diagnostic.py:65-71 rebuild_xq_partial(codes, cb, n_layers)`；层码本：`tools/sid_consistency_contrast.py:96-104 load_codebook_weights()` |
| 代码证据（定义源头） | `rqvae/models/rq.py:39-56`：`residual=x`；循环内 `x_res,_,idx = quantizer(residual)`；**`residual = residual − x_res`**；**`x_q = x_q + x_res`**（累计和）；`vq.py:63-99`：`x_q=codebook[argmin]`，straight-through `x+(x_q−x).detach()`（eval 下恒等于 `x_q`） |
| 输入与输出 | codes(N,4) + 层码本 → 累计 (N,32) 张量；(N,4) codes npy 已在盘 |
| 复用等级 | 累计向量 **直接复用**；**residual 需要新增**（一行：`res_l(z) = z − rebuild_xq_partial(codes, cb, l)`） |
| 缺失项 | ⚠ **没有现成的"逐层 residual"或"逐层单层量化向量 x_res_l=codebook_l[code_l]"导出**（`sf_rq.py:93-125` 有重算 residual 的逻辑，但属 E48 SFRQ 方向，非通用工具） |
| 风险 | `rebuild_xq_partial` 用的是**保存的 codes**。碰撞消解条目（第4列≠0）的码未必是该层 argmin → 由 codes 重建的累计向量与模型单次前向的 `x_q` 在少数条目上可能有微差（**待验证**，需 E3 sanity 比对） |
| 建议 | E3 增加一项 sanity：对全量 item 比对 `rebuild_xq_partial(codes,cb,3)` 与 `model(x, use_sk=False)` 的 `x_q` |

### ⑥ 最终 SID token 的生成、映射、碰撞处理及 collision bit

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | `rqvae/generate_sid_codes_param.py`（参数化，E3 用）；同源 legacy：`generate_code_Sport_768d.py`、`generate_sid_codes_sport_768d.py`、`generate_code_Sport_v2_xq.py`；产物 `data/{Ds}/{tag}_sid.npy`(N,4) |
| 代码证据 | 初筛 `use_sk=False`（纯 argmin）：`:71`；碰撞循环（≤30 轮，用 `use_sk=True`，且先 `vq_layers[:-1].sk_epsilon=0`）：`:83-110`；**第 4 列 = 组内序号** `:122-128`（`np.hstack` 一列 0 后按重复组内 enumerate 赋 i） |
| 输入与输出 | ckpt + fused parquet → codes(N,4) int npy |
| 复用等级 | **直接复用** |
| 缺失项 | 第 4 列语义（组内序号 vs "碰撞位"）无独立文档；**当某 3-token 组 >256 件时第 4 列会 ≥256**（T5 侧会越界，见风险 1）——实测三数据集均未触发（最大 2） |
| 风险 | 见 §3 风险 1（token 区间不互斥）；见风险 5（孤儿 codes 别名） |
| 建议 | 冻结"collision 列 = 消解序号"的定义，并断言 `max(col3) < 256` |

### ⑦ `per_level_alignment_diagnostic.py` 的输入、输出、指标与限制

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | `tools/per_level_alignment_diagnostic.py`；产物 `doc/per_level_alignment_diagnostic.md` |
| 代码证据 | 测量空间 `:176-184`（content=`item_emb_768d_contentonly.parquet`、beh=`behavior_emb.parquet`、sasrec=`sasrec_output/.../*_best.pth`）；变体表 `:48-73`；两种分组 `:111-118`（**累积前缀** `prefix_groups` / **单层码字** `level_groups`）；指标 `:121-133`（组内平均成对 cos − 随机对 cos）；`n_sem=min(3, cols-1)` 排除碰撞列 `:213`；健康度 `:240-251` |
| 输入与输出 | codes npy + ckpt 码本 + 三个参照空间 → markdown 报告 |
| 复用等级 | **直接复用** |
| 缺失项 | ⚠ 只测"**同组内聚集度**"，不测 Top-K overlap/Spearman/edge AUC；⚠ 行为侧参照**固定用 TA-SID 的 `behavior_emb.parquet`**，对 Shuffled/ItemCF/Raw-CoOcc 变体而言其"δ_beh"是相对**公共行为参照**而非该变体自己的行为空间（`:177`，注释已写"跨变体共享"） |
| 风险 | 前缀分组在深层退化（脚本已用 `cover<0.05` 标 ⚠）；`random_pair_mean_cos` 用固定随机对（`:188-194`，seed=0）无 degree 匹配 |
| 建议 | 保留为"层内聚集度"补充指标；**主判据仍用 `behavior_structure_probe` 的 degmatch edge AUC** |

### ⑧ Top-K overlap / Spearman / edge AUC 的现有实现

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | edge AUC：`tools/sid_consistency_contrast.py:184-197`、`tools/behavior_structure_probe.py:253-255`、`tools/sid_capacity_diagnostic.py:96-101`；Top-K overlap(Jaccard)：`tools/analyze_emb_orthogonality.py:31-36,78-93`；Spearman：`analyze_emb_orthogonality.py:124`、`tools/behavior_space_diagnostic.py:132`、`tools/stage_probe.py:297-299` |
| 代码证据 | **edge AUC 主口径** = `behavior_structure_probe.py:148-278`：E 图 `build_graph`（train history+target、window=3、去对角）、正样本 = 频次最高 `TOPK_POS=1000` 条边、负样本 **`sample_neg_degmatch` 按 ⌊log2(deg)⌋ 分桶的度匹配**（`:210-240`）、`cos_sim`、`hamming(n_tokens=3)`、`auc_l0_match`；seed=2024、N_NEG=20000 |
| 输入与输出 | codes + 各空间 embedding → per-(空间×变体) AUC 表（CSV/markdown） |
| 复用等级 | `behavior_structure_probe` **直接复用**（已是 degmatch 主口径）；`analyze_emb_orthogonality` **小幅修改后复用** |
| 缺失项 | 无独立的"Top-K 邻居 overlap（连续空间之间）"通用工具（Jaccard 只在 orthogonality 脚本里，且用 4096d `item_emb.parquet`） |
| 风险 | 见 §3 风险 4（orthogonality Analysis-3 的行号/ItemID 混用；Jaccard 用 4096d 而非 768d） |
| 建议 | E1/E2 的 Top-K overlap 新建一个小工具，统一走 768d + `item_id_to_row`；edge AUC 一律走 degmatch |

### ⑨ SID prefix 诊断能力

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | `tools/per_level_alignment_diagnostic.py:111-118`（`prefix_groups` 累积前缀 / `level_groups` 单层） |
| 代码证据 | 前缀组内 cos 与随机对基线之差 `:215-238`；覆盖度退化告警 `:228,273` |
| 输入与输出 | codes(N,4) → 前缀/单层两套分组指标 |
| 复用等级 | **直接复用** |
| 缺失项 | 无"前缀 Hamming 距离分布"或"前缀命中率"这类离散口径（现有只有连续空间的组内 cos） |
| 风险 | 前缀分组会退化（深层几乎全单例），只能看 L0/L1 |
| 建议 | E4 用 `hamming(codes[:, :l])` 的离散口径补一个"前缀可分性"指标 |

### ⑩ Codebook capacity scan 的配置、checkpoint、结果与容量范围

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | 训练：`rqvae/run_e41_k0_sweep.sh`、`run_e41_k0_1536.sh`、`run_k0_stress.sh`；后处理：`run_e41_postprocess.sh`；T5 重训：`run_e43_toys_t5.sh`；诊断：`tools/sid_capacity_diagnostic.py`；结果：`doc/e41_K0{512,768,1536,2048}.md`、`doc/e41_k0_sweep_results.md`、`doc/e42_adaptive_k0_results.md`、`doc/e4x_*_K0*.md` |
| 代码证据 | **只改 L0**：`run_e41_k0_sweep.sh:13-20`（`--num_emb_list ${K0} 256 256`，`e_dim=32`、`layers 512 256 128 64`、`sk_epsilons 0.0 0.0 0.003`、epochs=3000 全部固定）；扫描实现 `sid_capacity_diagnostic.py:63-71`（**在 z 空间做单层 k-means 重量化**，k=256..8192） |
| 输入与输出 | fused 896d → ckpt / codes；诊断报告 markdown |
| 复用等级 | **直接复用**（诊断）；重训 **暂不启动**（本轮只盘点） |
| 已有容量范围 | RQ-VAE ckpt 存在：Toys K0∈{256,512,768,1024(L0Big),1536,2048}、Sport K0∈{64,128}(stress)；codes 齐套（实测 `col0max`=510/767/1021/1535/2047） |
| 缺失项 | ⚠ **T5 只在 K0∈{512,1024} 重训过**（`model/logs/*_L01024_*`、`*_L0512_*`，各 3 seed）；K0∈{1536,2048,768} **无 T5**；768 连 codes 均无 |
| 风险 | 见 §3 风险 1、2 |
| 建议 | 容量↔推荐效果的任何结论都必须标注"K0≠256 时 token 空间非分层互斥"，或先修 `item2code` 再复算 |

### ⑪ 生成式推荐评估脚本、日志、随机种子及指标口径

| 项 | 内容 |
|---|---|
| 实际脚本/产物 | 训练/评测：`model/main.py`（`evaluate()` `:154-178`）；离线评测：`tools/eval_trained_model.py`；invalid rate：`tools/eval_invalid_rate.py`；日志 `model/logs/*_eval.log` |
| 代码证据 | `set_seed` `main.py:180-186`；best 选择 = `NDCG@20` `:285-294`；beam search `generate(max_length=5, num_beams=beam_size)` `:90-97`；指标 `recall_at_k/ndcg_at_k` `:127-135`；**vocab_size=1025** `:202`；T5 seed 默认 2025，多 seed 用 2025/2026/2027 |
| 输入与输出 | T5 ckpt + codes + test.parquet → Recall@K/NDCG@K（K∈{5,10,20,50,100}） |
| 复用等级 | **直接复用** |
| 缺失项 | 无统计显著性内置（另有 `tools/paired_significance*.py` 承担）；test 指标在**每次刷新 val 最好时**打印（`:288`），故日志里有多条 Test 行 |
| 风险 | ⚠ **eval log 的 NDCG 列数随 `--topk_list` 变化**（3 列 vs 5 列），`$NF` 取数在 5 列下拿到的是 @100 而非 @20 —— 必须按 Metric 表头映射取值 |
| 建议 | 取数统一用 `tools/e49_summary.py` 的 `parse()`（已按表头解析）或显式列映射 |

---

## 2. 重点问题核实（B）

**2.1 所有表示能否按相同 item_id 顺序严格对齐？**
能，但**必须用 `ItemID→row` 映射**，不能假设 `ItemID=row+1`。
- 证据：`sid_consistency_contrast.py:135-147`；实测 `item_emb_768d.parquet` 的 `ItemID` 在 Sport/Toys 为 `1..N`（`is_1..N=True`），**Beauty 为排列**（`is_1..N=False`，12101 项）。
- fused 的行序 = content 行序（`fuse_embeddings.py:79-80` assert + `:110,114-117` 写 `c_ids`）。
- RQ-VAE 输入行序 = fused parquet 行序（`datasets.py:13-14` 直接按行读），codes 行号 = 该行序（`generate_sid_codes_param.py:62-79`，`shuffle=False`）。
- **结论：全链可行且严格对齐，前提是所有下游脚本都走 `item_id_to_row`。**

**2.2 原始行为关系、PPMI、SVD 向量分别在哪里构建和保存？**
- 构建：`build_transition_graph.py:82`(C)、`:126`(PPMI)、`:273`(SVD)；变体 `behavior_variants.py:79/114/149`。
- 保存：**只有 SVD 后的向量**（`_save_output` `:303-309` → `behavior_emb*.parquet`）。**C 与 PPMI 不落盘**（§1② 缺失项）。

**2.3 各量化阶段输出究竟是累计重构、单层 code，还是 residual？**
- **层 code**：`codes[:, l]`（`generate_sid_codes_param.py:118-121`）。
- **累计重构**：`rebuild_xq_partial(codes, cb, n_layers)` = Σ_{l<n} codebook_l[code_l]（`sid_per_layer_diagnostic.py:65-71`）；与 `rq.py:48` 的 `x_q = x_q + x_res` 定义一致。
- **residual**：**无现成导出**。定义在 `rq.py:47`（`residual = residual − x_res`），可由 `res_l = z − rebuild_xq_partial(codes,cb,l)` 推出（**需新增**）。
- ⚠ 由 codes 重建的累计向量依赖"保存的 codes 即模型所选码"；碰撞消解条目未必是该层 argmin（**待验证**，见 §1⑤ 风险）。

**2.4 是否能在不重训的条件下导出各阶段表示？**
能。全部为只读前向或矩阵重建：
| 阶段 | 导出方式 | 现有实现 |
|---|---|---|
| 原始关系 C | 重建 | `build_cooccurrence_matrix`（但**需自行落盘**） |
| PPMI | 重建 | `compute_ppmi` / `behavior_variants --variant ppmi` |
| SVD 行为向量 | 已在盘 | `behavior_emb*.parquet` |
| 融合输入 | 已在盘 | `fused_item_emb_768d.parquet` |
| Encoder z | 现算 | `encode_z` |
| 各层累计向量 | 现算 | `rebuild_xq_partial` |
| 各层 residual | 现算（**需新增**） | — |
| codes / collision | 已在盘 | `*_sid.npy` |

**2.5 collision bit 是否被错误地当作普通量化层纳入诊断？**
- **语义层诊断中未纳入**：`per_level_alignment_diagnostic.py:213`（`n_sem=min(3, cols-1)`）、`behavior_structure_probe.py:248`（`hamming(n_tokens=3)`）都排除第 4 列。
- **T5 输入中第 4 列是独立的第 4 个 token**：`dataset.py:85`（`offsets = c + i*256 + 1`，i 遍历 4 列），vocab=1025=4×256+1（`model/main.py:202`）。这是有意设计，与"3 语义 token + 1 collision token"一致。
- **结论：没有把 collision 当量化层；但要警惕 2.6 的 token 区间问题。**

**2.6 行为图 / PPMI / SVD 是否仅使用训练集交互构建？**
- 是。`build_transition_graph.py:227-234` 与 `behavior_variants.py:201-208` 都只读 `train.parquet`，并映射 `history+[target]`。（`preprocess_data.py:112` 下 train 序列 = `seq[:-2]`，**不含 val/test target**。）
- 诊断脚本的"行为参照 E"亦只用 train：`behavior_structure_probe.py:155-161`、`sid_consistency_contrast.py:117-131`。
- ⚠ **不一致**：`top_transition_pairs` 只遍历 `history`（丢弃 target，`:122`），而行为向量 C 用 `history+target`（含最后一个 train 项）。**待验证**其对 E1 的影响量级。

**2.7 容量扫描是只修改第一层 K0，还是同时修改全部三层？其他配置是否一致？**
- **只改 L0**：`run_e41_k0_sweep.sh:16`（`--num_emb_list ${K0} 256 256`）；`run_k0_stress.sh:41-42` 同。
- 其余全部固定：`e_dim=32`、`layers 512 256 128 64`、`sk_epsilons 0.0 0.0 0.003`、`kmeans_init True`、`epochs=3000 batch=1024 lr=1e-3`。
- ⚠ 训练 seed：`main.py:46-48` 默认 `seed=2024`（legacy），sweep 脚本**未显式传 seed** → 各 K0 用的是同一 seed 2024，可比。

**2.8 容量变化后是否需要重新生成 SID 并重新训练 T5 才能公平比较推荐效果？**
- **需要**。codes 与 T5 都随 K0 变：codes 由新 ckpt 生成（`run_e41_postprocess.sh:19-23`），T5 用新 codes 重训（`run_e43_toys_t5.sh:21-31`）。
- 已有覆盖：**K0∈{512,1024}**（Toys，3 seed）已重训+评测；其余无 T5（§1⑩）。
- ⚠ **但当前重训流程有缺陷**：T5 侧 token 映射硬编码 256 步长（见风险 1），K0≠256 时各层 token 不互斥。

**2.9 现有指标实现是否存在采样偏差/零向量/重复物品对/数据泄漏/索引错配？**
| 隐患 | 位置 | 判定 |
|---|---|---|
| 随机对含自环 + 重复 | `sid_consistency_contrast.py:176-181`（`integers(0,N,(n,2))`，未排除 `i==j`） | **存在**；`per_level_alignment_diagnostic.py:188-191` 用 `mask = ri!=rj` 排除了自环 |
| 不控流行度 | 同上的随机对基线 | **存在**；主口径 `sample_neg_degmatch`（`behavior_structure_probe.py:210`）已堵 |
| 零向量 | `normalize_rows` clamp 1e-9 保留 0（`sid_per_layer_diagnostic.py:74-77`）；冷启动物品行为向量为 0（`build_transition_graph.py:291-294`） | **存在**；AUC 中零向量恒给 cos=0，可能抬高区分度，需在协议里定"冷物品是否入样" |
| 重复物品对 | 随机对未去重 | **存在**（对 AUC 影响小，但会影响 bootstrap 独立性） |
| 数据泄漏 | 行为图仅 train | **无** |
| 索引错配 | `analyze_emb_orthogonality.py:160-171` 把 `transition_pairs_768d.npy` 的**行号**当 **ItemID** 再查 `id_to_idx` | **存在（实锤）** |
| ItemID/行号 | Beauty 非 1..N | **存在**，已被 `item_id_to_row` 堵 |

---

## 3. 最重要的实现风险（按优先级）

**风险 1（最高）— 容量扫描的 T5 侧 token 空间非分层互斥。**
`model/dataset.py:85` 的 `item2code` 把 layer 步长硬编码为 `codebook_size=256`；`model/main.py:202` 的 `vocab_size=1025` 也固定。K0≠256 时实测 `codes[:,0].max()` 达 510/1021/1535/2047（Toys L0512/L0Big/L01536/L02048），于是：
- L0 token 落到 `[1,K0]`，与 L1`[257,512]`/L2`[513,768]`/L3`[769,1024]` **区间重叠** → 同一 embedding 行在不同位置承载不同 (层,码) 语义；
- K0≥1536 时 L0 token > 1024，**超过 vocab_size=1025** → 若直接跑 T5 会越界（当前 K0∈{1536,2048} 未跑 T5，故未炸）。
- 实测 token 元组在**整行**层面仍唯一（`(c0,c1,c2,c3)→token` 是单射，`unique_tok==N`），所以不是"碰撞"，而是**层间 token 别名 / 非正交 token 空间**。
- 影响：K0∈{512,1024} 的 T5 推荐效果与 K0=256 的 T5 效果**不可直接归因于容量**，混入了 token 空间设计差异。→ **凡引用容量↔NDCG 结论必须加此说明，或先修 `item2code` 支持变 K0 再复算。**
- 佐证：`doc/e41_k0_sweep_results.md` 与 `doc/e42_adaptive_k0_results.md` 顶部已有 E45/E46 推翻横幅（Toys 塌缩=seed2024 单例）。

**风险 2 — 容量扫描的 T5 覆盖不全，"最优 K0"无法从推荐端证伪。**
只有 K0∈{512,1024} 有 T5 3-seed；K0∈{768,1536,2048} 无 T5、768 连 codes 都无。容量端有 z-gap 曲线（`doc/e41_k0_sweep_results.md`）却无对应下游。**当前不能声称"扩容改善推荐效果"。**

**风险 3 — "行为关系"存在三套互不一致的定义。**
(a) 行为向量 C = window=3 共现 **含对角**（`build_transition_graph.py:100-121` 无 `setdiag(0)`）；
(b) 诊断参照 E = 同源 C 但 **去对角**（`behavior_structure_probe.py:164-165`）；
(c) 转移对正样本 = 只遍历 `history`、**丢弃 target**（`sid_consistency_contrast.py:122-128`）。
→ E1 若把 (a)(b)(c) 混用，会得到不可解释的差异。**须在 P1 冻结唯一参照。**

**风险 4 — 既有指标的索引错配与口径漂移。**
`tools/analyze_emb_orthogonality.py:160-171` 把 transition pairs 的行号当 ItemID（实锤 bug），且 Analysis-1 的 Jaccard 用 **4096d** `item_emb.parquet` 而非 768d。`tools/behavior_space_diagnostic.py:73-85` 的 `cons_auc` 对 itemcf 有系统性定义优势（脚本注释已承认，`doc/e49_behavior_ablation…` 相关）。**这些历史数字不能直接搬进新研究。**

**风险 5 — 产物别名与孤儿文件。**
实测 `data/Toys/Toys_ta_sid_768d_sid.npy` 的 L0 用码仅 **62/256**（孤儿，已被 e48 作废），健康 2024 codes 是 `Toys_ta_sid_768d_seed2024_sid.npy`（L0=207）。`sid_consistency_contrast.py:52-53` 已订正，但**新脚本很容易再引到孤儿文件**。Sport 的 `col3` 全 0（无碰撞）；Beauty 用码 204/256。

---

## 4. 产物与脚本索引（供后续节点直接引用）

```
构建:  rqvae/preprocess_data.py · build_transition_graph.py · behavior_variants.py
       rqvae/fuse_embeddings.py · rqvae/ablation_data.py
量化:  rqvae/main.py · models/{rqvae,rq,vq,layers}.py · generate_sid_codes_param.py
       run_e41_k0_sweep.sh · run_e41_k0_1536.sh · run_e41_postprocess.sh · run_k0_stress.sh
诊断:  tools/sid_per_layer_diagnostic.py    (input/z/xq[L0..full] 逐层 AUC)
       tools/sid_capacity_diagnostic.py    (码本容量 + z-gap + 利用率)
       tools/per_level_alignment_diagnostic.py (前缀/单层分组 delta-cos)
       tools/behavior_structure_probe.py   (E 图 degmatch edge AUC; 主口径)
       tools/behavior_space_diagnostic.py  (PPMI 前的 freq-leak / cons AUC)
       tools/stage_probe.py                (A输入/B z/C xq[full] 三级分解)
       tools/analyze_emb_orthogonality.py  (Top-K Jaccard; 有索引 bug)
       tools/sid_consistency_contrast.py   (CONFIG / item_id_to_row / compute_auc)
T5:    model/main.py · dataset.py · tools/eval_trained_model.py · tools/eval_invalid_rate.py
       tools/e49_summary.py (日志→NDCG 的唯一权威解析)
数据:  data/{Sport,Toys,Beauty}/{train,valid,test}.parquet
       data/{Ds}/{item_emb_768d,item_emb_768d_contentonly,behavior_emb*,
                 fused_item_emb_768d,*_sid.npy}
ckpt:  rqvae/ckpt/{Sport,Toys,Beauty}_*  ·  model/ckpt/tiger_*.pth
日志:  rqvae/logs/*  ·  model/logs/*_eval.log
```

---

## 5. 待验证清单（P0 未闭合项）

1. `rebuild_xq_partial(codes, cb, 3)` 与模型单次前向 `x_q`（`use_sk=False`）在碰撞消解条目上是否有微差 → E3 sanity。
2. `top_transition_pairs`（丢 target）与行为向量 C（含 target）的定义差对 E1 的影响量级。
3. 统一 `random_pair_sims` 与 `sample_neg_degmatch` 两套负样本口径后，既有 AUC 数字的相对序是否保持。
4. `analyze_emb_orthogonality.py` Analysis-3 的索引错配在 Beauty 上的实际错配比例（可用 `item_id_to_row` 复算对照）。
5. K0∈{1536,2048} 若跑 T5，需确认 `vocab_size` 与 `item2code` 的改动方案（本轮不启动）。
6. `model/dataset.py:124` 的 `np.array([PAD_TOKEN]*4)` 兜底（未知 item）是否与"4 个 PAD token"在 eval 中的解码口径冲突（历史 `eval_invalid_rate` 相关）。

---

## 6. 第二轮补充审计整合（2026-10-13，量化前后行为结构保持诊断能力）

> 本节是**补审**结果，不重复 §1–§5 已确认的代码能力。
> 补审范围：`z_e→q1/q12/q123` 与 `behavior→各阶段` 的**Top-K 邻域保持**诊断能力。

### 6.1 实际材料核实（**路径均已逐一打开确认**）

| 材料 | 实际路径 | 大小/时间 | 与本研究的真实关系 |
|:--|:--|:--|:--|
| P0 任务书 | `research/Behavior Structure Preservation Audit.md` | 5,839 B / 10-13 14:41 | 第一轮 P0 的任务定义（11 环节 A + 9 问题 B + 2 产出 C） |
| 第一轮 P0 产出 | `research/behavior_structure_audit_inventory.md` | 27,430 B / 10-13 14:49 | 本文件 |
| P0 计划草案 | `research/experiment_plan.md` | 12,405 B → **已更新为 v2** | P1/Sanity/E1–E4 |
| **更早的可行性审计** | **``research/ code_audit.md``**（⚠ **文件名有一个前导空格**） | 39,738 B / 10-08 12:56 | **RQ1–RQ6 的代码可行性审计**，含"连续 vs 离散表示的 5 个同名张量"陷阱表（§4）与可行性矩阵（§5）。**第一轮 P0 未引用它** |
| 文献裁决 | `research/literature_audit_result.md` | 34,461 B / 10-08 | RQ1–RQ6 的**文献**裁决（Narrow），非代码 |
| 方案 v0.2 | `doc/behavior_structure_study_plan.md` | 25,438 B / 10-07 | RQ1/RQ2/RQ3 定位 + G1–G5 缺口清单 + 执行清单 §10 |
| 结构量化文献库 | `research/audit_structquant/papers/` | 30+ pdf/txt | **非代码**，本次不涉及 |
| 空文件 | `research/problem_audit.md`、`research/mechanism_hypothesis.md` | **0 B** | 未使用，本轮不依赖 |
| 协议文档 | — | **不存在** | 无任何 `*protocol*` 文件；P1 需新建 |
| 邻域/Jaccard 结果产物 | — | **不存在** | `doc/*.md` 中仅**正文提及**，无任何 CSV/markdown 数字产物 |

### 6.2 补审核心事实（第二轮新确认，附代码证据）

| # | 事实 | 证据 |
|:--|:--|:--|
| F1 | **全仓无 Neighbor Recall@K** | `grep "def .*recall"` 只命中 `model/main.py:127 recall_at_k`（**T5 推荐 Recall@K**，用户–商品排序，与邻域无关） |
| F2 | **全仓无标准 Jaccard@K** | `grep "def .*jaccard"` 零命中；唯一"Jaccard"出现在 `analyze_emb_orthogonality.py:86`，实为 `overlap/k` |
| F3 | **唯一的 Top-K 实现使用欧氏距离** | `analyze_emb_orthogonality.py:33-34` `np.linalg.norm(vecs - vecs[idx])`；仅在**已归一化**空间与余弦同序 |
| F4 | **ties 未处理** | `analyze_emb_orthogonality.py:36` `np.argsort(dists)[:k]`（默认 quicksort，**非稳定**）。量化空间中大量 item 的 `q1/q12/q123` 逐位相同（`rebuild_xq_partial` 为精确码本查表），**并列必然发生且次序未定义 ⇒ 不可复现** |
| F5 | **BSR 只有设计、无实现** | `doc/exp/创新点2验证方案1.md` §2.1；全仓零 `.py`。且该方向已 STOP（`doc/paper_outline_ta_sid.md:141`、`doc/paper_draft_zh_sec54_56.md:10`、`doc/paper_draft_zh_method_46.md:6`） |
| F6 | **raw z 可得但无入口** | `sid_per_layer_diagnostic.py:120` 返回**已归一化** z；`novelty_probe_ab.py:148-159 encode` 返回**未归一化** z，但无 CLI/落盘 |
| F7 | **z / xq 从不落盘** | 全仓 `grep np.save\|to_parquet` 只命中 codes 与 embedding（`generate_code*.py`、`prepare_data.py`、`fuse_embeddings.py` 等），**无任何脚本保存 z 或 xq** |
| F8 | **现有结构探针全是 edge-AUC 口径** | `sid_per_layer_diagnostic.py`、`sid_capacity_diagnostic.py`、`checkpoint_selection_probe.py`、`checkpoint_trajectory_probe.py`、`behavior_structure_probe.py`、`behavior_space_diagnostic.pair_auc:73-85`、`rho_regime_attribution.py`、`stage_probe.py` —— **无一为 Top-K 邻域保持** |
| F9 | **存在第三套 AUC 实现** | `behavior_space_diagnostic.py:73-85 pair_auc` 用 `roc_auc_score`，负样本 `rng.integers` **未排除自环/重复**；与 `sid_consistency_contrast.compute_auc:184-197`、`behavior_structure_probe.auc_of:253` 三者口径互不相同 |
| F10 | **"邻域"的近义物是 pair-set 而非邻居表** | `behavior_space_diagnostic.build_pair_sets:88-112` 给 `adj`（序列相邻）vs `far`（同序列相隔 ≥4），是**配对类型**定义，**不是** Top-K 邻居集合，二者不可互换 |

### 6.3 按用户指定五类归档的问题清单

#### 类别 1 — 已解决 / 已有可靠实现

| # | 项 | 位置 |
|:--:|:--|:--|
| 1.1 | `q1/q12/q123` 重建（=累积前缀和） | `sid_per_layer_diagnostic.py:65-71 rebuild_xq_partial`；`sid_consistency_contrast.py:107-114 rebuild_xq` |
| 1.2 | `xq[full]` 归一化 + 该空间余弦 | `sid_consistency_contrast.py:160-164 quantized_space` |
| 1.3 | `z_e` 提取（**两种**：归一化 / raw） | `sid_per_layer_diagnostic.py:87-120`（归一化）；`novelty_probe_ab.py:148-159`（raw） |
| 1.4 | item 对齐 canonical 入口（堵 Beauty 非 `1..N`） | `sid_consistency_contrast.py:135-157` |
| 1.5 | 主判据正负样本（E 图 + degmatch） | `behavior_structure_probe.py:148-178, 210-240`；采样路径已抽成可复用函数 `stage_probe.py:141-159 build_samples` |
| 1.6 | 变体×数据集×种子 → ckpt/codes 表（含 seed 回落） | `stage_probe.py:69-112 VARIANTS/FUSED_896D`、`:131-138 resolve` |
| 1.7 | Top-K 取邻居（**self 已排除**） | `analyze_emb_orthogonality.py:31-36 topk_sims`（`dists[idx]=np.inf`） |
| 1.8 | 离散前缀分组（**collision 列已排除**） | `per_level_alignment_diagnostic.py:111-118`，`n_sem=min(3, cols-1)` `:213` |
| 1.9 | 前缀碰撞率 / 逐层用码数 | `checkpoint_trajectory_probe.py:98-109 code_structure` |
| 1.10 | 行为结构分层的**素材**（计数、频率、adj/far） | `build_transition_graph.py:82`；`behavior_structure_probe.py:148-178`；`behavior_space_diagnostic.py:65-70, 88-112` |
| 1.11 | 分群基建（head/mid/tail） | `longtail_analysis.py:36 assign_groups` |
| 1.12 | 逐层重放基建（残差顺序正确，旧 bug 已记） | `rqvae/sf_rq.py:92 rederive_codes_sequential`；`rqvae/verify_variant_pairing.py:5-17` |

#### 类别 2 — 已有实现但需修正 / 验证

| # | 问题 | 位置 | 处置归属 |
|:--:|:--|:--|:--|
| 2.1 | Top-K 用**欧氏**而非余弦 | `analyze_emb_orthogonality.py:33-34` | P1.2 |
| 2.2 | `overlap/k` 被当作 Jaccard（**不是标准 Jaccard**） | `:86` | P1.3 |
| 2.3 | ties 无规则 ⇒ **不可复现** | `:36` | P1.2 |
| 2.4 | 行号当 ItemID（实锤错配） | `:160-171` | P1.0 / R-G |
| 2.5 | Jaccard 用 **4096d** 而非 768d | `:47, :58` | P1.0 / R-G |
| 2.6 | 零/冷行为向量无任何邻居指标处理 | `sid_per_layer_diagnostic.py:74-77`；`build_transition_graph.py:291-294` | P1.2 / P1.0-6 |
| 2.7 | 三套负采样口径并存（`random_pair_sims` 含自环+未去重+无 degmatch；`sample_neg_degmatch`；`mask=ri!=rj`） | `sid_consistency_contrast.py:176-181`；`behavior_structure_probe.py:210-240`；`per_level_alignment_diagnostic.py:188-191` | P1.0-5 |
| 2.8 | 三套"行为关系"定义（C 含对角 / E 去对角 / `top_transition_pairs` 丢 target） | `build_transition_graph.py:100-121`；`behavior_structure_probe.py:164-165`；`sid_consistency_contrast.py:122-128` | P1.0-1 |
| 2.9 | 第三套 AUC 实现（`roc_auc_score`，负样本未去自环） | `behavior_space_diagnostic.py:73-85` | P1.0-5 |
| 2.10 | §5 的 6 条待验证项 | 见 §5 | Sanity |

#### 类别 3 — 缺少实现，需新增

| # | 缺口 | 复核方式 | 归属 |
|:--:|:--|:--|:--|
| 3.1 | **Neighbor Recall@K**（任何空间） | F1 | P1.3 |
| 3.2 | **标准 Jaccard@K** | F2 | P1.3 |
| 3.3 | `z_e→q1/q12/q123` 的**邻域保持曲线** | 全仓无此组合 | P1.3 / E5 |
| 3.4 | `behavior→z/q1/q12/q123` 的**行为结构保持** | F5 | P1.4 / E5 |
| 3.5 | **统一样本集跨空间可比**的邻域探针 | 各脚本各自 `seed=0` 采样 | P1.2/P1.3 |
| 3.6 | z / xq 的**落盘缓存** | F7 | P1.1 |
| 3.7 | **单层** `residual` / `x_res` / 单层 loss | `research/ code_audit.md` RQ2:174-182 | P1.5 / R-B |
| 3.8 | **结构类型 × 阶段**分层诊断 | `research/ code_audit.md` RQ3:214-218 | P1.6 / R-C |
| 3.9 | 5 类表示**逐 item 对齐导出**探针 | `research/ code_audit.md` RQ1 缺口 | P1.1 / R-A |
| 3.10 | **T5 token embedding 层**结构探针 | `doc/behavior_structure_study_plan.md` G4 | R-D |
| 3.11 | **ties 规则**实现 | F4 | P1.2 |
| 3.12 | 协议文档本身 | §6.1 | P1.0 |

#### 类别 4 — 实验可信度前置问题

| # | 问题 | 依据 | 对 P1 的约束 |
|:--:|:--|:--|:--|
| 4.1 | **定义优势**：判据图 E 与 TA-SID 行为向量同源（同 `window=3`）⇒ TA-SID 在强边桶上被放大 | `behavior_structure_probe.py:148-178`；`study_plan §3.3 边界 3` | P1.6 必须显式标注 |
| 4.2 | **content 地板 0.875–0.908**：不报 Content-only，AUC 会被误读为"行为保持度" | `study_plan §3.2/§3.3` | protocol 强制并列 Content-only |
| 4.3 | **跨数据集绝对值不可比**（Sport/Toys/Beauty 区间不同） | `study_plan §3.2` | protocol 强制数据集内标准化 |
| 4.4 | 容量扫描 T5 侧 **token 空间非互斥**（`item2code` 硬编码 256） | `model/dataset.py:85`；`model/main.py:202`；§3 风险 1 | 引用容量结论必须附说明 |
| 4.5 | **孤儿 codes 文件**（Toys L0=62 已作废） | §3 风险 5 | 一律走 `stage_probe.VARIANTS` |
| 4.6 | `svds` 默认 **v0 不可复现** | `build_transition_graph.py:278`；对照 `behavior_variants.py:149`（显式 v0） | E1 必须走确定性路径 |
| 4.7 | 4 个同名 "quantized representation"，`out`(896d) 最易误用 | `research/ code_audit.md` §4 Q5 | protocol 明令禁 `out` |
| 4.8 | `set_seed` 在建模型**之后** ⇒ 3-seed 不控 T5 初始化 | `research/ code_audit.md` RQ6 | 论文侧已有记录，P1 不处理 |
| 4.9 | 产物覆盖风险（重跑脚本可覆盖 `doc/*.md`） | 历史教训（Beauty Tier-2 汇总脚本） | P1 全程写入独立临时目录 |

#### 类别 5 — 原计划中尚未完成的研究诊断

| # | 原计划遗漏 | 说明 |
|:--:|:--|:--|
| 5.1 | **整条"邻域保持"路线** | v1 的 E1–E4 全部是**配对级 edge-AUC** 路线；`Top-K overlap / Spearman` 只在 §3 P1 清单里被"列为候选"，**从未落到 E 节点**。补审确认该路线**零实现** |
| 5.2 | **K 值选择** | v1 只说"K 值需定，参考既有 K=20"；补审发现三类 K 语义不同（k=20 邻居数 / `TOPK_POS=1000` 边数 / eval `5,10,20,50,100`），**不可互套** |
| 5.3 | **ties 与零向量在邻居指标下的具体规则** | v1 第 5 条只说"需定冷物品是否入样"，未意识到**量化空间的大规模并列**会使邻居序不可复现 |
| 5.4 | **`research/ code_audit.md` 的 RQ1/RQ2/RQ3 缺口未进 plan** | 5 类表示导出、单层残差、结构分层（→ R-A/R-B/R-C） |
| 5.5 | **`study_plan` 的 G3/G4 未进 plan** | 三数据集成表、T5 embedding 层（→ R-F/R-D） |
| 5.6 | **`analyze_emb_orthogonality.py` 的索引 bug 未进 plan** | 已知实锤，但 v1 未列为修复项（→ R-G） |

### 6.4 补审结论

1. **表示层已就绪**：`z_e`（含 raw）、`q1/q12/q123`、`xq[full]`、`codes`、item 对齐**全部可现成构造**，
   且 `q1/q12/q123` 的重建是**精确码本查表**（`rebuild_xq_partial`），无额外假设。
2. **度量层全空**：**没有任何** Neighbor Recall@K、标准 Jaccard@K，也**没有任何** `z→q*` 或
   `behavior→q*` 的邻域保持诊断。现有全部为 edge-AUC。
3. **唯一可复用的邻域件带三个口径问题**：欧氏距离（F3）、非标准 Jaccard（F2）、ties 未定义（F4）。
4. **最小新增面 = 3 个件**：统一的 Top-K 采集器（余弦 + tie 规则 + 零向量策略）、
   Recall/Jaccard 计算器、跨空间对齐的表示导出层。其余全部复用。
5. **AUC ≠ BSR ≠ 邻域保持**：三者在仓库中必须并列声明、互不替代。

> 本节的整合结果已写入 `research/experiment_plan.md` v2 的 P1.0–P1.7 与 E5/R-A–R-G。
> **2026-10-14 后续**：P1.0 的三项开放决议已冻结并落成 `research/behavior_structure_protocol.md`（v1.0）：
> 主参照 = **去对角 C**（PPMI 降辅助）｜tie = **余弦降序 + canonical index 升序**｜
> 零向量 = **`U_beh` 剔除 / `U_cont` 保留** 两套 universe。本清单 §6.3 的类别 2/3 由此全部有了归属。
