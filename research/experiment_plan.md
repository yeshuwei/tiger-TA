# 行为结构保留研究 — 实验计划（**v2**：P1 完整化，含 Sanity / E1–E5 / R-A–R-G）

> **版本**：v2（2026-10-13）。v1 = 仅 P0 产出（P1 只列 8 条冻结项、无实施任务）。
> **v2 改动**：P1 从"冻结清单"扩为"**9 条冻结项 + 7 个实施任务（P1.0–P1.7）**"；
> 补入第二轮审计暴露的**邻域保持 / 行为结构保持**诊断缺口（E5 与 P1.1–P1.4）；
> 补入 `research/ code_audit.md`（RQ1–RQ6）与 `doc/behavior_structure_study_plan.md`（G1/G3/G4）
> 中**原 P0 计划未纳入**的缺口（R-A–R-G）；新增"P1 之后的任务与依赖顺序"一节。
> **状态**：**草案，不执行**。本次只制定 P1 计划，不跑 P1、不训练、不改模型。
> **首轮范围**：**仅 Sport 最小诊断**；Beauty、Toys 留作后续复核。
> **容量扫描**：**仅盘点**（见 `research/behavior_structure_audit_inventory.md` §1⑩），本轮**不启动重跑**。
> **共享只读工具链**：`tools/behavior_structure_probe.py`、`tools/sid_per_layer_diagnostic.py`、
> `tools/sid_consistency_contrast.py`、`tools/stage_probe.py`、`rqvae/build_transition_graph.py`，
> 以及 P1 新建的 `tools/structure_probe_common.py`。
> **审计整合记录**：`research/behavior_structure_audit_inventory.md` §6。
> **口径冻结**：`research/behavior_structure_protocol.md`（**v1.0，2026-10-14 已定稿**）——
> P1.0 的三项开放决议（主参照 / tie 规则 / 零向量）已闭环，见本文件 §P1.9。

首轮 Sport 用到的既有产物（已在盘，无需重训）：
`data/Sport/train.parquet`、`item_emb_768d.parquet`、`behavior_emb.parquet`、
`fused_item_emb_768d.parquet`、`Sport_ta_sid_768d_sid.npy`、
`rqvae/ckpt/Sport_TA_SID_768d/*/best_collision_model.pth`。

---

## P0 — 只读审计（本节点已执行）

- **研究问题**：现有代码/配置/checkpoint/日志/结果中，哪些实验能力已存在、可复用、缺失？
- **输入**：仓库全部源码、`rqvae/ckpt/`、`model/ckpt/`、`rqvae/logs/`、`model/logs/`、`doc/`、`data/`。
- **操作**：逐文件读实现（不凭文件名猜测），离线只读复核 SID code 列结构、ItemID 顺序、T5 token 映射。
- **输出**：`research/behavior_structure_audit_inventory.md`（11 环节表 + 重点问题核实 + 风险清单）。
- **依赖**：无（纯只读）。
- **风险**：把历史实验记录误当已复现证据；把计划功能误写成已实现。
- **停止标准**：11 个环节全部有 `文件:行号` 级证据；无法确认项显式标 **待验证**。
- **状态**：**已完成**（见审计清单 §5 的 6 条待验证）。

---

## P1 — 指标协议冻结 + 支撑工具就位（**本轮只制定，不执行**）

> **P1 相对 v1 草案的改动见 §P1.8。** P1 产出两类东西：**(a) 一份冻结决议文档**，**(b) 一套只读公共工具**。
> P1 **不产生任何研究结论**；它的唯一目的是保证 E1–E5 之间的数字**可比、可复现**。
> 全部 P1 任务状态：**⬜ 未开始**（本次仅制定计划）。

---

### P1.0 冻结决议文档 `research/behavior_structure_protocol.md`（**✅ 已定稿 v1.0，2026-10-14**）

- **研究目的**：在 E1–E5 开跑前把"度量定义 / 空间定义 / 采样定义"钉死，杜绝口径中途漂移。
- **现有代码基础**：既有实现 `behavior_structure_probe.py`、`sid_consistency_contrast.py`、
  `stage_probe.py`、`per_level_alignment_diagnostic.py`；审计清单 §2、§3、§6。
- **本次交付**：**只写文档，不改代码**。协议已落盘于 `research/behavior_structure_protocol.md`，
  含 14 节：主参照 / 空间清单 / 相似度 / Top-K 与并列 / 零向量 / 两类诊断 / 指标定义 /
  K 与抽样与对齐 / 采样协议 / bootstrap / 评估纪律 / 验收标准 / 变更记录 / 仍未决。逐条决议如下：
  1. **行为关系主参照 → 已冻结：去对角原始共现 C 为主参照**，PPMI 降为**辅助参照**，
     SVD 行为嵌入为**压缩后向量参照**（协议 §1）。
     **对原任务书的偏离及理由**（协议 §1.2 R1–R4）：① 保持与既有 54/54 路 AUC 证据链可比
     （现有主口径 E 图**本来就是**去对角 C，`behavior_structure_probe.py:164-165`）；
     ② PPMI 的 `max(·,0)` 截断会抹平弱边，使"结构丢失"的诊断混入 PPMI 自身效应，无法归因
     （`build_transition_graph.py:126-145`）；③ C 是无参参照；④ **避免把同一矩阵的两种变换当成
     两条独立证据**（协议 §1.4 强制声明）。
     同时冻结"含/去对角"与"含/不含 target"：**去对角 + `history+target`** 为主口径，
     `history`-only（`sid_consistency_contrast.py:122-128`）降为敏感性。
     ⚠ 必须记录：行为**向量**用**含对角** C 算（`build_transition_graph.py:100-121`），
     与参照空间**不是同一张矩阵**。
  2. **空间清单**（**新增，v1 遗漏**）：把 `input(896d) / content(768d) / behavior(128d) / z(32d) /
     q1 / q12 / q123(=xq[full]) / codes` 全部列入，并声明**每个空间的维度与语义**。
     ⚠ 依据 `research/ code_audit.md` §4 Q5：仓库存在 **4 个同名 "quantized representation"**
     （累加 `x_q` / straight-through / 单层 `x_res` / 解码器 `out` 896d）——**必须明文禁止使用 `out`**。
  3. **连续度量**：余弦相似度（**非欧氏**，见 P1.2 与 §P1.8-②）。Spearman 用于跨空间秩相关。
  4. **Top-K 邻域规则 → 已冻结**（协议 §3/§4/§8）：
     **度量 = 余弦**（禁用欧氏，`analyze_emb_orthogonality.py:33-34` 是欧氏）；**方向 = 降序**；
     **self 强制排除**；**候选集合 = 全库 N**；
     **排序键 = 相似度（降序）+ canonical item index（升序）两级**，
     **禁止**用共现频次作次级键（否则评估对象影响评估排序），实现须 `stable` / `lexsort`；
     必须报告 `tie_share@K`、`collapse`、逐层碰撞率，并做**并列敏感性检查**（阈值 0.005，协议 §4.3）；
     **K ∈ {5,10,20,50,100}，主 K = 20**；query 默认全量、分块计算，回落抽样 `n_query=2000, seed=2024`。
  5. **对照物品对采样**：正样本 = 频次最高 `TOPK_POS=1000` 边（`behavior_structure_probe.py:148-178`）；
     负样本 = `sample_neg_degmatch`（⌊log2(deg)⌋ 分桶，`:210-240`）+ `sample_neg_uniform` 参照；
     **统一 `SEED=2024`、`N_NEG=20000`**；⚠ **弃用** `sid_consistency_contrast.random_pair_sims`
     （含自环 + 未去重 + 无 degree 匹配，`:176-181`）作为主口径，仅留作敏感性。
  6. **零向量 / 冷物品 → 已冻结为两套 universe**（协议 §5）：
     **(甲) 行为结构保持主分析**（`Bh → z/q1/q12/q123`）用 **`U_beh = {i : ‖Bh[i]‖₂ > 0}`**，**剔除**零向量，
     并**强制单独报告** `n_zero / zero_ratio / coverage`；
     **(乙) 连续→量化邻域保持分析**（`z → q1/q12/q123`、`X → z`）用 **`U_cont = 全部 N`**，
     **不因行为向量为零而额外剔除**。
     纳入零向量的补充分析**必须单独成篇**，且必须声明"零向量之间的邻居排序完全由次级键决定，
     是**人为并列**、不是真实行为关系"。
     成因已核实：`build_transition_graph.py:291-294` 的 `if i not in seen_items: behavior_emb[i] = 0.0`；
     `normalize_rows` 用 `clamp(1e-9)` **保留**零向量（`sid_per_layer_diagnostic.py:74-77`）。
     ⚠ 待测：`‖Bh‖=0` 集合与 `deg=0` 集合的差集（协议 §14 U1）。
     ⚠ 既有 `sample_neg_degmatch` **已**把 `deg==0` 排除在分桶外（`behavior_structure_probe.py:213`），但**无任何**邻域指标做剔除。
  7. **SID prefix similarity**：连续口径（前缀组内 cos，`per_level_alignment_diagnostic.py:111-118`）
     vs 离散口径（前缀 Hamming，`behavior_structure_probe.py:248`）**并列声明**，E4 两者都出、不混算。
  8. **bootstrap**：单位（item / user / edge）、CI 水平、随机种子；⚠ 现有随机对**未去重**，
     重复对破坏独立性。
  9. **正交评估纪律**：行为轴与内容语义轴**分别**评估；**禁止**把融合空间 cosine 解释为纯行为相似度
     （融合空间 = `LN(content)⊕LN(beh)`，`fuse_embeddings.py:96-110`）。
- **输入**：审计清单 §2/§3/§6；`research/ code_audit.md` §4。
- **输出**：`research/behavior_structure_protocol.md`（**已产出，v1.0，14 节**）。
- **验收标准**：上列 9 项每项有明文决议且指向具体函数；空间清单覆盖全部 8 个空间并列明禁用空间 `out`；
  口径层**无任何"待定"条目**（§14 的 6 项是**实现期待测事实**，不是口径未决）。
- **前置依赖**：P0 完成（已完成）。
- **风险**：口径中途变更导致 E1–E5 不可比 → 变更必须记入协议 §13。
- **完成状态**：✅ **口径层已完成（2026-10-14）**；实现层面的 6 项待测事实待 P1.1–P1.7 回填。

---

### P1.1 统一 item 对齐与多空间表示导出（**新增工具** `tools/structure_probe_common.py`）

- **研究目的**：让所有下游诊断共用**同一个 item 顺序**与**同一份表示缓存**，消除各脚本各读各的错配风险。
- **现有代码基础**：`item_id_to_row` / `pairs_to_rows`（`sid_consistency_contrast.py:135-157`，已堵
  Beauty 非 `1..N`）；`encode_z`（`sid_per_layer_diagnostic.py:87-120`，**返回已归一化 z**）；
  `rebuild_xq_partial`（`:65-71`）；`load_input_emb`（`:80-84`）。
- **需新增或修改**：
  1. 建 `tools/structure_probe_common.py`，提供 `build_spaces(ds, variant, seed)` →
     返回**逐 item 对齐**的 `{input896, content768, behavior128, z, q1, q12, q123, codes}` 字典。
  2. **raw `z_e` 的暴露**：现有 `encode_z` 归一化后返回；raw 版本已在
     `novelty_probe_ab.encode`（`:148-159`，**未归一化**）中实现但无入口。**统一在 common 中显式返回
     两者**（`z_raw` 与 `z_norm`），并在 protocol 中写明用哪个。
  3. **缓存落盘**：全仓 `grep np.save|to_parquet` 确认**没有任何脚本保存 z / xq**（只保存 codes 与
     embedding）。P1.1 需把 z/xq 缓存到**独立临时目录**（禁止覆盖 `data/`、`rqvae/ckpt/`、`doc/`）。
  4. **两套 item universe 的构造与上报**（协议 §5）：`U_beh = {i : ‖Bh[i]‖₂ > 0}`、
     `U_cont = 全部 N`；并**实测 `n_zero / zero_ratio / coverage`**，
     以及 `‖Bh‖=0` 集合与 `deg=0` 集合的**差集**（回填协议 §14 U1）。
- **输入**：`data/{Ds}/{item_emb_768d,item_emb_768d_contentonly,behavior_emb*,fused_item_emb_768d}.parquet`、
  `*_sid.npy`、`rqvae/ckpt/<tag>/*/best_collision_model.pth`。
- **输出**：对齐后的 numpy/torch 缓存 + 一个 `spaces` 字典。
- **验收标准**：所有空间 `N` 相同；`codes` 行序与 `input896` 行序逐位一致；
  对 Sport/Toys 与 `ItemID-1` 等价、对 Beauty 不等价（断言显式记录）。
- **前置依赖**：P1.0-6（零向量规则）。**被** E1/E2/E3/E5 依赖。
- **风险**：**孤儿 codes 文件**——实测 `data/Toys/Toys_ta_sid_768d_sid.npy` 的 L0 用码仅 62/256
  （已被 e48 作废），健康文件是 `Toys_ta_sid_768d_seed2024_sid.npy`（L0=207）。变体/种子一律走
  `stage_probe.VARIANTS:78-112` + `resolve:131-138`，禁止手写文件名。
- **完成状态**：✅ **已完成（2026-10-14）**。新增 `tools/structure_probe_common.py`（只读 CPU，8.1 s）；
  产出 `research/p11_alignment_report.md` 与 `research/_p11_cache/{Sport,Toys,Beauty}_TA-SID_seed2024/`
  （各 8 文件，共 28 MB，仅 z/q/codes）。
  **实测要点**：① 三集 `codes`/`fused`/`item_emb_768d` **行序逐位一致**，Beauty 的 `ItemID` 是排列
  （Sport/Toys 连续）；② `U_beh` 三集齐备，coverage 0.9985/0.9980/0.9973，`zero_ratio` 0.0015/0.0020/0.0027；
  ③ 协议 §14 **U1 闭合**（`‖Bh‖=0` ≡ `deg=0` ≡ 补(seen)，对称差 = 0）；
  ④ 协议 §14 **U2 闭合**（codes 重建 ≠ `use_sk=False` 前向，209/226/235 行、全在第 2 层，裁决维持 §2.1）；
  ⑤ `collapse`：`q1` 0.983–0.986、`q12` 0.246–0.288、`q123` 0.0000–0.0008（已回填协议 §4.2，文字订正）。
  详见协议 §14.1。

---

### P1.2 Top-K 邻域采集器（**新增**，含并列与零向量规则）

- **研究目的**：提供**唯一**的 Top-K 邻居定义，供 P1.3/E5 的所有空间复用。
- **现有代码基础**：**全仓唯一实现** `topk_sims`（`analyze_emb_orthogonality.py:31-36`），已用
  `dists[idx]=np.inf` **排除 self**。
- **需新增或修改**（**以下三条已由协议 §3/§4 冻结，实现须逐条对齐**）：
  1. **距离度量**：现用 `np.linalg.norm(vecs - vecs[idx])` = **欧氏距离**（`:33-34`），
     不是余弦。对已归一化空间二者**同序**，但对 **raw z**、**raw behavior SVD** 等未归一化空间
     **不同序**。P1.2 统一为**余弦**（或对每个空间先归一化再取欧氏），并在 protocol 声明。
  2. **并列 ties**：现用 `np.argsort`（默认 quicksort，**非稳定**）。量化空间中**大量 item 的
     q1/q12/q123 逐位相同**（`rebuild_xq_partial` 是精确码本查表），并列必然发生且**次序未定义
     ⇒ 结果不可复现**。**已冻结**（协议 §4.1）：**两级键 = 余弦（降序）+ canonical item index（升序）**，
     实现用 `np.lexsort` 或 `argsort(kind="stable")`；**禁止**以共现频次作次级键；并须输出
     `tie_share@K` / `collapse` / 逐层碰撞率，且完成 §4.3 的并列敏感性检查（阈值 0.005）。
  3. **零 / 冷启动向量**：零向量与任何向量的余弦为 0，其 Top-K **完全由 tie 规则决定** ⇒ 若入样会
     系统性污染。必须实现 P1.0-6 的入样/剔除/单列上报策略。
- **输入**：任一空间矩阵 (N,d) + 待查 item 子集（`U_beh` 或 `U_cont`）+ K（**K ∈ {5,10,20,50,100}，主 K=20**，协议 §8.1）。
- **输出**：`(N_query, K)` 的邻居 index 矩阵（+ 可选相似度），**全部排除 self**。
- **验收标准**：同一输入连跑两次**逐位一致**；对同一空间，显式断言 `neighbors[i]` 不含 `i`；
  对已知全同向量块能给出**确定性**且**可解释**的邻居序。
- **前置依赖**：P1.0-4。**被** P1.3 / E5 依赖。
- **风险**：K 值无先例——现有 k=20（正交性脚本）、`TOPK_POS=1000`（E 图**边数**，非邻居数）、
  eval `topk_list=5,10,20,50,100`，三者语义不同，不可互套。
- **完成状态**：⬜ 未开始。

---

### P1.3 邻域保持指标族（**新增**：Neighbor Recall@K / 标准 Jaccard@K / Top-K overlap）

- **研究目的**：给出**与 edge AUC 正交**的"邻域保持"诊断量，用于回答"A→B 阶段邻居集合变了多少"。
- **现有代码基础**：**无**。已核实：全仓 `def .*recall` 只命中 `model/main.py:127`
  （**T5 推荐 Recall@K**，与邻域无关）；`def .*jaccard` **零命中**。
- **需新增或修改**：
  1. `neighbor_recall@K(A, B)`：`|N_K^A(i) ∩ N_K^B(i)| / K`，对 item 子集取均值。
  2. `jaccard@K(A, B)`：`|A∩B| / |A∪B|`。⚠ **必须与现有 `overlap/K` 区分**——
     `analyze_emb_orthogonality.py:86` 的 `jaccard = overlap / k` **不是** Jaccard
     （两侧集合等大时标准 Jaccard = `overlap/(2k−overlap)`）。
  3. `topk_overlap@K`：沿用现有 overlap/k 作为**第三个**独立量（保留历史可比性）。
     规范名称为 **`Overlap@K`**；既有文档中任何把该量称作 "Jaccard" 的表述**一律作废**（协议 §7-B）。
  4. 输出的**空间对**：`z→q1`、`z→q12`、`z→q123`（量化前后保持），
     `behavior→z`、`behavior→q1`、`behavior→q12`、`behavior→q123`（行为结构保持，即 BSR 的度量形式）。
- **输入**：P1.1 的 spaces 缓存；P1.2 的邻居矩阵。
- **输出**：`(空间对 × K)` 的保持度表（CSV + markdown）。
- **验收标准**：**恒等空间对**（A=B）的 Recall/Jaccard 必须 = 1.0（自检）；
  对**已知人工构造**的扰动空间，指标单调随扰动强度下降。
- **前置依赖**：P1.0-4/5、P1.2。**被** E5 依赖。
- **风险**：把该指标**当作方法创新**——P1 明确纪律：**它只是诊断量，不是新方法**（同 §4 约束）。
- **完成状态**：⬜ 未开始。

---

### P1.4 行为结构保持（BSR）口径落地

- **研究目的**：把"行为结构在量化中保留多少"从**设计文档**变为**可执行口径**。
- **现有代码基础**：**仅有设计文档** `doc/exp/创新点2验证方案1.md`（§2.1 Behavior Structure
  Retention，Step 3 算 BSR@10/@20），**全仓无任何 `.py` 实现**。且该方向已被 **STOP**
  （`doc/paper_outline_ta_sid.md:141`、`doc/paper_draft_zh_sec54_56.md:10`、
  `doc/paper_draft_zh_method_46.md:6` 均禁用该术语）。
- **需新增或修改**：P1 只做**口径一层**——把"行为结构保持"定义为 P1.3 的
  `neighbor_recall/jaccard(behavior → 各阶段)`，**不再复活 `BSR@K` 这一术语，也不再声称它是方法**。
- **输入**：`behavior_emb*.parquet`（128d）；P1.1 的 spaces。
- **输出**：进入 protocol 的定义条目（不单独出产物）。
- **验收标准**：protocol 中该定义与既有 AUC 口径**并列且互不替代**（明文写"**AUC ≠ BSR ≠ 邻域保持**"）。
- **前置依赖**：P1.3。
- **风险**：术语复活触发论文侧的红线（第二创新已裁撤）。
- **完成状态**：⬜ 未开始。

---

### P1.5 逐层单层 residual / `x_res` / 单层 loss 只读重放（RQ2 缺口）

- **研究目的**：把"哪一层丢结构"从**累计口径**升级为**单层增量口径**。
- **现有代码基础**：累计层**已有**（`rebuild_xq_partial`，`sid_per_layer_diagnostic.py:65-71`）；
  逐层重放基建**已有**（`rqvae/sf_rq.py:92 rederive_codes_sequential`）。
- **需新增或修改**：一个**只读旁路**（**不改 `rq.forward`**）——模仿真实残差顺序逐层记录
  `residual` 与 `x_res = codebook_l[idx_l]`，并单独算 `F.mse_loss`。约 30 行。
  ⚠ `rq.forward` 只返回**累积** `x_q` 与**平均** loss（`rq.py:39-56`），**不逐层暴露**。
  ⚠ 重放**必须**用真实残差顺序（`verify_variant_pairing.py:5-17` 记录了旧 `argmin(z, cb_l)` bug）。
- **输入**：ckpt + codes + 输入 embedding。
- **输出**：`(N, L, 32)` 的逐层 `x_res` 与 `(L,)` 的单层 loss。
- **验收标准**：`Σ_l x_res_l == rebuild_xq_partial(codes, cb, L)` 逐位一致（≤1e-6）。
- **前置依赖**：P1.0（空间定义）。**被** E3 增强项依赖。
- **风险**：SID decoding ambiguity——第 2 层受 `use_sk` 影响，重放默认 `use_sk=False`，
  与 codegen 主 pass 一致（L2 有 1–2% 行被碰撞消解循环改写，`generate_sid_codes_param.py:83-110`）。
- **完成状态**：⬜ 未开始。

---

### P1.6 结构类型 × 阶段 的分层诊断（RQ3 缺口）

- **研究目的**：回答"**哪种**行为结构最易丢"，而非只报整体 AUC。
- **现有代码基础**：三类结构的**计算全有**——共现计数（`build_transition_graph.py:82`；
  `behavior_structure_probe.py:148-178` 返回 `counts`）、item 频率（`behavior_space_diagnostic.py:65-70`）、
  任意空间 KNN（P1.2）；分群基建 `longtail_analysis.py:36 assign_groups`。**缺的是连接脚本**。
- **需新增或修改**：把正样本集按 **count 分位数**（强/弱边）、**⌊log2(deg)⌋ 分桶**、
  **head/mid/tail** 切分，逐桶算 `input → z → q1 → q12 → q123` 的 AUC **与** P1.3 的邻域保持率。
- **输入**：P1.1/E 图；P1.2/P1.3 工具。
- **输出**：`(结构桶 × 阶段)` 的 AUC/保持度表 + Δ（input→xq[full] 的损失按桶分解）。
- **验收标准**：各桶样本量、度分布不匹配程度**显式报告**；桶间比较必须带 degmatch 负样本。
- **前置依赖**：P1.2、P1.3、E1 的参照定义。
- **风险**：**流行度捷径**——强边天然连高频 item（`behavior_structure_probe.py:210-240` 的
  degmatch 已缓解）；**定义优势**——E 图与 TA-SID 行为向量同源。
- **完成状态**：⬜ 未开始。

---

### P1.7 验收门禁（P1 结束前必须全绿）

- **研究目的**：确保 P1 的工具在进入 Sanity/E1 之前**自洽**。
- **需新增或修改**：无新代码，一条 checklist。
- **验收标准（全部满足才可进 Sanity）**：
  1. P1.2 的邻居采集器**两次运行逐位一致**；
  2. P1.3 的**恒等空间对** Recall/Jaccard = 1.0；
  3. P1.4 的一致性断言（`Σ x_res = 累积`）通过；
  4. P1.1 的 `N` 对齐与 `codes↔input` 行序断言通过；
  5. protocol 无"待定"条目；
  6. 全程**未覆盖** `data/`、`doc/`、`rqvae/ckpt/`、`model/` 下任何既有产物。
- **前置依赖**：P1.0–P1.6。
- **完成状态**：⬜ 未开始。

---

### P1.8 相对 v1（原草案）的新增与调整

| # | 原 v1 P1 | v2 处置 | 依据 |
|:--:|:--|:--|:--|
| ① | 8 条冻结项 | **扩为 9 条**：新增"**空间清单**（8 个空间，禁 `out`）"与"**Top-K 邻域规则**（K/self/tie/零向量/碰撞）"；原第 2 条"K 值参考 20"拆出并入新条 | `research/ code_audit.md` §4 Q5；本轮补审 |
| ② | 未提距离度量 | **新增**：明确**余弦**而非欧氏；现有 `topk_sims` 是欧氏 | `analyze_emb_orthogonality.py:33-34` |
| ③ | 未提 Jaccard | **新增**：区分 `overlap/K` 与**标准 Jaccard** | `analyze_emb_orthogonality.py:86` |
| ④ | 未提 ties | **新增**：并列规则为**必须实现项**（量化空间大量重复向量） | 本轮补审 |
| ⑤ | 未提冷物品入样 | **升级**：原仅"需定"，现列为 P1.2 的**实现要求** | `build_transition_graph.py:291-294` |
| ⑥ | 无公共工具任务 | **新增 P1.1/P1.2/P1.3/P1.7**：表示导出、邻居采集、指标族、验收门禁 | 本轮补审 |
| ⑦ | 无单层残差 | **新增 P1.5** | `research/ code_audit.md` RQ2:174-182 |
| ⑧ | 无分层诊断 | **新增 P1.6** | `research/ code_audit.md` RQ3:214-218 |
| ⑨ | 无 BSR 条目 | **新增 P1.4**，但**只作诊断口径**，不复活术语 | `doc/exp/创新点2验证方案1.md`（仅设计） |

### P1.9 口径冻结后的增量（2026-10-14，相对 v2）

| # | v2 的开放项 | 冻结结果 | 落点 |
|:--:|:--|:--|:--|
| ⑩ | 主参照 C vs PPMI | **C（去对角）为主**，PPMI 降辅助；**禁止把二者当独立证据** | 协议 §1；本文件 P1.0-1 |
| ⑪ | tie 规则未定 | **余弦（降序）+ canonical index（升序）**；禁用频次作次级键；强制 `tie_share`/`collapse`/敏感性检查 | 协议 §4；本文件 P1.0-4、P1.2 |
| ⑫ | 零向量"是否入样" | **两套 universe**：`U_beh` 剔除、`U_cont` 不剔除；强制报告覆盖度；补充分析必须单列 | 协议 §5；本文件 P1.0-6、P1.1 |
| ⑬ | 类 A / 类 B 未分 | **强制分开**；新增 `Neighbor Recall@K`、标准 `Jaccard@K`、`Overlap@K` | 协议 §6/§7；本文件 P1.3 |
| ⑭ | `overlap/K` 名不副实 | **更名 `Overlap@K`**；旧称一律作废 | 协议 §7-B |
| ⑮ | 无 K 值决议 | **K ∈ {5,10,20,50,100}，主 K=20** | 协议 §8.1 |
| ⑯ | 无术语红线 | **AUC ≠ BSR ≠ 邻域保持**；诊断指标非方法创新 | 协议 §0/§6/§11 |

**P1 不包含**：任何模型改动、任何训练、任何对既有结果的重算。

---

## Sanity Check — 工具链自洽性（在任何 E 之前）

- **研究问题**：现有工具在同一份输入上是否给出与历史记录一致的结果，且无已知缺陷？
- **输入**：Sport 的 codes + ckpt + 三个参照空间。
- **操作**：
  1. 用 `sid_per_layer_diagnostic.py --datasets Sport` 复算，比对 `KNOWN_AUC`（`:58-62`，TA-SID 0.962 / Content 0.900）
     的 `xq[full]` sanity 行（脚本内建 `±0.005` 判定）。
  2. **新增**：对全量 Sport item 比对 `rebuild_xq_partial(codes,cb,3)` 与 `model(x, use_sk=False)` 的 `x_q`
     （验证"保存的 codes 即模型所选码"，见审计清单 §5-1）。
  3. 复核 `item_id_to_row` 在 Sport 上等价于 `ItemID-1`（Sport 为 1..N），但**断言不可外推到 Beauty**。
- **输出**：一份 sanity 摘要（记录偏差量级；**不写入既有无产物**）。
- **依赖**：P1 的采样口径冻结。
- **风险**：sanity 用错 codes 别名（**孤儿文件** `Toys_ta_sid_768d_sid.npy` 教训；Sport 无此问题）。
- **停止标准**：`xq[full]` 与历史记录偏差 <0.005；步骤 2 的最大逐行偏差可解释（如仅在碰撞条目）。
  不满足 → 暂停 E1，先查实现。

---

## E1 — 原始行为关系 vs SVD 行为向量

- **研究问题**：从原始共现关系（C / PPMI）到 128d SVD 行为向量，Top-K 邻居、Spearman、
  degmatch edge AUC 发生了多少变化？哪些结构被保留、哪些被抹平？
- **输入**：`data/Sport/train.parquet`；由 `build_transition_graph.build_cooccurrence_matrix`
  + `compute_ppmi`（或 `behavior_variants.py --variant ppmi`，确定性 v0）现场重建的 C / PPMI；
  已在盘的 `behavior_emb.parquet`。
- **操作**：
  1. 重建 C 与 PPMI（**只读，落盘到独立临时目录，不覆盖现有产物**）。
  2. 在三个空间（C 的加权邻接 / PPMI / SVD-embed）上，用**同一组**正负样本（P1 冻结口径）算 degmatch edge AUC。
  3. 算 SVD 与 PPMI 的 Top-K 邻居 overlap（K 按 P1）与距离秩 Spearman。
  4. 报告 PPMI 的 freq-leak（可选复用 `tools/behavior_space_diagnostic.py:132`）作为流行度偏差诊断。
- **输出**：一张 `空间 × 指标` 表 + 简短解读（哪些结构在 SVD 后仍可分）。
- **依赖**：P1 冻结；Sanity 通过；现场重建的 C/PPMI（**因原始矩阵未落盘**）。
- **风险**：SVD 的随机 v0 不可复现（`build_transition_graph.py:278`）；对角含/去不一致（审计风险 3）。
  → 统一走 `behavior_variants.svd_embed`（显式 v0）与"去对角"定义。
- **停止标准**：AUC/overlap 在固定 seed 下重跑两次逐位一致 → 否则先修 v0。
  若 SVD 的 degmatch AUC 与 PPMI 差异 <0.01，记"该环节基本无损"，不再深挖。

---

## E2 — SVD 行为向量 vs 融合输入及 Encoder latent

- **研究问题**：行为信号进入融合(896d)再经 encoder 变到 z(32d) 后，行为可分辨结构损失多少？
  融合空间的行为可分辨性有多少来自 content 部分？
- **输入**：`behavior_emb.parquet`(128d)、`fused_item_emb_768d.parquet`(896d，**必须用 768d 版**)、
  `encode_z` 产出的 z(32d)。
- **操作**：在 (a) 行为空间、(b) 融合空间整体、(c) 融合空间的 **content 子块**（前 768 维）、
  (d) encoder z 四个空间上，用同一正负样本算 degmatch edge AUC + Top-K overlap(相对行为空间)。
- **输出**：四空间 AUC 表 + "行为可分辨性沿 行为→融合→z 的衰减曲线"。
- **依赖**：E1 的参照定义；`encode_z`（`sid_per_layer_diagnostic.py:87`，返回**已归一化** z）。
- **风险**：⚠ **维度混淆**——`fused_item_emb.parquet` 在 Beauty 是 896d，但 Sport/Toys 是 4224d
  （`tools/stage_probe.py:24-30` 已记录）；本轮固定用 `fused_item_emb_768d.parquet`（三数据集均 896d）。
  ⚠ 融合空间 cos 不能当纯行为相似度（P1-8）。
- **停止标准**：内容子块的行为 AUC 若已接近融合整体，则记"融合空间的行为增益主要来自 content"，
  停止对融合空间的行为解读。

---

## E3 — Encoder latent vs RVQ 各层累计量化表示

- **研究问题**：转移/共现结构在 RQ-VAE 的哪一层被保留或丢失？是"量化环节"丢的，还是 encoder 前就弱？
- **输入**：z、`rebuild_xq_partial(codes, cb, l)`（l=1,2,3）、层码本。
- **操作**：逐层算 degmatch edge AUC：`input(896d) / z / xq[L0] / xq[L0+L1] / xq[full]`；
  另算 z→xq[full] 的逐层**残差**（`res_l = z − 累计_l`，**需新增一行**）的 AUC 曲线。
- **输出**：Sport 的逐层 AUC 表 + z↔xq[full] 的 Δ 与解读（量化是否丢结构）。
- **依赖**：Sanity 步骤 2（codes↔模型 x_q 一致性）；`load_codebook_weights`。
- **风险**：由 codes 重建的累计向量 ≠ 模型单次前向 x_q（碰撞条目）；**residual 无现成实现**需自写；
  与 `sid_capacity_diagnostic.py` 的 z-gap 口径需一致（随机对 5000 vs 3000 的历史漂移教训，见
  `doc/e41_k0_sweep_results.md` 协议注记）。
- **停止标准**：`z→xq[full]` 若 Δ > −0.02（近无损），记"量化环节基本保有结构"，不再追容量；
  若 Δ < −0.05，进入"量化环节丢失"分支（**仅描述，不设计修复**）。

---

## E4 — 完整量化向量 vs 最终 SID 离散代码结构

- **研究问题**：从 xq[full]（连续 32d）到离散 SID 码（3 语义 token + 1 collision 序号），
  行为/共现结构以 Hamming/前缀口径还剩多少？离散化是否引入额外损失？
- **输入**：`codes(N,4)`、xq[full]、E 图。
- **操作**：
  1. 离散口径 edge AUC：`hamming(codes[:, :3])`（`behavior_structure_probe.py:248`）与 `L0-match`
     （第一列相等率，`:273-275`），与同一样本集的 xq[full] 连续 AUC 并列。
  2. 前缀可分性：按前 l 列分组的"组内正样本命中率 / 平均组大小"，l=1,2,3（离散版
     `per_level_alignment_diagnostic.prefix_groups`）。
  3. **碰撞列校验**：`collision = (codes[:,3]!=0)` 比例及其与 AUC 的关系（Sport 实测 0%）。
- **输出**：Sport 的 `连续 vs 离散` 对照表 + 前缀曲线。
- **依赖**：E3 完成；P1-6（prefix similarity 定义）。
- **风险**：⚠ `hamming` 用前 3 列（**正确排除 collision**，`n_tokens=3`），新增脚本需保持一致；
  ⚠ collision 列**不是**学习码字，仅碰撞消解序号（`generate_sid_codes_param.py:122-128`），
  不得当第 4 个语义层；⚠ K0≠256 的 codes 不纳入本轮（token 空间非互斥，见审计风险 1）。
- **停止标准**：若离散 AUC 与 xq[full] 连续 AUC 的**秩序**在 6 个变体上一致（ρ=1），
  记"离散化未重排结构"；否则逐对列出倒置对。

---

## P1 之后的任务：保留项、新增项与依赖顺序

> 原计划的 Sanity / E1–E4 **全部保留**（本节逐条标明其在 v2 中的变化）；下列 R-A–R-F 来自
> `research/ code_audit.md`（RQ1–RQ6）与 `doc/behavior_structure_study_plan.md`（G1/G3/G4），
> **原 P0 计划未纳入，本次补齐但不执行**。

### 保留项（原样）

| 节点 | 内容 | v2 变化 |
|:--|:--|:--|
| **Sanity** | 工具链自洽性；`xq[full]` 对 `KNOWN_AUC` 的 ±0.005 判定；`codes ↔ 模型 x_q` 一致性 | **不变**；新增 1 条：走 P1.2 邻居采集器的确定性自检 |
| **E1** | 原始行为关系（C/PPMI） vs SVD 行为向量 | **不变**；须先按 P1.0-1 冻结"含/去对角 + 含/不含 target"；SVD 走 `behavior_variants.svd_embed:149`（**显式 v0**），禁用 `build_transition_graph.py:278` 的默认随机 v0 |
| **E2** | SVD 行为向量 vs 融合输入及 Encoder latent | **不变**；须用 `fused_item_emb_768d.parquet`（三数据集统一 896d） |
| **E3** | Encoder latent vs RVQ 各层累计量化表示 | **增强**：在累计口径之外，接入 P1.5 的**单层增量口径** |
| **E4** | 完整量化向量 vs 最终 SID 离散代码结构 | **不变**；连续/离散两个 prefix 口径**并列**，不混算 |

### 新增项（来自两轮审计的缺口）

| 节点 | 研究问题 | 现有代码基础 | 需新增 | 前置 | 成本 |
|:--|:--|:--|:--|:--|:--|
| **E5 ★** | **量化前后邻域保持与行为结构保持**：`z→q1/q12/q123` 与 `behavior→z/q1/q12/q123` 的 Neighbor Recall@K / 标准 Jaccard@K 曲线 | **无**（全仓零实现） | 复用 P1.1/P1.2/P1.3 即可，**无需新方法** | P1 全绿 | CPU |
| **R-A** | 5 类表示（content / behavior / fusion / z / xq / SID）**逐 item 对齐导出**探针 | 各部分均有，缺统一导出 | P1.1 即此项 | P1.1 | CPU |
| **R-B** | 单层 residual / `x_res` / 单层 loss | 累计层已有；`sf_rq.rederive_codes_sequential:92` 可扩 | P1.5 即此项 | P1.5 | CPU |
| **R-C** | 结构类型（边强/弱、head/mid/tail、近邻对）× 阶段的**分层**结构保持 | 三类结构的计算全有，缺连接脚本 | P1.6 即此项 | P1.2/P1.3 | CPU |
| **R-D**（G4） | T5 **token embedding 层**是否保持行为结构（链条现止于 `xq`） | T5 ckpt + `model/dataset.py:71 item2code` 可用 | 新探针（可行性待试） | E4 | 低–中 |
| **R-E**（G1） | 局部 vs 全局的**单变量隔离**（补 2×2 右列 A1=Basket-Raw / A2=Basket-PPMI） | `stage_probe.VARIANTS:78-112`、`FUSED_896D:69-76` 已就位 | **需 GPU 重训** | 用户授权 | 2 臂 ≈ 20–22 h（可只做 1 臂 ≈ 10–11 h） |
| **R-F**（G3） | 三数据集 × 6 变体 AUC **统一成表**（Beauty 六行散在 Tier-1 文件） | 数字全在盘 | 仅文档 | — | 0 |
| **R-G** | `tools/analyze_emb_orthogonality.py` 的**索引错配修正**（`:160-171` 把行号当 ItemID）与 4096d→768d 口径统一 | 有 bug | 小改 | P1.0 | 0 |

### 依赖顺序

```
P0（✅ 已完成）
 └─ P1（本轮制定；P1.0 protocol ⊣> P1.1 ─┬─ P1.2 ─┬─ P1.3 ─ P1.4
                                        │        └─ P1.6
                                        ├─ P1.5
                                        └─────────> P1.7 门禁
      ├────────────────────────────────────────────────> Sanity Check
      │                                                   ├─ E1（C/PPMI vs SVD）
      │                                                   │    └─ E2（SVD vs 融合 vs z）
      │                                                   │         └─ E3（z vs 逐层 xq*；+ 单层增量）
      │                                                   │              └─ E4（xq[full] vs 离散 SID）
      │                                                   │                   └─ R-D（T5 embedding 层）
      │                                                   └─ E5 ★（邻域保持 / 行为结构保持全链）
      │                                                        └─ R-C（结构类型 × 阶段分层）
      └─ R-F（成表，0 成本，可并行） · R-G（正交性脚本修正，可并行）
                                                          R-A ≡ P1.1 · R-B ≡ P1.5
                                                          R-E（GPU，需独立授权，不属本轮）
```

### 执行纪律（贯穿全部节点）

- 每步结束**先复核"与历史记录/§5.3 sanity"**再进入下一步；任一步 sanity 失败则回退查实现。
- Beauty/Toys 在**首轮不跑**；K0 扫描**不重训、不重跑**，仅沿用审计清单 §1⑩ 的盘点结论。
- **全程只读**：不覆盖 `data/`、`doc/`、`rqvae/ckpt/`、`model/` 下任何既有产物；新增中间产物
  一律落到独立临时目录。
- **不预设**行为结构一定在哪一级量化中丢失，也**不提前得出**机制结论；
  **不把诊断指标本身宣称为方法创新**。
- 凡引用"容量 ↔ 推荐效果"结论，必须附带 token 空间非互斥的说明（审计清单 §3 风险 1）。
