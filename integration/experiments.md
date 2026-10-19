# CSG4Rec + ETEGRec → TIGER 实验总表

- **版本**：v1.0
- **日期**：2026-10-14
- **配套文档**：[plan.md](plan.md)（移植计划）、[status.md](status.md)（进度）
- **冻结基线**：`5edfca08b22f23a2e965f5e21062adb53e1214b`（Clean TIGER）

> **本文件不含任何推荐效果结果。** E1–E10 与 C1 / C2 / C3 的**工程 / 机制验证**（A 类）已执行并通过并记入 §2；**B / C 类（推荐效果）全部仍为 `PLANNED`**。
> 结果只能来自真实运行（见 §4 结果管理规则）。**不得以"实现完成"推断任何效果。**

> **状态图例**（本文件专用）：`PLANNED` = 未执行 ｜ `A-PASS` = 工程 / 机制验证已通过（**不代表效果**）｜ `RUNNING` ｜ `DONE`（附真实产物）

---

## 1. 实验总表

| Task | 移植内容 | 验证实验 | 类型 | 论文价值 | 状态 |
|------|----------|----------|------|----------|------|
| E1 | T5 hidden-state interface | EXP-E1 Clean TIGER consistency test | A | ❌ 不进论文 | ✅ **A-PASS**（15/15） |
| E2 | SIA adapter | EXP-E2 SIA adapter interface sanity test | A | ❌ 不进论文 | ✅ **A-PASS**（29/29 + 29/29） |
| E3 | SIA code-distribution extraction | EXP-E3 SIA code-distribution diagnostic | A + B | ⚠️ 视结果 | ✅ A 部分 A-PASS（39/39）；B 部分 PLANNED |
| E4 | SIA loss | EXP-E4 TIGER vs TIGER + SIA | B + C 候选 | ⚠️ 视结果 | ✅ **机制 A-PASS**（43/43）；**效果 PLANNED** |
| E5 | PSA decoder interface | EXP-E5/E6 TIGER+SIA vs TIGER+SIA+PSA | B + C 候选 | ⚠️ 视结果 | ✅ **机制 A-PASS**（54/54）；**效果 PLANNED** |
| E6 | PSA loss | EXP-E5/E6 TIGER+SIA vs TIGER+SIA+PSA | B + C 候选 | ⚠️ 视结果 | ✅ **机制 A-PASS**（54/54）；**效果 PLANNED** |
| E7 | Tokenizer / recommender freeze interface | **EXP-E7** 阶段契约工程验证 + EXP-E7-E9 Fixed training vs Alternating | A + (B + C 候选) | ❌ 工程项（EXP-E7 不进论文）｜⚠️ 视结果（EXP-E7-E9） | ✅ **EXP-E7 A-PASS**（66/66）；**EXP-E7-E9 PLANNED** |
| E8 | SID refresh | **EXP-E8** SID refresh 验收 + EXP-E7-E9 Fixed training vs Alternating | A + (B + C 候选) | ❌ 工程项（EXP-E8 不进论文）｜⚠️ 视结果（EXP-E7-E9） | ✅ **EXP-E8 A-PASS**（55/55）；**EXP-E7-E9 PLANNED** |
| E9 | Alternating cycle | **EXP-E9** 交替调度工程验证 + EXP-E7-E9 Fixed training vs Alternating | A + (B + C 候选) | ❌ 工程项（EXP-E9 不进论文）｜⚠️ 视结果（EXP-E7-E9） | ✅ **EXP-E9 A-PASS**（90/90）；**EXP-E7-E9 PLANNED** |
| E10 | ETEGRec integration smoke test | EXP-E10 TIGER + ETE 集成 smoke | A + C 候选 | ➖ 非论文项（门禁） | ✅ **A-PASS**（64/64，CPU + CUDA） |
| C1 | CF embedding interface | EXP-C1 CF embedding sanity check | A | ❌ 不进论文 | ✅ **A-PASS**（132/132，CPU + CUDA） |
| C2 | Collaborative relation (kNN) | EXP-C2 Collaborative kNN diagnostic | A + B | ❌ 工程项（B 部分见 §11 限制） | ✅ **A-PASS**（127/127）；**关系无效果结论** |
| C3 | Q-space collaborative constraint（construction half） | **EXP-C3** Q-space collaborative structure constraint | A | ❌ 不进论文 | ✅ **A-PASS**（99/99，CPU + CUDA）；**约束构造无效果结论** |
| C4 | CSG loss（objective half） | EXP-C4 Q-space constraint training sanity test | A | ❌ 不进论文 | ✅ **A-PASS**（87/87） |
| C5 | CSG trainer integration | EXP-C5/C6 TIGER vs TIGER + CSG | A + B + C | ⚠️ 视结果 | ✅ **机制 A-PASS**（99/99）；**效果 PLANNED** |
| C6 | CSG smoke test | EXP-C5/C6 TIGER vs TIGER + CSG | A + B + C | ⚠️ 视结果 | ✅ **A-PASS**（63/63）；**效果 PLANNED** |
| I1 | TIGER + CSG（= R13 格 #2） | EXP-I1 | C | ✅ 候选主表 | PLANNED |
| I2 | TIGER + ETE + ALT（= R13 格 #7） | EXP-I2 | C | ✅ 候选主表 | PLANNED |
| I3 | TIGER + ALT（= R13 格 #4） | EXP-I3 | C | ✅ 候选主表 | PLANNED |
| I4 | Full（= R13 格 #8） | EXP-I4 | C | ✅ 候选主表 | PLANNED |

---

## 2. 【工程验证】（A 类）

> 目的：确认接口和原始 TIGER 行为**没有被破坏**。**不过不进论文**（除非被要求提供可复现性附录）。

### EXP-E1 — Clean TIGER consistency test

| 项 | 内容 |
|---|---|
| 对应 Task | E1 |
| 前置 | 无 |
| 验证内容 | ① loss 行为不变；② logits 行为不变；③ `encoder_last_hidden_state` 可获得；④ `decoder_hidden_states[-1]` 可获得 |
| 判定 | ①–② 必须 `torch.equal` **逐位相同**（不接受 allclose）；③–④ 形状为 `(B, L_enc, d_model)` / `(B, L_dec, d_model)` |
| 输出 | 一致性测试脚本 + 逐位比较日志 |
| 状态 | ✅ **A-PASS**（15/15，CPU）— `integration/exp_e1_consistency.py` |

### EXP-E2 — SIA adapter interface sanity test

| 项 | 内容 |
|---|---|
| 对应 Task | E2 |
| 验证内容 | tensor shape、latent dimension（=32）、forward、backward、gradient |
| 判定 | 输出 `(B, 32)`；梯度 `is not None` 且非全零；padding 位置不影响有效样本输出 |
| 状态 | ✅ **A-PASS**（29/29 接口 + 29/29 adapter，CPU）— `integration/exp_e2_encoder_interface.py`、`integration/exp_e2_sia_adapter.py` |

### EXP-E3 — SIA code-distribution diagnostic

| 项 | 内容 |
|---|---|
| 对应 Task | E3 |
| 类型 | **A + B** |
| 验证内容 | ① 可得到 3 个 RQ level 的 code distribution；② target item 可得到对应 distribution；③ distribution shape 正确 `(B, 256)` × 3；④ **不包含 collision token**；⑤ **序列侧路径断言**：进入 RQ quantizer 的 query 就是 E2 adapter 的输出，该 path 上无 `RQVAE.encoder` 被调用；⑥ **目标侧路径断言**：query 是 `RQVAE.encoder(target item embedding)` 的输出（经过 encoder） |
| 判定 | softmax 后每行和为 1；用 SID 第 4 列非零的 item 显式断言 collision 未被纳入；关闭开关时 `get_indices` 逐位不变；⑤⑥ 的检查方式不限（调用路径记录 / 张量来源标记 / hook / 等价运行时检查），但不得改变模型设计 |
| 诊断记录 | Adapter 输出的 L2 norm 分布；Adapter 输出到最近 codeword 的距离分布；`RQVAE.encoder(item_embedding)` 输出到最近 codeword 的对应距离分布；两类统计的量级比较。**⚠️ Diagnostic Only — 不作为 E3 的阻断性 pass/fail 条件**；仅记录，不预设异常、不设阈值、不预设必须同量级、不自动触发 warmup（见 [plan.md](plan.md) E3 诊断） |
| 状态 | ✅ A 部分 **A-PASS**（39/39，CPU + CUDA）— `integration/exp_e3_code_distribution.py`；B 部分 PLANNED |
| R3 修正后新增断言 | code logits **非负**（`+squared L2`）；target 侧用独立 `cdist` 复算三层一致；quantizer 选中的码字 = logits 的 **argmin**（level 0/1） |

### EXP-E4/E5/E6 — SIA / PSA 机制验证（A 类）

> 这组是 E4 / E5 / E6 的**机制与回归验证**（loss 机制本身 + 关闭即等价），**不是** §3 的 EXP-E4 / EXP-E5/E6 效果实验。

| 项 | 内容 |
|---|---|
| 对应 Task | E4、E5、E6 |
| 验证内容 | ① loss 可算、标量、有限；② 梯度正常（非零、有限）且方向正确；③ 两侧路径与张量来源断言；④ `lambda_* = 0` 时总 loss 与全部参数梯度对 Clean TIGER **逐位相同**（`torch.equal`）；⑤ 原始 TIGER 的 loss / logits / `generate()` 逐位回归；⑥ 与独立 float64 参考实现 / 闭式解吻合 |
| 判定 | 全部为 `torch.equal` 或独立参考比对，不使用 allclose 作结论性判定 |
| 状态 | ✅ **A-PASS** — E4 43/43、E5/E6 54/54（均 CPU + CUDA）；脚本 `integration/exp_e4_sia_loss.py`、`integration/exp_e5_psa.py` |
| 备注 | E5/E6 的 54 项含 R3 新增的 4 项：默认温度 0.07、双向 **求和**（非平均）、**cosine 缩放不变性**、受控构造自检 |

### EXP-E7 — 阶段契约工程验证（A 类）

> 这是 E7 的**工程 / 契约验证**（阶段冻结范围、梯度流、loss 生效、码本状态、checkpoint 恢复、回归），**不是** §3 的 EXP-E7-E9 效果实验。

| 项 | 内容 |
|---|---|
| 对应 Task | E7 |
| 脚本 | `integration/exp_e7_stage_freeze.py`（10 节，66 项） |
| 验证内容 | ① 阶段契约与 API；② 逐阶段 `requires_grad`；③ 逐阶段 loss 开关；④ **（参数状态 × loss 状态）一致性**（E7 验收 ⑤）；⑤ 码本初始化 / 封存（`vq.initted` 坑）；⑥ Tokenizer 阶段走**未修改的** `rqvae/trainer.py`；⑦ Recommender 阶段走**未修改的** `model/main.py::train()`；⑧ 阶段切换前后的**优化器参数覆盖**；⑨ checkpoint 阶段记录与 resume 校验；⑩ E7 未启用时 Clean TIGER 逐位不变 |
| 关键判定 | ① 验收 ①/②：Tokenizer 阶段 T5 与 adapter **0 张量移动**、RQ-VAE **23/23 移动**；Recommender 阶段反向。② 冻结参数在 epoch 末**梯度全为 `None`**（0/23），且**前向确实执行**（encoder 8 次 / decoder 4 次）——"frozen ≠ `eval()`"。③ 负对照：裸 `load_state_dict` + train 模式前向**确实会**被惰性 k-means 覆盖码本（证明该坑真实），`load_tokenizer_state_dict` 后同一前向**逐位不变**。④ `lambda_sia/psa = 0` 与 E7 未启用时，TIGER 的 loss / logits / `generate()` 对 `5edfca0` **逐位相同**（`torch.equal`） |
| 状态 | ✅ **A-PASS** — 66/66（**CPU + CUDA**） |
| 顺带查实 | **adapter 不在 `TIGER.parameters()` 内** ⇒ Recommender 阶段优化器必须显式覆盖 adapter；冻结的 `model/main.py::__main__` 入口（`optim.Adam(model.parameters())`）**不会**训练 adapter（E9/E10 需遵守） |

### EXP-E8 — SID refresh 验收（A 类）

> E8 的**工程验收**（版本化命名、覆盖写入、失败状态、ckpt↔SID 追溯、真实生成），**不是** §3 的效果实验。

| 项 | 内容 |
|---|---|
| 对应 Task | E8 |
| 脚本 | `integration/exp_e8_sid_refresh.py`（8 节，55 项）；**含真实生成**（Toys 11924 item / 4096-d / 真实 ckpt，CPU + CUDA） |
| 验证内容 | ① 覆盖全部 item；② 保留 `use_sk` + collision handling（零重复行）；③ token ∈ [0,256)；④ 与上一轮差异率；⑤ ckpt↔SID 可回溯。另含：版本化命名与自动发现、**覆盖写入拒绝**、**失败态零残留**、**publish 原子性**、校验规则的阴性对照 |
| 关键判定 | ① 同设备重跑 **差异率 = 0**（逐位确定性）；② CUDA 上重新生成与 clean_tiger 基线 SID **逐位相同**（证明复用 `generate_code.py` 无偏差、无新编码机制）；③ **CPU vs CUDA 差异 3.25%**，只落在最后一层/collision 列 ⇒ **manifest 记录 device，diff rate 须对照该噪声底**；④ `generate_code.py` **不能在 `python -I` 下运行**（`from datasets import EmbDataset` 会命中 HuggingFace 包），须纯解释器 + `cwd=rqvae/` |
| 状态 | ✅ **A-PASS** — 55/55（**CPU + CUDA**） |
| 实测成本 | 单次全量 SID 生成 ≈ 18 s（CPU 与 CUDA 相近） |

### EXP-E9 — Alternating cycle 工程验证（A 类）

> E9 的**调度工程验收**（交替顺序、SID 交接、失败中止、断点续跑、种子控制），**不是** §3 的效果实验。

| 项 | 内容 |
|---|---|
| 对应 Task | E9 |
| 脚本 | `integration/exp_e9_alternating.py`；4 节（纯调度 → 注入 fake phase 的分支/失败/恢复 → **真实链路** 2 个完整 cycle → 回归），覆盖 E9 规格的 12 项 |
| 验证内容 | ① ≥2 个完整 cycle 端到端；② `Tokenizer → Refresh → Recommender` 顺序；③ 下一 Recommender 消费的是**新** SID；④ refresh 失败在 Recommender **之前**中止；⑤ 从 `refresh_done` 恢复**不重复 refresh**；⑥ `recommender_done` 进入下一 cycle；⑦⑧⑨ 阶段冻结与更新范围；⑩ 评估只挂在 Recommender 之后；⑪ 种子可控且按 phase 派生；⑫ E1–E8 仍全绿 |
| 关键判定 | 相位用 **0-based** `epoch_idx % cycle == 0`（与官方 `trainer.py:513` 一致；1-based 会整体反相，脚本内有对照断言）；Tokenizer 阶段 RQ-VAE **15/15** 张量动、T5/双 adapter **0 张量**动；Recommender 阶段 RQ-VAE **0/15** 动且冻结参数**零梯度**（前向仍执行，是 PSA 靶侧）、T5 26/26 + adapter 4 张量动；**同 seed 重跑 SID 逐位一致**；`verify_sid_consistency` 逐 item 校验 dataset↔SIA↔PSA 三方路由一致（用别的 SID 建的 dataset/mount 立刻失败） |
| 状态 | ✅ **A-PASS** — 90/90（**CPU + CUDA**，含 2 个真实 cycle） |
| 备注 | 全程**从不 `publish`**（canonical SID 是冻结基线产物），交接靠每 cycle 的 `--code_path` 指向不可变版本文件 |

### EXP-C1 — CF embedding sanity check

| 项 | 内容 |
|---|---|
| 对应 Task | C1 |
| 脚本 | `integration/exp_c1_cf_embedding.py`；8 节 132 项（合成 fixture → 真实数据 → **SASRec 抽取契约复算** → 缺失/OOV → ItemCF 备用源 → torch 交接 → 边界 → 回归） |
| 验证内容 | ① item 数量；② embedding dimension；③ **无错误 item mapping**；④ 来源可追溯；另加：`ItemID → 行号` 双向映射、缺失 item 策略、save→reload 逐位、dtype/device |
| 判定 | 行数 == `item_emb.parquet` 行数；`ItemID → 行号` 映射逐条校验（**禁止 `a-1` 式推断**）；**padding 行显式丢弃并断言全零** |
| 实测（正式源 = SASRec item embedding） | `(N+1, 50)` 的 `item_emb.weight`（Toys 11925 / Sport 18358 / Beauty 12102），上行全零；规范表 = `W[canonical ItemID]`，**不是 `W[1:]`**（Beauty 上 `W[1:]` 会错配 **4158** 行，脚本有负对照） |
| 实测（备用源 = ItemCF） | `(n_items, 128)`，Toys/Sport/Beauty 冷启动全零行 **24 / 28 / 33**；仍可加载、`is_default_source=False` |
| 状态 | ✅ **A-PASS** — 132/132（**CPU + CUDA**），且 E1–E10 回归在同一脚本内复跑通过 |

### EXP-C2 — Collaborative kNN diagnostic

| 项 | 内容 |
|---|---|
| 对应 Task | C2 |
| 类型 | **A + B** |
| 脚本 | `integration/exp_c2_collab_relation.py`（8 节 127 项）；**0 个既有文件被修改** |
| 验证内容 | ① cosine similarity；② kNN relation；③ relation 数量；④ relation 是否合理 |
| 判定 | 抽样手算校验 cosine；边数符合 `n_items × k` 量级并记录；定性抽检高相似 item 是否真的共现/同类并留档；无自环 |
| 状态 | ✅ **A-PASS** — **127/127**（exit 0；含 E1–E10 回归 64/64） |
| 关系定义 | cosine（`dot/(‖a‖‖b‖)`，float32）+ top-k，**k = 20**（继承 CSG4Rec `--knn_k` 默认值，**未调参**），`k_eff = min(k, n-1)`；**有向、未对称化**；**对角线显式 mask**（不用参考实现的 `indices[:, 1:]`）；零范数行双向排除并标 `-1`/`NaN`；同分按行号升序（stable argsort，完全确定） |
| 实测（正式源 SASRec 50d） | Toys 238480 边 / Sport 367140 / Beauty 242020，**均 = n_items × 20，0 自环**；互反边占比 0.652 / 0.612 / 0.694；平均 cos rank0→rank19 0.670→0.506（Sport 0.651→0.515，Beauty 0.653→0.496） |
| 正确性证据 | 合成 fixture 与**手算**逐位一致（含并列 tie-break、零范数孤立）；真实数据与**独立 float64 暴力复算** top-k 集合+顺序全同（`max\|Δcos\| ≤ 1.9e-7`）；**最优性**：第 k 名 cos 高于所有非邻居；与 **sklearn `NearestNeighbors(metric='cosine')`** 交叉复算一致；存/取**逐位一致**；篡改 pair 文件 / 交换规范 id / k 不符 / 文件缺失 **全部拒绝加载** |
| 行空间护栏 | artifact 自带 `item_ids` + 摘要，`load` 不匹配即拒绝；**Beauty 上 `row+1` 会让 136490/242020（56%）的边指向错商品**，Toys/Sport 看不出差别 |
| 参考实现缺陷（实测） | CSG4Rec 的 `indices[:, 1:]` 位置式丢弃**在零范数行上会留下自环**：ItemCF 备用源 **6/24（Toys）、11/33（Beauty）**；官方 SASRec 源无零范数行 ⇒ 主线 0/0/0 看不出差别，**极易不知不觉继承** |
| ⚠️ B 部分限制 | 共现富集（Toys 23.7× / Sport 27.3× / Beauty 22.4×）是在 valid+test 上测的，而本项目 valid/test 是**同批用户序列的前缀**（与训练语料高度重叠）⇒ **不构成效果证据**。它只证明两件事：邻居是行为相关商品；**该探针能抓住行→ItemID 映射错误**（打乱映射后富集塌回 ≈1.0×） |
| 未变 | **无任何推荐效果结论**；C3–C6 与全部 B / C 类效果实验仍为 `PLANNED` |

### EXP-C3 — Q-space collaborative structure constraint（construction half）

| 项 | 内容 |
|---|---|
| 对应 Task | C3 |
| 类型 | **A** |
| 脚本 | `integration/exp_c3_qspace_constraint.py`（9 节 99 项）；**0 个既有文件被修改** |
| 验证内容 | ① 约束读的确实是**量化空间**张量（运行时来源断言）；② 关系对含 batch 外 item 时行为明确；③ `lambda_csg = 0` 时无额外计算与读取；④ 量化算法本体未被改动 |
| 判定 | 运行时断言张量来源（`x_q == Σ_l codebook_l[indices_l]`，且 ≠ encoder latent）；batch 外邻居**丢弃并计数**；`lambda_csg = 0` 时 `enabled=False` 且 `forward()` 抛异常；`git status` 零改动 |
| Q-space 定义 | **`x_q`** —— `RQVAE.forward(return_xq=True)` 返回的量化表示（各 RQ 层码本查表之和，带 straight-through）。**排除 E3 code logits**：`rqvae/models/vq.py:107` 的 logits 由 `latent`（量化**前**的 encoder 输出）算出 ⇒ 约束它们就是约束 encoder 向量，即 plan.md 禁止的 latent alignment 换了个名字 |
| 产出 | `ConstraintBatch`：**每 batch 的约束中间量**（in-batch 位置对 + 两侧 gather 的 `x_q`/code indices + 关系 cosine 作权重 + 丢弃计数）。**不含标量 loss**——那是 C4 |
| 参考实现 | `CSGT/models/vq.py:68-106,195,198-200`（`diversity_loss`）：`sim = x_q @ embᵀ` → `F.cross_entropy(sim, y_true)`，`y_true` 是**码字 id**；逐层算、`rq.py:64` 平均、`--beta` 加权 |
| ⚠️ 参考仓库缺陷 | **`CSGT/models/rqvae.py` 在快照里不存在**（无归档 / 无 `.pyc` / 无第二副本），而那正是 `global_knn` 等 6 个开关的**唯一读取点** ⇒ **CSG4Rec 的 kNN loss 确切形式不可恢复**，本次**不复现**。详见 [plan.md §15.2](plan.md) |
| 适配（**非**原始机制） | 监督由 code-to-code 改为 **item-to-item**（用 C2 的关系）；out-of-batch **丢弃并计数**（依据本仓库 `rqvae/trainer.py::_get_transition_loss:213-219` 的既有先例；参考实现的做法在缺失文件里，**未知**）；开关用 `lambda_csg`（默认 0），**不复用** `--use_knn` / `trans_graph_path` 语义 |
| 实测（三数据集，256-item batch） | in-batch 对数 Toys 738 / Sport 118 / Beauty 333；哨兵 **0/0/0**；`slots == pairs + sentinel + out_of_batch` 三数据集恒成立；重建偏差 ≤ 5.96e-08，latent 间距 0.131~0.229 |
| ⚠️ 度量口径 | 上述"对数"是 **batch 几何**（256 行若邻居均匀随机，期望仅 0.43 对/item；Sport 实测 0.46 正落其上，Toys 2.88 是表头行互锁），**不是效果指标** |
| ⚠️ tokenizer 口径 | 仓库中**只有 Toys** 有训在**规范 4096d content 表**上的 RQ-VAE；Sport / Beauty 只有融合表上的 `*_TA_SID_4096d`。故三数据集那轮统一用**同架构 fixture tokenizer**（冻结架构 + 用模型自己的 `init_emb` 在该数据集真实 latent 上 k-means 初始化，**不训练**），Toys **额外**用真实训练过的 tokenizer 复跑：**对数与 pair 轴逐位相同**，code 分布不同 ⇒ A/C 与 B 可分离 |
| 状态 | ✅ **A-PASS** — **99/99**（exit 0；含 E1–E10 回归 64/64，**CPU + CUDA**） |
| 未变 | **无任何推荐效果结论**；C4–C6 与全部 B / C 类效果实验仍为 `PLANNED` |

### EXP-C4 — Q-space constraint training sanity test（objective half）

| 项 | 内容 |
|---|---|
| 对应 Task | C4 |
| 验证内容 | ① loss 可计算；② gradient 正常；③ RQ-VAE 可以正常训练 |
| 判定 | 标量、无 NaN；梯度非零；码本使用率无异常塌缩；`lambda_csg = 0` 时总 loss 逐位等价于 Clean TIGER |
| 状态 | PLANNED（C3 已交付其输入：`ConstraintBatch`） |

### EXP-E10 — ETEGRec integration smoke test

| 项 | 内容 |
|---|---|
| 对应 Task | E10 |
| 类型 | **A + C 候选** |
| 脚本 | `integration/exp_e10_smoke.py`（5 节 64 项）；**0 个既有文件被修改** |
| 验证内容 | 全链跑通 + 每个新增模块的实际执行证据（23 个入口的调用计数探针，**每个探针带阴性对照**） |
| 判定 | exit 0；各模块调用次数 > 0；门禁 G0（关闭即等价）与 G1（历史机制零激活）通过 |
| 关键判定 | 全链 2 个完整 cycle（`E10Syn` fixture，1024 item）；G0 **非同源**逐位对比 `5edfca0`（T5 单 epoch loss / 89 个梯度 / 92 个 state 张量、`get_indices`、`RQVAE.forward` 4-tuple、`generate()` token、eval 指标全部逐位同）；G1 tripwire **0 次**触发、链上 3 个 `Trainer` 的历史开关全为 0 且 `trans_dict/align_table/align_proj` 全 None；幂等 / resume / refresh 失败中止全绿 |
| 状态 | ✅ **A-PASS** — 64/64（**CPU + CUDA**），exit 0 |
| 备注 | ⚠️ E10 子进程给 E9 设 `EXP_E9_REGRESSION=skip` ⇒ 此处 E9 报 81/81 而非独立跑的 90/90，**两者相加无重复计数** |

### EXP-C6 — CSG integration smoke test

| 项 | 内容 |
|---|---|
| 对应 Task | C6 |
| 类型 | **A + C 候选** |
| 验证内容 | 全链跑通 + C1→C5 各模块执行证据 |
| 判定 | exit 0；调用次数 > 0；门禁 G0/G1 通过；无 User SID 痕迹 |
| 状态 | PLANNED |

---

## 3. 【机制验证】（B 类）

> 目的：确认新机制**实际产生了预期效果**。**有论文价值才纳入最终论文实验。**
> **设计源 = [plan.md §19](plan.md)（R13，2026-10-18）。** 执行顺序固定：**Stage 0（λ 标定）→ Stage 1（本节）→ Stage 2（§4）**，不得跳阶。
> 口径：**768d content-only**，`D_target = 768`，基线 = `model/ckpt/tiger_content768d_baseline_*_{seed2025,2026,2027}.pth`（9/9 已存在）。

### EXP-L0 — λ 标定（Stage 0，**先于一切效果实验**）

| 项 | 内容 |
|---|---|
| 目的 | 在 [plan.md §19.4](plan.md) 的预注册规则下选出 `lambda_csg` 与 `lambda_sia = lambda_psa` 的工作点 |
| 数据集 / seed | **Toys / 2025**（成本 48.7 min；768d collision 0.000419 为三数据集最干净） |
| 臂 | 5 路：`λc-{1e-4,1e-3,1e-2}`（C=1,E=0）+ `λe-{0.1,1.0}`（C=0,E=1） |
| 主判据 | `ΔNDCG@20 = 臂 − 基线(seed 2025)` |
| 副判据 | tokenizer collision ≤ 2× 基线；三层码本不塌缩（排除用，不用于选优） |
| 选点规则 | 取**不劣化（Δ ≥ −0.001）且不塌缩的最大 λ**；三档全劣化 ⇒ 回落锚点 `1e-4 / 0.1` 并如实记录 |
| 状态 | ⬜ **PLANNED** |
| ⚠️ | λ_sia = λ_psa = **0.1 是 `alt_pipeline.py:187` 的实现默认值，本文件从未记录其来源**——本次按待标定值处理，不当既定值 |

### EXP-B1 — 8 格机制筛选（Stage 1）

| 项 | 内容 |
|---|---|
| 目的 | 用**单数据集单 seed** 判定哪些格值得升格进 C 类；**不做显著性、不进论文主表** |
| 数据集 / seed | **Toys / 2025**（若 NO-GO，须在 Sport 复筛一次再决定，见 plan §19.5） |
| 实跑 | 8 路 = 2×2×2 的 7 个新格（#2–#8）+ **1 个 SIA-only 子臂**（见下） |
| 预算 | T5 **200 ep 上限 + early_stop 10**，8 格严格相同；A=1 用 `n_cycles = 200` |
| 记录 | Recall@5/10/20、NDCG@5/10/20；collision；码本 distinct；**每 cycle SID diff rate**；tokenizer optimizer steps |
| 门禁 | G-B1 链路 exit 0 / G-B2 不塌缩 / G-B3 无 `ΔN@20 < −0.002` / **G-B4 A=1 的 SID diff > 0**（否则判 A 惰性） |
| 状态 | ⬜ **PLANNED** |

**8 格与旧实验 ID 的映射**（旧 ID 不废弃，作为别名保留）：

| # | C | E | A | 名称 | 旧 ID | 承担的原实验 |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | TIGER（基线，已有） | — | 全部对照的父组 |
| 2 | 1 | 0 | 0 | +CSG | **I1** | **EXP-C5/C6**（TIGER vs TIGER+CSG） |
| 3 | 0 | 1 | 0 | +ETE | I2 的固定调度版 | **EXP-E4**（+SIA 子臂）/ **EXP-E5/E6**（SIA vs SIA+PSA） |
| 4 | 0 | 0 | 1 | +ALT | **I3** | **EXP-E7-E9**（Fixed vs Alternating） |
| 5 | 1 | 1 | 0 | CSG+ETE | — | 新增交互格 |
| 6 | 1 | 0 | 1 | CSG+ALT | — | 新增交互格（需 P1） |
| 7 | 0 | 1 | 1 | ETE+ALT | **I2** | 旧 I2 的定义 |
| 8 | 1 | 1 | 1 | Full | **I4** | 旧 I4（需 P1） |

> **额外子臂 `#3-sia`（+SIA only，PSA 关）**：2×2×2 把 ETE 当作**一个因子**，无法区分 SIA 与 PSA 各自的贡献。为保住 EXP-E5/E6 的原始问题（"PSA 的增量"），Stage 1 追加 1 路 `λ_psa = 0`、`λ_sia > 0`，与 #3 只差 PSA 一项。**这是对 D4（1×1 全格）的 +1 路扩展，成本约 0.8 h，已标注待用户追认。**

**EXP-E7-E9 的"阶段生效统一验证"归属不变**：仍由本组承担——逐阶段打印（参数状态 × loss 状态）矩阵；Tokenizer 阶段 `L_SIA = L_PSA = 0`、`L_RQ-VAE > 0`、T5 `requires_grad = False`；Recommender 阶段相反。E9 的 A-PASS 已机器验证该矩阵，效果实验只需**如实记录**每格的阶段日志。

---

## 4. 【正式实验】（C 类）

> 目的：最终论文结果。矩阵结构见 [plan.md §19.2](plan.md)，统计口径见 [§19.6](plan.md)。

### Stage 2 实验矩阵（2×2×2 全因子）

| # | C | E | A | 名称 | 别名 | 状态 |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | **TIGER**（768d content-only 基线） | — | ✅ 产物已有（9/9），**不重跑** |
| 2 | 1 | 0 | 0 | **+CSG** | EXP-I1 | PLANNED |
| 3 | 0 | 1 | 0 | **+ETE** | — | PLANNED（需 P2） |
| 4 | 0 | 0 | 1 | **+ALT** | EXP-I3 | PLANNED |
| 5 | 1 | 1 | 0 | **CSG+ETE** | — | PLANNED（需 P1+P2） |
| 6 | 1 | 0 | 1 | **CSG+ALT** | — | PLANNED（需 P1） |
| 7 | 0 | 1 | 1 | **ETE+ALT** | EXP-I2 | PLANNED（需 P1） |
| 8 | 1 | 1 | 1 | **Full** | EXP-I4 | PLANNED（需 P1） |

**规模**：通过 G-B 的格数 × 3 数据集 × 3 seed。最坏 63 路（~68 h），典型 45 路（~48 h）。

**统计**：主指标 NDCG@20，配对显著性 `tools/paired_significance_ablation.py`，std `ddof=1`；交互项用 2×2×2 **边际均值**算，不用单格相减。

### 收敛规则

- 若某控制组相对其父组**无正向增益**，则从主表降级为附录讨论，并如实报告。
- 若 `C×E` / `C×A` / `E×A` 存在**显著交互**，主结论必须写成"仅在 X 条件下有效"，**不得**写成"普遍有效"。
- **ALT 的预先判定**：A=1 且 C=0 时 tokenizer 目标与起点均与 A=0 相同 ⇒ **预期 #4 ≈ #1**。若 G-B4 的 SID diff ≈ 0，ALT 直接降级，**不得为保住 I3/I4 而改判据**。
- **后续根据实际结果决定最终论文是否保留所有控制组。**

---

## 5. 实验结果管理规则

1. **实际实验结果必须来自真实运行。**
2. **不允许 Claude 编造结果。** 任何未经真实运行的数字一律不得写入本文件或论文；占位一律使用 `PLANNED` / `N/A`，不得使用貌似合理的估计值。
3. **每个实验必须记录**：
   - **Task ID**
   - **config**（完整超参与开关状态）
   - **dataset**
   - **seed**
   - **metrics**（按指标逐项，注明 `@K`）
   - **checkpoint**（路径 + 生成时间）
   - **commit**（代码版本 hash）
   - **conclusion**（结论；无增益时如实写"无增益"）
4. **工程验证实验不一定进入论文。**（A 类默认不进）
5. **机制验证结果如果有论文价值，再纳入最终论文实验。**（B 类→C 类需显式升格决策）
6. **最终论文实验计划不能在代码尚未验证前完全锁死。**
7. **根据阶段性结果逐步收敛最终论文实验设计。**

### 5.1 记录模板

```markdown
#### <EXP-ID> — <名称>

- Task ID        :
- 类型            : A / B / C
- config         : <完整超参 + 各开关状态（lambda_* / 冻结策略 / 迭代轮数）>
- dataset        : <Toys / Sport / Beauty / ...>
- seed           : <RQ-VAE seed / T5 seed 分别记录>
- metrics        : Recall@5=<>, Recall@10=<>, Recall@20=<>
                   NDCG@5=<>,   NDCG@10=<>,   NDCG@20=<>
- checkpoint     : <路径> (<生成时间>)
- commit         : <hash>
- 对照           : <baseline 与其 commit>
- conclusion     : <结论，含"无增益"这类如实表述>
- 备注           : <异常、口径差异、未决问题>
```

### 5.2 已知必须遵守的取数纪律（来自本项目历史教训）

| 纪律 | 原因 |
|---|---|
| NDCG 取值必须**按 Metric 表头映射**，不得用 `$NF` | `eval_trained_model.py` 的 NDCG 行列数随 `--topk_list` 变化（3 列 vs 5 列），`$NF` 在 5 列下取到的是 @100 而非 @20 |
| 行号必须走显式映射表 | 数据集 `ItemID` 可能是排列而非"行号+1"，`a-1` 会静默取错商品 |
| 判定用逐位相同，不用 md5 / allclose | 跨环境 parquet md5 会漂移；结论性比较须为同环境内逐位一致 |
| 完成时刻以产物 `mtime` 为准，不以驱动脚本的批量打印行为准 | 驱动日志的 per-seed 行是批量打印，不代表真完成时刻 |
| 随机性须可复现 | `DataLoader` 的 `n_workers > 1` 会因 OS 调度导致跨运行不可复现；std 统一 `ddof=1` |

---

## 6. 当前状态摘要

```
实验总数 : 20 (E: 10, C: 6, I: 4) + R13 的 2 个阶段实验 (EXP-L0 / EXP-B1)

工程/机制验证 (A)    : 16 / 16 通过 (E1-E10 + C1-C6，见 §2)
推荐效果实验 (B)     : 0 执行       (EXP-L0 / EXP-B1 均 PLANNED)
推荐效果实验 (C)     : 0 执行       (8 格中 1 格=已有基线，7 格 PLANNED)
训练运行             : 0
失败                 : 0
```

> **⚠️ 区分**："A 类全通过" = 接口未被破坏 + loss 机制正确 + 阶段契约成立 + SID 可正确重生成 + 交替调度可正确交接 + CF 向量可正确加载且无错映射 + 协同关系可正确复现并落盘 + 协同关系可正确投影到量化空间 + CSG 全链跑通 + 关闭即逐位等价。**它不等于**"SIA / PSA / 交替训练 / CSG 有推荐增益"，后者**完全未验证**。
>
> **R13 的口径变更（2026-10-18）**：效果实验一律走 **768d content-only**（`D_target = 768`），不沿用 §1.3 的 4096d。若要复跑 §2 的 A 类实验，其 fixture 口径不受影响（它们不消费真实 content 维度）。

**论文价值符号约定**（全文一致）：

| 符号 | 含义 |
|---|---|
| ✅ | 进入论文 |
| ❌ | 不进入论文 |
| ➖ | 非论文实验 / 工程门禁 |

**下一个要执行的实验**：E 线 **E1–E10 已全部实现并通过 A 类验证**，可执行 **EXP-E7-E9** 与 B 类效果实验；C 线 **C1、C2、C3 已完成**（A-PASS 132/132、127/127、99/99），下一个是 **C4**（CSG loss——C3 已交付其输入 `ConstraintBatch`）。
> ⚠️ B / C 类效果实验开工前须先解决 [status.md](status.md)「已知问题」第 3 条（Toys 缺训练充分的 4096d RQ-VAE）。

---

## 7. 修订记录 R3 — E1–E6 源码核验修正（2026-10-14）

E1–E6 实现完成后对 ETEGRec 官方源码做了 SIA / PSA Source-of-Truth Audit，据此修正 4 处（详见 [plan.md §9](plan.md)）：

| 编号 | 位置 | 修正 |
|---|---|---|
| R3-1 | `tiger_ext/psa_loss.py` | PSA similarity：裸点积 → **cosine** |
| R3-2 | `tiger_ext/psa_loss.py` | PSA temperature 默认值 1.0 → **0.07** |
| R3-3 | `tiger_ext/psa_loss.py` | PSA 对称项 `0.5*(CE1+CE2)` → **`CE1+CE2`** |
| R3-4 | `rqvae/models/vq.py` | SIA code logits `-raw_d` → **`+raw_d`** |

修正后 E1–E6 全部 A 类验证重跑通过（209 项），原始 TIGER 逐位回归通过。
**R3 只影响机制忠实度，不含任何效果结论。**

---

## 8. 修订记录 R4 — E7 实现（2026-10-14）

执行 E7（Tokenizer / recommender freeze interface），实现 [plan.md](plan.md) §1.4 / E7 表定义的阶段契约（差异说明见 [plan.md §10](plan.md)）。

| 项 | 内容 |
|---|---|
| 产物 | `tiger_ext/stage.py`（`StageFreeze` 控制层）、`integration/exp_e7_stage_freeze.py`（66 项） |
| `model/` / `rqvae/` 改动 | **无**（`git diff --stat` 与 E6 结束时相同） |
| 新增工程实验 | **EXP-E7**（§2 新增条目，A 类，❌ 不进论文） |
| 结果 | **66/66**（CPU + CUDA） |
| 回归 | E1–E6 的 **209 项全部重跑通过**（CPU；E3/E4/E5 另跑 CUDA）+ 原始 TIGER 逐位回归 |
| 闭环 | status.md「已知问题」第 1 条（阶段冻结）、第 2 条（`vq.initted`）**均已关闭** |
| 未变 | EXP-E7-E9 与全部 B / C 类实验**仍为 `PLANNED`**；本项目**仍无任何效果数据** |

---

## 9. 修订记录 R5 — E8 实现（2026-10-14）

执行 E8（SID refresh），按 [plan.md](plan.md) E8 表的验收标准 ①–⑤ 落地。

| 项 | 内容 |
|---|---|
| 产物 | `tiger_ext/sid_refresh.py`（版本化命名 + provenance manifest + 原子提交；只驱动 `rqvae/generate_code.py` 的既有 CLI）、`integration/exp_e8_sid_refresh.py`（55 项） |
| `model/` / `rqvae/` / `clean_tiger/` 改动 | **无**（`git status` 与 E7 结束时逐字相同） |
| 新增工程实验 | **EXP-E8**（§2 新增条目，A 类，❌ 不进论文） |
| 结果 | **55/55**（CPU + CUDA，含真实生成：11924 item / 4096-d / 真实 ckpt） |
| 关键发现 | ① 同设备重跑 SID **逐位确定性**，CUDA 上重新生成与基线 SID **逐位相同**；② **SID 与设备相关（CPU vs CUDA 3.25%，仅最后一层/collision 列）**⇒ diff rate 有噪声底，manifest 记录 device；③ `generate_code.py` **不能**在 `python -I` 下运行 |
| 未决（E9） | E8 只产出新版本 + manifest；**谁把 T5 指向新 SID**（`publish`）由 E9 调度决定 |

---

## 10. 修订记录 R6 — E9 / E10 实现（2026-10-14）

执行 E9（Alternating cycle）与 E10（集成 smoke），E 线实现部分**至此收口**。

| 项 | 内容 |
|---|---|
| 产物 | E9 = `tiger_ext/alternating.py`（纯调度，**不 import torch/numpy**）+ `tiger_ext/alt_pipeline.py`（4 个 phase 体，全部调用冻结代码）+ `integration/exp_e9_alternating.py`（90 项）；E10 = `integration/exp_e10_smoke.py`（64 项，**唯一产物**） |
| `model/` / `rqvae/` / `clean_tiger/` 改动 | **无**（`git status` 与 E8 结束时逐字相同：仍只有 `model/main.py`、`rqvae/models/rq.py`、`rqvae/models/vq.py` 三个 E1–E6 遗留 modify） |
| 新增工程实验 | **EXP-E9**、**EXP-E10**（§2 新增/更新条目，A 类，❌ 不进论文） |
| 结果 | E9 **90/90**（CPU + CUDA，含 2 个真实 cycle）；E10 **64/64**（CPU + CUDA），exit 0 |
| 关键判定（E9） | 0-based `epoch_idx % cycle`（1-based 会整体反相）；SID 交接靠 `--code_path` 指向不可变版本，**从不 `publish`**；`verify_sid_consistency` 逐 item 验 dataset↔SIA↔PSA 三方路由（负对照已实测）；refresh 失败 → `RefreshAborted`，Recommender **绝不运行** |
| 关键判定（E10） | G0 **非同源**逐位对比 `5edfca0`（loss / 89 梯度 / 92 state / `get_indices` / 4-tuple / token / eval 全同）；G1 tripwire **0 次**触发；23 个入口调用计数探针**每个都带阴性对照**；幂等 / resume / 失败中止全绿 |
| 回归 | E1–E9 九个 suite 全绿 **411/411**（E10 内层给 E9 设 `EXP_E9_REGRESSION=skip`，故报 81/81；与独立跑的 90/90 **相加无重复**） |
| 未变 | EXP-E7-E9 与全部 B / C 类效果实验**仍为 `PLANNED`**；本项目**仍无任何效果数据** |

---

## 11. 修订记录 R7 — C1 实现 + CSG CF 来源裁定（2026-10-14）

执行 C1（CF embedding interface），CSG 线开工。本轮同时**裁定了正式 CF 表征的来源**。

**来源裁定（用户裁决，推翻首轮的 ItemCF 路线）**

| 项 | 内容 |
|---|---|
| 正式来源 | **SASRec item embedding** — `sasrec_output/<D>/seed2025_best.pth` 的 `item_emb.weight`，**50d** |
| 依据 | CSG4Rec 的 `--cf_emb` 默认即 SASRec 的 item embedding 产物（`CSGT/main.py:54` → `Instruments-32d-sasrec.pt`；`CSGT/datasets.py:24-29` 直接 `np.load` 后**按行**与 `text_emb_np` 对齐）⇒ **其 CF 向量 = SASRec item embedding 表，不是 ItemCF 矩阵** |
| 为何不复刻生成端 | 参考快照**不含 SASRec 训练器**（`CSGT/models/` 仅 `layers.py`/`rq.py`/`vq.py`）⇒ 取本仓库**同语义**的 SASRec 基线 `tools/run_sasrec.py`；seed 选 **2025**（v2 对齐协议；v1 的 `sasrec_<D>_best.pth` 有 off-by-one eval bug） |
| ItemCF 文件的处置 | **保留、不删除**。它**不是**正式 CSG 来源（口径 = 本项目 ItemCF cosine + SVD 128d，与 CSG4Rec 语义不同）；在接口中按 `source="itemcf"` 显式加载，`is_default_source=False` |

| 项 | 内容 |
|---|---|
| 产物 | `tiger_ext/cf_embed.py`（`CFEmbedding` 接口，**纯 numpy 叶子，不自算任何表征**）+ `integration/exp_c1_cf_embedding.py`（132 项）+ `integration/artifacts/cf_embed/{Toys,Sport,Beauty}.json`（正式源 provenance）+ `fallback_{Toys,Sport,Beauty}.json`（备用源） |
| 既有 tracked 文件改动 | **零**（`git status --porcelain` 与开工前逐字相同） |
| 新增工程实验 | **EXP-C1**（§2 新增条目，A 类，❌ 不进论文） |
| 结果 | **132/132**（CPU + CUDA），exit 0 |
| 抽取契约（实测） | ckpt 的 `item_emb.weight` = **`(N+1, 50)` float32**（11925 / 18358 / 12102）；**第 0 行 = padding 且恒全零**；ItemID 空间恰为 1..N ⇒ 规范表 = `W[canonical ItemID]`，**不是 `W[1:]`** |
| 负对照 | Beauty 上 `W[1:]` 会错配 **4158** 行（第 8 行 ItemID=10 应取 `W[10]` 而非 `W[9]`）——脚本内已实测断言 |
| 明确不做 | 不加 projection / normalization（**不因 128→50 补维度**）、不改 `CFEmbedding` 核心接口、不实现 C2 的 cosine/kNN、不实现 C3 约束、不实现 C4 loss（AST 断言模块内无 `cosine/knn/loss` 顶层符号、无任何 `nn.*`/`F.*` 层构造） |
| 回归 | **E1–E10 全部重跑通过**（同一脚本内驱动 `exp_e10_smoke`，64/64） |
| 未变 | C2–C6 与全部 B / C 类效果实验**仍为 `PLANNED`**；本项目**仍无任何效果数据** |

---

## 12. 修订记录 R8 — C2 实现（2026-10-14）

执行 C2（Collaborative relation / kNN），把 C1 的 CF 表变成 item-to-item 的 top-k 协同关系。

**关系定义（先以 plan.md C2 规格为准；与 CSG4Rec 原实现的差异逐条记录，不自行扩大设计）**

| 项 | 裁决 | 依据 |
|---|---|---|
| 相似度 | `cos(a,b) = dot(a,b)/(‖a‖₂‖b‖₂)`，float32；先对表做一次 L2 归一化，再按 block 做归一化 matmul | CSG4Rec `metric='cosine'`（sklearn 的 cosine **距离** = `1-cos`，只用排序） |
| **k** | **20**（`k_eff = min(k, n-1)`，`k` 与 `k_eff` 分开记录） | CSG4Rec `--knn_k` 默认值 `CSGT/main.py:39`。**未重新调参**——调 k 属设计决策，不属本 Task |
| 方向性 | **有向**，每行独立取自己的 top-k，**不做对称化** | 忠实于 `global_knn` 的逐行语义 |
| 自环 | **显式 mask 对角线** + `validate()` 断言 0 自环 | 见下方"参考实现的缺陷" |
| 零范数行 | source / target **双向排除**，标 `-1` / `NaN` | cosine 在该行无定义 |
| tie-break | 相似度降序 → 行号升序（stable argsort，完全确定） | `argpartition` 会在第 k 位放行任意并列项，与所声明的 tie-break 矛盾 |
| 行空间 | C1 的规范行序；artifact 自带 `item_ids` + 摘要，`load` 不匹配即拒绝 | plan.md C2 验收⑤ |

**与 CSG4Rec 原实现的差异（5 条，均写入 artifact 的 provenance）**

1. 用 numpy 分块 matmul 而非 sklearn（不引入新依赖，且让自环排除显式化）。**已与 sklearn `NearestNeighbors(metric='cosine')` 交叉复算，三数据集 top-k 完全一致。**
2. **自环靠 mask，不靠丢第 0 列**。参考实现的 `indices[:, 1:]` 假设"自己永远排第一"；**在零范数行上该假设失效**——ItemCF 备用源上参考写法留下自环 **6/24（Toys）、11/33（Beauty）**。官方 SASRec 源无零范数行，两者结果 0/0/0 相同 ⇒ **该缺陷在主线上不会暴露**。
3. 零范数行双向排除并标 `-1`/`NaN`（参考实现把排序交给求解器）。
4. 边落盘，并附 ItemID 伴随文件与 id 摘要（参考实现只在内存里存数组）。
5. 独立 artifact + 独立 loader，**不复用历史 `trans_graph_path` 开关语义**（plan.md 明令）；只沿用 `(M,2)` 数组形态。

| 项 | 内容 |
|---|---|
| 产物 | `tiger_ext/collab_relation.py`（`CollabRelation` + `build_relation(_for_dataset)`，numpy 叶子）+ `integration/exp_c2_collab_relation.py`（127 项）+ `integration/artifacts/collab_relation/<D>_k20.{neighbors,similarity,pairs,pairs_itemid,item_ids}.npy` + `<D>_k20.json` |
| 既有 tracked 文件改动 | **零**（`git status --porcelain` 与 C2 开工前逐字相同） |
| 新增工程实验 | **EXP-C2**（§2 更新条目，A 类，❌ 不进论文） |
| 结果 | ✅ **A-PASS — 127/127**（exit 0） |
| 实测 | Toys 238480 边 / Sport 367140 / Beauty 242020，**均 = n_items × 20，0 自环**；互反边占比 0.652 / 0.612 / 0.694（有向）；平均 cos rank0→rank19 = 0.670→0.506 / 0.651→0.515 / 0.653→0.496 |
| 正确性 | 合成 fixture 与**手算**一致；真实数据与**独立 float64 暴力复算** 的集合+顺序全同（`max\|Δcos\| ≤ 1.9e-7`）；**最优性**：第 20 名 cos 高于所有非邻居；与 sklearn 交叉复算一致；存/取逐位一致；篡改/错 id/k 不符/缺文件 **全部拒绝加载** |
| 行空间护栏 | **Beauty 上 `row+1` 会让 136490/242020（56%）的边指向错商品**；Toys/Sport 因 ItemID 恰为 1..N 而看不出差别 |
| ⚠️ B 部分限制 | 共现富集 Toys 23.7× / Sport 27.3× / Beauty 22.4×，但 valid/test 是**同批用户序列的前缀**（与训练语料高度重叠）⇒ **不构成效果证据**。它证明的是"邻居是行为相关商品"+"探针能抓住映射错误"（打乱映射后富集塌回 ≈1.0×） |
| 回归 | **E1–E10 全部重跑通过**（同一脚本内驱动 `exp_e10_smoke`，64/64） |
| 未变 | C3–C6 与全部 B / C 类效果实验**仍为 `PLANNED`**；本项目**仍无任何效果数据** |

---

## 13. 修订记录 R9 — C3 实现（Q-space collaborative structure constraint，2026-10-14）

执行 C3（Q-space collaborative constraint），把 C2 的协同关系**投影到 RQ-VAE 的量化空间**上。C3 只交付**约束中间量**，**不算 loss**（那是 C4）。

**先说参考实现的结论（写代码之前确认的 7 问，详见 [plan.md §15.1](plan.md)）**

| # | 问题 | 答案 |
|---|---|---|
| 1 | CSG4Rec 的 Q-space 是什么 | **量化后的码本查表向量** `x_q = self.embedding(indices)`（`CSGT/models/vq.py:195`），即取到码字、但**尚未** straight-through 的那个张量 |
| 2 | 约束作用在哪个张量 | 同一个 `x_q`，被 `diversity_loss`（`vq.py:68-106`）消费，调用点在 `:198-200` |
| 3 | quantized vector / code logits / 别的 | **quantized vector**：打分是 `sim = torch.matmul(x_q, emb.t())`（`:84`） |
| 4 | constraint 怎么算 | 屏蔽自身（`:87-90`）后 `F.cross_entropy(sim, y_true)`，`y_true` 是**码字 id**（同簇另一个码字，`:72-79`）；逐层算、`rq.py:64` 平均、`--beta` 加权 |
| 5 | relation 怎么参与 | **在可恢复的代码里，relation 不是 CF kNN**，而是对**码本**做 constrained-kmeans 得到的 10 簇（**code-to-code**） |
| 6 | 哪些是原始机制 | `x_q` 的选择、`x_q × 码本` 的打分形式、`cross_entropy` 形态、逐层平均 + `--beta` 加权 |
| 7 | 哪些 TIGER 没有 | ① **CF kNN 进入 loss 的那一半**；② `use_constrain_loss` / `--alpha` 开关 |

### 13.1 ★ 决定性发现：`CSGT/models/rqvae.py` 不存在

参考快照的 `CSGT/models/` 下**只有** `layers.py` / `rq.py` / `vq.py`，**没有** `rqvae.py`；无归档、无 `.pyc`、本机无第二份。而 `CSGT/main.py:9` 与 `generate_indices.py:15` 都 import 它，`global_knn` / `use_knn` / `use_knn_argument` / `use_cf_loss` / `use_pairwise` / `use_kd` **只在这里被读**。

⇒ **CSG4Rec 的 kNN loss 的确切函数形式不可恢复，本次不复现。** 这与 C1（SASRec trainer 缺失）、C2（RQVAE model 缺失）同类。已追到的调用链：`datasets.py:36-38` 构建 → `trainer.py:155,192` 随 batch 停在 CPU → `:159` 交给 `model.compute_loss`；逐 epoch 打印有 `knn` 槽（`:241`）；`trained_loss = {"total","rqvae","recon","cf"}`。

### 13.2 A / B / C 三分（plan.md 明令不得混成一个 embedding alignment）

| 部分 | 是什么 | 拥有者 | 形态 |
|---|---|---|---|
| **A** collaborative relation | C2 的 `CollabRelation` | C2 | 只经 `CollabRelation.load(prefix, expected_item_ids=..., expected_k=20)`；**本模块从不打开 `.npy`**（AST 钉死） |
| **B** Q-space representation | `x_q` | RQ-VAE 自己的 forward | `capture_qspace()`，**当场自证**张量来源 |
| **C** constraint | A 投影到 B | C3 | `ConstraintBatch`：位置对 + 两侧 gather 的 Q-space + cosine 权重 + 丢弃计数 |

### 13.3 表示选择：`x_q`，排除 code logits（结构性理由）

| 候选 | 裁决 | 依据 |
|---|---|---|
| `x_q` | ✅ 采用 | 是参考实现存活的约束所消费的量，且**量化后** |
| code logits | ❌ 排除 | `rqvae/models/vq.py:76` 的 `raw_d = d`，`d` 是 `latent`（**量化前** encoder 输出）到码字的平方 L2 ⇒ **约束 logits = 约束 encoder 向量**，正是 plan.md 禁止的 latent alignment，只是改了个名字 |

**来源断言**（把"量化空间"从标签变成可测量属性）：`x_q ≈ Σ_l codebook_l[indices_l]`（用**该模型自己的**码本重算，`atol=rtol=1e-4`；实测偏差 7.45e-09 ~ 5.96e-08），**且** `x_q` 必须不同于同一次前向的 `z`（实测差 0.131 / 0.154 / 0.229）。4 个负对照全部生效：交 latent → 拒；篡改 code index → 拒；层数不对 → 拒；`rows` 对不上 → 拒。

### 13.4 与参考实现的差异（5 条，均写入 `PORT_DIFFERENCES`）

| # | 项 | **原始机制** | **当前适配** | 理由 |
|---|---|---|---|---|
| 1 | 监督来自谁 | code-to-code（码本同簇另一码字） | **item-to-item**（C2 的协同邻居） | plan.md C3 的输入就是"C2 的关系对" |
| 2 | loss | 内联 `beta * diversity_loss` | **完全不计算** | plan.md 拆成 C3/C4；在此算会越界，且会逼着猜那个**缺失文件** |
| 3 | batch 外邻居 | **未知**（在缺失文件里） | **丢弃并计数** | 唯一不需额外信息的策略；本仓库既有先例 `rqvae/trainer.py::_get_transition_loss:213-219` |
| 4 | 量化哪个张量 | ST **之前**的码本查表 | `forward(return_xq=True)` 的 `x_q`：**值相同**，带 ST 梯度路径 | TIGER 只暴露 ST 之后的；**梯度路由不同，C4 必须知道** |
| 5 | 开关 | `--use_knn` / `--knn_weight` | `lambda_csg`（默认 0，本模块自己的 mount） | plan.md 要求独立开关；不复用 `trans_graph_path` |

**未做**：warmup / projection / 额外 regularization / concat CF embedding / CF↔latent 对齐 / User SID / 改 RQ-VAE 架构 / 改 T5 / C4 的 loss / C5 的 trainer integration。

### 13.5 约束的数学形式（产出张量，不产出标量）

对 batch 行集 `R`（`|R|=B`）与有向关系 `E`：

```
P  = {(i,j) ∈ E : i ∈ R 且 j ∈ R}     # in-batch 边，其余丢弃并计数
anchor_pos[p] = pos_R(i_p)             # 位置，不是行号
target_pos[p] = pos_R(j_p)
w[p]          = sim(i_p, j_p)          # 关系自带 cosine，原样透传
X[p] = x_q[i_p],  Y[p] = x_q[j_p]      # 两侧各 gather 一次
```

**C3 产出这些张量，不产出标量**。目标函数形态（同族于参考的 `x_q × 码本` 打分）归 C4。

### 13.6 实测结果

| 数据集 | 关系边数 | 256-item batch 内对数 | 丢弃（batch 外） | 哨兵 | 有对的锚点 | 重建偏差 | latent 间距 |
|---|---|---|---|---|---|---|---|
| Toys | 238480 | **738**（2.88/item） | 4382 | 0 | 175/256 | 2.98e-08 | 0.154 |
| Sport | 367140 | **118**（0.46/item） | 5002 | 0 | 88/256 | 5.96e-08 | 0.229 |
| Beauty | 242020 | **333**（1.30/item） | 4787 | 0 | 181/256 | 2.98e-08 | 0.131 |

> ⚠️ **对数不是效果指标**，是 batch 几何。256 行若邻居均匀随机，期望仅 `256/11924 × 20 ≈ 0.43` 对/item —— Sport 的 0.46 **正落在该期望上**，Toys 的 2.88 是**表头行邻居互锁**。同一段代码在两个数据集差 6 倍，恰好说明它取决于"抽了哪些行"。

**tokenizer 口径（重要）**：仓库中**只有 Toys** 有训在**规范 4096d content 表**上的 RQ-VAE（`clean_tiger/ckpt/Toys/rqvae/Oct-14-2026_05-55-09/`；已核对训练用的 smoke parquet 与 `data/Toys/item_emb.parquet` 的 ItemID 与 embedding **逐位相同**）。Sport / Beauty 只有融合表上的 `*_TA_SID_4096d`。故：

- 三数据集那轮统一用**同架构 fixture tokenizer**（冻结架构 + 模型自己的 `init_emb` 在该数据集真实 latent 上 k-means 初始化，**不训练**），保证三行可比；
- Toys **额外**用**真实训练过的** Clean TIGER tokenizer 复跑：**对数 738 与 pair 轴逐位相同**，但 code 分布不同（真实 256 项用到 71 个 L0 码 / fixture 99 个）⇒ **A/C 与 B 是可分离的**。

### 13.7 下游（C4 起）必须知道的边界

1. **C3 不产出 loss、不产出梯度**。`lambda_csg=0`（默认）时 `enabled=False`，`forward()` **抛异常**而非静默返回 0（同 `SIAMount`/`PSAMount`）；调用方必须守卫 `enabled`。
2. **梯度路由**：C4 拿到的是 **ST 之后**的 `x_q`（值 = 码本求和，梯度回 encoder）；参考实现用 ST 之前那个（梯度只到码本）。**照抄参考的反传语义会得到不同的梯度图。**
3. **`x_q` 来自 RQ-VAE 时不要重新 gather**（`rqvae/models/rqvae.py:64-67` 的警告）；`rebuild_quantized()` **只用于校验**。
4. **in-batch 策略是适配**，不是原始机制；参考实现的策略**未知**。C4 若要改策略，那是一个新的设计决策。
5. **有向**，未对称化（互反 61–69%）。
6. **`rows` 恒在 CPU**（整数身份）；Q-space 张量跟模型走。CUDA 臂已验证 "CPU 索引 → CUDA 表示取行" 可用。
7. **k=20 是继承来的**，不是搜出来的。

| 项 | 内容 |
|---|---|
| 产物 | `tiger_ext/qspace_constraint.py`（`QSpaceConstraint` / `QSpaceBatch` / `ConstraintBatch` / `capture_qspace` / `rebuild_quantized` / `rows_of_item_ids` / `load_constraint_for_dataset`）+ `integration/exp_c3_qspace_constraint.py`（9 节 99 项）+ `tiger_ext/__init__.py` 的再导出 |
| 既有 tracked 文件改动 | **零**（`git status --porcelain` 与 C3 开工前逐字相同：仍只有 E1–E6 那 3 个文件） |
| 新增工程实验 | **EXP-C3**（§2 更新条目，A 类，❌ 不进论文）；C4 从 EXP-C3/C4 中**独立出来** |
| 结果 | ✅ **A-PASS — 99/99**（exit 0） |
| 回归 | **E1–E10 全部重跑通过**（同一脚本内驱动 `exp_e10_smoke`，64/64） |
| 未变 | C4–C6 与全部 B / C 类效果实验**仍为 `PLANNED`**；本项目**仍无任何效果数据** |

> **R10 追加的复现性更正**：13.6 末段的 "71 / 99 个 L0 码" 是**单次抽样**（该 fixture tokenizer 走冻结的 `init_emb`，其 `KMeans` 无 `random_state`，默认线程下跨进程不可复现）。该节的**结论不受影响**——"A/C 与 B 可分离"依据的是 **pair 轴逐位相同（738）**，而配对与 tokenizer 无关。C4 已把 fixture 改成可复现的 seeded 初始化并记录了实测，详见 [plan.md §16.9](plan.md)。

---

## 14. 修订记录 R10 — C4 实现（Q-space collaborative constraint objective，2026-10-14）

执行 C4（Q-space collaborative constraint objective），把 C3 的 `ConstraintBatch` 变成**标量 loss**。**C4 只做这一件事**：不接训练循环、不改 E7、不改 Tokenizer/Recommender stage、不改 RQ-VAE / T5 / SIA / PSA / C1 / C2 / C3、不加 projection / warmup / regularization / 新 loss 分支。**完成后停止，不进 C5。**

**先说参考实现的结论（写代码之前确认的 10 问，详见 [plan.md §16.1](plan.md)）**

| # | 问题 | 答案（代码依据） |
|---|---|---|
| 1 | **query** | anchor item 在该层的**码字** `W_l[Ia[:,l]]`（`CSGT/models/vq.py:195`、`:84`） |
| 2 | **candidate / target** | candidate = **该层自己的码本** `emb = self.embedding.weight`（`:69`）；target = **一个码字 id**（`:92`），取协同邻居在该层的码 |
| 3 | **similarity logits** | **裸点积** `X @ W_lᵀ`（`:84`）；cosine 在 `:83` 被注释掉；`temp=1` 硬编码（`:70`）⇒ 无温度 |
| 4 | **y_true** | **码字 id（int 类标签）**，与参考同类型。不用相似度、不用软标签 |
| 5 | **多邻居** | 每对**独立一行 CE**（n 邻居 = n 行），per-level 归约 = **pair 均值**；**不是** multi-positive |
| 6 | **有向 pair** | **不合并**：`(a→b)` 与 `(b→a)` 两行独立 CE。实测 `L(0→1)=0.7201` vs `L(1→0)=2.6657` |
| 7 | **C2 的 cosine weight** | **不进 objective**。`diversity_loss` 签名 `(x_q, indices, indices_cluster, indices_list)`（`:68`）**没有 weight 参数** ⇒ "C3 construction 中保留，C4 objective 不使用"，并实测其不影响 loss |
| 8 | **reduction** | 层内 = CE 默认**行（pair）均值**（`:92`）；层间 = 对有 pair 的层取 mean（`rq.py:64`）。**不是 anchor 均值**（实测 1.8179 vs 1.5353） |
| 9 | **无有效 pair** | 返回 **0 维 `0.0`**（先例 `rqvae/trainer.py::_get_transition_loss` / `rqvae/trans_loss.py::ContrastiveLoss`）；**开关关闭是另一回事，必须抛** |
| 10 | **同 anchor 多 target** | **独立 CE**（= 第 5 问），**不发明** multi-positive（实测 1.3329 ≠ 1.5353） |

### 14.1 ★ `emb` 是什么（brief 要求的确认）与由此**被迫**得出的 level-wise

`emb = self.embedding.weight`（`vq.py:69`）⇒ **该 RQ 层自己的码本**，不是全局 embedding、不是 CF embedding、不是别的层。
⇒ `x_q` 是**各层码字之和**，而 `emb` 只有**单层**码本，`sum(x_q) @ embᵀ` 在参考实现里**无对应物**。因此 level-wise **不是我们加的**，是参考实现**唯一能表达**的形态——brief 的"优先保持完整 `x_q`"与代码证据在此**指向同一结论**。

### 14.2 ★ 可复原 / 不可复原（**不伪造**）

- **可复原**：`vq.py:68-106` 的 `diversity_loss`——`x_q × 码本` 打分、`1e12` 自身屏蔽、`F.cross_entropy(sim, y_true)`、`temp=1`、逐层调用 + 层间平均 + `beta` 加权。
- **不可复原**：**CSG4Rec 的 kNN loss 本体**——`CSGT/models/rqvae.py` 在快照中**不存在**（无归档、无 `.pyc`、无第二副本）；`global_knn` / `use_knn` / `use_cf_loss` / `use_pairwise` / `use_kd` / `compute_loss` / `get_indices` 全在里面。
- **裁决**：**不复刻、不猜、不补造**。复现的是**存活约束的形态**，正样本换成 C2 的 item-to-item 关系。`provenance()["is_a_reproduction"]` 恒为 `False`。

### 14.3 最终数学形式（冻结，见 [plan.md §16.4](plan.md)）

```
keep[p] = (Ia[p,l] != Iy[p,l])                # 参考的正样本必为不同码字 (vq.py:77-78)
X  = W_l[Ia[keep,l]]                          # query，W_l 来自 rqvae.rq.get_codebook()（可微）
S  = X @ W_lᵀ                                 # candidate = 该层全部 n_e 个码字，裸点积
S[p, Ia[p,l]] -= 1e12                         # 自身屏蔽，vq.py:87-90 逐字
CE_l = F.cross_entropy(S, Iy[keep,l])         # 行（pair）均值
L_csg = mean{ CE_l : P_l > 0 }                # 层间均值
```

无 weight、无温度、无归一化、无 projection、无 warmup、无额外 regularization、无 detach、无新可训练参数。

### 14.4 与参考实现的差异（6 条，均写入 `_PORT_DIFFERENCES`）

| # | 项 | **原始机制** | **当前适配** | 理由 |
|---|---|---|---|---|
| 1 | loss 本体 | **未知**（在缺失文件里） | **不是复刻**：存活约束形态作用于 C2 关系 | 文件缺失；补造即伪造 |
| 2 | 正样本来源 | 同簇**另一码字**（`vq.py:72-79`） | C2 协同邻居在该层的码字 | plan.md C3 输入就是 item-to-item；被约束量/候选集/打分/target 类型全不变 |
| 3 | query 梯度路由 | ST **之前**的码本查表（`:195`） | 同一次查表（`rqvae/models/vq.py:90`）⇒ **同样路由** | 数值相同；C3 另交的 ST 后张量其 encoder 通路实测存在但**本 objective 不用** |
| 4 | 正样本必不同码 | 由构造保证（`:77-78`） | **丢弃并逐层计数** | 参考没有"自身为目标"的算术；保留会让屏蔽列赢，loss ≈ 1e12 |
| 5 | 层 | 每层总贡献 | 无 pair 的层**略去**，均值取有定义的层 | 用制造的 0 摊薄 scale 更糟 |
| 6 | 开关 | `--use_knn` / `--knn_weight` / `--use_constrain_loss` / `--beta` | `lambda_csg` 默认 0，**不接入任何总 loss** | plan.md 要求独立开关；C4 不得碰 trainer |

**未做**：接训练循环 / 改 E7 / 改 stage / 改 RQ-VAE 架构 / 改 T5 / 改 SIA / 改 PSA / 改 C1·C2·C3 / projection / warmup / regularization / 新 loss 分支 / User SID / concat CF embedding / CF↔latent 对齐 / T5 logits↔SID logits 对齐。**已完成即停，未进 C5。**

### 14.5 梯度（brief §六，实测）

| 检查 | 结果 |
|---|---|
| `L_csg` 是否可微 | ✅ `requires_grad=True`、`grad_fn=MeanBackward0` |
| `dL/dW_l` | ✅ ≠ 0（查询侧 + 候选侧两项，与 `sim = x_q @ embᵀ` 同构） |
| 是否到 encoder | ❌ **不到**——indices 来自 `argmin/argmax`（`rqvae/models/vq.py:78,86`）无梯度。**与参考实现同路由**（它打分 ST 前张量 `:195`，`:213` 才重写 `x_q`） |
| ST 通路是否完好 | ✅ C3 的 `x_q` 直接 backward ⇒ **10 个 encoder 参数拿到梯度**（实测）。objective 不用它，但**没破坏它** |
| 是否 detach / 改 `requires_grad` / 加 projection | ❌ 都没有（AST 逐函数检查 + `[8]` 逐条断言；仅有的两处 `.detach()` 都在 `float(...)` 统计里） |
| E7 是否被改 | ❌ **未改**（brief §六 明文） |

> **"让约束反传到 encoder"是 C5 的决策，不是 C4 的遗漏**——已记入 `QSL_GRADIENT_POLICY`。

### 14.6 `lambda_csg` 语义（与 C3 一致）

| 取值 | 行为 |
|---|---|
| `== 0`（默认） | **抛 `QSpaceDisabled`**，**不执行 objective、不静默返回人造 zero** |
| `> 0` | `lambda_csg * L_csg`；**仍未接入任何总训练 loss**（那是 C5） |
| 空 pair（与开关无关） | 返回 0 维 `0.0` + `stats["reason"]`——**数据结果，不是接线错误**，与上一行严格区分 |

### 14.7 实测结果（真实三数据集 + 真实已训 tokenizer，全部可复现）

| 数据集 | 关系边 | batch 内 pair | `L_csg` | per-level | 层内 same-code 丢弃 | `\|dL/dW\|` per level |
|---|---|---|---|---|---|---|
| Toys（fixture tok.） | 238,480 | 738 | **5.539878** | [5.5378, 5.5413, 5.5405] | 60 | [1.10883, 0.14682, 0.16637] |
| Sport（fixture tok.） | 367,140 | 118 | **5.556832** | [5.5875, 5.5418, 5.5412] | 10 | [1.46451, 0.22270, 0.35763] |
| Beauty（fixture tok.） | 242,020 | 333 | **5.559719** | [5.5967, 5.5410, 5.5415] | 29 | [1.09656, 0.15401, 0.22607] |
| Toys（**真实已训** ckpt） | 238,480 | 738 | **5.582809** | [5.6668, 5.5405, 5.5412] | — | [0.756085, 0.130800, 0.078298] |

> ⚠️ **这不是效果数据。** `log(255) = 5.5413` 是"码本完全无信息"的值，四行都在它附近 ⇒ **未训练码本应有的样子**。**不得**把 `L_csg` 的大小读成"约束有效 / 无效"，也不得与其他方法的 loss 数值比较。
> ⚠️ **与 §13.6 的同名 pair 数相同（738 / 118 / 333），但 same-code 丢弃数与码分布变了**——因为 fixture tokenizer 换了初始化（原因见 14.9 ①）。pair 数不变是**预期的**：配对只取决于关系。

**合成 fixture（独立 numpy float64 复算，不重跑同一段 torch）**：6 对 / 2 层，手算 per-row `[1.3132617, 0.3132617, 0.3132617, 1.3132617, 1.3132617]` 与 `[0.1269280, 4.0181499, 4.0181499, 0.1269280]`，层均值 `[0.9132616875, 2.0725389695]`，`L_csg = 1.492900328499`，丢弃 3。**被拒绝的归约**：anchor 均值 1.817900328、multi-positive 1.332852904（本实现 pair 均值 1.535298347）。**weight 不变性**：零/极大/取负/打乱下 `0.765279412` 逐位不变。**数值稳定**：零码本 ⇒ 恰好 `log(n_e-1)`；scale 1e3/1e-8 有限且在解析界内；NaN 码本 ⇒ NaN（不粉饰）；越界/负 code index、层数不符、形状不符、单码字层 **全拒**。**如实记录的退化读数**：64×32 随机 fixture 上只有 level 0 的非零（[3.617317, 0.0, 0.0]，因 64 个残差被 256 码字完全吸收），真实内容上三层全非零（上表）。

### 14.8 工程断言覆盖面（EXP-C4 = 10 节 + `[6b]`，共 87 项）

| 节 | 内容 |
|---|---|
| `[1]` | 手算 fixture：logits → target → CE，逐行/逐层均值/总和/丢弃数 |
| `[2]` | 多邻居：每对一行、归约是 **pair 均值**（并实测拒绝 anchor 均值与 multi-positive） |
| `[3]` | 有向：(a→b) 与 (b→a) 两行，**不合并** |
| `[4]` | 空 pair：显式 0.0 + `reason`；某层全丢时只对有定义的层取均值 |
| `[5]` | **C2 weight 被携带但未被读取**（四种扰动逐位不变 + 源码不读 `batch.weight`） |
| `[6]` | 梯度 → 码本；ST 通路到 encoder 的独立测量；不 detach / 不 backward / 不持参数 |
| `[6b]` | 真实三数据集 + 真实已训 tokenizer；fixture 可复现性 |
| `[7]` | 数值稳定与拒绝面（NaN / ±Inf / 空张量 / batch=1 / 单对 / 越界 index / 单码字层） |
| `[8]` | 禁项扫描（无 projection / concat / warmup / 正则 / 新参数 / 不 import 内核）+ **6 个 sha256 pin** 证明 C1·C2·C3·E7·SIA·PSA 逐字节未变 |
| `[9]` | `lambda_csg=0` 的干净路径（抛异常而非人造零；关闭时 0 参数 0 buffer） |
| `[10]` | 回归：`exp_e10_smoke` 仍 **64/64** |

### 14.9 ★ R10 查实的两个附带问题（**不改冻结面**）

**① 冻结的 k-means 助手跨进程不可复现**：`rqvae/models/layers.py:86` 的 `KMeans(...)` **无 `random_state`**。实测 2026-10-14（3000×32 float32，固定 numpy seed）：默认线程 **4/4 次 `cluster_centers_` 哈希互不相同**，`OMP_NUM_THREADS=1` 时 **3/3 相同**。污染源分离（真实 Toys content）：已训 ckpt **3/3 稳定**、torch-seeded init + encoder forward **3/3 稳定**、**只有走 `init_emb` 的不稳**。⇒ **不改冻结核心**，C4 的 fixture 改为"该层 residual 的 seeded 子样本"并加同进程重建断言。**连带更正 §13.6**（见该节 R10 追加说明）。与 [[md5-is-not-a-data-fingerprint]]、C2 记录的 `WarpSampler n_workers` 属同一类陷阱。

**② E10 的 CSG 守卫对 C4 假阳性**：`exp_e10_smoke.py:863` 是 `grep -rniE "csg_|lightgcn|graph_conv"`（**`-i`**）。C4 的常量原用 `CSG_` 前缀（为避开 C2/C3 已导出的名字），**恰好含 `csg_`** ⇒ E10 掉到 **63/64**。**裁决：改 C4 的名字（`CSG_*` → `QSL_*`、`qspace_csg_loss` → `qspace_constraint_loss`），不改 E10**（§九 禁止改 E1–E10）⇒ E10 回到 **64/64**。这不是绕过守卫——守卫的意图是"CSG 线不得被静默激活"，而 C4 与 C3 同性质：`tiger_ext/` 下**未接入任何运行时**的独立目标函数（`[8]` 逐条断言）。

| 项 | 内容 |
|---|---|
| 产物 | `tiger_ext/qspace_loss.py`（`QSpaceLoss` / `QSpaceLossMount` / `qspace_constraint_loss` / `codebook_stack` / `verify_indices_match_qspace` + 14 个 `QSL_*` 记录常量）+ `integration/exp_c4_qspace_loss.py`（10 节 + `[6b]`，87 项）+ `tiger_ext/__init__.py` 再导出 |
| 既有 tracked 文件改动 | **零**（`git status --porcelain` 与 C3 结束时逐字相同：仍只有 E1–E6 那 3 个文件） |
| 新增工程实验 | **EXP-C4**（A 类，❌ 不进论文）——C4 从 CRISP 的 C3/C4 拆分中**独立出来** |
| 结果 | ✅ **A-PASS — 87/87**（exit 0） |
| 回归 | **`exp_e10_smoke` 重跑 64/64，exit 0**（修复 ② 之后） |
| 未变 | C5 / C6 与全部 B / C 类效果实验**仍为 `PLANNED`**；本项目**仍无任何效果数据**；`lambda_csg` 默认 0，**未接入任何训练循环** |

---

## 15. 修订记录 R11 — C5 实现（CSG trainer integration，2026-10-14/15）

> **命名去歧**：本节的 **C5 = 工程接入任务**（把 C4 的目标函数接进冻结训练循环），验证实验是 **`integration/exp_c5_csg_integration.py`（A 类工程门禁，❌ 不进论文）**。总表 §1 里名为 **EXP-C5/C6 的"TIGER vs TIGER + CSG"效果实验（B + C 类）仍然是 `PLANNED`**，两者不是同一件事。

| 项 | 内容 |
|---|---|
| 产物 | `rqvae/trainer.py`（**加一处受守卫的挂载**）+ `tiger_ext/qspace_train.py`（新驱动入口 + `CSGHook` + `CST_*` 常量）+ `integration/exp_c5_csg_integration.py`（11 节）+ `tiger_ext/__init__.py` 再导出 |
| **既有 tracked 文件改动** | **`rqvae/trainer.py` 一个**（plan.md §C5 明文授权）——这是我方**第一次**修改冻结内核内的文件；`git diff` = **7 hunks、+34/−5**，`sha256(diff)[:16] = a8da64e3291fa1c5` |
| 新增工程实验 | **EXP-C5**（A 类，❌ 不进论文） |
| 结果 | ✅ **A-PASS**（见 §15.7 覆盖面；CPU + CUDA） |
| 回归 | 7 份既有 harness 重跑，**每一条 FAIL 都是本次授权修改造成的 tracked-file guard**，已机械归因（见 §15.5） |
| 未变 | **EXP-C5/C6 效果实验仍为 `PLANNED`**；本项目**仍无任何推荐效果数据**；`lambda_csg` 默认 **0** |

### 15.1 total loss 与挂载点

```
L_total = L_RQVAE + lambda_csg * L_csg
L_RQVAE = loss_recon + quant_loss_weight * quant_loss      # 定义未改（rqvae/models/rqvae.py:87-97）
```

挂载插在既有的 `L_align` 之后、`loss.backward()` 之前，**形状与 `L_trans`/`L_balance`/`L_align` 完全同构**；权重只乘一次（`CSGHook` 返回的就是 `lambda_csg * L_csg`）；只有 `requires_grad and isfinite` 才真的加进 `loss`（C4 空 pair 返回**无梯度**的 `0.0`，所以这一步必要）。**`csg=None` 时所有新增表达式与改前求值完全相同**（G0 逐位实测，见 §15.4）。

**参考侧**：`CSGT/models/vq.py:209` 可复原出 `codebook_loss + mu*commitment + beta*diversity`，`CSGT/train_tokenizer.sh:8` 发布 `--beta 0.0001` ⇒ 参考对同形项的权重 = **1e-4**（尺度匹配锚点）。`CSGT/models/rqvae.py` 在快照与 zip 里都不存在 ⇒ 参考的 `total` 公式与 `knn` 项**不可复原**，本实现**不声称复刻**。

### 15.2 梯度策略（brief §二四项，全部实测）

| 路由 | 判定 | 证据 |
|---|---|---|
| **P1 仅 codebook** | **采纳（默认）** | 三个 RQ level 码本均有梯度（`Σ|dL/dW_l|` = 7.99e-04 / 3.41e-05 / 2.62e-05）；**0 个 encoder 张量**从此目标拿到梯度。这也是**参考自身的路由**（`vq.py:195` 的约束在 ST 之前，indices 出自 argmin/argmax） |
| **P2 ST → encoder** | **实测可达，不采纳** | `sum(x_q).backward()` 触达 **10 个 encoder 张量**；`max|x_q − z| = 3.29e-02` 证两路由是不同量。不采纳原因：要么替换目标 query（= 重定义 `L_csg`），要么改 `rqvae/models/{rq,vq}.py`（= 改冻结内核）⇒ **边界冲突·待裁决** |
| **P3 baseline 不启用** | **采纳（默认）** | `λ=0` ⇒ `enabled=False`、`n_calls=0`、G0 逐位相等；禁用时 mount **抛 `QSpaceDisabled`** 而非人造零 |

**"真的进了 backward"三重独立证明**：① 同初始状态同 batch 下 `L_total(λ=1e-3) − L_total(λ=0) = 5.583858e-03`；② 梯度可加性 `max |g_total − g_base − g_objective| = 8.846e-13`（相对 5.274e-08），逐码本张量；③ 码本张量 **3/3** 随 λ 改变、encoder 张量 **0/10** 改变。

### 15.3 lambda 候选与相对贡献（**报告，不排名**）

| λ | `λ·E[L_csg] / E[L_RQVAE]` | 读数 |
|---|---|---|
| 1e-4 | **1.551%** | 参考自己的同形项权重（**尺度匹配**）；RQ-VAE loss 仍主导 |
| 1e-3 | **15.535%** | 本次 sanity check 用的值 |
| 1e-2 | **154.972%** | 约束成为较大项 |
| （λ=0） | 0 | 控制臂：逐位等价于原 TIGER |

`E[L_RQVAE] = 0.035458`、`E[L_csg] = 5.508495`（20 步）。四个臂全部有限、无 NaN、无塌缩。**工作点不由 C5 选择**——它属效果实验。

### 15.4 tokenizer sanity check（真实数据；Toys 前 2048 行 × 4096d，冻结配方逐字照搬）

`num_emb_list=[256,256,256]`、`e_dim=32`、`layers=[512,256,128,64]`、`sk_epsilons=[0,0,0.003]`、`kmeans_init=True`、AdamW lr=1e-3、`quant_loss_weight=1.0`、batch 1024、10 epoch × 2 batch = **20 步**、`OMP_NUM_THREADS=1`。

| 项 | 实测 |
|---|---|
| forward/backward | 20 步全有限，`L_total ∈ [0.0232, 0.0834]`，无 NaN/Inf |
| reconstruction | epoch-1 `0.0302` → epoch-10 `0.0180`（下降） |
| 保存路径 | 冻结 trainer 自写 `best_loss_model.pth`（未被接线破坏） |
| code index | 2048 行全在 `[0, 255]` |
| 码本使用 | L0 256/256（max share 0.047）、L1 256/256（0.098）、L2 137/256（0.106）——**无塌缩** |
| 零码字（**容差 1e-12**，且读**任何 forward 之前**的训练码本） | 逐层最小行范数 **6.903e-02 / 2.410e-02 / 6.368e-03** |
| L_csg 参与 total loss | 20/20 步有 pair、有 3 层 loss；累计 **109240 pair / 60 level-loss** |
| λ=0 保持原行为 | 20 步 `L_total` 逐位 == `L_RQVAE`（且控制跑确在训练：20 个互异 `L_RQVAE`） |
| disabled / empty-pair 语义 | disabled ⇒ 抛 `QSpaceDisabled`；空 pair ⇒ 无梯度 `0.0` + `reason="no pairs"`，`L_total == L_RQVAE` 逐位 |
| 历史机制 | `L_trans`/`L_balance`/`L_align` 在一次 λ>0 跑里**各 0 次**；负对照 `lambda_balance=1.0` ⇒ 同一探针**触发 20 次** |
| **G0（验收③）** | pristine 源取自 `git show 5edfca0:rqvae/trainer.py`，同进程同 loader 同种子：**(a) 全默认**与 **(b) `lambda_balance=1.0`** 两种配置下，**逐步 loss 序列逐位相同**、**最终 `state_dict` 23/23 张量逐位相同**、初始 23/23 先证可比 |

### 15.5 ★ 授权偏离：既有 harness 的 tracked-file guard **必然**失败（如实记录，不掩盖）

按用户 2026-10-14「方案 3」：**不改任何既有 harness、不改 `PRE_*_MODIFIED`**，由 EXP-C5 §[10] 单独报告并**机械归因**（2026-10-15 实测）：

| harness | 结果 | 期望失败 | 内容 |
|---|---|---|---|
| `exp_e8_sid_refresh` | **54/55** | 1 | 自身 modified-set guard |
| `exp_e9_alternating` | **80/81** | 1 | 自身 guard（`EXP_E9_REGRESSION=skip`，避免重复计 E1–E8 的 330 项） |
| `exp_e10_smoke` | **60/64** | 4 | 自身 guard + 它重跑的两个子 suite 的 guard + 二者 aggregate |
| `exp_c1_cf_embedding` | **130/132** | 2 | 自身 guard + 内嵌 E10 重跑 |
| `exp_c2_collab_relation` | **125/127** | 2 | 同上 |
| `exp_c3_qspace_constraint` | **96/99** | 3 | 两条自身 guard + 内嵌 E10 重跑（另有 4 条是**回显**的 E10 输出，不计入） |
| `exp_c4_qspace_loss` | **84/87** | 3 | modified-set guard + **文本 guard**（"no frozen file mentions C4 or its switch"）+ 内嵌 E10 重跑 |

**归因规则（写进 harness，全部机械）**：**R1** = 点名一条 modified-set guard；**R2** = 点名一份被重跑的 suite，其缺口**恰好等于**期望值、其自身失败也全被这些规则解释（并**核对行内引用的计数与实测一致**——所以"写着 61/64 的行"不会被"实测 60/64"解释掉）；**R3** = E10 对两个受 guarded 子 suite 的 aggregate，缺口 = 两者期望缺口之和（2 = 1 + 1）。实测：**每一条 FAIL 行都被命中**（`unexplained: none`），且**每份 harness 的"自身失败行数"恰好等于它的缺口**（回显行被单独分离，不当作它自己的结果）。

**C4 那条文本 guard 单独归因**：重跑 C4 自己的 grep ⇒ **1 个** frozen 文件命中，正是被授权的 `rqvae/trainer.py`，命中的 **3 行（119/121/366）全部是注释**；冻结树里没有任何可执行行点名该开关。

### 15.6 ★ 顺带发现：k-means 把**首 batch** 变成拟合集 × C3 provenance guard（**必须记住**）

冻结配方 `kmeans_init=True` 下，`rqvae/models/layers.py:70-93` 的 k-means **拟合它看到的第一批点**；若该 batch 的不同点数 ≤ 码字数，则 RQ 残差被精确复现 ⇒ **`x_q == z` 逐位**、RQ-VAE 自己的码本梯度 ≈ 0。**实测**：`kmeans_init=True` 首 batch **1024 行 → `max|x_q − z| = 4.211e-02`**（guard-clean）；首 batch **256 行 → `0.000e+00`**（落进 C3 的 1e-4 容差 ⇒ **guard 拒绝**）；`kmeans_init=False`、1024 行 → `1.101e+00`。拟合 batch 上 `|dL_RQVAE/dW| ≈ 2e-10`，下一 batch ≈ 3.0。
**裁决**：**不放宽 C3 的 guard、不改 C3**；C5 的 sanity check 用冻结 batch size 1024；EXP-C5 §[6](c0) 把该退化**端到端测出来**（"整个数据集只有一个 item"的跑，第一 step 就被 C3 拒绝并点明原因）；§[6](c) 用 `kmeans_init=False` 才能走到空 pair 路径。数字全部记入 `CST_PENDING_DECISIONS` 第 6 条。
**连带规矩**：**先消耗 k-means 初始化（跑完首 batch），再测梯度/加法性**——否则相对误差会被 `|dL_RQVAE/dW| ≈ 2e-10` 的退化分母放大（本次实测：错在拟合 batch 上得 7.4e-03，改到 batch 1 后 5.3e-08）。

### 15.7 工程断言覆盖面（EXP-C5 = 11 节，**99/99 checks passed**）

| 节 | 覆盖 |
|---|---|
| `[1]` | total-loss 恒等式（逐步）、hook 计数、λ=0 不进分支 |
| `[2]` | 进了 backward 的三重证明（标量差 / 梯度可加性 / 码本动而 encoder 不动） |
| `[3]` | P1 / P2 / P3 三路由逐条实测 |
| `[4]` | 真实数据训练 sanity（NaN / 重建下降 / code usage / 零码字 / 约束每步活着）+ **码本健康与 λ=0 对照配对** |
| `[5]` | λ 三候选 + 控制，相对贡献与"无塌缩" |
| `[6]` | C4 语义在 trainer 内保持：disabled、空 pair、**k-means 退化端到端 (c0)**、provenance 数值入 trace、越界 rows 被拒 |
| `[7]` | 历史机制**惰性**（探针 + 负对照） |
| `[8]` | **最小挂载的机器核验**（用户方案 3 的 8 条）+ 7 个内核 pin + 6 个 C1–C4/E7/SIA/PSA pin + 无 `csg_` 标识符 |
| `[9]` | **G0**：全默认与 `lambda_balance=1.0` 两种配置下与 pristine 逐位等价 |
| `[10]` | 回归 + **回显分离** + R1/R2/R3 归因 |
| `[11]` | 新入口可跑、默认关闭、enable-without-rows 被拒、**k-means × guard 三组实测** |

**本次对 C5 自己 harness 的三处修正（改变了读数，如实记录）**：

1. **计数行必须按 suite 自己的 `EXP-<id>` 读，且该行必须恰好出现一次。** C3/C4 的回归段会打印它内嵌的 E10 重跑的**整段捕获输出——包括 E10 自己的 `EXP-E10: n/64 checks passed` 摘要行——而且打印在自己的摘要之前**，所以"匹配任意 `EXP-` 前缀"（甚至完全非锚定）的第一条命中都是 **E10 的数字被当成该 suite 的总数**（实测两者双双被读成 `60/64`）。最终按 suite 自己的 id 建正则（`suite_summary_re(exp_id)`），并**新增一条检查**断言该行**恰好出现 1 次**（否则读数 `-1/-1` 并失败）。修后为真实 **96/99**、**84/87**——**更严格**而非更宽松。
2. **`[FAIL]` 行按 suite 分离"自身的"与"回显的"**（`_split_echoes()`），并**新增一条检查**断言"自身失败行数 == 该 suite 自己的缺口"。C1–C4 在重跑的 suite 失败时会把它捕获的输出原样打印，E10 的 4 条失败行因此**逐字出现在它们的 stdout 里**；这是与第 1 条互相独立的一个坑（计数来自摘要行，失败来自 `[FAIL]` 行）。
3. 码本零行判据由 `== 0` 改为 **1e-12 容差**并**加配对对照**——冻结 `use_sk=True` forward 自己替换死码，替换后 L2 第 0 行范数 **6.628e-17**，用"精确等于 0"会把它判成健康；实测 **λ=0 对照跑一模一样**（`5.854e-17`，同为 L2 一行）⇒ 该行是**冻结 forward 的死码替换**，不是约束造成。

> **教训（写给后续 harness）**：当一份 harness 会**回显**它重跑的子套件输出时，"读计数"必须**按被测量对象的身份**去读（自己的 `EXP-<id>`），不能**按格式**去读（任意 `EXP-` 前缀 / 任意 `N/M checks passed`）——日志格式是弱契约，会被嵌套输出骗到。同类坑在本 port 已出现两次（另一次是 `$NF` 取 NDCG 列，见 [[eval-log-ndcg-column-trap]]）。

### 15.8 下游（C6 起）必须知道的边界

1. `L_csg` **只回传码本（P1）**；要让 ST 把梯度带给 encoder 是**新决策**，当前边界下做不到而不越界。
2. **开关默认 0**；`csg=None`（所有既有调用点）时新增表达式与改前**求值完全相同**。
3. **空 pair（数据结果）与开关关闭（接线错误）严格分离**；trainer 在加进 `loss` 前检查 `requires_grad`。
4. 新增训练入口只有 `tiger_ext/qspace_train.py`；`rqvae/main.py` 未动；E9 的 `tokenizer_phase` 仍以 `csg=None` 构造 trainer。
5. **CSG 只属 Tokenizer 阶段**（它是 `L_RQVAE` 的一项，E7 契约已限定）⇒ **E7 无需任何改动**。
6. **batch 几何是硬约束**：首 batch 行数必须让 k-means 初始化"花掉"（batch 1024 guard-clean，256 不 clean）。
7. `rqvae/trainer.py` 现在**是**被修改的 tracked 文件 ⇒ 任何以 `PRE_*_MODIFIED` 为判据的既有 harness 都会报 guard FAIL。**这是授权状态，不是回归**（见 §15.5，并在 [status.md](status.md) 的「已知问题」登记）。

**⚠️ 无效果结论**：C5 只证明"该约束**真的接进了 backward**、梯度只到码本、关掉即与原 TIGER 逐位等价、既有门只有授权 guard 一条偏离"。**不含任何推荐效果结论**（1.55%–155% 只是**尺度读数**）。**EXP-C6 已于 R12 通过；`lambda_csg` 工作点与全部 B / C 类效果实验仍为 `PLANNED`。**

---

## 16. 修订记录 R12 — C6 实现（CSG smoke test，2026-10-15）

**判定**：**EXP-C6 63/63 checks passed，exit 0**（`EXP_C6_TOYS=all EXP_C6_REGRESSION=all`，2026-10-15；CPU）。

**定位**：全链路 smoke test + 最小工作点验证，**不是**效果实验。四个问题：① 四个 λ 能否跑通完整链；② CSG 是否真的进入训练；③ 完整 pipeline 是否被破坏；④ `kmeans_init` 哪个更安全。

**实验矩阵（不新增 λ、不做超参搜索）**：
| 臂 | λ | 数据 | 结果 |
|---|---|---|---|
| A | 0 | 合成 fixture | 全链 exit 0，3.3s |
| B | 1e-4 | 合成 fixture | 全链 exit 0，2.2s |
| C | 1e-3 | 合成 fixture | 全链 exit 0，2.2s |
| D | 1e-2 | 合成 fixture | 全链 exit 0，2.2s |
| — | 0 | Toys（真实 4096d） | 全链 exit 0，65s |
| — | 1e-3 | Toys（真实 4096d） | 全链 exit 0，72s |

**§16.1 组合而非改写**：tokenizer 段走 C5 的 `train_tokenizer(hook=…)`；其余三段走 `TigerAlternatingPipeline`。**`alt_pipeline.py` 零改动**（否则会重开 E9 并破坏 C1–C4 的 sha256 pin）。`phase_log` 只有 `refresh / recommender / evaluation` 三条是有构造原因的，tokenizer 段的证据另行给出（ckpt + steps + 有限 loss）。

**§16.2 CSG 进入训练的证据**：λ>0 各臂 `n_calls == n_steps`（合成 3/3、Toys 24/24）且 `n_nonempty == n_calls`、`n_zero = 0`；`L_total(1e-3) − L_total(0) = 8.114338e-03`；梯度可加性 `7.276e-12`（相对 9.245e-08）；码本张量 3/3 随 λ 变、**encoder 张量 0/6 变**（P1：仅码本）；无 NaN/Inf。λ=0 臂 `n_calls = n_disabled_calls = 0`。

**§16.3 λ=0 逐位等价（全链级 G0）**：`λ=0` 的 hook 与 `hook=None` 在 SID、tokenizer `state_dict`（15/15）、recommender `state_dict`（29/29）、`recalls`/`ndcgs` 上**全部逐位相同**（`torch.equal` / `np.array_equal`，非 `allclose`）。

**§16.4 pipeline 完整性**：E7 stage manifest 齐备；tokenizer 组全冻结 / recommender 组全可训；SID manifest 按哈希绑定 tokenizer ckpt；handoff `{sia: 0, psa: 0}`；历史机制 `L_trans` / `L_balance` / `L_align` 各 **0 次**（负对照触发）；**SIA/PSA mount 未被进入**（forward 计数 0，负对照证明计数器活着）。

**§16.5 kmeans_init 裁决（Q4）**：
| 数据 | kmeans_init | rebuild | latent | step0 \|dW\| | 首 epoch distinct | max_share | safe |
|---|---|---|---|---|---|---|---|
| 合成 | True | 2.384e-07 | 1.435e-01 | 1.184e-04 | 128/128/128 | 0.041 | ✅ |
| 合成 | False | 6.072e-07 | 4.703e+00 | 8.190e-02 | 22/20/65 | 0.321 | ✅ |
| Toys | True | 2.980e-08 | 4.211e-02 | 1.105e-04 | 256/256/240 | 0.146 | ✅ |
| Toys | False | 6.566e-08 | 1.101e+00 | 9.036e-03 | 42/46/256 | 0.523 | ✅ |

**裁决 = 维持冻结配方 `kmeans_init=True`**：batch 1024 下两者都过 C3 的两条首 batch guard，但 `True` 的码本覆盖率显著更好（`False` 的 level 0 只用掉 42/256）。**未引入新初始化机制、未改 RQ-VAE 核心、未改 batch size。**C5 的退化情形是 **batch 几何**性质（batch 256 ⇒ `x_q == z` 逐位），继承的约束是"每个首 batch 都要量"。

**§16.6 真实数据臂与 confound**：真实臂只能是 Toys（C1/C2 产物仅 Toys/Sport/Beauty；MI 两者皆缺）。Toys 2-epoch 4096d tokenizer 的 **collision rate ≈ 0.9225（λ=0）/ 0.9233（λ=1e-3）**，**Recall/NDCG 全为 0** —— 属 status.md「已知问题」第 3 条的 4096d 欠训练 confound，**与 CSG 无关**，不得据此对约束作任何判断。合成臂的 eval 在 1024 item / 1-epoch T5 下同样是退化的（全 0），其逐位等价的意义是"两边同为 0 的逐位相同"，不是"评估有效"。

**§16.7 regression**：`[10]` 在 C6 内重跑 `exp_c5_csg_integration.py` → **99/99 checks，exit 0，0 FAIL**；该一次重跑传递覆盖七份更早 suite 的 R1/R2/R3 逐行归因。

**§16.8 未复核项**：C6 自己的 harness **不修改**任何既有 harness / E7 / E10，也**不通过改守卫去凑 PASS**；C5 授权修改引起的既有 harness 的 tracked-file guard FAIL 状态**由 C5 的 §[10] 负责归因，C6 不重述**（C6 只机器核验"tracked 改动集合没有扩大"）。

**⚠️ 无效果结论**：C6 的 Recall/NDCG 仅证明"CSG 开启后完整 TIGER pipeline 可正常运行"。**不得**作为 CSG 有效/无效的论据；`lambda_csg` 工作点选择属 C6 之后的效果实验。

---

## 17. 修订记录 R13 — 效果实验设计（2026-10-18）

**触发**：C6 smoke PASS（R12）后工程侧收口，按 §5.2 进入 B / C 类。本节记录**设计的三处结构改动**与**三条设计前核实的事实**。**本节不含任何效果结果。**

### 17.1 三处结构改动

| # | 改动 | 原状 | 现状 |
|---|---|---|---|
| 1 | **content 口径** | §1.3 的 4096d（Toys/Sport/Beauty） | **768d content-only**，`D_target = 768` |
| 2 | **矩阵结构** | §7「建议 5 组并列」 | **2×2×2 全因子 8 格**（C × E × A），旧 I1–I4 降为别名 |
| 3 | **λ 工作点** | 无登记（`λ_sia = λ_psa = 0.1` 是实现默认值） | **预注册的「先扫后定」**（EXP-L0，5 路 × Toys × seed 2025） |

### 17.2 三条设计前核实的事实（均来自真实产物，不是推断）

| 事实 | 证据 |
|---|---|
| **768d 彻底解除 4096d 的 Toys blocker** | 768d collision：Toys **0.000419** / Sport 0.000545 / Beauty 0.001653；对照 C6 的 4096d Toys **0.9225**、Recall/NDCG 全 0 |
| **768d 基线 9/9 已存在** | `model/ckpt/tiger_content768d_baseline_{toys,sport,beauty}_seed{2025,2026,2027}.pth`；实测 N@20(seed2025) = 0.0341 / 0.0274 / 0.0368 |
| **`alt_pipeline.py` 不在任何源码 pin 里** | C4/C5 的 `PINNED_SHA16` 只钉 `cf_embed` / `collab_relation` / `qspace_constraint` / `stage` / `psa_loss` / `sia_loss` / `qspace_loss`；`tiger_ext/` 未被 git 跟踪 ⇒ **P1 不会重开 E9/E10 守卫** |

> ⚠️ **一处口头判断的更正**：设计讨论中曾说"`alt_pipeline.py` 被 C3/C4/E9 的 sha256 pin 钉住，改它会重开 E9"。**这是错的**——那些 sha256 是**产物**（ckpt / SID）哈希，不是源码 pin。实际代价是"改后重跑 EXP-E9 / EXP-E10 确认默认路径不变"，不是"守卫变红"。

### 17.3 两条新增接线

| 编号 | 内容 | 状态 |
|---|---|---|
| **P1** | `TigerAlternatingPipeline` / `tokenizer_phase` 增加可选 `csg=` 透传 | ✅ **ACCEPTED（2026-10-18，见 §17.6）** |
| **P2** | 新驱动 `tiger_ext/fixed_train.py`（冻结 tokenizer + `model/main.py::train` 单次多 epoch + `SIAMount`/`PSAMount`） | ✅ **IMPLEMENTED（2026-10-18，待验收，见 §17.6b）**；启动门禁 **CONDITIONAL**（§17.6c 差分回归；§17.6d 修后配方核对；**B6 已修 §17.6e、B7 已修 §17.6f、R1 已审计关闭 §17.6g、R2 已冻结关闭 §17.6h**，剩 R3） |

**P2 的必要性**：不能复用 `recommender_phase` 连调 200 次——每 phase 是独立 Trainer（重建 optimizer + linear schedule），200 × 1 epoch ≠ 一次 200-epoch 训练，训练体制不同。

### 17.4 预算等价规则（本设计最容易被忽略的一条）

T5 预算 8 格严格相同（200 ep + early_stop 10）；A=1 用 `n_cycles = 200` 对齐。**这是"同 T5 预算 + 不同 tokenizer 调度"，不是"同 tokenizer 预算"**（A=1 额外 200 ep 精化）。

由此的**预注册预期**：A=1 且 C=0 时 tokenizer 目标与起点均与 A=0 相同 ⇒ **#4 预期 ≈ #1**；A 的增益只可能经 `A×C` 出现。若 SID diff ≈ 0 ⇒ 判 A 惰性、ALT 降级，**不得改判据**。

### 17.5 未改动项

- `clean_tiger/`（永久只读）、`model/`、`rqvae/`、`tools/`、`doc/` **零改动**。
- E9 / E10 / C1–C6 的既有 harness **未改**。
- **ST→encoder 路由仍为 P1（梯度只到码本）**，不因效果实验而改 objective。

### 17.6 P1 实现与验证（2026-10-18，**ACCEPTED**）

**改动面**：只动 `tiger_ext/alt_pipeline.py`（未跟踪文件），9 hunks / 净 +76 行（626 → 702），
`sha256[:16] = fc1b8652a72c04ce`。RQ-VAE 核心、`rqvae/trainer.py`、`model/` **零改动**。

| 项 | 内容 |
|---|---|
| 生产文件 | `tiger_ext/alt_pipeline.py`（未跟踪）｜`sha256[:16] = fc1b8652a72c04ce` |
| 测试 harness | `integration/exp_p1_csg_passthrough.py`｜`sha256[:16] = dd267ea07c145356` |
| 新参数 | `TigerAlternatingPipeline(..., csg=None)`，末位关键字 |
| 透传点 | `tokenizer_phase` 的 `RQVAETrainer(args, rqvae, len(loader), csg=csg)` |
| 配套（必需） | 仅当 `csg.enabled` 时 `EmbDataset(..., return_index=True)`；否则 trainer 的 `batch_indices is not None` 守卫会**静默丢弃** `L_csg` |
| 记录 | `result["csg"]` / `run_config()["csg"]`，且**只在挂载 hook 时出现** ⇒ `csg=None` 时输出字典逐键与 P1 之前相同 |
| 新守卫 | 开启 hook 但数据无行索引 / hook 未到达 trainer / 非 hook 对象 / 带参数的 hook —— 四种都抛错而非静默 |

**EXP-P1（`integration/exp_p1_csg_passthrough.py`，CPU，27/27 通过）**：

| 断言 | 结果 |
|---|---|
| `csg=None` 走的是 pre-P1 调用点（`EmbDataset(path)`、3 参 `Trainer`） | PASS（shim 断言 `return_index=False`、`csg is None`） |
| `csg=None` 与 pre-P1 逐位等价 | PASS：**23/23 张量 `torch.equal`**、`best_loss` 相同、collision 相同、结果键集与 `run_config` 键集相同 |
| `lambda_csg=0` 的 hook 惰性 | PASS：与 `csg=None` 逐位相同、`n_calls=0`、`n_disabled_calls=0` |
| enabled hook 确实到达 trainer | PASS：`attached=True`、`return_index=True`、`n_calls=4 = n_batches`、constraint/mount 各 4 次 |
| enabled hook 真的起作用 | PASS：4/4 步产出非零且 `requires_grad` 的 `L_csg`，state_dict 与 `csg=None` **不同** |
| 负向控制 | PASS：非 hook ⇒ `TypeError`；带参数 hook ⇒ `ValueError`；构造期即拒绝 |

#### 门禁口径（差分回归，**所有内核外改动共用**；权威表述见 plan §19.7）

| 项 | 规则 |
|---|---|
| 基线 | **C5 之后、本次改动之前**的同一套 suite 结果（同 commit / 同 device / 同依赖） |
| 判定 | 改动后检查项**逐条与基线相同**（判定 + detail；仅时间戳、临时目录名可不同）⇒ 无新增失败 |
| 禁止 | 以历史绝对分数（EXP-E9 90/90、EXP-E10 64/64）作通过条件 —— 那是 **C5 之前**的读数，已不可达 |
| 禁止 | 表述为"全部测试通过"；基线中既有失败必须原样列出并标注归属 |

**差分回归结果（P1 前 vs P1 后）**：

| suite | 基线 | P1 后 | 逐项比对 | 新增失败 |
|---|---|---|---|---|
| EXP-E9 | 87/90 | **87/90** | **90/90 检查行逐字节相同**（含 detail） | **0** |
| EXP-E10 | 60/64 | **60/64** | 62/64 行逐字节相同；2 行仅差临时目录名（#4，仍 PASS）与进度条时间戳（#62，仍 FAIL、数值同为 80/81） | **0** |

**基线中既有的 7 项失败（原样列出；归属 C5，不属于 P1，本任务不修复）**：

| suite | 失败项 |
|---|---|
| E9 + E10 | `model/ and rqvae/ carry no E9/E10 change -- modified: model/main.py, rqvae/models/rq.py, rqvae/models/vq.py, rqvae/trainer.py` |
| E9 + E10 | `exp_e8_sid_refresh: 54/55` |
| E10 | `exp_e9_alternating: 80/81` |
| E9 | `(12) the E1-E8 suites are unaffected by E9 (329/330)` |
| E10 | `the E1-E9 suites are unaffected by E10 (409/411)` |

> ⇒ **P1 的验收结论 = 零新增失败**，不是"全绿"。上表在 P1 之前即存在。

**已知限制**：① 差分回归只在 **CPU** 复跑，CUDA 未复跑；② 基线是**逐字还原**的 pre-P1 文件（30227 字节，与改动前一致），不是 git 存档（`tiger_ext/` 未被跟踪）；③ `csg` 记录字段"仅挂载时出现"是不对称设计，下游需接受 `config.get("csg") is None`；④ ~~P2 未实施，格 #3 / #5 仍无执行路径~~ → **2026-10-18 P2 已实现，见 §17.6b**。

### 17.6b P2 实现与验证（2026-10-18，**IMPLEMENTED，待验收**）

**改动面**：**纯新增** `tiger_ext/fixed_train.py`（未跟踪，628 行，`sha256[:16] = b3ec855be760bafc`）
+ 新增 harness `integration/exp_p2_fixed_train.py`（675 行，`sha256[:16] = 555ef348da5e0e24`）。
`model/`、`rqvae/`、`clean_tiger/` 与既有 harness **零改动**（mtime 仍为 2026-10-14），且
`grep -rn fixed_train model rqvae clean_tiger tools` 为空 ⇒ 无既有模块依赖它。

| 项 | 内容 |
|---|---|
| 入口 | `FixedSidTrainer(...).run()`；CLI `python -m tiger_ext.fixed_train`（`--rqvae_config` 接受 JSON overrides） |
| 固定 tokenizer/SID | `StageFreeze.apply("recommender")` 冻结 RQ-VAE + 严格封存码本；SID 只读一次并 `validate_sid`；**每 epoch 后**复核 digest / sha256 / `initted` |
| 多轮训练 | 单一 `Adam`（T5 ∪ 两 adapter，与 RQ-VAE 零交）+ 每 epoch `model/main.py::train` → valid `evaluate` → 改进时 test + `best.pth` → `early_stop=10` |
| ETE | `λ>0` 才构造 `SIAMount`/`PSAMount`；`λ=0` 传 `sia=None, psa=None`（Clean TIGER 分支）；不改任何损失/温度/优化策略 |
| CSG | 构造期**拒绝** `csg=`（`FixedSidError`）；格 #5 只能经 CSG-trained ckpt + 其 SID 进入 |
| 落盘 | `best.pth` / `last.pth` = 纯 T5 `state_dict`（`tools/eval_trained_model.py` 可 strict load）；adapter 单独存 `{best,last}.{sia,psa}_adapter.pth` |

**EXP-P2 = 47/47（CPU）**：SID 逐字节不变 + tokenizer digest 逐位不变（每 epoch 复核）；
`train` 每 epoch 恰好一次、T5 权重确实移动；`λ=0` 两 mount 不构造且 `train()` 收到 `None`、
adapter 逐位不动，SIA-only 时 **PSA adapter 逐位不动**、both 时两者都动；
`λ=0` 一次运行与手工重放 `model/main.py::__main__` 循环**逐位相同**（50/50 张量 + 每 epoch loss 一致）；
`recommender_phase` 被替换成抛错仍能跑完；改进分支实跑到达（小词表 split NDCG 0→0.6309，
test 评估 4 次 = 改进 4 次，`best.pth` ≠ `last.pth`）；CLI 端到端 exit 0 并写 `run.json`。

**已知限制**：① 全部验证在 **CPU**，CUDA 未复跑；② 小词表 split 用 `batch_size=16` 仅为凑梯度步，正式臂仍 `T5_BATCH=256`；③ 无改进时只写 `last.pth`（与 `model/main.py` 一致）；④ `tiger_ext/__init__.py` 未改（不 re-export）；⑤ 未启动任何真实效果实验；⑥ 纯新增文件 ⇒ 不触发 P1 那套差分回归。

### 17.6c P2 后启动门禁（2026-10-18，**CONDITIONAL**）

任务边界：只做门禁检查，**不跑 EXP-L0 / EXP-B1**，不改模型/训练逻辑/损失公式。完整证据见 [plan.md §19.11](plan.md)。

#### (1) 差分回归（EXP-E9 / EXP-E10）

| suite | 本次 | 基线（C5 后 / P1 前） | 失败项 | 新增失败 |
|---|---|---|---|---|
| EXP-E9 | **87/90** | 87/90 | 3/3 相同 | **0** |
| EXP-E10 | **60/64** | 60/64 | 4/4 相同 | **0** |

失败项与 §17.6 记录的 7 项**逐条一致**（tracked-file guard ×2、`exp_e8_sid_refresh 54/55`、`exp_e9_alternating 80/81`、嵌套 `329/330` 与 `409/411`），全部归属 **C5**。原始日志：`integration/artifacts/p2_gate/p2gate_{e9,e10}.log`。
`tiger_ext/alt_pipeline.py` 的 `sha256[:16]` 仍为 `fc1b8652a72c04ce`（与 P1 验收记录相同）⇒ P2 之后没有代码变化。

> ⚠️ **诚实标注**：磁盘上没有 P1 前的基线日志文件 ⇒ 本次是**按总数 + 失败项身份 + 嵌套计数 + 代码哈希**核对，**不是逐字节 diff**。P1 验收那次才是逐字节比对。

#### (2) 格 #5 链路（CSG ckpt → 其 SID → 固定 SID 的 ETE 训练）

实测（CPU）：`refresh_sid` 从 768d 基线 ckpt 生成 SID（collision 0.000419、`ok=True`）→ sidecar manifest `verify_sid_manifest` 通过（`ckpt 8069f998c144` / `sid bb2693a6c71e`）→ `FixedSidTrainer` 用该 ckpt+SID 跑 1 epoch（λsia=λpsa=0.1）：**routing 0/11924 mismatch**、`tokenizer_unchanged=True`、`sid_unchanged=True`。
当时 `FixedSidTrainer` **不校验** SID manifest 与 ckpt 是否同源（错配 ckpt 未被拒）⇒ 阻塞 **B4**；**2026-10-19 已修复**：训练器在构造期与 `run()` 起点复用 `sid_refresh.verify_sid_manifest`，错配在训练前抛 `SidProvenanceError`；缺 manifest 默认硬失败，显式 opt-out 才放行并记录（EXP-B4 38/38，见 [plan.md §19.11.8](plan.md)）。

#### (3) EXP-L0 五臂运行清单（澄清，不改设计）

| # | 臂 | C | E | λ_csg | λ_sia | λ_psa | tokenizer | SID |
|---|---|---|---|---|---|---|---|---|
| 1 | `λc-1e-4` | 1 | 0 | 1e-4 | 0 | 0 | 新训 3000 ep CSG | 该 ckpt refresh |
| 2 | `λc-1e-3` | 1 | 0 | 1e-3 | 0 | 0 | 新训 3000 ep CSG | 该 ckpt refresh |
| 3 | `λc-1e-2` | 1 | 0 | 1e-2 | 0 | 0 | 新训 3000 ep CSG | 该 ckpt refresh |
| 4 | `λe-0.1` | 0 | 1 | 0 | 0.1 | 0.1 | 冻结 768d 基线 ckpt | `Toys_content768d_baseline_sid.npy` |
| 5 | `λe-1.0` | 0 | 1 | 0 | 1.0 | 1.0 | 同臂 4 | 同臂 4 |

固定条件：Toys × T5 seed 2025（RQ-VAE seed 2024）、768d content-only、T5 200 ep + early_stop 10、`lr 1e-4 / batch 256 / infer 96 / beam 30 / topk 5,10,20`、**五臂 A=0**、RQ-VAE 配方逐字继承 [plan.md §19.1](plan.md)（含 `warmup_epochs=50`、`align_weight=0.0`）。
**澄清**：三个 λc 臂各有自己的 tokenizer 与 SID ⇒ Stage 0bis 是 **3 次 3000-ep 训练**（原 §19.9 的"1 个"已改）。

#### (4) 阻塞项

| # | 阻塞 | 影响 | 状态 |
|---|---|---|---|
| **B1** | `qspace_train` 硬编码 `item_emb.parquet`（Toys 上为 4096d），无 CLI override | λc 三臂会用错表 ⇒ 违反 768d 冻结 | ✅ **已修 2026-10-18**（`--embedding-path` + `resolved/validate_embedding_path`） |
| **B2** | `qspace_train` 无法设 `warmup_epochs`（默认 0，配方要 50） | λc 三臂配方不符 ⇒ 违反 §19.10 门禁第 3 条 | ✅ **已修 2026-10-18**（`--warmup-epochs`，默认 0） |
| **B3** | 历史 ckpt 的 `data_path` 是仓库相对路径，而生成器以 `cwd=rqvae/` 跑 | 只影响复算历史 ckpt 的 SID（`qspace_train` 产物用绝对路径，不受影响） | ✅ **已修 2026-10-18**（`resolve_data_path` + 临时绝对化副本，见下） |
| **B4** | `FixedSidTrainer` 不校验 SID manifest ↔ ckpt 同源 | 格 #5 lineage 只靠驱动自律 | ✅ **已修 2026-10-19**（构造期 + `run()` 起点校验；缺 manifest 默认失败 + 显式 opt-out） |
| **B5** | `qspace_train.py` 无 `if __name__ == "__main__"` 守卫 | 文档化 CLI 惰性（`python -m` exit 0 零输出）；须程序调用 `main(argv)` | ✅ **已修 2026-10-18** |

**B1/B2/B5 修复证据**：[plan.md §19.11.6](plan.md)；harness `integration/exp_p3_qspace_cli.py`（`sha256[:16]=2faddf50c189df48`）**EXP-P3 = 29/29**，CPU + 合成数据 + Toys 1-epoch smoke，未启动 EXP-L0。只改 `tiger_ext/qspace_train.py`（`ea224a1d56e8c5eb` → `e10f24cf0378a816`），其余生产文件零改动。
**实测传参链路**：`--warmup-epochs 50` → `CSGTrainerConfig` → `namespace()` → `rqvae_namespace(..., warmup_epochs=)` → `args.warmup_epochs` → `trainer.py:33 warmup_steps=50×data_num` → `:82` scheduler → 末步 LR `2.000e-05`（warmup=0 时为 `0.000e+00`）⇒ 真进优化器。
**新登记小坑**：`sid_refresh.ckpt_args` 按 `_CKPT_ARG_KEYS` 白名单过滤，**不含 `warmup_epochs`**（ckpt 里本身有）；审计须用未过滤的 `torch.load(...)['args']`。

**B3 修复证据**：[plan.md §19.11.7](plan.md)；只改 `tiger_ext/sid_refresh.py`（`6c54fab7c4ce6fa9` → `983121cb3d9cd17b`），**未改 `rqvae/`、未加符号链接、不依赖调用方 cwd**。语料实测 708 个 ckpt：`data/...` 627、`../data/...` 61、绝对 20 ⇒ 两种相对约定都要处理。`resolve_data_path` 只试 `repo/<p>` 与 `repo/rqvae/<p>` 两个固定根，缺失/歧义一律报错；`run_generator` 传给冻结生成器的是一份**只改 `args.data_path` 的临时副本**（权重 23/23 张量逐位相同，两种形态生成的 SID 逐位相同）。harness `integration/exp_b3_data_path.py`（`44a7234aa5164789`）**EXP-B3 = 30/30**；回归 **EXP-E8 = 54/55**（唯一失败仍是 tracked-file guard，零新增失败）、**EXP-P3 = 29/29**。已知限制：11 个 `../data/Sports/...` 历史 ckpt 因原表已随数据集改名消失而**响亮拒绝**（不跨表猜）。

**B4 修复证据**：[plan.md §19.11.8](plan.md)；只改 `tiger_ext/fixed_train.py`（`b3ec855be760bafc` → `8fde074584fe443f`），`sid_refresh.py`（`983121cb3d9cd17b`）、`model/`、`rqvae/`、`clean_tiger/` 零改动。`verify_sid_provenance()` 复用 `verify_sid_manifest`（错 ckpt / 错 SID 都抛 `SidProvenanceError`），在 `__init__` 与 `run()` 起点各校验一次。**缺 manifest 默认硬失败**；显式 `allow_unregistered_sid=True`（CLI `--allow_unregistered_sid`）才放行并在 `run.json` 记 `mode="unregistered"`，versioned 文件名仍校验 `ckpt_sha8`；`--sid_manifest` 覆盖 publish 只复制 `.npy` 的场景。记录的合法无 manifest 场景 = R13 λe 两臂的 legacy `Toys_content768d_baseline_sid.npy`（建议用 `refresh_sid` 复算）与 publish 目标路径。harness `integration/exp_b4_provenance.py`（`1e02a89004c508e4`）**EXP-B4 = 38/38**（含真实 768d ckpt + 真 refresh manifest 的接受与另一真实 tokenizer 的拒绝）；回归 **EXP-P2 = 47/47**、**EXP-E8 = 54/55**（零新增失败）、**EXP-B3 = 30/30**。**不是效果证据**（仅 1-epoch 机制 smoke）。

**判定**：门禁第 2 条 ✅；第 3 条原 ❌ 的**直接理由（B1/B2）已消除**，且 B3/B4 的 lineage 缺口已修复。**修后逐项核对已于 2026-10-19 完成（§17.6d）**：λc 配方通过，但推荐器侧发现 **B6 硬阻塞** ⇒ **GATE = CONDITIONAL（维持）**，EXP-L0 仍不启动。**2026-10-19 晚：B6 已修复（§17.6e）**，推荐器侧改由显式 `--embedding_path` 进入；**同日 B7 已修复（§17.6f，EXP-B7 48/48）** ⇒ 剩余阻塞为 R1/R2/R3；**R1 已于同日审计关闭（§17.6g，EXP-R1 44/44）**。

### 17.6d 修后配方逐项核对（2026-10-19，**GATE = CONDITIONAL，B6**）

任务边界：只核对，**未启动 EXP-L0 / EXP-B1，未改任何生产文件**（`fixed_train.py e10f24cf…` 以外的四个 `tiger_ext` 文件哈希与 §17.6c 记录相同；`fixed_train.py` 仍为 B4 后的 `8fde074584fe443f`）。完整逐项表与命令见 [plan.md §19.11.9](plan.md)；原始产物 `integration/artifacts/r13_gate/`。

#### (1) 通过的项

| 项 | 实测证据 | 结论 |
|---|---|---|
| λc RQ-VAE 配方 | 正式 CLI 解析出的 namespace 与基线 ckpt `args`：29 个字段中 **23 个逐值相同**；`ckpt_dir` 为 run-specific、`data_path` 仅绝对/相对写法（同一文件）；`tau/lambda_trans/lambda_balance/trans_graph_path` 缺键⇒`getattr` 取冻结默认 | ✅ |
| warmup=50 / 3000 ep / 768d | `torch.load(ckpt)['args']`（未过滤）实测 `warmup_epochs=50`、`num_workers=4`、`batch_size=1024`、`eval_step=50`、`data_path` 为绝对 768d；正式解析 `epochs=3000` | ✅ |
| λc → SID 同源 | 1-epoch CSG smoke（`λ=1e-4`，hook 12/12）⇒ `refresh_sid` 成功（11924 行、collision 0.00763、`ok=True`）⇒ `verify_sid_manifest=True` | ✅（机制 smoke） |
| λe 冻结 SID | 从同一 ckpt 在 **cuda:0** 复算 ⇒ 与 `Toys_content768d_baseline_sid.npy` **sha256 完全相同**（`bb2693a6…`，`rows_changed=0`）；已写 sidecar 登记；`.npy` 未变 | ✅ |
| B4 默认策略 | 登记前无 sidecar ⇒ `SidProvenanceError`；**未使用** `--allow_unregistered_sid` | ✅ |
| T5 共同配置 | 运行时默认抓取：Toys / seed 2025 / 200 ep / early_stop 10 / lr 1e-4 / batch 256 / infer 96 / beam 30 / `topk=[5,10,20]`；协议与 `model/main.py` 逐行对应（EXP-P2 已证 λ=0 50/50 张量逐位相同） | ✅ |
| 产物与记录 | `run_dir/{best,last}.pth`、adapter 各两份、`run.json`（config+result）、best 的 `.stage.json`；SID/ckpt 哈希与 provenance 均在 `run.json` | ✅ |

#### (2) 新阻塞 B6（推荐器侧写死 4096d 表）

`FixedSidTrainer` 仍写死 `data_root/<dataset>/item_emb.parquet`（Toys = **4096d**），用真实基线 ckpt 构造时报：

```
size mismatch encoder.mlp_layers.1.weight: checkpoint [512,768] vs model [512,4096]
size mismatch decoder.mlp_layers.13.weight/bias: [768,512]/[768] vs [4096,512]/[4096]
```

⇒ **λc / λe 五臂全部无法开跑**（两边的 tokenizer 都是 768d）。隔离实验：用诊断 shadow root（`item_emb.parquet` 链接到 768d 表）构造成功 —— `in_dim=768`、provenance `verified=True`、两 mount 均活、tokenizer 23/23 冻结、optimizer 4,694,048 参数 ⇒ 唯一变量就是 embedding 表路径。**shadow root 仅诊断，不是批准的运行方式。** 最小修正 = 给 `FixedSidTrainer` 加 `embedding_path`/`--embedding_path`（默认不变，与 B1 同构），属生产改动，待批准。

#### (3) 其他未决项

| # | 项 | 处理 |
|---|---|---|
| ~~B7~~ ✅ **已修复（§17.6f）** | `qspace_train` 曾不播种 RNG；`--seed 2024` 仅落盘 | 已由 `apply_seed` + `seeded_kmeans` 关闭（EXP-B7 48/48） |
| R1 | 新 ckpt `args` 缺 `tau/lambda_trans/lambda_balance/trans_graph_path`（有效值由 `getattr` 取冻结默认） | 补键（值不变）或写入实验记录 |
| ~~R2~~ ✅ **已关闭（§17.6h）** | 两套 CLI 默认值均不等于 EXP-L0 要求（qspace_train 的 epochs/batch/eval/warmup/embedding/workers/device/ckpt-dir；fixed_train 的 λ） | 只用 §19.11.13 的五条冻结命令（EXP-R2 274/274），禁止依赖默认值 |
| R3 | 尚无 EXP-L0 driver | 启动前落 driver，固定五条命令与输出目录 |

#### (4) 判定

**GATE = CONDITIONAL**：第 2 条 ✅；第 3 条 λc 侧 ✅ 但推荐器侧 **B6 不通过**，另有 B7/R1/R2/R3。**不得启动 EXP-L0。** 所有运行均为 1-epoch / 构造级机制 smoke，**不是效果证据**，不能用于选点。（B6/B7 后已于同日修复，见 §17.6e / §17.6f；R1 亦已审计关闭，见 §17.6g；本条保留为当时的判定。）

> **后续（2026-10-19 晚）**：B6 已按批准的最小方案修复，见 §17.6e；同日 B7 亦已修复，见 §17.6f。本节 B6/B7 相关行由两节取代；R1 亦已审计关闭（§17.6g），R2 亦已冻结关闭（§17.6h）。GATE 仍 CONDITIONAL（仅 R3）。

### 17.6e B6 修复：显式 embedding 表（2026-10-19，**GATE 仍 CONDITIONAL**）

只改 `tiger_ext/fixed_train.py`（`8fde074584fe443f` → **`389e90854b5f6f13`**，808 → 924 行）并新增 harness `integration/exp_b6_embedding_path.py`；`model/`、`rqvae/`、`clean_tiger/`、`sid_refresh.py`、`qspace_train.py` 零改动，未启动 EXP-L0/EXP-B1，未处理 B7。

**改动**

1. `FixedSidTrainer(..., embedding_path=None)` + CLI `--embedding_path`；`None` 严格等价旧行为（`data_root/<dataset>/item_emb.parquet`），显式路径按给定文件读取、无回退。
2. 构造期新增宽度门禁：`tokenizer_input_dim(ckpt)` 从 `encoder.mlp_layers.1.weight` 读 `in_dim`，与所读表列数不符即**在建模前**抛 `FixedSidError`（`embedding dimension mismatch: <table> (<explicit|default>) has D columns, but <ckpt> was built with in_dim=E (...)`）。
3. `run.json`（config+result）与 `best.pth.stage.json` 记录 `embedding_path / embedding_path_explicit / embedding_sha256 / embedding_dim / embedding_n_items / tokenizer_input_dim / tokenizer_input_shape`；`_check_frozen` 每 epoch 复核表未被改写，`result` 增 `embedding_unchanged`。
4. B4 未放松：provenance 校验顺序与判据不变（`verify_sid_manifest`，构造期 + `run()` 起点）。

**测试**

| suite | 结果 | 说明 |
|---|---|---|
| EXP-B6（新增，`integration/exp_b6_embedding_path.py`） | **31/31** | 默认兼容 / 显式被真实读取 / 维度错误建模前失败且无 run 目录 / 48d 诱饵默认表不回退 / 错 ckpt 与无 manifest 仍拒 / CLI 三态 / 真实 768d Toys 对（**无 shadow data root**）构造成功且默认 4096d 表被清晰拒绝 |
| EXP-P2 | **47/47** | 机构与清洁路径逐位等价回归 |
| EXP-B4 | **38/38** | provenance 策略回归 |
| EXP-E8 | **54/55** | 唯一失败仍是既有 tracked-file guard（列出 `model/main.py`、`rqvae/{models/rq.py,models/vq.py,trainer.py}`），零新增 |

> 陷阱（已在 harness 注释钉住）：`data/Toys/fused_item_emb_768d.parquet` **实为 896d**；768d baseline ckpt 真正读的是 `item_emb_768d_contentonly.parquet`。首版 [7] 用错表被门禁当场拒绝——正是本修复要防的错配。

**不是效果证据**：全部训练为 1 epoch 合成 fixture 机制 smoke，真实输入组 0 epoch。

**剩余阻塞**：B7（`qspace_train` 不播种 RNG，待裁决）；R1（新 ckpt `args` 缺 `tau/lambda_*` 键）（→ 已由 §17.6g 审计关闭）、R2（CLI 默认值 ≠ EXP-L0 要求）（→ 已由 §17.6h 关闭）、R3（无 EXP-L0 driver）。⇒ **GATE 仍 CONDITIONAL，EXP-L0 不启动。**

### 17.6f B7 修复：seed 真正控制 CSG 训练的随机过程（2026-10-19，**GATE 仍 CONDITIONAL**）

只改 `tiger_ext/qspace_train.py`（`e10f24cf0378a816` → **`305c50a7bf530f53`**，693 → 832 行）并新增 harness `integration/exp_b7_seed_control.py`；`fixed_train.py`、`sid_refresh.py`、`alt_pipeline.py` 与 `model/`、`rqvae/`、`clean_tiger/` 零改动；未启动 EXP-L0/EXP-B1；模型/loss/配方/epochs/R13 定义未改。

**修复内容**

1. `apply_seed(seed)`：`random` + numpy + torch CPU + torch CUDA，且 `cudnn.deterministic=True` / `benchmark=False`（与 `rqvae/main.py:77-84` 同构）；在 `train_tokenizer` 的 dataset/model/lazy-k-means **之前**调用。
2. DataLoader shuffle 使用独立 `torch.Generator().manual_seed(config.seed)`，不再吃全局 torch 流。
3. `seeded_kmeans(seed)`：冻结 `rqvae/models/layers.py:86` 的 `KMeans(...)` 无 `random_state` ⇒ 运行期把该模块的 `KMeans` 名称换成**逐次注入 `random_state`** 的工厂（私有 `RandomState(seed)`），`finally` 还原；**不编辑 `rqvae/`**。上下文覆盖整个 `fit`（k-means 在首 batch lazy 初始化）。冻结的 `num_clusters > B` 子采样 `RandomState(0)` 不变，报告中记为 `kmeans_subsample_random_state: 0`。
4. `report["seeding"]` 15 字段：四条流、`torch_initial_seed`、cuDNN、shuffle 来源/seed、`kmeans_random_state_calls`、注入来源、子采样 seed、`cuda_nondeterminism`。

**CUDA 不承诺逐位复现**：embedding backward atomics、cuDNN kernel 选择、多流/多卡归约、`num_workers>0` worker 到达顺序；**未启用** `torch.use_deterministic_algorithms`（可能中止冻结循环，且不改配方）。CPU 为逐位验证路径。

**测试**

| suite | 结果 | 关键读数 |
|---|---|---|
| EXP-B7（新增） | **48/48** | 同 seed 同进程（两次运行间烧掉 6000 个随机值）：init `aa17a452…`、final `be74cfe…`、codes 220 distinct、`best_loss=0.982522`、3 个 k-means draws 全同；同 seed **跨三进程**（两子进程 + 进程内）`comparable()` 完全一致；异 seed 4321：init `a51be790…`、draws、final、codes（234 distinct）全变，表 sha 相同；KMeans 组先证明冻结调用依赖 numpy 全局流（seed 1 vs 2 中心不同），再证明注入后两段不同全局流中心逐位相同 |
| EXP-P3 | **29/29** | CLI/表/warmup 不受影响 |
| EXP-C4 | **84/87** | 与 C5 后记录的基线逐条相同（modified-set guard + `trainer.py:121` 文本 guard + 内嵌 E10 重跑），**零新增失败** |
| EXP-C6 | **63/63** | 内嵌 EXP-C5 **99/99, exit 0**；七份旧 suite 归因仍 `unexplained: none` |

**不是效果证据**：1 epoch × 320 行合成表机制 smoke，无 Recall/NDCG，不用于 λ 选点。

**残余**：R1（新 ckpt `args` 缺 `tau/lambda_*` 键）（→ 已由 §17.6g 审计关闭）、R2（CLI 默认值 ≠ EXP-L0）（→ 已由 §17.6h 关闭）、R3（无 EXP-L0 driver）⇒ **GATE 仍 CONDITIONAL，EXP-L0 不启动**；B1–B7 已全部关闭。

### 17.6g R1 审计：新 ckpt 缺失训练参数的实际生效值（2026-10-19，**GATE 仍 CONDITIONAL**）

只做核实：**未修改任何生产文件**（`qspace_train.py` / `fixed_train.py` / `sid_refresh.py` / `alt_pipeline.py` 哈希逐字节未变），新增 harness `integration/exp_r1_ckpt_args_audit.py`，审计表 `integration/artifacts/r1_ckpt_args/exp_r1_audit.json`；未启动 EXP-L0 / EXP-B1；未改模型/loss/R13 定义。权威记录 [plan.md §19.11.12](plan.md)。

**缺键范围更正**：不是 4 个而是 **7 个** —— `tau / lambda_trans / lambda_balance / trans_graph_path`（新 ckpt 缺，基线 ckpt 有） + `align_weight / align_tau / align_data_path`（两类 ckpt 都缺）。namespace 构造（`alt_pipeline.py:153-163`，26 键）与 `trainer.py:444` 的 `"args": self.args` 原样落盘决定了这一点。

**实际生效值**（用 ckpt args 构造**真实 `RQVAETrainer`** 后读取，不是 `getattr` 推演）：

| 字段 | 新 ckpt 有效值 | 基线 ckpt 有效值 | R13 冻结值 | 判定 |
|---|---|---|---|---|
| `tau` | 0.1（`contrastive_loss.tau=0.1`） | 0.1 | 0.1 | ✅ |
| `lambda_trans` | 0.0（`trans_dict=None`） | 0.0 | 0.0 | ✅ |
| `lambda_balance` | 0.0 | 0.0 | 0.0 | ✅ |
| `trans_graph_path` | None | None | None | ✅ |
| `align_weight` | 0.0（`align_proj=None`） | 0.0 | 0.0 | ✅ |
| `align_tau` | 0.1 | 0.1 | 0.1 | ✅ |
| `align_data_path` | None（`align_table=None`） | None | None | ✅ |

三份 ckpt（当前代码现产 / `lc_smoke` warmup=50 / 冻结 768d 基线）**21/21 全部等于冻结值**。默认出处：`trainer.py:57,58,63,86,87,88,100`；其默认值与 `git show 5edfca0:rqvae/main.py:52-68` 的 argparse 默认值 **7/7 逐值相同**（AST 提取）。必读的 11 个直读字段在新 namespace 中全部存在。

**"安全默认" vs "不确定"的判别证据**：① 注入 `lambda_balance=0.7` / `tau=0.07` → Trainer 实际取到 0.7 / 0.07（有键必用）；② `lambda_trans=0.7` + 坏路径 → 静默降为 0.0（另一类失败，R13 不走）；③ 删除直读字段 `warmup_epochs` → `AttributeError`，删除可选字段则正常（必需/可选由代码强制）；④ 新 ckpt 上 `hasattr` 为 False 而 `getattr` 得 0.0。

**下游**：`generate_code.py` 不读这 7 项（模型结构键全有）；`fixed_train.py` 从不读 tokenizer ckpt 的 `args`；`sid_refresh.ckpt_args` 白名单 11 键不含它们（仅 provenance 记录缺口）；全仓 34 个 ckpt-args 读取文件中 0 个触碰这 7 项。

**残余观察（不阻断）**：`lambda_csg`（处理变量）不在 ckpt args，仅在运行报告 —— 建议 R3 driver 落记录；`sid_refresh` 白名单亦不含 `warmup_epochs`；基线 ckpt 缺 `seed`（白名单按 `hasattr` 记录，不报错）。

**测试**

| suite | 结果 | 说明 |
|---|---|---|
| EXP-R1（新增） | **44/44**，`CLOSED-BY-AUDIT` | 含正反控 |
| EXP-P3 | **29/29** | CLI → namespace |
| EXP-P2 | **47/47** | 冻结 Trainer 重放 |
| EXP-B6 | **31/31** | tokenizer 宽度/显式表 |

**不是效果证据**：唯一训练为 1 epoch × 320 行合成表，无 Recall/NDCG。

**结论**：**R1 关闭（审计即可，不改生产代码）**。**GATE 仍 CONDITIONAL**：剩 **R2**（CLI 默认值 ≠ EXP-L0）（→ 已由 §17.6h 关闭）、**R3**（无 EXP-L0 driver）⇒ EXP-L0 不启动。

### 17.6h R2 关闭：EXP-L0 五条冻结命令（2026-10-19，**GATE 仍 CONDITIONAL**）

只做命令冻结与核对：**未修改任何生产文件**（四个 `tiger_ext` 文件哈希逐字节未变，EXP-R2 [6] 4/4），**未新增 driver**（R3 仍开），**未启动 EXP-L0 / EXP-B1**，未改 R13 配方。新增 harness `integration/exp_r2_exp_l0_cli.py`，审计表 `integration/artifacts/r2_exp_l0_cli/exp_r2_audit.json`。权威记录 [plan.md §19.11.13](plan.md)。

**五条命令（仓库根目录；三组 CSG + 两组 ETE）**

```bash
# λc 三臂：--lambda-csg 与 --ckpt-dir 同步取 {1e-4|1e-3|1e-2}
python -m tiger_ext.qspace_train \
  --dataset Toys --data-root data \
  --artifact-dir integration/artifacts/collab_relation --k 20 \
  --lambda-csg {1e-4|1e-3|1e-2} \
  --epochs 3000 --batch-size 1024 --eval-step 50 \
  --num-workers 4 --warmup-epochs 50 \
  --embedding-path data/Toys/item_emb_768d_contentonly.parquet \
  --device cuda:0 --seed 2024 \
  --ckpt-dir rqvae/ckpt/EXP_L0_lc{1e-4|1e-3|1e-2}

# λe 两臂：{le0.1|le1.0} ↔ {0.1|1.0}
python -m tiger_ext.fixed_train \
  --dataset Toys --data_root data \
  --embedding_path data/Toys/item_emb_768d_contentonly.parquet \
  --sid_path data/Toys/Toys_content768d_baseline_sid.npy \
  --sid_manifest data/Toys/Toys_content768d_baseline_sid.npy.sid.json \
  --tokenizer_ckpt rqvae/ckpt/Toys_Content768d_Baseline/Jul-31-2026_21-37-33/best_collision_model.pth \
  --arm "EXP-L0 λe-{0.1|1.0}" \
  --lambda_sia {0.1|1.0} --lambda_psa {0.1|1.0} \
  --seed 2025 --num_epochs 200 --early_stop 10 \
  --batch_size 256 --infer_size 96 --beam_size 30 --lr 1e-4 \
  --num_workers 1 --device cuda:0 \
  --run_dir integration/artifacts/exp_l0/le{0.1|1.0}/run
```

λc 臂还需 refresh SID（不在这五条之列，SID 路径是运行时产物）：`SR.refresh_sid('Toys', "$RUN/best_collision_model.pth", 'integration/artifacts/exp_l0/lc{λ}/sid', device='cuda:0')`，再用 `sid_versions/*.npy` 的最新版本作 `--sid_path`、同目录 `.sid.json` 作 `--sid_manifest`、`$RUN/best_collision_model.pth` 作 `--tokenizer_ckpt`、`--lambda_sia 0 --lambda_psa 0`。

**逐项核对结论**（真实解析器解析，不训练）：CSG 的 λ/epochs/batch/eval-step/workers/warmup/seed/device/embedding/ckpt-dir 与 §19.1 逐值一致；ETE 的 λ_sia=λ_psa、seed、200 ep + early_stop 10、batch/infer/beam/lr、冻结 tokenizer、已登记 baseline SID + manifest、无 `--allow_unregistered_sid`、独立 run_dir 与 §19.3 / §19.11.3 一致。

**具体发现（均为记录项，不改配方）**

1. **flag 拼写**：`qspace_train` 用短横线 `--embedding-path`（`qspace_train.py:778`），`fixed_train` 用下划线 `--embedding_path`（`fixed_train.py:845`）；argparse 不做别名，任务描述中的 `--embedding_path` 字面写进 CSG 命令会 **exit 2**（实测）。
2. **`align_weight 0.0`**：`qspace_train` 无此 flag，只能由 `trainer.py:57` 的 `getattr` 默认满足（R1 已用真实 Trainer 复核）。§19.1 的"显式传"在无 CLI 的前提下是**效果等价**，不是字面满足。
3. **`--num_workers 1`（ETE）** 取 `alt_pipeline.NUM_WORKERS=1`；旧 `model/main.py` 的 `GenRecDataLoader` 默认是 4。§19.3 未钉该项；`shuffle=True` 的 `RandomSampler` 在主进程生成索引，worker 数不改变 batch 顺序 ⇒ 无语义差异。
4. **896d 诱饵**：`fused_item_emb_768d.parquet` 实测 **896d**，五条命令 0 次引用；选中的 content-only 表实测 **768d / 11924 行**，与冻结 ckpt `in_dim=768` 一致；默认 `item_emb.parquet` 是 **4096d**（R2 的原始风险）。
5. **λc 的 SID 版本号**由 `base_dir` 内容决定（全新目录为 `v0001`），driver 必须解析而不得硬编码；**`--ckpt-dir` 是父目录**，Trainer 自动追加时间戳子目录（`rqvae/trainer.py:42-45`）。

**测试**

| suite | 结果 | 说明 |
|---|---|---|
| **EXP-R2（新增）** | **274/274**，`CLOSED-BY-COMMAND-FREEZE` | 五条命令逐参数 + 路径/宽度 + 默认值反证 + 作用域 |
| EXP-P3 | **29/29** | CSG CLI → namespace |
| EXP-P2 | **47/47** | 冻结 Trainer 重放 |
| EXP-B6 | **31/31** | tokenizer 宽度/显式表 |

**不是效果证据**：全部为解析/构造级核对，未训练任何模型。

**结论**：**R2 关闭（命令已冻结，无需改生产代码）**。**GATE 仍 CONDITIONAL，唯一剩余阻断 = R3（无 EXP-L0 driver）**；EXP-L0 不启动。

### 17.6i R3 关闭：EXP-L0 自动执行 driver（2026-10-19，**GATE 仍 CONDITIONAL**）

只做 driver 与机制测试：**未修改任何生产文件**（四个 `tiger_ext` 文件哈希逐字节未变，EXP-R3 [9] 4/4）；**未启动 EXP-L0 / EXP-B1**；未改 R13 配方。新增 `integration/exp_l0_driver.py`、`integration/exp_r3_l0_driver.py`、`integration/artifacts/r3_l0_driver/{exp_r3.log,exp_r3_audit.json}`。权威记录 [plan.md §19.11.14](plan.md)。

**driver 做了什么**

- 五臂固定顺序：`lc1e-4 → lc1e-3 → lc1e-2`（每臂 tokenizer → refresh → recommender，λsia=λpsa=0）、`le0.1 → le1.0`（冻结 baseline tokenizer + 已登记 baseline SID，λsia=λpsa=λ）。
- 每次调用一个全新 run root：`integration/artifacts/exp_l0/runs/<run_id>/<arm>/{ckpt,sid,run,logs}`；`<arm>` 非空或已有 SID 版本 ⇒ preflight 失败，**不启动任何子进程**。
- 产物动态识别：ckpt = 训练前后子目录差集中的唯一新目录；SID = refresh 前后 `list_versions` 差集中的唯一新版本；不硬编码时间戳/版本号。
- refresh 后校验 `sid_path`、SID sha256、ckpt sha256、文件名版本与 manifest 一致，并要求 `.sid.json` sidecar 存在；任一不符 ⇒ 该臂 refresh 失败。
- 任一阶段非 0 退出 ⇒ 立即停止该臂及所有后续臂，**不回退到旧 ckpt / 旧 SID / 缺失 manifest**。
- 总记录 `integration/artifacts/exp_l0/driver_run.json`（最新）+ `runs/<run_id>/driver_run.json`（该次不可变副本）：五臂冻结参数（尤其 `lambda_csg` / `lambda_sia` / `lambda_psa`）、逐阶段命令/退出码/开始结束时间/日志路径、输入输出路径、ckpt/SID/manifest 哈希与验证结果、失败阶段与原因、768d 表路径/宽度/sha、seed、device、SID provenance。
- `--dry-run`：输出五臂计划、依赖关系、完整 argv；**不执行子进程、不建目录、不刷新 SID、不写记录**。

**与 §19.11.13 冻结命令的关系**：配方与输入路径逐 token 一致（EXP-R3 [2]）。唯一差异是**输出位置按运行隔离**——`--ckpt-dir` → `<run_root>/<arm>/ckpt`、`--run_dir` → `<run_root>/<arm>/run`、λc refresh 的 `base_dir` → `<run_root>/<arm>/sid`。这是 R3 任务"禁止复用旧产物、每次独立输出目录"的直接要求，属输出位置而非配方；测试逐臂钉死除这些路径外无参数漂移。

**测试**

| suite | 结果 | 说明 |
|---|---|---|
| **EXP-R3（新增）** | **105/105**，`CLOSED-BY-MECHANISM-TESTS` | 五臂配置 8、R2 命令对比 13、dry-run 10、合成全链路 24、动态版本 3、失败即停 12、provenance 篡改 18、旧产物不复用 7、CLI dry-run 4、生产文件/git 6 |
| EXP-R2 | **274/274** | 零新增失败（其 “no EXP-L0 driver added” 只匹配 `*.sh`，不触发） |
| EXP-P1 | **27/27** | 默认路径逐位不变 |
| EXP-P2 | **47/47** | 冻结 Trainer 重放 |
| EXP-P3 | **29/29** | CSG CLI → namespace |
| EXP-B6 | **31/31** | tokenizer 宽度/显式表 |

**失败注入结果**（全部由 spy runner 在合成 fixture 上完成，无真实训练）

- 阶段失败：λc-1e-3 tokenizer rc=3 / λc-1e-4 refresh rc=1 / le0.1 recommender rc=2 ⇒ 记录 `status=failed`、`failed_stage` 正确、后续臂全部未启动。
- provenance 篡改 6 种：SID sha 错、ckpt sha 错、manifest 版本与文件名不一致、缺 sidecar、refresh 产出两个新 SID、无新 SID ⇒ 均在 refresh 阶段失败，recommender 从未运行。
- 旧产物：非空 arm dir、已有 SID 版本 ⇒ preflight 失败且零子进程；旧 run root 的 ckpt/SID 在本次命令中零引用。

**dry-run 示例**：`python -I integration/exp_l0_driver.py --dry-run --arms lc1e-3,le1.0 --run-root /tmp/exp_l0_dry` 输出两臂 4 个阶段的完整 argv；实测 run root 与 `integration/artifacts/exp_l0` 均未被创建。

**不是效果证据**：全部为 mock/spy、合成 fixture 与静态核对；未训练任何模型，无 Recall/NDCG，不能用于 Stage 0 选点。

**结论**：**R3 的代码与机制测试通过（`CLOSED-BY-MECHANISM-TESTS`）**。按任务要求，**GATE 仍 CONDITIONAL**：下一步是**独立门禁复核**（重跑 §19.11.9 相关行并裁决 R3），复核通过前 **EXP-L0 不得启动**。

### 17.6j 独立门禁复核（2026-10-19，**GATE = CONDITIONAL：M1/M2 两个必须修复项**）

**复核对象**：`plan.md §19.11.9–§19.11.14`、`integration/exp_l0_driver.py`、`integration/exp_r3_l0_driver.py`、`integration/artifacts/r3_l0_driver/exp_r3_audit.json`、R2 冻结命令与 `tiger_ext/*.py`。复核 harness `integration/exp_r3_gate_review.py`（46 项，`REVIEW 44/46`，两个失败即 M1/M2），日志 `integration/artifacts/r3_gate_review/review.log`、审计 `…/exp_r3_gate_review.json`。

**结论一览**

| 核查项 | 结论 |
|---|---|
| 重跑冻结门禁测试（区分历史失败） | **PASS**（R1 44 / R2 274 / R3 105 / P1 27 / P2 47 / P3 29 / B4 38 / B6 31 / B7 48 全同；E9 87/90、E10 60/64 失败行逐条相同） |
| 独立核对五臂配置与执行顺序 | **PASS**（11 条 argv 与从 §19.11.13 独立重写模板逐 token 全等；真实解析器 15 项配方字段逐值相等） |
| 真实执行安全性 | **FAIL**（M1/M2，见下） |
| 配方只改输出位置 | **PASS** |
| 启动前置条件 | **PASS**（768d 表/baseline ckpt+in_dim/baseline SID+manifest/CUDA/数据/磁盘） |
| 不改生产代码、不启动实验 | **PASS**（哈希与 git 状态不变，`integration/artifacts/exp_l0/` 不存在） |

**必须修复项**

- **M1 空臂假成功**：`--arms ""` ⇒ 退出 0、`status=completed`、0 臂。修复：`select_specs` 对空选择抛 `DriverError`。
- **M2 异常丢记录**：stage 抛异常（Popen 失败、`run.json` 损坏等）⇒ 异常穿出，`driver_run.json` 完全未写。修复：`execute_run` 用 `try/except BaseException/finally` 保证记录落盘并标记 failed。fail-fast 与"不误记成功"未受影响，缺的是整轮记录。

两项均在 `integration/` 内，**本次未实施**（只报告证据与最小补丁）；修复后须重跑 EXP-R3 与复核相关行。

**残余风险（不阻断）**：记录非增量落盘（SIGKILL 会丢整轮 JSON，日志仍在）、driver 不回读 `fixed_train` 的 `run.json` provenance、`--run-id` 与 `--run-root` 同时给出时记录取目录名、`sid_stats` 缺键视为通过、§19.11.13 的 `{ckpt_sha12}` 应为 `sha8`、refresh 无超时、`topk_list` 无 CLI 直传但实际生效值与记录一致（`best_ndcg@20` 取自 `run.json`，不解析日志列）。

**不是效果证据**：本次未训练、未刷新正式 SID、未产生 Recall/NDCG。

**判定**：**GATE = CONDITIONAL**；EXP-L0 / EXP-B1 未启动。详见 [plan.md §19.11.15](plan.md)。

### 17.6k M1/M2 最小修复与回归验收（2026-10-19，**修复完成，等待独立复核**）

**改动范围**：仅 `integration/exp_l0_driver.py`（+113 / −41，4 个 hunk）与 `integration/exp_r3_l0_driver.py`（新增 §[10]/[10b]，25 项）。生产代码、R2 冻结配方与五臂命令**零改动**；未启动 EXP-L0 / EXP-B1。

- **M1**：`select_specs("")`/`, `/`" , "` ⇒ `DriverError`；`main()` rc=2 且不建 run root、不写记录。`execute_run([])` 亦 `raise DriverError`，杜绝零臂 `completed`。
- **M2**：`execute_run` 用 `try/except BaseException` + `finalize()`；异常先以 `status="failed"` 落盘 `driver_run.json`（含 `failure={arm,stage,type,message,traceback}`、`arms_completed`、`arms_not_started`）再 `raise`。在飞阶段由 `"running"` 改 `"failed"`。`main()` 的未预期异常 rc=1。

**失败注入证据**（`integration/artifacts/r3_m1m2/`）：Popen `OSError` ⇒ 记录 `failed/tokenizer`，后 4 臂未启动、runner 仅 1 次调用；`run.json` 损坏 ⇒ 记录 `failed/recommender`，`le1.0` 未启动、runner 仅 1 次调用；成功对照 ⇒ `completed` 且 `failure=null`。

**回归**：R1 44/44、R2 274/274、**R3 130/130（+25）**、P1 27/27、P2 47/47、P3 29/29、B4 38/38、B6 31/31、B7 48/48、E9 87/90、E10 60/64（E9/E10 失败行与历史逐条相同，零新增）；独立 harness **REVIEW 46/46**（原 44/46）。

**判定**：修复完成，等待独立门禁复核；**本任务不宣布 GATE PASS**，EXP-L0 / EXP-B1 未启动。详见 [plan.md §19.11.16](plan.md)。

### 17.6l M1/M2 修复的独立复核与最终门禁裁决（2026-10-19，**GATE = PASS**）

**只复核、未改代码、未启动实验。** EXP-R3 **130/130**、独立 harness **REVIEW 46/46**、R1 44/44、R2 274/274、P1/P2/P3 27/27·47/47·29/29、B3/B4/B6/B7 30/30·38/38·31/31·48/48；**历史失败逐条复现、零新增**：E8 54/55（1 行）、E9 87/90（3 行）、E10 60/64（4 行），全部为 C5 授权修改 `rqvae/trainer.py` 引起的 tracked-file guard 及其汇总行。

失败注入独立重放：空臂 rc=2 且零副作用；`OSError` ⇒ `failed/tokenizer`、`arms_not_started` 4 臂、runner 仅 1 次、无残留 `running`；`JSONDecodeError` ⇒ `failed/recommender`、`le1.0` 未启动；成功对照 `completed` + `failure=null`。

配方/输入：8 条 argv 与 §19.11.13 逐 token 相同（仅 run-scoped 路径不同）；768d 表 11924×768、ckpt in_dim 768、基线 SID (11924,4) 且 sidecar 双哈希实测相等、`sid_stats.ok=true`；生产四哈希与 `git status` 未变；`integration/artifacts/exp_l0|exp_b1` 不存在、无相关进程。

**判定：GATE = PASS**，可进入 Stage 0（EXP-L0）。本次**仍未启动**正式实验，等待用户明确指令。详见 [plan.md §19.11.17](plan.md)。

### 17.6m Stage 0（EXP-L0）启动与完成记录（2026-10-19，**DONE / COMPLETED**）

**授权**：用户于 2026-10-19 明确授权启动 EXP-L0（GATE = PASS 之后）。**未修改任何代码**；生产四哈希启动前存档：`qspace_train 305c50a7…`、`fixed_train 389e9085…`、`sid_refresh 983121cb…`、`alt_pipeline fc1b8652…`；driver `244b71b6…`。git HEAD `5edfca0`，工作区与门禁复核时逐行相同。

**启动方式**：`python -I integration/exp_l0_driver.py --quiet`（后台，nohup），执行顺序 `lc1e-4 → lc1e-3 → lc1e-2 → le0.1 → le1.0`，run root = `integration/artifacts/exp_l0/runs/20261018T202823Z`，整轮记录 = `integration/artifacts/exp_l0/driver_run.json`。每阶段日志 `<run_root>/<arm>/logs/{tokenizer,refresh,recommender}.log`。

**环境**：GPU 0 = RTX 4090（24G，启动前 529 MiB / 0%），`/data` 余 972G，`python` = `/home/media601/anaconda3/bin/python`（torch 2.6.0+cu124，CUDA 可用）。

**配方**：五臂参数与 plan §19.11.13 逐 token 一致（仅输出位置 run-scoped）；`--embedding-path`（CSG）/`--embedding_path`（ETE）各自正确；`--allow_unregistered_sid` 未出现。

**结果（2026-10-19 10:38:23 CST 完成，6 h 10 m）**：`driver_run.json` **`status=completed`**、`failure=null`、`arms_not_started=[]`、五臂全列；15 个 stage 全部 `completed` / `exit 0`；五个 `run.json` 实测 sha256 与 driver 记录逐一相等。

| arm | 类型 | λ | best epoch | 执行 epochs | valid NDCG@20 | valid Recall@20 | test NDCG@20（仅记录） | 耗时 |
|---|---|---|---|---|---|---|---|---|
| lc1e-4 | CSG | λc=1e-4 | 96 | 107 | **0.040444** | 0.095454 | 0.033515 | 4020.8 s |
| lc1e-3 | CSG | λc=1e-3 | 94 | 105 | 0.039014 | 0.092334 | 0.032033 | 4081.6 s |
| lc1e-2 | CSG | λc=1e-2 | 66 | 77 | 0.037674 | 0.090261 | 0.031311 | 3016.4 s |
| le0.1 | ETE | λsia=λpsa=0.1 | 112 | 123 | 0.039269 | 0.093042 | 0.031531 | 4571.9 s |
| le1.0 | ETE | λsia=λpsa=1.0 | 113 | 124 | 0.033796 | 0.079700 | 0.028671 | 4667.8 s |

五臂全部触发 early stop。CSG 侧 `--lambda-csg` 实为 1e-4/1e-3/1e-2（tokenizer 日志），hook `n_calls=n_nonempty=36000`、`n_disabled_calls=0`；三条 SID 均 `(11924,4) int64`、`ok=true`、sha 互不相同（dup rate 0.000587/0.000587/0.001761）。ETE 侧 `mounts.sia/psa=true`，SID 为冻结 baseline。

**结论边界**：单 seed、无方差/显著性；CSG 与 ETE 是不同 pipeline（非单变量对照）；本轮无 λ=0 对照臂；λ 越大越低、best epoch 同步左移仅属筛选观察。**不构成效果结论。**

**lc1e-3 摘要抽取（2026-10-19 07:14）**：新增只读工具 `integration/summarize_l0_run.py`，对 `lc1e-3/run/run.json` 生成 `lc1e-3/run/run_summary.json`（11 KB，未复制 105 条 epoch 明细）。原文件未改动（`sha256 0eb49a31…`，mtime 07:03:38 不变）。缺失字段 6 项：`config.lambda_csg`、`config.sid_shape`、每 epoch 的 `L_T5/L_SIA/L_PSA/L_RQVAE` 分量值（run.json 只存 `train_loss` 聚合值）；均已标记 `null`，未猜测补值。`result.losses` 是**阶段契约静态标志**，实际挂载以 `result.mounts` + λ 为准（本臂 SIA/PSA 均未挂载）。

**lc1e-2 摘要抽取（2026-10-19 08:13）**：结构键集与 lc1e-3 **逐项相同**（config 41 / result 45 / epoch 12 键），复用同一只读脚本生成 `lc1e-2/run/run_summary.json`（11 KB）。原文件未改动（`sha256 63f89f8c…`，mtime 08:04:13 不变）。同样缺失 6 项（`lambda_csg`、`sid_shape`、4 个逐分量 loss），未补值。注意本臂 SID `raw_duplicate_rate=0.001761`（lc1e-4/1e-3 为 0.000587），`sid_stats.ok=true`。

**le0.1 摘要抽取（2026-10-19 09:52）**：结构键集与前两臂**逐项相同**（config 41 / result 45 / epoch 12 键），复用同一只读脚本生成 `le0.1/run/run_summary.json`（11,219 B）。原文件未改动（`sha256 254e0d14…`，mtime 09:20:30 不变）。同样缺失 6 项（`lambda_csg`、`sid_shape`、4 个逐分量 loss），未补值。本臂为 **ETE**：λsia=λpsa=0.1 **真实挂载**（`result.mounts` sia/psa/constructed 均为 true），与 CSG 三臂的 λ=0 未挂载不同；SID 为冻结 baseline `Toys_content768d_baseline_sid.npy`（`bb2693a6…`，`sid_unchanged=true`），tokenizer 冻结 23 张量 / 0 可训练，adapter 可训练 4 张量。`raw_duplicate_rate=0.000419`，`sid_stats.ok=true`。

**le1.0 摘要抽取（2026-10-19 10:42）**：结构键集与前几臂**逐项相同**（config 41 / result 45 / epoch 12 键），复用同一只读脚本生成 `le1.0/run/run_summary.json`（11,214 B）。原文件未改动（`sha256 37617f76…`，mtime 10:38:23 不变）。同样缺失 6 项（`lambda_csg`、`sid_shape`、4 个逐分量 loss），未补值。本臂 λsia=λpsa=**1.0** 且 `mounts.sia/psa=true`（与 le0.1 同为 ETE，仅 λ 不同）；SID/tokenizer/embedding 三家哈希与 le0.1 **逐个相同**（同一冻结基线，`sid_unchanged`/`tokenizer_unchanged`/`embedding_unchanged` 均 true）。⚠️ `train_loss` best epoch = **14.5643**（le0.1 为 3.2810），因 `L_total = L_RQVAE + λ·(L_SIA+L_PSA)` 且 λ 为 1.0 ⇒ **`train_loss` 跨 λ 臂不可直接比较**。

### 17.6n 指标口径审计：EXP-L0 × Clean TIGER（2026-10-19，**只读 / 完成**）

**任务边界**：只读。**未修改代码/配置/产物，未重跑训练**。新增权威文档 [doc/exp_l0_baseline_metric_audit.md](../doc/exp_l0_baseline_metric_audit.md)；plan 侧 [§19.11.19](plan.md)。

**结论**：基线 `NDCG@20=0.0341` / `Recall@20=0.0795` 是 **test**（seed 2025 单次）；EXP-L0 `run.json` 的 `best_ndcg@20` 是 **valid**。两者**不可直接相减**，此前 `lc1e-4 +18.6%` **撤回**。

**基线首次取得 valid 记录**：[model/logs/tiger_content768d_baseline_toys.log](../model/logs/tiger_content768d_baseline_toys.log) 的 epoch 76 行给出 valid **N@20 0.03980784884796207 / R@20 0.09360632170083488**，同 epoch test 为 **0.03408397088203524 / 0.07945402289552642**。三个独立来源（训练日志、eval 日志、本次只读复算）逐位一致。

**同口径复核表**：

| arm | valid N@20 | Δ vs baseline valid | test N@20 | Δ vs baseline test |
|---|---|---|---|---|
| baseline | 0.039808 | — | 0.034084 | — |
| lc1e-4 | 0.040444 | **+0.000636 (+1.60%)** | 0.033515 | **−0.000569 (−1.67%)** |
| lc1e-3 | 0.039014 | −0.000794 | 0.032033 | −0.002051 |
| lc1e-2 | 0.037674 | −0.002134 | 0.031311 | −0.002773 |
| le0.1 | 0.039269 | −0.000539 | 0.031531 | −0.002553 |
| le1.0 | 0.033796 | −0.006012 | 0.028671 | −0.005413 |

**没有任何一臂在 test 上超过基线**；lc1e-4 的差距（0.00057）小于基线 3-seed 极差（0.0013）⇒ 判「无差异」。原 `plan §19.11.18` 表中的 valid/test 两列本身无误，**错的是把它们与 0.0341 的 test 交叉比较**。

**评估流程（代码证实）**：选择指标 = valid；test 仅在 valid 改善时评估（`n_test_evals == n_best_saves`，本批 46/47/41/46/50），**不进任何控制流**。`run.json` 不存「最佳 epoch 的 test」顶层标量，正确取法是 `result.epochs[best_epoch].test_ndcgs`。

**可比性差异**：① 基线 T5 初始化在 `main.py` 中**先建模型后 seed**（未定种），EXP-L0 先 seed 后建；② worker 4 vs 1；③ adapters 始终构造但 λ=0 时无梯度且不入 `best.pth`（92 keys 与基线一致）；④ CSG 三臂各自重训 tokenizer（seed 2024）+ 刷新 SID，ETE 两臂用基线 tokenizer/SID；⑤ valid 目标是 `seq[-2]`、test 目标是 `seq[-1]`，**不同任务**，系统性 gap ≈0.006（−14%）。

**规则影响**：test 口径下 CSG 仅 `lc1e-4` 满足 Δ ≥ −0.001 ⇒ λc=1e-4；ETE 无合格臂 ⇒ λe 回退 0.1 并记「未找到正向工作点」。valid 口径下合格集 {lc1e-4, lc1e-3, le0.1}，但 lc1e-3 的 L0 distinct 80% 触发塌缩判据 ⇒ 排除后同样落在 λc=1e-4 / λe=0.1。**「合格」≠「提升」。**

**未确认**：基线启动时是否外部定种、ckpt 重命名无哈希证据、eval 日志不记 `--code_path`、EXP-L0 每臂仅 1 seed。

### 17.6o 代码发布记录（2026-10-19，**DONE**）

集成代码已发布到 GitHub `yeshuwei/tiger-TA` 的独立分支 `integration-review-20261019`，提交 `7a578bac…`（"Publish integrated CSG and ETE implementation for review"），父提交 = Clean TIGER 基线 `5edfca08…`。`main` 与 `experiments-2026-09-25` 未被改动或强推。

上传 = 4 个既有文件的可选钩子（`model/main.py`、`rqvae/models/rq.py`、`rqvae/models/vq.py`、`rqvae/trainer.py`）+ `tiger_ext/` 17 个源文件 + `integration/` 29 个 harness 与 3 份文档 + `integration/README.md` + `doc/exp_l0_baseline_metric_audit.md` + `.gitignore` 三条新规则。排除 `data/`、权重、SID/embedding、`integration/artifacts/`（678 MB）与缓存/日志。详见 [plan.md §19.11.20](plan.md)。
