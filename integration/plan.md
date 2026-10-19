# CSG4Rec + ETEGRec → TIGER 工程移植计划

- **版本**：v1.1（R3–R12 修订记录见 §9–§18；**R13 效果实验设计见 §19**）
- **日期**：2026-10-18
- **状态**：**工程侧收口**——E1–E10 / C1–C6 全部实现且 **A-PASS**（C6 = 63/63，R12）。**下一步 = 效果实验**：按 §19 的 2×2×2 设计执行 Stage 0（λ 标定）→ Stage 1（B 类筛选）→ Stage 2（C 类正式）。**推荐效果仍未验证，本文件不含任何效果数字。**
- **冻结基线**：`5edfca08b22f23a2e965f5e21062adb53e1214b`（`freeze clean TIGER baseline`）
- **基线入口**：`bash clean_tiger/run_clean_tiger.sh {rqvae|sid|t5|eval|verify|all}`
- **配套文档**：[experiments.md](experiments.md)（实验总表）、[status.md](status.md)（进度）

> **本节为 M0.5 阶段的历史约束，已不适用**（M0.5 已 `DONE / CLOSED`，E 线自 E1 起开工）。
> 原文：本文件在 M0.5 阶段只做**计划**，不产生任何代码改动；M0.5 期间 `model/`、`rqvae/`、`clean_tiger/`、`tools/`、`doc/` 全部只读。
> 现状：E1–E6 对 `model/main.py`、`rqvae/models/{rq,vq}.py` 做了**默认关闭的加法式**改动，并新增 `tiger_ext/`；`clean_tiger/`、`tools/`、`doc/` 仍为只读。详见 [status.md](status.md) 边界状态与 §9。

---

## 0. 总原则

### 0.1 TIGER 是唯一主干

```
item content representation → RQ-VAE (tokenizer) → Semantic ID → T5 (recommender) → Recommendation
```

- **唯一主干**：所有实验都运行在 Clean TIGER 的这条主链上。
- **不做 fork**：不为 CSG4Rec / ETEGRec 各建一套训练框架。移植物必须以**插件形式**挂在 TIGER 既有接口上。
- **可回退**：任何移植机制都必须能被开关彻底关闭，关闭后必须**逐位等价于** Clean TIGER（见 §4 门禁 G0）。
- **不污染基线**：`clean_tiger/` 是冻结参照，移植代码不得修改它；对比实验通过新入口驱动。

### 0.2 移植边界

#### CSG4Rec —— **只移植 4 项**

| # | 移植内容 | 说明 |
|---|---|---|
| 1 | item-level collaborative embedding | 仅 item 侧协同向量，形状 `(n_items, d_cf)` |
| 2 | collaborative similarity / kNN relation | CF 向量 → cosine → kNN 关系对 |
| 3 | Q-space collaborative constraint | 约束作用在**量化空间**，不是输入空间 |
| 4 | 对应 loss / trainer 接口 | loss 与训练循环挂钩 |

#### CSG4Rec —— **明确不移植**

| 不移植项 | 理由 |
|---|---|
| User SID | 与 TIGER 的 item-SID 主干正交，引入会改变任务定义与评测口径 |
| CSG4Rec Generation 模块 | 生成式协同模块自带一套序列/图结构，等于第二套主干 |
| CSG4Rec 自己的 RQ-VAE | TIGER 的 tokenizer 是唯一 tokenizer（否则 SID 语义不可比） |
| CSG4Rec 完整训练框架 | 本项目不做框架合并 |

#### ETEGRec —— **只移植 3 项**

| # | 移植内容 | 说明 |
|---|---|---|
| 1 | SIA（Semantic ID Alignment） | 序列表示 ↔ 目标 item 的**码分布**对齐 |
| 2 | PSA（Preference/Personalized Semantic Alignment） | decoder 表示 ↔ 目标 item 的**重建表示**对齐 |
| 3 | Alternating Optimization | tokenizer 与 recommender 交替训练 |

#### ETEGRec —— **明确不移植**

| 不移植项 | 理由 |
|---|---|
| ETEGRec RQ-VAE | 同上，TIGER tokenizer 唯一 |
| SASRec | 与 T5 主干重复，且本项目已有独立 SASRec 基线记录 |
| semantic embedding table | 等于第二套 item 表示源，破坏"content → RQ-VAE"单一表征链 |
| ETEGRec token embedding architecture | 会改变 T5 输入嵌入语义 |
| ETEGRec 完整 trainer | 同上 |
| ETEGRec 完整 T5 | 同上；TIGER 的 T5 结构不得改动 |

---

## 1. 接口事实基线（只读核实）

以下事实已在只读检查中实测确认，作为后续所有 Task 的设计依据。**若后续实测与此不符，应先更新本节，再改 Task 设计。**

| 事实 | 值 | 证据 |
|---|---|---|
| RQ-VAE latent 维度 | `e_dim = 32` | `clean_tiger/config.sh`；smoke ckpt `args.e_dim` |
| 码本结构 | `num_emb_list = [256, 256, 256]` | 同上 |
| SK epsilon | `[0.0, 0.0, 0.003]` | 同上 |
| SID 形态 | `(n_items, 4)` int64，前 3 列 = 3 个 RQ level，第 4 列 = collision token | smoke SID `(11924, 4)`，第 4 列 467 条非零 |
| T5 类别 | `T5ForConditionalGeneration` | [model/main.py:40](../model/main.py#L40) |
| T5 `forward` 返回值 | **只有** `(outputs.loss, outputs.logits)` | [model/main.py:76](../model/main.py#L76) |
| SID → T5 token 布局 | `offsets = c + i*256 + 1`；L0→`[1,256]`、L1→`[257,512]`、L2→`[513,768]`、collision→`[769,1024]`；`vocab_size=1025` | [model/dataset.py:85](../model/dataset.py#L85) |
| 序列构造 | history 每个 item 展开为 4 个 token 后拼接；`max_len=20`（≤5 个 item）；左侧 pad `0` | [model/dataset.py:64-69](../model/dataset.py#L64-L69) |
| `generate` 路径 | `max_length=5, num_return_sequences=beam_size`，**不暴露 decoder hidden** | [model/main.py:90](../model/main.py#L90) |
| RQ-VAE 对外接口 | `get_indices(xs, use_sk=False)` → 硬索引；`forward(x, use_sk, return_z, return_xq)` → `(out, rq_loss, indices[, z][, x_q])` | [rqvae/models/rqvae.py:82](../rqvae/models/rqvae.py#L82), [rqvae/models/rq.py:39](../rqvae/models/rq.py#L39) |
| 量化器**不输出** code logits | `ResidualVectorQuantizer.forward` 只回 `(x_q, mean_losses, all_indices)` | [rqvae/models/rq.py:56](../rqvae/models/rq.py#L56) |
| transformers 版本 | 4.57.6 | 运行时实测 |

### 1.1 E1 的可行性已预先验证

对 `T5ForConditionalGeneration` 实测（toy config，只读）：

```
output_hidden_states=True  →  encoder_last_hidden_state : (B, L_enc, d_model)   ✅ 恒有
                               decoder_hidden_states[-1] : (B, L_dec, d_model)   ✅ 开启后才有
hasattr(outputs, "decoder_last_hidden_state") = False   ← 该字段在 4.57.6 不存在
torch.equal(o.logits, o2.logits) = True                 ← 开启前后 logits 逐位一致
torch.equal(o.loss,   o2.loss)   = True                 ← 开启前后 loss   逐位一致
```

**三条结论**：
1. E1 的"获取两种 hidden 且不改变原始行为"在技术上成立。
2. decoder hidden 必须定义为 `decoder_hidden_states[-1]`，**不能**引用不存在的 `decoder_last_hidden_state`。
3. **E5 的 PSA 只能在 teacher-forcing 的 `forward` 路径上取 decoder hidden**；`generate()` 的 beam search 循环不暴露 hidden，不得依赖它。

### 1.2 仓库内已有的可复用先例（避免重复造轮子）

| 先例 | 位置 | 与本计划的对应 |
|---|---|---|
| item 级协同向量 | `data/<D>/behavior_emb_itemcf.parquet`，`(n_items, 128)` | **C1 的备用输入**（ItemCF cosine + SVD，非正式来源；正式来源见 [C1 决策点](#c1--cf-embedding-interface)与 §13 R7） |
| 关系对 → `trans_dict`（`item_idx → [positive item_idx]`） | [rqvae/trainer.py:99-108](../rqvae/trainer.py#L99-L108)，由 `trans_graph_path` `.npy` 形态 `(N, 2)` 构建 | **C2 的 kNN 关系可直接复用该数据格式与加载方式** |
| 关系驱动对比损失 | `rqvae/trans_loss.py::ContrastiveLoss`（L_trans） | **C4 的结构参照** |
| 量化表示 → **外部嵌入**的对比对齐（带投影头） | `Trainer._get_align_loss` + `align_proj = nn.Linear(e_dim, D, bias=False)`（E50 b2 的 L_align，**已裁撤**） | ⚠️ **不是 PSA 的参照**。其实际逻辑是 `latent(32) → projection → 外部嵌入` 的 InfoNCE，与 PSA 的 `decoder hidden → 重建表示` 在**输入来源 / 投影维度 / 对齐目标**三处均不同。仅可参考"投影头 + in-batch InfoNCE"的**代码组织形式**，不得复用其 latent → external embedding 的对齐逻辑 |

> 复用格式 ≠ 复用机制。上表只说明"数据结构与代码骨架可以借用"，不表示可以直接打开这些历史开关。

### 1.3 内容 / 重建维度（`D_target`）

> **⚠️ R13（2026-10-18）已改口径**：本系列效果实验统一改用 **768d content-only**（`item_emb_768d_contentonly.parquet`），`D_target = 768`。理由与证据见 **§19.1**。下表保留为 **4096d 历史口径**，不得用于 R13 之后的实验。

PSA 的对齐目标是 tokenizer 的**重建表示**。该维度在 [rqvae/models/rqvae.py](../rqvae/models/rqvae.py) 中已确认（`out = self.decoder(x_q)`）：

```
D_target = RQ-VAE decoder 输出维度 = item content embedding 维度
```

当前各数据集实测值（`data/<D>/item_emb.parquet`）：

| 数据集 | n_items | `D_target`（content dim） |
|---|---|---|
| Toys | 11924 | **4096** |
| Sport | 18357 | **4096** |
| Beauty | 12101 | **4096** |
| Electronics | 63001 | 768 |
| Musical_Instruments | 900 | 768 |
| Video_Games | 10672 | 768 |

PSA adapter 的维度关系因此是 **`d_model → D_target`**（`d_model` 来自 T5 decoder hidden）。**不得擅自修改维度。**

### 1.4 训练阶段与 loss 生效表（E7 的阶段契约）

| | Tokenizer 阶段 | Recommender 阶段 |
|---|---|---|
| **RQ-VAE** | **trainable** | frozen |
| **T5** | **frozen** | **trainable** |
| `L_RQ-VAE` | **生效** | 不生效 |
| `L_SIA` | **不生效** | **生效** |
| `L_PSA` | **不生效** | **生效** |
| `L_T5` | 不生效 | **生效** |

> **`L_SIA` 与 `L_PSA` 均挂载于 Recommender（T5）训练阶段；Tokenizer 阶段二者不参与训练。**
>
> **不得**把 `L_SIA` 挂载于 RQ-VAE / Tokenizer training loop。
>
> 依据：SIA 的序列侧来自 **T5 encoder**（E2），PSA 的输入是 **T5 decoder hidden**（E5），二者都需要 T5 可训练；而 Tokenizer 阶段按 E7 冻结 T5。

---

## 2. ETEGRec → TIGER 接口路线（E1–E10）

### 依赖图

```
E1 (hidden 接口)
 ├─→ E2 (SIA adapter) ─→ E3 (码分布抽取) ─→ E4 (SIA loss)
 └─→ E5 (PSA decoder 接口) ─────────────────→ E6 (PSA loss)
                                                  │
E2/E3 需要 RQ-VAE 侧暴露 code logits（新增，默认惰性）
                                                  ↓
                            E7 (freeze 接口) ─→ E8 (SID refresh) ─→ E9 (交替循环) ─→ E10 (smoke)
```

---

### E1 — T5 hidden-state interface

| 字段 | 内容 |
|---|---|
| **Task ID** | E1 |
| **任务名称** | T5 hidden-state interface |
| **来源机制** | 通用 seq2seq 表示抽取（ETEGRec 的 SIA/PSA 都需要 T5 hidden，ETEGRec 自身实现不作为移植来源） |
| **TIGER 对应接口** | `TIGER.forward()` @ [model/main.py:59](../model/main.py#L59)；底层 `self.model` = `T5ForConditionalGeneration` |
| **输入** | 与原 `forward` 完全相同的 `(input_ids, attention_mask, labels)` |
| **输出** | 原 `(loss, logits)` **不变**；额外可选返回 `encoder_last_hidden_state` 与 `decoder_hidden_states[-1]` |
| **需要新增什么** | `forward` 的一个可选开关（如 `return_hidden=False`）；开启时透传 `output_hidden_states=True` 并把两个 hidden 一并返回 |
| **需要修改什么** | [model/main.py](../model/main.py) 的 `TIGER.forward`（**唯一改动点**；不改 T5 结构、不改 loss、不改 logits、不改训练循环） |
| **前置 Task** | 无（本 Task 是 ETEGRec 路线的根） |
| **验收标准** | ① `return_hidden=False` 时返回值与 Clean TIGER **逐位相同**；② 开启时 `encoder_last_hidden_state.shape == (B, L_enc, d_model)`、`decoder_hidden_states[-1].shape == (B, L_dec, d_model)`；③ 开启前后 `loss` / `logits` **逐位相同**（`torch.equal`，非 allclose）；④ `generate()` 路径不受任何影响 |
| **对应验证实验** | **EXP-E1**（Clean TIGER consistency test，类型 A，见 [experiments.md](experiments.md)） |

> **硬约束**：decoder hidden 定义为 `decoder_hidden_states[-1]`。transformers 4.57.6 无 `decoder_last_hidden_state`，不得引用。

---

### E2 — SIA adapter

| 字段 | 内容 |
|---|---|
| **Task ID** | E2 |
| **任务名称** | SIA adapter |
| **来源机制** | ETEGRec SIA 的表示侧入口 |
| **TIGER 对应接口** | 输入 = E1 的 `encoder_last_hidden_state`；输出 = TIGER RQ-VAE latent space（`e_dim = 32`） |
| **输入** | `encoder_last_hidden_state`，形状 `(B, L_enc, d_model)` |
| **输出** | `(B, 32)` 的序列 latent 表示 |
| **需要新增什么** | 新模块（建议置于新目录，**不放** `rqvae/` 也不放 `model/` 的内核路径）：mean pooling（按 `attention_mask` 加权）→ `Linear(d_model, 32)` adapter |
| **需要修改什么** | 无需修改 TIGER 既有文件 |
| **前置 Task** | E1 |
| **验收标准** | ① shape 正确 `(B, 32)`；② forward 可跑；③ backward 可跑；④ **梯度可回传**到 adapter 参数（`grad is not None` 且非全零）；⑤ padding 位置不参与 pooling（改 padding 内容不改变同一有效样本的输出）；⑥ **序列侧不经过 `RQVAE.encoder`**（见下方硬约束，检查落点在 E3） |
| **对应验证实验** | **EXP-E2**（SIA adapter interface sanity test，类型 A） |

> **硬约束**：latent 维度固定 32 = TIGER `e_dim`。**不得**引入 ETEGRec 自己的 RQ-VAE。

> **SIA 两侧前向路径（E2/E3 共同锁定，不得改动）**：
>
> ```
> 序列侧：T5 encoder hidden → adapter → 32-d query → RQ quantizer
> 目标侧：target item embedding → RQ-VAE.encoder → RQ quantizer
> ```
>
> - ✅ 序列侧 Adapter 输出**直接作为 RQ quantizer 的 query**；
> - ❌ 序列侧**不得**经过 `RQ-VAE.encoder`；
> - ✅ 目标侧**必须**经过完整 `RQ-VAE.encoder → RQ`；
> - ❌ 不得改成 `T5 logits ↔ SID token logits`；
> - ❌ 不得引入 User SID。
>
> 依据：TIGER 的码本位于 `RQVAE.encoder` 的输出流形上（[rqvae/models/rqvae.py](../rqvae/models/rqvae.py)：`z = self.encoder(x)` → `self.rq(z)`）。序列侧若再套一层 encoder，SIA 就对不齐同一个量化空间了。

---

### E3 — SIA code-distribution extraction

| 字段 | 内容 |
|---|---|
| **Task ID** | E3 |
| **任务名称** | SIA code-distribution extraction |
| **来源机制** | ETEGRec SIA 的码空间对齐信号 |
| **TIGER 对应接口** | TIGER RQ-VAE 量化器 [rqvae/models/rq.py](../rqvae/models/rq.py)、[rqvae/models/vq.py](../rqvae/models/vq.py)；目标侧复用 `EmbDataset` + `RQVAE.get_indices` 的同一条前向 |
| **输入** | ① 序列 latent（E2 输出，`(B, 32)`）；② 目标 item content embedding（`(B, D_content)`，来自 `data/<D>/item_emb.parquet`） |
| **输出** | 两侧各 **3 组** code logits，每组 `(B, 256)`；即 `seq_logits[3]` 与 `tgt_logits[3]` |
| **需要新增什么** | RQ-VAE **量化器侧的惰性开关**：让每一层 VQ 额外返回该层的 code logits（由该层的 **squared L2 distance** 得到，**符号取正**、温度可配；R3 修正，见 §9 R3-4）。默认关闭，返回结构不变 |
| **需要修改什么** | [rqvae/models/vq.py](../rqvae/models/vq.py) 与 [rqvae/models/rq.py](../rqvae/models/rq.py) 各加一个可选输出；**不改** 量化算法本体（argmin / sinkhorn / STE / loss 一律不动） |
| **前置 Task** | E2；本 Task 自身引入的量化器开关 |
| **验收标准** | ① 能取到 3 个 level 的分布，形状 `(B, 256)` × 3；② target item 侧同样可得；③ 分布经 softmax 后每行和为 1；④ **只作用于 3 个 RQ level，第 4 个 collision token 不参与**（用 SID 第 4 列非零的 467 条 item 显式断言）；⑤ 关闭开关时 `get_indices` 输出与 Clean TIGER **逐位相同**；⑥ **序列侧路径断言**：进入 RQ quantizer 的 query **就是** E2 adapter 的输出张量本身，且该 path 上**没有** `RQVAE.encoder` 被调用；⑦ **目标侧路径断言**：进入同一个 RQ quantizer 的 query 是 `RQVAE.encoder(target item embedding)` 的输出，即**经过** `encoder` |
| **对应验证实验** | **EXP-E3**（SIA code-distribution diagnostic，类型 A+B） |

> **关键点**：TIGER 现有量化器**只吐硬索引**，不吐 logits。E3 的改动是"暴露"而非"改变"。
> **禁止**：把 SIA 做成 4 个 level、或让 collision token 进入 KL。

> **⑥⑦ 的实现方式不限**（调用路径记录、张量来源标记、hook、等价运行时检查均可），但**不得改变模型设计**。

#### E3 诊断：SIA query 与 RQ codebook 空间（**Diagnostic Only — 非阻断**）

仅**记录**，用于判断 Adapter 输出是否进入合理的 RQ latent / codebook 可及范围。至少记录：

1. Adapter 输出的 **L2 norm** 分布；
2. Adapter 输出**到最近 codeword 的距离**分布；
3. `RQ-VAE.encoder(item_embedding)` 输出**到最近 codeword 的对应距离**分布；
4. 上述两类统计的**量级比较**。

> ⚠️ **以上内容仅用于诊断和分析，不作为 E3 的阻断性 pass / fail 条件。**
>
> 该诊断用于判断 SIA 是否存在 **query-space mismatch**。**不预设异常。**
>
> **不得**：
> - 设置人为阈值；
> - 预设"必须同量级"；
> - 因诊断结果自动触发 warmup；
> - 增加 projection；
> - 增加 loss；
> - 修改 RQ-VAE 或 codebook。
>
> 若未来诊断发现明显 mismatch，**另开 Task 讨论**，不在 E3 内自行处理。

---

### E4 — SIA loss

| 字段 | 内容 |
|---|---|
| **Task ID** | E4 |
| **任务名称** | SIA loss |
| **来源机制** | ETEGRec SIA（语义 ID 对齐） |
| **TIGER 对应接口** | 接收 E3 的两组 3-level 分布；**挂载于 Recommender（T5）训练阶段**，作为 T5 训练循环中的一个可加权项（见 §1.4 阶段契约） |
| **输入** | `seq_logits[3]`、`tgt_logits[3]`，各 `(B, 256)` |
| **输出** | 标量 `L_SIA` |
| **需要新增什么** | 新 loss 模块：对每个 level 做 `softmax(logits/T)` → **symmetric KL**，3 个 level 上取平均/求和；权重 `lambda_sia` 默认 0 |
| **需要修改什么** | **Recommender（T5）侧**训练循环增加一处受 `lambda_sia > 0` 守卫的挂载；**不得**改动既有 loss 的构成；**不得**挂到 RQ-VAE / Tokenizer loop |
| **前置 Task** | E3 |
| **验收标准** | ① 标量、可反传、梯度非 NaN；② 分布完全一致时 `L_SIA ≈ 0`；③ 分布差异增大时 `L_SIA` 单调增大（方向性测试）；④ `lambda_sia = 0` 时总 loss 与 Clean TIGER **逐位相同** |
| **对应验证实验** | **EXP-E4**（TIGER vs TIGER + SIA，类型 B + C 候选） |

> **明确禁止**：不得实现成 **T5 output logits ↔ target SID token logits**。
> SIA 对齐的是 **RQ-VAE 码本空间的分布**，不是 T5 词表上的 token 分布。两者形状都是"256"，极易混淆——实现时必须用 §1 的 token 布局表反查，确认对齐对象是 `codebook`（0–255）而非 `vocab`（0–1024）切片。

> **生效阶段**：`L_SIA` **仅**在 Recommender（T5）阶段生效；Tokenizer 阶段不参与训练（§1.4）。
> **生效性的验证归属 E7–E9，不在 E4**：E4 只验证 SIA **loss 机制本身**（forward / backward / gradient / shape / 两侧路径 / loss 正确性）。阶段生效由 **E7–E9** 统一验证（见 E7 的"阶段生效统一验证项"）。

---

### E5 — PSA decoder interface

| 字段 | 内容 |
|---|---|
| **Task ID** | E5 |
| **任务名称** | PSA decoder interface |
| **来源机制** | ETEGRec PSA 的表示侧入口 |
| **TIGER 对应接口** | E1 的 `decoder_hidden_states[-1]` |
| **输入** | `decoder_hidden_states[-1]`，形状 `(B, L_dec, d_model)` |
| **输出** | `(B, D_target)` 的 PSA 表示，`D_target` 定义见 §1.3 |
| **需要新增什么** | 取 **序列位置 0** → PSA adapter（`Linear(d_model, D_target)`）；`d_model → D_target` 的维度关系见 §1.3 |
| **需要修改什么** | 无需修改 TIGER 既有文件 |
| **前置 Task** | E1 |
| **验收标准** | ① 取的是**最后一层 decoder hidden 的序列位置 0**，即 `decoder_hidden_states[-1][:, 0, :]`（**不是** encoder hidden，**也不是**"第 0 层"）；② shape 正确；③ 可 forward/backward 且梯度非零；④ 训练态必须走 teacher-forcing `forward`，**不得**依赖 `generate()` |
| **对应验证实验** | 并入 **EXP-E5/E6** |

> **表述约定**：`decoder_hidden_states[-1][:, 0, :]` 中的 `0` 是**序列位置 0**（decoder 输入首位的 `decoder_start_token_id`），层维度取**最后一层**。见 §1.1：transformers 4.57.6 无 `decoder_last_hidden_state` 字段，只能取 `decoder_hidden_states[-1]`。
> **依据 §1.1 第 3 条**：`generate()` 的 beam search 不暴露 decoder hidden，PSA 训练期只能用带 `labels` 的 `forward`。

---

### E6 — PSA loss

| 字段 | 内容 |
|---|---|
| **Task ID** | E6 |
| **任务名称** | PSA loss |
| **来源机制** | ETEGRec PSA（偏好/个性化语义对齐） |
| **TIGER 对应接口** | 目标侧：目标 item content embedding → TIGER RQ-VAE → **重建表示**（[rqvae/models/rqvae.py](../rqvae/models/rqvae.py) 中 `out = self.decoder(x_q)`，即 `RQVAE.forward` 的第一个返回值 `out`） |
| **输入** | PSA 表示 `(B, D_target)`；目标 item 的重建表示 `(B, D_target)` |
| **输出** | 标量 `L_PSA` |
| **需要新增什么** | 新 loss 模块：symmetric contrastive loss（in-batch 负样本），权重 `lambda_psa` 默认 0。similarity / temperature / 双向系数按 ETEGRec 官方实现钉死：**cosine + 0.07 + 两方向求和（无 0.5）**（R3 修正，见 §9 R3-1~R3-3） |
| **需要修改什么** | **Recommender（T5）侧**训练循环增加一处受 `lambda_psa > 0` 守卫的挂载；**不得**挂到 RQ-VAE / Tokenizer loop |
| **前置 Task** | E5 |
| **验收标准** | ① 标量、可反传、无 NaN；② 正样本为自身时 loss 下降方向正确；③ 温度/负样本数变化行为合理；④ `lambda_psa = 0` 时总 loss 与 Clean TIGER 逐位相同 |
| **对应验证实验** | **EXP-E5/E6**（TIGER + SIA vs TIGER + SIA + PSA，类型 B + C 候选） |

> **PSA 的定义（不得改动）**：
>
> ```
> T5 decoder hidden 的序列位置 0  →  adapter  →  tokenizer reconstruction representation
> ```
>
> **不是** `RQ latent → external embedding` 的 latent alignment。
> **不得**重新引入已经裁撤的 E50 b2 / latent alignment 机制。

> **维度**：`D_target = RQ-VAE decoder 输出维度 = item content embedding 维度`（§1.3；当前 Toys / Sport / Beauty = **4096**，Electronics / Musical_Instruments / Video_Games = 768）。adapter 维度关系为 `d_model → D_target`。

> **设计注意**：目标侧取的是**重建表示**而非原始 content embedding——这与 SIA 取"码分布"形成互补（E4 在离散码空间，E6 在连续重建空间）。
> **生效阶段**：`L_PSA` **仅**在 Recommender（T5）阶段生效；Tokenizer 阶段不参与训练（§1.4）。
> **生效性的验证归属 E7–E9，不在 E6**：E6 只验证 PSA **loss 机制本身**（decoder hidden 序列位置 0 / adapter / reconstruction target / forward / backward / gradient / loss 正确性）。阶段生效由 **E7–E9** 统一验证。

---

### E7 — Tokenizer / recommender freeze interface

| 字段 | 内容 |
|---|---|
| **Task ID** | E7 |
| **任务名称** | Tokenizer / recommender freeze interface |
| **来源机制** | ETEGRec Alternating Optimization 的阶段切分（移植其调度逻辑，不移植其 trainer） |
| **TIGER 对应接口** | Tokenizer 侧 = [rqvae/main.py](../rqvae/main.py) + `rqvae/trainer.py`；Recommender 侧 = [model/main.py](../model/main.py) |
| **输入** | 阶段标识（`tokenizer` / `recommender`） |
| **输出** | 被正确冻结/解冻的参数集合；各阶段可独立跑通 |
| **需要新增什么** | 一个冻结控制层：Tokenizer 阶段冻结 T5；Recommender 阶段冻结 RQ-VAE/tokenizer；并在 resume 时校验冻结一致性 |
| **需要修改什么** | 只在新入口/新控制层内实现；**不修改** 两侧训练循环的内部逻辑 |
| **前置 Task** | E6 |
| **验收标准** | ① Tokenizer 阶段 T5 参数 `requires_grad=False` 且更新后 `state_dict` 逐位不变；② Recommender 阶段 RQ-VAE 参数同理；③ 两阶段各自可独立跑通；④ 冻结标记在 checkpoint / 日志中可审计；⑤ **参数状态与 loss 状态一致**（下表逐行核对，不得出现"参数已冻结但 loss 仍在生效"之类的错配） |
| **对应验证实验** | **EXP-E7-E9**（Fixed training vs Alternating training，类型 B + C 候选） |

**E7 freeze / unfreeze 与 loss 生效阶段表**（与 §1.4 一致，本表为 E7 的实现契约）：

| 组件 / loss | Tokenizer 阶段 | Recommender 阶段 |
|---|---|---|
| **RQ-VAE** | **trainable**（解冻） | **frozen** |
| **T5** | **frozen** | **trainable**（解冻） |
| adapter（E2 / E5） | frozen | **trainable** |
| `L_RQ-VAE` | **生效** | 不生效 |
| `L_SIA`（E4） | **不生效** | **生效** |
| `L_PSA`（E6） | **不生效** | **生效** |
| `L_T5` | 不生效 | **生效** |

> 阶段切换时必须显式重置 `requires_grad` 与 loss 开关，并在日志中打印本阶段的（参数状态 × loss 状态）矩阵，作为 ⑤ 的审计依据。

> **实现状态（R4，2026-10-14）**：已实现并通过验收 ①–⑤。控制层 = `tiger_ext/stage.py`（`StageFreeze`），
> 验证脚本 = `integration/exp_e7_stage_freeze.py`（**66/66，CPU + CUDA**）。本 Task **未修改** `model/` 与 `rqvae/`，
> 不改变两侧训练循环内部逻辑；`L_SIA` / `L_PSA` 的数学定义与 E4 / E6 核验通过时完全一致。
> 与 ETEGRec 官方交替训练的差异见 **§10 R4**。

#### E7 阶段生效统一验证项（E7–E9）

**loss 在正确训练阶段生效**这一验证，统一由 **E7–E9** 承担，不在 E4 / E6：

| 检查 | Tokenizer 阶段 | Recommender 阶段 |
|---|---|---|
| `L_SIA` 调用次数 | **= 0** | **> 0** |
| `L_PSA` 调用次数 | **= 0** | **> 0** |
| `L_RQ-VAE` 调用次数 | **> 0** | **= 0** |
| `L_T5` 调用次数 | **= 0** | **> 0** |
| T5 `requires_grad` | **False** | **True** |
| RQ-VAE `requires_grad` | **True** | **False** |

分工：

```
E4 / E6   →  验证 loss 机制本身（forward / backward / gradient / shape / 路径 / loss 正确性）
E7–E9     →  验证 loss 在正确的训练阶段生效
```

> 该分工使 E4 / E6 的验收**不依赖 E7**；prerequisite 链保持 `E1 → E2 → E3 → E4 → E5 → E6 → E7 → E8 → E9`，E7 的核心设计（阶段划分与冻结/解冻）不变。

---

### E8 — SID refresh

| 字段 | 内容 |
|---|---|
| **Task ID** | E8 |
| **任务名称** | SID refresh |
| **来源机制** | ETEGRec 交替循环中"码本更新后重算语义 ID"的环节 |
| **TIGER 对应接口** | [rqvae/generate_code.py](../rqvae/generate_code.py)（含 SK 去重循环与 duplicate 消解） |
| **输入** | 当前迭代的 TIGER RQ-VAE checkpoint；`data/<D>/item_emb.parquet` |
| **输出** | 新的全量 item SID `.npy`（`(n_items, 4)` int64） |
| **需要新增什么** | 每轮迭代的 SID 版本化命名与登记（写进 manifest，记录 ckpt、commit、时间） |
| **需要修改什么** | 无（复用现有 `generate_code.py`，通过 CLI 传参） |
| **前置 Task** | E7 |
| **验收标准** | ① SID 重新生成覆盖**全部** item；② **保持 TIGER 原有 collision handling**（`use_sk` 去重循环 + 第 4 列 duplicate 消解），不得自创去重逻辑；③ 所有 token ∈ `[0, 256)`；④ 与上一轮 SID 的差异率被记录；⑤ 本轮 SID 与所用 ckpt 的对应关系可回溯 |
| **对应验证实验** | 并入 **EXP-E7-E9** |

> **实现状态（R5，2026-10-14）**：已实现并通过验收 ①–⑤。新增 = `tiger_ext/sid_refresh.py`（**只驱动** `rqvae/generate_code.py` 的既有 CLI，**不自带任何编码机制**）+ 验证脚本 `integration/exp_e8_sid_refresh.py`（**55/55，CPU + CUDA**）。
> 本 Task **未修改** `rqvae/generate_code.py`、`clean_tiger/`、`model/`——`git status` 与 E7 结束时逐字相同。
> 实测：同设备重新生成与 clean_tiger 基线产出的 SID **逐位相同**；**SID 与设备相关**（CPU vs CUDA 差异 3.25%，只落在最后一层/collision 列）⇒ manifest 记录 device，diff rate 须对照该噪声底。详见 **§11 R5**。

---

### E9 — Alternating cycle

| 字段 | 内容 |
|---|---|
| **Task ID** | E9 |
| **任务名称** | Alternating cycle |
| **来源机制** | ETEGRec Alternating Optimization |
| **TIGER 对应接口** | E7 的冻结控制层 + E8 的 SID refresh |
| **输入** | 迭代轮数、每轮预算 |
| **输出** | 每轮的 tokenizer checkpoint + SID 版本 + recommender checkpoint |
| **需要新增什么** | 交替调度器：`Tokenizer → SID refresh → Recommender → Tokenizer → …`；每轮落盘 manifest |
| **需要修改什么** | 无既有文件 |
| **前置 Task** | E7、E8 |
| **验收标准** | ① 至少完成 2 个完整交替周期；② 每轮的 ckpt / SID / 指标链条可追溯；③ 中途中断可续跑；④ 随机性有种子控制（参照本项目既有 `--n_workers 1` / `ddof=1` 的可复现教训） |
| **对应验证实验** | **EXP-E7-E9**（对照：固定 tokenizer 的原始 TIGER 式训练 vs Alternating，类型 B + C 候选） |

> **实现状态（R6，2026-10-14）**：已实现并通过验收 ①–④。新增 = `tiger_ext/alternating.py`（`AlternatingScheduler` 纯调度 + `CycleManifest`）+ `tiger_ext/alt_pipeline.py`（把 E7 阶段机 / E8 refresh 和两个**冻结**训练循环接起来）+ 验证脚本 `integration/exp_e9_alternating.py`（**90/90，CPU + CUDA**）。`model/` / `rqvae/` / `clean_tiger/` **零改动**。详见 **§12 R6**。

---

### E10 — ETEGRec integration smoke test

| 字段 | 内容 |
|---|---|
| **Task ID** | E10 |
| **任务名称** | ETEGRec integration smoke test |
| **来源机制** | 集成验收（非来源机制） |
| **TIGER 对应接口** | E1–E9 全链 |
| **输入** | 极小预算配置（参照 `clean_tiger/smoke_test.sh` 的思路） |
| **输出** | 全链跑通的日志与产物清单；每个新增模块"是否被执行"的证据 |
| **需要新增什么** | 集成 smoke 脚本 + 执行证据探针（模块调用计数器，含阴性对照） |
| **需要修改什么** | 无既有文件 |
| **前置 Task** | E9 |
| **验收标准** | ① 全链 exit 0；② 每个新增模块的实际调用次数 > 0；③ 关闭所有开关时与 Clean TIGER **逐位等价**（门禁 G0）；④ 无历史机制（L_trans / L_balance / 历史 L_align / behavior / fused）被激活 |
| **对应验证实验** | **EXP-E10**（TIGER + ETE 正式模型实验的前置门禁） |

> **实现状态（R6，2026-10-14）**：已实现并通过验收 ①–④。新增 = `integration/exp_e10_smoke.py`（**64/64，CPU + CUDA**），**0 个既有文件改动**。全链 2 个完整交替周期 + G0 逐位等价 + G1 零激活 + 幂等 / resume 全绿；① 全链 exit 0 ✅ ② 每个新增模块调用次数 > 0（含**阴性对照**）✅ ③ 关闭开关时对 `5edfca0` 逐位等价 ✅ ④ 无历史机制被激活 ✅。详见 **§12 R6**。

---

## 3. CSG4Rec → TIGER 接口路线（C1–C6）

### 依赖图

```
C1 (CF embedding) ─→ C2 (kNN relation) ─→ C3 (Q-space 约束) ─→ C4 (CSG loss)
                                                                    │
                                                                    ↓
                                          C5 (trainer 集成) ─→ C6 (smoke)
```

---

### C1 — CF embedding interface

| 字段 | 内容 |
|---|---|
| **Task ID** | C1 |
| **任务名称** | CF embedding interface |
| **来源机制** | CSG4Rec 的 item-level collaborative embedding |
| **TIGER 对应接口** | 独立于 TIGER 主链的**旁路输入**；不得进入 RQ-VAE 输入路径 |
| **输入** | **SASRec item embedding**（`sasrec_output/<D>/seed2025_best.pth` 的 `item_emb.weight`，`(N+1, 50)`）；ItemCF 文件 `data/<D>/behavior_emb_itemcf.parquet` 保留为备用来源 |
| **输出** | `(n_items, 50)` float32 item 级协同向量 + 显式 `ItemID → 行号` 映射表 |
| **需要新增什么** | CF 向量来源与其加载/校验模块（**仅 item 级**） |
| **需要修改什么** | 无 |
| **前置 Task** | 无 |
| **验收标准** | ① item 数量与 `item_emb.parquet` 行数一致；② 维度明确且被记录；③ **无错误 item mapping**——必须显式校验 `ItemID` 与行号的对应关系（本项目已有"Beauty ItemID 是排列 ≠ 行号+1"的踩坑记录，此处必须按映射表取值，禁止 `a-1` 式取值）；④ 向量来源可追溯 |
| **对应验证实验** | **EXP-C1**（CF embedding sanity check，类型 A） |

> **C1 决策点 —— 已裁决（R7，2026-10-14）**：CF 向量采用
> **(b′) 从仓库既有 SASRec checkpoint 提取 item embedding 表**（`item_emb.weight`，50d，N+1 行含 padding）。
>
> 裁决依据来自对 `reference_repos/CSG4Rec_original` 的代码审计：CSG4Rec 的 `--cf_emb` 默认值就是 **SASRec 的 item embedding 产物**（`CSGT/main.py:54`，`./RQ-VAE/ckpt/Instruments-32d-sasrec.pt`），由 `CSGT/datasets.py:24-29` 直接 `np.load` → `torch.from_numpy` → 与 `text_emb_np` **行对齐**使用。原快照**不含 SASRec 训练器**（`CSGT/models/` 只有 `layers.py` / `rq.py` / `vq.py`，无 `models.rqvae`），无法逐字复现生成端；但本仓库已有**同一语义**的 SASRec 基线（`tools/run_sasrec.py`，`nn.Embedding(item_num + 1, 50, padding_idx=0)`），可直接取其 `item_emb.weight`。
>
> 因此 **ItemCF-128 不作正式 CSG 来源**（其口径是本项目的 ItemCF cosine + SVD 行为变体，与 CSG4Rec 的 SASRec CF 语义不同）；该文件**保留、不删除**，在接口中作为**备用来源**可按 `source="itemcf"` 显式加载。
>
> **由本次裁决引出的两条硬约束**：① **不允许 `item_id - 1`**——必须按显式 `ItemID → 行` 映射取值（Beauty 有 4158 行会被 `a-1` 取错）；② **不允许因为 128d → 50d 而新增 projection / normalization**——C1 接口本就要求下游不得假定固定维度。padding 行（第 0 行）**显式丢弃并断言为全零**。
>
> **硬约束**：**不引入 User SID**、不引入用户侧建模。

> **实现状态（R7，2026-10-14）**：已实现并通过验收 ①–④（**EXP-C1 132/132**）。新增 = `tiger_ext/cf_embed.py`（`CFEmbedding` 接口 + 两种来源的加载器）+ `integration/exp_c1_cf_embedding.py`。**既有 tracked 文件零改动**（`git status` 与开工前逐字相同）。详见 **§13 R7**。

---

### C2 — Collaborative relation

| 字段 | 内容 |
|---|---|
| **Task ID** | C2 |
| **任务名称** | Collaborative relation |
| **来源机制** | CSG4Rec 的协同相似度 / kNN 关系 |
| **TIGER 对应接口** | 复用既有"关系图 `.npy` 形态 `(N, 2)` → `trans_dict`"的加载骨架（[rqvae/trainer.py:99-108](../rqvae/trainer.py#L99-L108)），**但使用独立文件与独立开关**，不得复用历史 `trans_graph_path` 开关语义 |
| **输入** | C1 的 `(n_items, d_cf)` |
| **输出** | 关系对文件（`(N, 2)` int）+ 关系统计（边数、度数分布、对称性） |
| **需要新增什么** | cosine 相似度 → top-k kNN → 关系对落盘 + 诊断产物 |
| **需要修改什么** | 无 |
| **前置 Task** | C1 |
| **验收标准** | ① cosine 计算正确（抽样手算校验）；② kNN 关系数量符合 `n_items × k` 的量级且被记录；③ **关系是否合理**——用定性样例（高相似 item 是否真的共现/同类）做人工抽检并留档；④ 关系对无自环；⑤ 行号映射与 C1 一致 |
| **对应验证实验** | **EXP-C2**（Collaborative kNN diagnostic，类型 A+B） |

> **实现状态（R8，2026-10-14）**：已实现并通过验收 ①–⑤（**EXP-C2 127/127**，exit 0）。新增 = `tiger_ext/collab_relation.py`（`CollabRelation` + `build_relation(_for_dataset)`，numpy 叶子）+ `integration/exp_c2_collab_relation.py`（8 节 127 项）+ `integration/artifacts/collab_relation/<D>_k20.{neighbors,similarity,pairs,pairs_itemid,item_ids}.npy` 与 `<D>_k20.json`。**既有 tracked 文件零改动**。详见 **§14 R8**。

---

### C3 — Q-space collaborative constraint

| 字段 | 内容 |
|---|---|
| **Task ID** | C3 |
| **任务名称** | Q-space collaborative constraint |
| **来源机制** | CSG4Rec 的协同约束（**移植目标是它作用在量化空间这一形式**） |
| **TIGER 对应接口** | TIGER RQ-VAE 的**量化后表示**：`x_q`（`RQVAE.forward(..., return_xq=True)`）或码分配分布（E3 的 code logits） |
| **输入** | C2 的关系对；一批 item 的量化表示 |
| **输出** | 每 batch 的约束张量/标量中间量 |
| **需要新增什么** | 约束构造模块：以 kNN 关系为监督，在量化空间上约束"协同近邻 → 码结构/量化表示相近"，形式需与 C4 的 loss 配套 |
| **需要修改什么** | 受开关守卫的挂载点（`lambda_csg` 默认 0） |
| **前置 Task** | C2 |
| **验收标准** | ① 约束确实读取的是**量化空间**张量（`x_q` 或 code logits），有运行时张量来源断言；② 关系对中含未在 batch 内的 item 时行为定义明确；③ `lambda_csg = 0` 时无任何额外计算与读取；④ 量化算法本体未被改动 |
| **对应验证实验** | 并入 **EXP-C3/C4**（Q-space constraint training sanity test，类型 A） |

> **两条明令禁止**（这是 C3 的成败判据）：
> 1. ❌ **不得**把 CF embedding concat 到 TIGER RQ-VAE 输入。
>    理由：那会退化成"content + behavior 融合"，本项目已有大量该路线的历史结论，且直接破坏"content → RQ-VAE"单一表征链。
> 2. ❌ **不得**直接做 CF embedding ↔ RQ-VAE latent 的简单 alignment。
>    理由：那只是把协同信号灌进 latent，属于输入空间/表示空间的直接对齐，**不是** Q-space 约束，会丢掉"约束量化行为"这一核心区别。

---

### C4 — CSG loss

| 字段 | 内容 |
|---|---|
| **Task ID** | C4 |
| **任务名称** | CSG loss |
| **来源机制** | CSG4Rec 的协同约束损失 |
| **TIGER 对应接口** | RQ-VAE 训练循环（结构参照 [rqvae/trans_loss.py](../rqvae/trans_loss.py) 与 trainer 的加权挂载） |
| **输入** | C3 的约束中间量 |
| **输出** | 标量 `L_CSG` |
| **需要新增什么** | 新 loss 模块；权重 `lambda_csg` 默认 0 |
| **需要修改什么** | trainer 增加一处受守卫的挂载 |
| **前置 Task** | C3 |
| **验收标准** | ① 标量、可反传、无 NaN；② 构造性测试：关系为纯噪声时 loss 行为与关系为真时**可区分**；③ `lambda_csg = 0` 时总 loss 与 Clean TIGER 逐位相同；④ 与 L_trans / L_balance / 历史 L_align 无耦合（各自独立开关） |
| **对应验证实验** | 并入 **EXP-C3/C4** |

---

### C5 — CSG trainer integration

| 字段 | 内容 |
|---|---|
| **Task ID** | C5 |
| **任务名称** | CSG trainer integration |
| **来源机制** | 工程集成（非来源机制） |
| **TIGER 对应接口** | [rqvae/main.py](../rqvae/main.py) + `rqvae/trainer.py` |
| **输入** | C4 的 loss + C1/C2 的旁路数据 |
| **输出** | 可运行的 RQ-VAE 训练入口（新入口，不动既有入口） |
| **需要新增什么** | 新驱动入口与配置；含"关闭即等价"的开关组 |
| **需要修改什么** | 仅受守卫的最小挂载；不改既有训练流程 |
| **前置 Task** | C4 |
| **验收标准** | ① Q-space 约束参与后可正常训练若干 step；② loss / 梯度 / 码本使用率无异常（无码本塌缩或 NaN）；③ 关闭开关后与 Clean TIGER 逐位等价（门禁 G0）；④ 历史机制（L_trans / L_balance / 历史 L_align）**调用次数为 0**（用计数器探针验证，含阴性对照） |
| **对应验证实验** | **EXP-C5/C6** 的前置门禁 |

---

### C6 — CSG smoke test

| 字段 | 内容 |
|---|---|
| **Task ID** | C6 |
| **任务名称** | CSG smoke test |
| **来源机制** | 集成验收（非来源机制） |
| **TIGER 对应接口** | C1–C5 全链 + Clean TIGER 的 SID/T5/eval 段 |
| **输入** | 极小预算配置 |
| **输出** | 全链日志 + 产物清单 + 执行证据 |
| **需要新增什么** | CSG smoke 脚本 |
| **需要修改什么** | 无 |
| **前置 Task** | C5 |
| **验收标准** | ① 全链 exit 0；② C1→C5 各模块调用次数 > 0；③ 关闭开关时逐位等价于 Clean TIGER；④ 无 User SID 引入的痕迹 |
| **对应验证实验** | **EXP-C5/C6**（TIGER vs TIGER + CSG，类型 B + C） |

---

## 4. 全局门禁

### G0 — 关闭即等价（每个 Task 都必须过）

所有新增机制默认权重为 0 / 开关为关。**关闭状态下，Clean TIGER 的全部可观测行为必须逐位等价**：

| 观测点 | 判定方式 |
|---|---|
| RQ-VAE loss | 同 seed 同数据，逐位相同 |
| RQ-VAE 码本 / 分配结果 | `get_indices` 输出逐位相同 |
| 生成的 SID `.npy` | 逐位相同 |
| T5 loss / logits | `torch.equal`（非 allclose） |
| T5 生成结果 | `generate()` 输出逐位相同 |
| 评测指标 | 逐位相同 |
| 文件读取集合 | 不含任何历史/新增旁路文件 |

> 判定必须用**逐位相同**（`torch.equal` / `np.array_equal`），不接受 allclose。历史上已有"`md5` 跨环境会漂移"的教训，故以同环境内的逐位比较为准。

### G1 — 历史机制零激活

`L_trans` / `L_balance` / 历史 `L_align` / behavior_emb / fused_item_emb 的调用次数必须为 **0**，且该结论必须由**带阴性对照的执行探针**给出（探针需证明自己能在机制被激活时报警）。

### G2 — 只读边界

`clean_tiger/` 是冻结参照，**永久只读**。移植代码一律放在 `integration/` 下的新目录与既有 `rqvae/` / `model/` 的受守卫挂载点。

---

## 5. 阶段性实验验证规则

> **本项目硬规则：每完成一个有实际代码意义的移植阶段，都必须进行对应验证实验，然后才能进入下一阶段。**
> **不要等所有模块全部迁移完成后才做实验。**

### 5.1 三类实验

| 类型 | 名称 | 目的 | 是否进论文 |
|---|---|---|---|
| **A** | 工程正确性验证 | 确认接口与原始 TIGER 行为**没有被破坏** | ❌ 不进论文 |
| **B** | 机制验证 | 确认新机制**实际产生了预期效果** | ⚠️ 有论文价值才纳入 |
| **C** | 正式论文实验 | 最终论文结果 | ✅ 进论文 |

### 5.2 阶段门禁

每个 Task 完成后必须：
1. 跑完其对应的 **A 类**实验并全部通过（否则不得进入下一 Task）；
2. 若该 Task 引入了新机制，再跑 **B 类**实验看有无效果；
3. 只有 A、B 都通过，才考虑把它升格为 C 类候选。

### 5.3 设计演进

最终论文实验计划**不能在代码尚未验证前完全锁死**。应根据阶段性 A/B 结果逐步收敛控制组集合（见 §7 的组合实验矩阵）。

---

## 6. 里程碑

| 里程碑 | 内容 | 出口条件 |
|---|---|---|
| **M0** | Clean TIGER 冻结 | ✅ DONE（commit `5edfca0`） |
| **M0.5** | 接口移植计划 + 阶段性实验验证计划 | 本文件 + [experiments.md](experiments.md) + [status.md](status.md) |
| M1 | E1 完成 | EXP-E1 通过（A） |
| M2 | E2–E4 完成 | EXP-E2 / E3 通过（A），EXP-E4 有结果（B） |
| M3 | E5–E6 完成 | EXP-E5/E6 有结果（B） |
| M4 | E7–E9 完成 | EXP-E7-E9 有结果（B） |
| M5 | E10 完成 | 门禁 G0/G1 通过；TIGER + ETE 可跑通 |
| M6 | C1–C4 完成 | EXP-C1 / C2 通过（A），EXP-C3/C4 通过（A） |
| M7 | C5–C6 完成 | 门禁 G0/G1 通过；TIGER + CSG 可跑通 |
| M8 | 组合实验 I1–I4 | 五个控制组结果齐备 |
| M9 | 正式论文实验 | 按 §7 收敛后的矩阵执行 |

---

## 7. 最终组合实验

> **⚠️ 本节已被 §19 R13 取代。**
> §7 的 5 组并列设计在结构上无法测 `C×E` 交互，且 `I4` 当时无代码路径。R13 收敛为 **2×2×2 全因子 8 格**，并保留 `I1–I4` 作为别名（`I1=#2`、`I2=#7`、`I3=#4`、`I4=#8`）。
> **下面保留原文作为设计演进记录，不得再据此开工。**

| ID | 组合 | 说明 |
|---|---|---|
| **I1** | TIGER + CSG | 只开 C 线 |
| **I2** | TIGER + ETE | 只开 E 线（SIA + PSA） |
| **I3** | TIGER + ALT | 只开交替优化（E7–E9） |
| **I4** | Full | C 线 + E 线 + ALT 全开 |

### 7.1 I2 / I3 的训练范式与潜在重叠

**I2 的训练范式**：I2 在**当前计划中定义为「ETE + Alternating」**。理由：按 §1.4 的阶段契约，`L_SIA` / `L_PSA` 只在 Recommender 阶段生效、adapter 也只在 Recommender 阶段可训练，而 tokenizer 侧需要随之更新才能让"目标码分布 / 重建表示"随训练演进——这正是 Alternating 存在的意义。因此 **ETE 自然包含交替优化**。

**I3 是什么**：I3 是**只开交替调度（E7–E9）、不带 SIA/PSA** 的控制组，即"纯 tokenizer 刷新 + 两阶段交替"本身的效果。

> ⚠️ **潜在重叠（不得预设正交）**：若 I2 确实采用了 Alternating，则 **I2 与 I3 不是正交的并列控制组**——I3 变成 I2 的一个消融（去掉 SIA/PSA，保留调度）。
>
> 因此 §7 不预设 I2/I3 完全正交。**两者的实际关系要在 E7–E9 实现完成后才能确定**：
>
> - 情况 A：实现中发现 ETE 可以在**固定 tokenizer**下训练（例如适配器与 T5 可训练、码本不动也能收敛）→ 则 I2 可改用固定训练，I2/I3 正交，5 组并列成立；
> - 情况 B：ETE 必须依赖 Alternating → I2 ⊃ I3，矩阵应改写为
>   `TIGER` / `TIGER + ALT` / `TIGER + ALT + SIA` / `TIGER + ALT + SIA + PSA` / `TIGER + CSG` / `Full`，
>   并把 I3 作为 I2 的消融行而非并列行。

**建议正式实验矩阵**（先按此准备，最终由阶段性结果收敛）：

```
TIGER            (Clean TIGER 冻结基线 = 5edfca0)
TIGER + CSG
TIGER + ETE      (范式见 7.1，待 E7-E9 实现后按情况 A/B 定稿)
TIGER + ALT
Full
```

> **后续根据实际结果决定最终论文是否保留所有控制组。**
> 例如：若 I3 相对 I2 无增益，则 ALT 从主表降级为附录讨论；若 I1 与 I2 存在明显交互，则需补 2×2 交互分析。
> **不提前下结论。**

---

## 8. 本阶段（M0.5）的禁止事项

M0.5 期间，以下动作一律不得执行：

- ❌ 修改 `model/`、`rqvae/`、`clean_tiger/`、`tools/`、`doc/`
- ❌ 执行 E1（T5 hidden-state interface 改造）
- ❌ 执行 C1（CF embedding 接入）
- ❌ 任何模型训练
- ❌ 移植 CSG4Rec / ETEGRec 的任何代码
- ❌ 复制任何源项目代码

M0.5 **只修改 `integration/` 下的管理文档**。

> 本节为 M0.5 阶段的历史约束。M0.5 已 `DONE / CLOSED`，E 线自 E1 起正常开工，本节不再适用。

---

## 9. 修订记录 R3 — E1–E6 源码核验与修正（2026-10-14）

**触发**：E1–E6 实现完成后，对 `/data/gtx/project/code/reference_repos/ETEGRec_original` 做了一次 SIA / PSA Source-of-Truth Audit，逐项比对机制定义。

**核验范围与结论**：

| 机制 | ETEGRec 原实现 | 核验结论 |
|---|---|---|
| SIA | `trainer.py::compute_discrete_contrastive_loss_kl`（`kl_loss`） | seq 侧 = `encoder_last_hidden_state` masked mean → `enc_adapter` → **直接进 `rq`，绕过 encoder**；tgt 侧 = 冻结 `semantic_embedding` → `RQVAE.forward` → 同一组 3 层码本；比较**逐层 code logits**（非 T5 CE logits）；symmetric KL，无 temperature、无 detach ⇒ **与 E2–E4 一致** |
| PSA | `trainer.py::compute_contrastive_loss`（`dec_cl_loss`） | seq 侧 = `decoder_hidden_states[-1][:, 0, :]` → `dec_adapter`（`MLPLayers([in,out])` 退化为 `Linear`）⇒ **与 E5 一致**；tgt 侧 = `model_id(x)[0]` = **RQ-VAE 重建** ⇒ **确认 E6"靶 = 重建"的选择是忠实的** |

**据此修正的 4 处（本次唯一改动）**：

| 编号 | 位置 | 修正 |
|---|---|---|
| R3-1 | `tiger_ext/psa_loss.py` | PSA similarity：裸点积 → **cosine**（两侧 `F.normalize`），对应官方 `sim='cos'` |
| R3-2 | `tiger_ext/psa_loss.py` | PSA temperature 默认值 1.0 → **0.07**（官方法调用点默认值，config `tau: 0.07`） |
| R3-3 | `tiger_ext/psa_loss.py` | PSA 对称项 `0.5*(CE1+CE2)` → **`CE1+CE2`**（官方为 `cl(a,b)+cl(b,a)`，无 0.5） |
| R3-4 | `rqvae/models/vq.py` | SIA code logits `-raw_d` → **`+raw_d`**（官方 `dist='l2'` 分支返回未取负的 `d`） |

> R3-4 备注：官方同一份代码的 `dist='dot'/'cos'` 分支返回 `-similarity`，与 `l2` 分支的 `+d` 方向相反，判断为 ETEGRec 自身的不一致；本计划选择**字面保真**（复刻 `dist='l2'` 行为）。

**核验后仍未复刻的 3 处（已记录，不在 R3 范围内）**：batch 内按 target 去重（`np.unique(..., return_index=True)`）、`warm_epoch=10` 的 loss 权重预热、rec/id 相位的 `requires_grad` 交替切换。
（**R4 更新**：第 3 处已由 E7 实现，见 §10；剩 2 处仍不复刻。）

**验证状态**：E1–E6 全部工程 / 机制验证通过（见 [experiments.md §6](experiments.md)），原始 TIGER 逐位回归通过。
**⚠️ 与本计划的方法学效果无关**：R3 只修正**机制忠实度**，**不代表** SIA / PSA 的推荐效果已被验证——推荐效果属 B / C 类实验，**尚未执行**（见 [experiments.md §3](experiments.md)）。

---

## 10. 修订记录 R4 — E7 实现（2026-10-14）

**触发**：执行 E7（Tokenizer / recommender freeze interface），实现 §1.4 / 本节 E7 表定义的阶段契约。

**产物（仅 3 个文件，均在冻结内核之外）**：

| 文件 | 内容 |
|---|---|
| `tiger_ext/stage.py` | 冻结控制层 `StageFreeze`；阶段契约常量 `STAGE_CONTRACT`；码本工具 `seal_codebooks` / `load_tokenizer_state_dict` / `codebook_initted` / `codebook_is_zero`；审计清单 `write_stage_manifest` / `read_stage_manifest` / `verify_stage_manifest` |
| `tiger_ext/__init__.py` | 导出上述符号 |
| `integration/exp_e7_stage_freeze.py` | 66 项验证（10 节） |

**`model/` 与 `rqvae/` 零改动**：`git diff --stat` 与 E6 时相同（仅 E1/E3/E4/E6 的既有加法式改动）。控制层不导入两侧训练循环，两侧训练循环也不导入控制层。

**设计要点（不新增机制，只做开关与校验）**：

1. `apply(stage)` 只做两件事：按契约显式重置各组 `requires_grad`，并以 `mounts_for_stage()` 在**调用方**重置 loss 开关（Tokenizer 阶段把 `sia` / `psa` 置 `None`，Recommender 阶段原样透传）。**不重写任何训练循环。**
2. `assert_consistent()` 实现验收 ⑤：既检查"该冻结的组里没有可训练参数"，也检查 **`LOSS_REQUIRES`**（生效的 loss 所需的组不得在同一阶段被冻结）。
3. `seal_codebooks()` 在进入 Recommender 阶段时把 `vq.initted` 置 `True`；`strict=True` 时**拒绝**封存全零（从未初始化）的码本，避免把占位码本冻结成假状态。
4. 审计：`signature()` / `<ckpt>.stage.json` 记录每组冻结切分、loss 开关与码本 `initted` 状态；`verify_stage_manifest()` 在 resume 时拒绝**跨阶段**或**冻结切分不一致**的 checkpoint。

**与 ETEGRec 官方交替训练的差异（已记录，本计划为准）**：

| 项 | ETEGRec 官方 | 本计划（§1.4 / E7 表） | 处理 |
|---|---|---|---|
| 相位切换条件 | `epoch_idx % cycle == 0`（`cycle=2`，另有 `warm_epoch=10`） | 由 E8 / E9 决定调度，E7 只提供阶段机 | **不移植**调度，E7 只实现"进入某阶段后的冻结/解冻与 loss 生效" |
| Tokenizer 阶段的 `kl_loss` / `dec_cl_loss` | `_train_epoch_id` **仍计算** `kl_loss` 与 `dec_cl_loss`（用 `id_*` 权重） | §1.4 规定 `L_SIA` / `L_PSA` **仅** Recommender 阶段生效 | **以本计划为准**（本计划是 E7 的显式契约；`warm_epoch` 与官方 trainer 全量复刻已列在"未复刻"清单） |
| `semantic_embedding` 例外 | Recommender 阶段除 `semantic_embedding` 外全部解冻 | TIGER 无对应模块 | 不适用 |

**顺带查实的两条集成事实（写入 [status.md](status.md)，E9 / E10 需遵守）**：

- **adapter 不在 `TIGER.parameters()` 内**。Recommender 阶段优化器必须显式覆盖 adapter；冻结的 `model/main.py::__main__` 入口用 `optim.Adam(model.parameters())`，**不会**训练 adapter。
- **"frozen" ≠ `eval()`**。Recommender 阶段 RQ-VAE 仍被**执行**（PSA 目标侧需要其重建），只是 `requires_grad=False`；实测一次 λ>0 的 epoch 中 RQ-VAE 前向被调用（encoder 8 次 / decoder 4 次）而 **0/23** 参数移动、epoch 末梯度全为 `None`。`L_RQ-VAE` 作为**训练目标**在 Recommender 阶段关闭，与其前向被调用不矛盾。

**验证状态**：EXP-E7 **66/66（CPU + CUDA）**；E1–E6 的 209 项 A 类验证在 E7 之后**全部重跑通过**。
**⚠️ 无效果结论**：E7 只验证阶段契约（参数冻结范围、梯度流、loss 生效、码本状态、checkpoint 恢复与回归）。**EXP-E7-E9 的 B / C 类效果实验仍未执行**，推荐效果未被验证。

---

## 11. 修订记录 R5 — E8 实现（2026-10-14）

**触发**：执行 E8（SID refresh），按本文件 E8 表的验收标准 ①–⑤ 落地。

**产物（2 个新文件，`model/` / `rqvae/` / `clean_tiger/` 零改动）**：

| 文件 | 内容 |
|---|---|
| `tiger_ext/sid_refresh.py` | 版本化命名（`<D>_sid_v<NNNN>_<ckpt_sha8>.npy`）+ provenance manifest（`<sid>.sid.json`，沿用 E7 的 sidecar 约定）+ 原子提交 + 可选 `publish` 交付；**通过既有 CLI 驱动** `rqvae/generate_code.py`，不自带编码机制 |
| `integration/exp_e8_sid_refresh.py` | 55 项验证（8 节，含**真实**生成：11924 item / 4096-d / 真实 ckpt，CPU + CUDA） |

**验收结果（EXP-E8，55/55，CPU + CUDA）**：

| 验收 | 结果 |
|---|---|
| ① 覆盖全部 item | 11924 行 == item embedding 行数（从 ckpt 的 `data_path` 独立读出） |
| ② 保留原有 collision handling | 复用 `generate_code.py` 逐字未改；`use_sk` 去重循环后**零重复行**（11924/11924 唯一；前置 3 列重复率 3.92%，最大组 21） |
| ③ token ∈ [0,256) | 实测 min 0 / max 255；**注意第 4 列是 duplicate 计数器**，其上界是"最大重复组大小"（本例 20），非结构性约束 |
| ④ 与上一轮差异率 | 同设备重跑 **= 0（逐位确定性）**；跨设备 **3.25%**（见下） |
| ⑤ ckpt ↔ SID 可回溯 | manifest 记录 ckpt sha256、generator sha256、repo commit+dirty、时间、**device**、ckpt 训练参数；`verify_sid_manifest()` 拒绝 ckpt 字节不符或 SID 被改动 |

**实测发现（重要，影响 E9）**：

1. **SID 与设备相关**：同一 ckpt 在**同设备**上重新生成与 clean_tiger 基线产出的 SID **逐位相同**（CUDA 上 100% 一致），但 **CPU 与 CUDA 相差 3.25%**（388/11924），差异**只落在第 2、3 列**（最后一层 + collision 列，SK 路径对边界 item 的判定）。⇒ manifest 记录 device；**diff rate 必须对照该噪声底**，低于它的差异不能解读为码本变化。
2. **`generate_code.py` 不能在 `python -I` 下运行**：它 `from datasets import EmbDataset`，隔离模式会解析到 HuggingFace 的 `datasets` 包（且 `-I` 丢弃 `PYTHONPATH`）。唯一可靠调用方式是纯解释器 + `cwd=rqvae/`（`run_generator` 即如此）。E9/E10 驱动该脚本时须遵守。
3. **实测成本**：单次全量 SID 生成 ≈ 18 s（Toys 11924 item，CPU 与 CUDA 相近）。

**未决（留给 E9）**：E8 只产出**新版本 + manifest**，**不决定**谁把 T5 的输入指向新 SID——`publish=...` 是显式 opt-in，默认不发布。该交接时机属 E9 的调度职责（本文件 E9 表的 `Tokenizer → SID refresh → Recommender` 顺序）。

**⚠️ 无效果结论**：E8 只验证"SID 能被正确、可追溯地重新生成"。**EXP-E7-E9 的 B / C 类效果实验仍未执行**，推荐效果未被验证。

---

## 12. 修订记录 R6 — E9 / E10 实现（2026-10-14）

**触发**：执行 E9（Alternating cycle）与 E10（ETEGRec integration smoke test），按本文件 E9 / E10 表的验收标准落地。**E 线的实现部分至此收口。**

**产物（3 个新文件，`clean_tiger/` / `model/` / `rqvae/` 零改动）**：

| 文件 | 内容 |
|---|---|
| `tiger_ext/alternating.py` | **纯编排层**：`AlternatingScheduler` 状态机 + `CycleManifest` + handoff 校验。**不 import torch/numpy** ⇒ 4 个 phase 体由调用方注入，分支 / 失败 / 恢复逻辑可在**零训练**下测 |
| `tiger_ext/alt_pipeline.py` | 4 个 phase 体：**只调用** E7 阶段机 + E8 refresh + 两个**冻结**训练循环，不自带机制 |
| `integration/exp_e9_alternating.py` | 90 项验证（4 节：纯调度 → 注入 fake phase → **真实链路 2 cycle** → 回归） |
| `integration/exp_e10_smoke.py` | 64 项验证（5 节：探针 + 阴性对照 → G0 非同源逐位 → 全链 2 cycle + G0/G1 → 边界 → 回归）。**唯一产物，0 个既有文件被改** |

**E9 周期规则（关闭 status.md 原「已知问题」第 4(b) 条留白）**：1 phase = 1 epoch；1 cycle = `1 Tokenizer + 1 Recommender`；相位判定用 **0-based** `epoch_idx % cycle == 0`，与 ETEGRec 官方 `trainer.py:513/542` 及 `config/scientific.yaml` 的 `cycle: 2` 一致。**1-based 读法会把 T/R 整体反相**，脚本内有对照断言。

**验收结果（EXP-E9，90/90，CPU + CUDA，含 2 个真实交替周期）**：

| 验收 | 结果 |
|---|---|
| ① ≥2 个完整交替周期 | ✅ 2 cycle 端到端；产物链含不可变 SID 版本 v1/v2/v3 |
| ② ckpt / SID / 指标链条可追溯 | ✅ `.stage.json` 阶段正确；SID sidecar 的 `ckpt_sha256` 指向**本 cycle** 的 tokenizer；evaluation 只挂 recommender epoch |
| ③ 中途中断可续跑 | ✅ 四状态机：`started` → 重跑 Tokenizer；`tokenizer_done` → Refresh + Recommender；`refresh_done` → **直接 Recommender（Refresh 不重复）**；`recommender_done` → 下一 cycle。恢复前校验 ckpt/SID 的 sha256，被篡改抛 `ResumeError` |
| ④ 随机性有种子控制 | ✅ 种子按 phase 派生；同 seed 重跑 SID 逐位一致 |

**实测关键判定（E9）**：Tokenizer 阶段 RQ-VAE **15/15** 张量动、T5 / 双 adapter **0 张量**动；Recommender 阶段 RQ-VAE **0/15** 动且 epoch 末**0 个冻结参数带梯度**（前向仍执行，是 PSA 靶侧），T5 26/26 + adapter 4 张量动；`verify_sid_consistency` 逐 item 验 dataset↔SIA↔PSA 三方路由一致（用别的 SID 建的 dataset/mount **立刻失败**，负对照已实测）；refresh 失败 ⇒ `RefreshAborted`，Recommender **绝不运行**；**从不 `publish`**，交接靠每 cycle 的 `--code_path` 指向不可变版本文件。

**三处必须上报的偏离（E9）**：① `PHASE_WARMUP_EPOCHS = 0`（冻结 `Trainer` 每 phase 重建 optimizer + linear schedule，50-epoch warmup 会超过单 phase 的 step 预算；实际值记入 `run.json`）；② Tokenizer 优化器**归 `rqvae/trainer.py` 所有**（该文件冻结），E9 不改它而是**断言** `trainer.optimizer` 的参数 id 集合 == `freeze.trainable_parameters('tokenizer')`；Recommender 优化器才由 E9 用 `trainable_parameters('recommender')` 显式建（含 T5 ∪ adapter）；③ **initial SID refresh 在本 port 没有训练消费者**——仍按规格执行，作为版本链根 + cycle-0 的 diff 基线。

**验收结果（EXP-E10，64/64，CPU + CUDA，exit 0）**：

| 验收 | 结果 |
|---|---|
| ① 全链 exit 0 | ✅ `E10Syn` fixture（1024 item × 32d，24 train / 8 valid）跑真实 2 cycle，CPU 8s / CUDA 8s |
| ② 每个新增模块调用次数 > 0 | ✅ E1–E9 共 **23 个入口**全部 > 0，且**每个探针都带阴性对照**（⇒ 关闭时的 0 是"测出来的"，不是"没测"） |
| ③ 关闭即逐位等价（G0） | ✅ **非同源**对比：从 `5edfca0` 用 `git show` 抽原始源码并排导入，同权重同输入下 T5 单 epoch mean loss / **89 个梯度** / **92 个 state 张量**、`get_indices`、`RQVAE.forward` 4-tuple、`generate()` token、eval 指标**全部逐位相同** |
| ④ 无历史机制被激活（G1） | ✅ tripwire **0 次**触发；链上 3 个 `Trainer` 的 `lambda_trans/lambda_balance/align_weight` 全为 0 且 `trans_dict/align_table/align_proj` 全 `None` |

**E10 另外两项**：**幂等 / resume**（完成后再 `run()` 是 no-op；refresh 失败 ⇒ `RefreshAborted` + state 停在 `tokenizer_done`，解封后恢复只补 cycle 1 的 tokenizer）；**关闭对照**（`lambda_sia=lambda_psa=0` ⇒ SIA/PSA 链 284→**0** 次调用、`TIGER.forward(return_hidden=True)` **0** 次、mount 以 `enabled=False` 到达冻结循环；单 batch 逐位复现 current 与 pristine 两条 clean TIGER 路径）。

**回归**：E1–E9 九个 suite **411/411** 全绿。⚠️ E10 内层给 E9 设 `EXP_E9_REGRESSION=skip` ⇒ E10 里 E9 报 **81/81** 而非独立跑的 **90/90**，两者相加**无重复**。

**⚠️ 无效果结论**：E9 / E10 都只验证"调度与集成正确"，**EXP-E7-E9 的 B / C 类效果实验仍未执行**，推荐效果未被验证。

---

## 13. 修订记录 R7 — C1 实现 + CSG CF 来源裁定（2026-10-14）

**触发**：执行 C1（CF embedding interface），CSG 线开工。本轮同时**裁定了正式 CF 表征的来源**（本文件 §3 C1 的"决策点"原为留白）。

### 13.1 来源裁定（用户裁决，推翻首轮的 ItemCF 路线）

| 项 | 内容 |
|---|---|
| **正式来源** | **SASRec item embedding** — `sasrec_output/<D>/seed2025_best.pth` 的 `item_emb.weight`，**50d** |
| 依据 | CSG4Rec 的 `--cf_emb` 默认值就是 SASRec 的 item embedding 产物（`CSGT/main.py:54` → `Instruments-32d-sasrec.pt`；`CSGT/datasets.py:24-29` 直接 `np.load` → `torch.from_numpy` → 与 `text_emb_np` **行对齐**）⇒ **其 CF 向量 = SASRec item embedding 表，不是 ItemCF 矩阵** |
| 为何不复刻生成端 | 参考快照**不含 SASRec 训练器**（`CSGT/models/` 仅 `layers.py`/`rq.py`/`vq.py`，无 `models.rqvae`）⇒ 取本仓库**同语义**的 SASRec 基线 `tools/run_sasrec.py`；seed 选 **2025**（v2 对齐协议的 3-seed 基线；v1 的 `sasrec_<D>_best.pth` 有 off-by-one eval bug，sha256 亦不同） |
| ItemCF-128 的处置 | **保留、不删除**。它**不是**正式 CSG 来源（口径 = 本项目 ItemCF cosine + SVD 128d，与 CSG4Rec 语义不同）；接口按 `source="itemcf"` 显式加载，`is_default_source=False` |

### 13.2 产物与验收

| 文件 | 内容 |
|---|---|
| `tiger_ext/cf_embed.py` | **纯 numpy 叶子**：`CFEmbedding` 接口（加载 / 校验 / 建 id 映射 / 回存 / 留 provenance），**不自算任何表征**。**核心接口未重写**，新增的只是两个加载器 `load_sasrec_item_embedding` / `load_cf_embedding` |
| `integration/exp_c1_cf_embedding.py` | 132 项验证（8 节，含 SASRec 抽取契约的独立复算与负对照） |
| `integration/artifacts/cf_embed/{Toys,Sport,Beauty}.json` | 正式源的 provenance manifest |
| `integration/artifacts/cf_embed/fallback_{Toys,Sport,Beauty}.json` | 备用源（ItemCF）的 provenance manifest |

**验收（EXP-C1，132/132，CPU + CUDA，exit 0）**：① item 数量 == `item_emb.parquet` 行数 ✅ ② 维度明确且被记录（50d）✅ ③ **无错误 item mapping** ✅ ④ 来源可追溯 ✅。**既有 tracked 文件零改动**（`git status --porcelain` 与开工前逐字相同）。

### 13.3 抽取契约（实测，三数据集）

ckpt 的 `item_emb.weight` = **`(N+1, 50)` float32**（Toys 11925 / Sport 18358 / Beauty 12102）；**第 0 行 = `padding_idx`，恒为全零**；ItemID 空间恰为 **1..N**，无 id 落在 padding 上 ⇒ 规范表 = **`W[canonical ItemID]`**，**不是 `W[1:]`**。

| 数据集 | n_items | d_cf | ckpt 行数 | `ItemID == 行+1` ? | 全零行（SASRec） | 全零行（ItemCF 备用源） |
|---|---|---|---|---|---|---|
| Toys | 11924 | 50 | 11925 | ✅ 是 | 0 | 24 |
| Sport | 18357 | 50 | 18358 | ✅ 是 | 0 | 28 |
| Beauty | 12101 | 50 | 12102 | ❌ **否，4158 行错位** | 0 | 33 |

**负对照（脚本内实测断言）**：Beauty 上朴素 `W[1:]` 会错配 **4158 行**（例：第 8 行 ItemID=10 应取 `W[10]`，`W[1:]` 给的是 `W[9]`）。另有三条 guard 各有负对照：非全零 padding 行 / id 空间含 0 / item 数与 ckpt 行数不符，**一律拒绝而不是静默取值**。

**两种来源的冷启动语义不同（如实记录，不粉饰）**：ItemCF 对"训练交互里没出现过的 item"写**全零行**（24/28/33）；SASRec 对"从未训练到的 item"**保留随机初始化** ⇒ **无全零行**（0/0/0，但这类行不含学到的信号）。**未知 id 与冷启动是两回事**：未知 id 默认抛 `MissingItemError`（`missing="zero"` 才给零向量），冷启动行合法返回并被 `is_zero_row` 标出。

### 13.4 明确不做（用户硬约束）

**不因 128→50 加 projection / normalization**（`CFEmbedding` 本就要求下游不得假定固定维度）、**不改 `CFEmbedding` 核心接口**、不实现 C2 的 cosine/kNN、不实现 C3 的 Q-space 约束、不实现 C4 的 loss、不引入 User SID。EXP-C1 用 AST 断言模块顶层无 `cosine/knn/loss` 类符号、无任何 `nn.*` / `F.*` 层构造。

### 13.5 下游（C2 起）必须知道的边界

1. **ItemID 映射**：Toys / Sport 的 ItemID 就是 `1..N`；**Beauty 是 `1..N` 的排列，4158 行错位** ⇒ 任何按行取值的代码在 Beauty 上都会静默取错。C1 一律走显式映射表，C2 起**沿用同一张表**。
2. **维度不是契约**：50 与 128 同走一个类，**C2 起不得硬编码 50**。
3. **"全零行 = 冷启动"只对 ItemCF 成立**（SASRec 源无全零行）。
4. **`torch.from_numpy` 共享 buffer**：`to_tensor()` 返回的是**副本**；下游若拿原始 numpy 表做原地归一化，会改写所有消费者看到的表。

**⚠️ 无效果结论**：C1 只验证"CF 向量能被正确、可追溯地加载且无错映射"，**不含任何推荐效果结论**；C2–C6 与全部 B / C 类效果实验仍为 `PLANNED`。

---

## 14 修订记录 R8 — C2 实现（Collaborative relation）

**日期**：2026-10-14 ｜ **Task**：C2 ｜ **判定**：**A-PASS（EXP-C2 127/127，exit 0）** ｜ **既有 tracked 文件改动：0**

### 14.1 实现

| 产物 | 内容 |
|---|---|
| `tiger_ext/collab_relation.py` | `CollabRelation`（关系容器 + 校验 + 统计 + provenance + 存取）、`build_relation` / `build_relation_for_dataset`、`topk_cosine_neighbors`、`l2_normalize`。**纯 numpy 叶子**：顶层只 import `numpy` / `os` / `json` / C1 的 `cf_embed`（复用同一个 `item_ids_sha256`，两模块的 row-space 摘要必然一致，不会各写一份而漂移） |
| `integration/exp_c2_collab_relation.py` | 8 节 127 项验证（合成 fixture + 手算 cosine + 独立 float64 复算 + sklearn 交叉复算 + 行空间契约 + 数量分布 + B 类共现探针 + 边界 + E1–E10 回归） |
| `integration/artifacts/collab_relation/<D>_k20.neighbors.npy` | `(n_items, k_eff)` **int32** 行号邻居，`-1` = 无邻居 |
| `integration/artifacts/collab_relation/<D>_k20.similarity.npy` | `(n_items, k_eff)` **float32** cosine，`NaN` = 同位置无邻居 |
| `integration/artifacts/collab_relation/<D>_k20.pairs.npy` | `(M, 2)` **int32** 行号边（**沿用历史 transition-pair 的 `(N,2)` 数组形态**，但不复用其开关语义） |
| `integration/artifacts/collab_relation/<D>_k20.pairs_itemid.npy` | `(M, 2)` int64，同一批边的 ItemID 形式（人工核验用；**这就是 off-by-one 会被抓住的地方**） |
| `integration/artifacts/collab_relation/<D>_k20.item_ids.npy` | `(n_items,)` int64 规范 ItemID（行空间契约本体） |
| `integration/artifacts/collab_relation/<D>_k20.json` | provenance + 统计（边数、度数直方图、对称性、逐 rank 相似度剖面、C1 来源 / checkpoint / hash、参考实现与差异清单） |

### 14.2 关系的确切定义

| 项 | 裁决 | 依据 |
|---|---|---|
| 相似度 | `cos(a,b) = dot(a,b) / (‖a‖₂‖b‖₂)`，float32；表先 L2 归一化一次，再按 block 做归一化 matmul | CSG4Rec `metric='cosine'`（sklearn 的 cosine **距离** = `1-cos`，只用其排序） |
| **k** | **20**，`k_eff = min(k, n_items-1)`；`k` 与 `k_eff` 分开记录 | CSG4Rec `--knn_k` 默认值（`CSGT/main.py:39`）。**未重新调参**——调 k 是设计决策，本 Task 只移植机制 |
| 方向性 | **有向**：每行独立取自己的 top-k，**不做对称化** | 忠实于参考实现（`global_knn` 逐行）；cosine 对称但 top-k 截断不对称，实测互反边占比 Toys 0.652 / Sport 0.612 / Beauty 0.694，已如实记录 |
| 自环 | **显式 mask 对角线**，然后 `validate()` 断言 0 自环 | **不用**参考实现的 `indices[:, 1:]` 位置式丢弃（见 14.3） |
| 零范数行 | 作为 source 与 target **双向排除**（cosine 无定义），标记 `-1` / `NaN` | 官方 SASRec 源实测 0 行；ItemCF 备用源 24/28/33 行 |
| tie-break | 相似度降序，再按行号升序（**stable argsort**，完全确定） | 用 `argpartition` 会在第 k 位放行任意并列项，与advertise的 tie-break 矛盾 |
| 行空间 | C1 的规范行序；artifacts 一律自带 `item_ids` + 摘要，`load` 不匹配即拒绝 | plan.md C2 验收⑤ |

### 14.3 与 CSG4Rec 原实现的差异（**先以 plan.md 为准，此处只记录**）

参考实现（`reference_repos/CSG4Rec_original/CSGT/datasets.py:36-38`）：

```python
nbrs = NearestNeighbors(n_neighbors=min(knn_k+1, self.N), metric='cosine').fit(cf_emb_np)
_, indices = nbrs.kneighbors(cf_emb_np)
self.global_knn = indices[:, 1:min(knn_k+1, self.N)]
```

| # | 差异 | 理由（实测） |
|---|---|---|
| 1 | 用 numpy 分块归一化 matmul，不用 sklearn | 不给移植叶子引入新依赖；且让自环排除变成显式的。**已与 sklearn 交叉复算**：三个数据集的 SASRec 表上 top-k **完全一致** |
| 2 | **自环靠 mask 对角线，不靠丢第 0 列** | 位置式丢弃假设"自己永远排第一"。**在零范数行上该假设失效**：ItemCF 备用源上参考写法留下自环 **6/24（Toys）、11/33（Beauty）**——正是 C2 验收④禁止的东西。官方 SASRec 源无零范数行，两者 0/0/0 结果相同 ⇒ **这个缺陷在主线上不会暴露，最容易不知不觉继承** |
| 3 | 零范数行双向排除并标 `-1`/`NaN` | 参考实现把这种行的排序交给求解器。另一种约定（cos 记 0）是数据不支持的断言，故不采用 |
| 4 | 边落盘 + ItemID 伴随文件 + id 摘要 | 参考实现只在内存里存一个数组，行空间没有任何载体 |
| 5 | 独立 artifact + 独立 loader，**不复用历史 transition 开关** | plan.md 明令"不得复用历史 `trans_graph_path` 开关语义"；只沿用 `(M,2)` 这一数组形态 |

**未做**（不自行扩大设计）：不加对称化、不调 k、不加权重、不加阈值截断、不实现 C3 的 Q-space 约束、不实现 C4 的 loss、不动 RQ-VAE / T5 / E1–E10。

### 14.4 实测结果（三数据集，正式源 = SASRec item embedding 50d）

| 数据集 | n_items | k_eff | 边数 | 自环 | 互反边占比 | rank0→rank19 平均 cos | cos 范围 |
|---|---|---|---|---|---|---|---|
| Toys | 11924 | 20 | **238480** = 11924×20 | 0 | 0.652 | 0.670 → 0.506 | [0.374, 0.988] |
| Sport | 18357 | 20 | **367140** = 18357×20 | 0 | 0.612 | 0.651 → 0.515 | [0.361, 0.990] |
| Beauty | 12101 | 20 | **242020** = 12101×20 | 0 | 0.694 | 0.653 → 0.496 | [0.380, 0.992] |

**正确性证据（EXP-C2 内）**：① 合成 fixture 上 cosine / top-k 与**手算完全一致**（含并列 0.0 按行号升序、零范数行双向孤立）；② 真实数据抽样行与**独立 float64 暴力复算**的 top-k **集合与顺序都相同**，`max|Δcos| ≤ 1.9e-7`；③ **最优性**：第 k 名邻居的 cos **高于所有非邻居**；④ 与 **sklearn `NearestNeighbors(metric='cosine')`** 交叉复算一致；⑤ 保存→加载**逐位一致**（neighbors/similarity/pairs/stats）；⑥ 篡改 pair 文件 / 交换两个规范 id / 期望 k 不符 / 文件缺失 —— 全部**拒绝加载**。

**ItemID 映射（验收⑤）**：pairs 存行号，但 artifact 自带 `item_ids` 与摘要。Beauty 上"把行号当 ItemID"（`row+1`）会让 **136490 / 242020（56%）的边指向错的商品**；Toys/Sport 因 ItemID 恰为 `1..N` 而**看不出差别**——所以该护栏只在 Beauty 上有效，已用负对照钉死。

### 14.5 B 类共现探针（验收③，**重要限制见下**）

在 valid+test 的篮子对上测"kNN 邻居是否真的共现"，与**同规模随机对**基线比：

| 数据集 | kNN 命中共现率 | 随机基线 | 富集倍数 | 打乱 row→ItemID 映射后 |
|---|---|---|---|---|
| Toys | 0.3819 | 0.01610 | **23.7×** | 0.0171（≈1.1×，**塌回基线**） |
| Sport | 0.2462 | 0.00900 | **27.3×** | 0.0087（≈1.0×） |
| Beauty | 0.3472 | 0.01547 | **22.4×** | 0.0137（≈0.9×） |

**⚠️ 这不是效果证据，也不独立**：SASRec 是在 `train.parquet` 上训的，而本项目的 valid/test history 是同一批用户序列的**前缀**（`valid.history ⊃ train.history + train.target`，实测），因此该共现统计与训练语料**高度重叠**。它在这里只承担两件它真能承担的事：**(i)** 邻居是行为相关的商品而非噪声；**(ii)** **行→ItemID 映射是对的**（打乱后富集立刻塌回基线 ⇒ 该探针确实对映射敏感）。Beauty 上 `row+1` 写法仍显示 0.2010（12.7× 于基线）——**看起来"依然富集"，但明显劣于正确映射的 0.3472**，这正是"off-by-one 会静默骗过一眼扫过"的量化证据。

### 14.6 下游（C3 起）必须知道的边界

1. **行空间**：C2 的邻居是**行号**，与 C1 的表行号、`item_emb.parquet` 的 ItemID 列同序。取 ItemID 必须经 `pairs_as_item_ids()` / `item_ids[row]`，**禁止 `row + 1`**（Beauty 56% 的边会错）。
2. **有向**：`i → j` **不蕴含** `j → i`（互反只占 61–69%）。C3 若要双向语义，必须显式声明，不要假设对称。
3. **`-1` / `NaN` 是合法取值**：SASRec 源上没有（0 行），但接口允许；C3 必须显式处理，不能假定每行都是满 k。
4. **k=20 是继承来的，不是搜出来的**：没有任何证据说 20 对此任务最优；它只是 CSG4Rec 的默认值。调 k 属于后续实验，不属 C2。
5. **共现探针不构成效果结论**：见 14.5 的限制。

**⚠️ 无效果结论**：C2 只验证"关系能被正确地、可追溯地、可复现地构造与落盘"，**不含任何推荐效果结论**；C3–C6 与全部 B / C 类效果实验仍为 `PLANNED`。

---

## 15 修订记录 R9 — C3 实现（Q-space collaborative constraint，construction half）

**日期**：2026-10-14 ｜ **Task**：C3 ｜ **判定**：**A-PASS（EXP-C3 99/99，exit 0）** ｜ **既有 tracked 文件改动：0**

### 15.1 开工前必须确认的 7 问（先从 `CSG4Rec_original` 实际代码确认，再写代码）

| # | 问题 | 答案（代码依据） |
|---|---|---|
| 1 | CSG4Rec 的 Q-space 到底是什么 | **量化后的码本查表向量**。`CSGT/models/vq.py:195` 的 `x_q = self.embedding(indices)` —— 即"取到码字、但**尚未**做 straight-through"的那个张量 |
| 2 | 约束作用在哪个具体张量 | 同一个 `x_q`，被 `diversity_loss`（`vq.py:68-106`）消费；调用点在 `vq.py:198-200`（`use_constrain_loss` 分支内） |
| 3 | 用 quantized vector / code logits / 别的 | **quantized vector（`x_q`）**。原实现里出现的是 `sim = torch.matmul(x_q, emb.t())`（`:84`），即"量化向量 × 整个码本" |
| 4 | constraint 的计算方式 | `sim_self[idx, indices[idx]] = 1e12` 屏蔽自身（`:87-90`）后 `loss = F.cross_entropy(sim, y_true)`；`y_true` 是**码字 id**，取"同一 constrained-kmeans 簇里的另一个码字"（`:72-79`）。逐 RQ 层算，`rq.py:64` 取平均，外层按 `--beta` 加权（`vq.py:209`） |
| 5 | relation 如何参与 | **在可恢复的那部分代码里，relation 不是 CF kNN**——它是 `constrained_km`（`trainer.py:95-107` / `vq.py:53-65` / `generate_indices.py:106-117`）对**码本**做的 10 簇划分，即 **code-to-code** 关系 |
| 6 | 哪些属于 CSG4Rec 原始机制 | `x_q` 这个量的选择、`x_q × 码本` 的打分形式、`cross_entropy` 的形态、逐层平均 + `--beta` 加权 |
| 7 | 哪些当前 TIGER 没有对应物 | ①**CF kNN 进入 loss 的那一半**（见 15.3）；②`use_constrain_loss` / `--alpha` 那套开关 |

### 15.2 ★ 决定性发现：`CSGT/models/rqvae.py` 在快照里**不存在**

| 项 | 内容 |
|---|---|
| 事实 | 参考仓库 `reference_repos/CSG4Rec_original/CSGT/models/` 下**只有** `layers.py` / `rq.py` / `vq.py`，**没有** `rqvae.py`；无归档、无 `.pyc`、本机无第二份副本 |
| 谁 import 它 | `CSGT/main.py:9`（`from models.rqvae import RQVAE`）、`CSGT/generate_indices.py:15` |
| 它原本装了什么 | `RQVAE`、`compute_loss`、`get_indices`——**以及** `global_knn` / `use_knn` / `use_knn_argument` / `use_cf_loss` / `use_pairwise` / `use_kd` 的**全部读取点** |
| 已追到的调用链 | `global_knn` 在 `datasets.py:36-38` 构建 → 随 batch dict 停在 CPU（`trainer.py:155,192`）→ 交给 `self.model.compute_loss(batch_items, epoch_idx, self.epochs, self.labels)`（`:159`）；逐 epoch 打印里有 `knn` 槽（`:241`）；`trained_loss = {"total","rqvae","recon","cf"}` |
| **后果** | **CSG4Rec 的 kNN loss 的确切函数形式无法从参考实现恢复**，本次移植**不复现它**。这与 C1（SASRec trainer 缺失）、C2（RQVAE model 缺失）是同一类问题 |

> 因此下表把「**原始机制**」（有代码依据）与「**当前适配**」（无依据时采用的最小方案）**分开陈述**，不把后者混进前者。

### 15.3 A / B / C 三分（plan.md 明令不得混成一个 embedding alignment）

| 部分 | 是什么 | 谁拥有 | 在本 Task 里的形态 |
|---|---|---|---|
| **A — collaborative relation** | C2 的 `CollabRelation` | **C2** | 只经 `CollabRelation.load(prefix, expected_item_ids=..., expected_k=20)` 读取；**本模块从不打开 `.npy`**（EXP-C3 用 AST + 源码计数钉死） |
| **B — Q-space representation** | `x_q` | **RQ-VAE 自己的 forward** | `capture_qspace()` 调 `RQVAE.forward(..., return_xq=True)` 并**当场自证**（见 15.4） |
| **C — constraint** | A 投影到 B 上 | **C3** | `ConstraintBatch`：in-batch 的 `(anchor, target)` **位置对** + 两侧各自 gather 出来的 Q-space 张量 + 关系自带的 cosine 作权重 |

### 15.4 表示选择：**`x_q`**，以及为什么**不**用 E3 code logits

| 候选 | 裁决 | 代码依据 |
|---|---|---|
| **`x_q`** | ✅ **采用** | 它就是 CSG4Rec 存活的约束所消费的量（`vq.py:195`），且是**量化后**的 |
| **code logits**（`vq.py:107`） | ❌ **排除** | `rqvae/models/vq.py:76` 的 `raw_d = d`，而 `d` 是 `latent`（**量化前的 encoder 输出**）到各码字的平方 L2。**约束 logits 就是在约束 encoder 向量**——那正是 plan.md C3 明令禁止的 latent-space alignment，只是换了个 Q-space 的名字。**这是结构性理由，不是偏好** |

**来源断言（把"量化空间"从标签变成可测量属性）**：`QSpaceBatch.validate()` 要求

1. `indices` 是 `(B, L)` 且 `L == len(num_emb_list)`（collision token 不是量化层，不得出现）；
2. `x_q ≈ Σ_l codebook_l[indices_l]`，**用该模型自己的码本重算**，容差 `atol=rtol=1e-4`（float32 下 `x + (y-x) ≠ y`，实测偏差 7.45e-09 ~ 5.96e-08）；
3. `x_q` 与**同一次前向**的 encoder latent `z` **必须不同**（实测差 0.131 / 0.154 / 0.229），否则"Q-space"是个空标签。

> `rqvae/models/rqvae.py` 的 docstring 警告**不要**为了反传而在外部重新 gather `x_q`（会切断 encoder 的梯度路径）。这里重算**只用于校验**，且取自 detach 后的副本；反传用的仍是 forward 返回的那个张量。EXP-C3 用负对照钉死：把 latent 当 `x_q` 交上去 → 拒绝；篡改一个 code index → 拒绝；层数不对 → 拒绝；`rows` 对不上 → 拒绝。

### 15.5 与 CSG4Rec 原实现的差异（5 条，均写入 `PORT_DIFFERENCES`）

| # | 项 | **原始机制**（有依据） | **当前适配**（本模块的选择） | 理由 |
|---|---|---|---|---|
| 1 | 监督来自谁 | code-to-code：正样本是码本同一 constrained-kmeans 簇里的**另一个码字**（`vq.py:72-79`） | item-to-item：正样本是 anchor item 的 **C2 协同邻居** | plan.md C3 的输入**就是** "C2 的关系对"。参考实现里那个"码的簇"是另一个对象，不采用 |
| 2 | loss | 内联在 quantiser 自己的 `forward` 里，`beta * diversity_loss`（`vq.py:198-209`） | **完全不计算**——C3 只产出中间量，目标函数归 C4 | plan.md 把两者拆成两个 Task；在此处算 loss 既越界，又恰好会逼着去猜那个**缺失的文件** |
| 3 | 未在 batch 内的邻居 | **未知**（消费代码在缺失文件里） | **丢弃并计数** | 唯一不需要额外信息的策略；本仓库既有的"关系 → Q-space"构造 `rqvae/trainer.py::_get_transition_loss:213-219` 正是这么做的（`if target_idx in idx_to_pos`） |
| 4 | 量化的是哪个张量 | `x_q`，**straight-through 之前**的码本查表（`vq.py:195`） | `forward(return_xq=True)` 返回的 `x_q`——**数值相同**（同为码本求和），但带 ST 梯度路径 | TIGER 只暴露 ST 之后的那个张量；约束看到的量一致，**梯度路由不同**，C4 必须知道 |
| 5 | 开关 | `--use_knn` / `--knn_weight`（发布配方 `knn_weight=0.01`） | `lambda_csg`，默认 0，挂在本模块自己的 mount 上 | plan.md 要求独立开关；历史 `trans_graph_path` 语义**明确不复用** |

**未做（不自行扩大设计）**：不加 warmup、不加 projection、不加额外 regularization、不 concat CF embedding 到 RQ-VAE 输入、不做 CF ↔ latent 距离对齐、不引入 User SID、不改 RQ-VAE 基础架构、不改 T5、不实现 C4 的 loss、不做 C5 的 trainer integration、不改 E1–E10。

### 15.6 约束的数学形式

对一个 batch（行集合 `R`，`|R| = B`）与 C2 的有向关系 `E = {(i,j)}`：

```
P  = {(i, j) ∈ E : i ∈ R 且 j ∈ R}                       # in-batch 边，其余丢弃并计数
anchor_pos[p] = pos_R(i_p)                                # 位置，不是行号
target_pos[p] = pos_R(j_p)
w[p]          = sim(i_p, j_p)                             # 关系自带的 cosine，原样透传
X[p]          = x_q[i_p]      ∈ R^{e_dim}                 # 锚点侧
Y[p]          = x_q[j_p]      ∈ R^{e_dim}                 # 目标侧
Ix[p], Iy[p]  = codes(i_p), codes(j_p)  ∈ Z^L             # 逐 RQ 层的码 index
```

**C3 产出 `(P, ...)` 这些张量，不产出标量**。可选的目标函数形态（**归 C4，本 Task 未实现**）与参考实现的 `x_q × 码本` 打分是同族的：`sim[p] = X[p] · Wᵀ`（`W` 为码本），`−log softmax(sim[p])[Iy...]`。

未在 batch 内的部分**被丢弃而不是被近似**；丢弃量 `dropped_out_of_batch` / `dropped_sentinel` 逐 batch 计数并记录。

### 15.7 `-1` / `NaN` 与 ItemID 映射

| 项 | 处理 |
|---|---|
| `-1` 邻居 | 与 `NaN` 相似度**同时**出现（C2 的不变量）。两者在任何张量运算**之前**被过滤；`-1` 若漏过去会把 `-1` 当索引取到 batch 的**最后一行** |
| 哨兵不一致 | `-1` 与 `NaN` 不吻合即 `raise QSpaceMismatch`（C2 保证不会发生 ⇒ 一旦发生，说明这个 relation 对象不是 C2 造的；EXP-C3 用两个方向的篡改对照证明该 guard 会响） |
| 统计 | `slots == pairs + dropped_sentinel + dropped_out_of_batch` 恒等式在合成与三数据集上均被断言 |
| ItemID → 行号 | 一律经 `CollabRelation.row_of` / `rows_of_item_ids`，**禁止 `id - 1`**（Beauty 上 4158/12101 行 ItemID ≠ row+1，`id - 1` 会静默取错商品） |
| 行空间 | 批次行越界 / 负数 → 拒绝；装载时 `expected_item_ids` 或 `expected_k` 不符 → **拒绝加载** |

### 15.8 实测结果（三数据集最小验证 + 真实 tokenizer）

| 数据集 | 关系边数 | 256-item batch 内 in-batch 对数 | 丢弃（batch 外） | 哨兵 | 有对的锚点数 | 重建偏差 | latent 间距 |
|---|---|---|---|---|---|---|---|
| Toys | 238480 | **738**（2.88/item） | 4382 | 0 | 175/256 | 2.98e-08 | 0.154 |
| Sport | 367140 | **118**（0.46/item） | 5002 | 0 | 88/256 | 5.96e-08 | 0.229 |
| Beauty | 242020 | **333**（1.30/item） | 4787 | 0 | 181/256 | 2.98e-08 | 0.131 |

**⚠️ 上述 in-batch 对数不是效果指标**，它是 batch 几何：256 行若邻居"均匀随机"，期望只有 `256/11924 × 20 ≈ 0.43` 对/item。Sport 实测 0.46 **正落在该期望上**，说明排序表的表头不聚集；Toys 的 2.88 则是**行号靠前的商品本身邻居互锁**。两个数据集差异如此之大，恰恰说明这个量取决于"抽了哪些行"，不取决于关系质量或 tokenizer。

**三数据集的 tokenizer 说明（重要）**：本仓库中**只有 Toys 有"训在规范 4096d content 表上"的 RQ-VAE**（`clean_tiger/ckpt/Toys/rqvae/Oct-14-2026_05-55-09/`，已核对：其训练用的 `clean_tiger/_smoke/Toys/data/Toys/item_emb.parquet` 与 `data/Toys/item_emb.parquet` 的 ItemID 与 embedding **逐位相同**）。Sport / Beauty 在 `rqvae/ckpt/` 下只有训在**融合表**上的 `*_TA_SID_4096d`，**没有**规范 content 表上的 tokenizer。因此：

- **三数据集那一轮**统一使用**同架构 fixture tokenizer**（冻结 TIGER 架构；码本用模型**自己的** `init_emb` 在该数据集真实 content latent 上做 k-means 初始化，**不训练**），使三行可比；
- Toys **额外**用**真实训练过的** Clean TIGER tokenizer 再跑一遍：**对数与 pair 轴逐位相同**（738 对），而 code 分布不同（真实 tokenizer 256 项用到 71 个 L0 码，fixture 用到 99 个）——**这正证明 A/C 与 B 是可分离的**。

### 15.9 下游（C4 起）必须知道的边界

1. **C3 不产出 loss、不产出梯度**。`lambda_csg = 0`（默认）时 `enabled=False`，`forward()` **抛异常**而不是静默返回 0（与 `SIAMount` / `PSAMount` 一致）；调用方必须守卫在 `enabled` 上。
2. **梯度路由**：C4 拿到的 `x_q` 是 **straight-through 之后**的张量（值 = 码本求和，梯度回 encoder）。参考实现用的是 ST 之前那个（梯度只到码本）。**C4 若照抄参考实现的反传语义，会得到不同的梯度图**。
3. **`x_q` 由 RQ-VAE 提供时不要重新 gather**（`rqvae/models/rqvae.py:64-67` 的警告）；`rebuild_quantized()` **只用于校验**。
4. **in-batch 策略是适配**，不是原始机制。参考实现的策略在缺失文件里，**未知**。C4 若要改策略（例如补第二遍前向），那是一个新的设计决策。
5. **有向**：pair 来自每个 anchor 自己的出边，**未对称化**（C2 互反只占 61–69%）。要对双向就显式要。
6. **`rows` 恒在 CPU**：它是整数身份（`(B,)` long），Q-space 张量才跟着模型走。CUDA 臂已验证 "CPU 索引张量对 CUDA 表示取行" 这条路可用。
7. **k=20 是继承来的**，不是搜出来的。

**⚠️ 无效果结论**：C3 只验证"C2 的关系能被正确地投影到 RQ-VAE 的量化空间上"，**不含任何推荐效果结论**；C4–C6 与全部 B / C 类效果实验仍为 `PLANNED`。

> **R10 追加的复现性更正（见 §16.9）**：15.8 表格里的 pair 数（738 / 118 / 333）不受影响——C3 已自证配对只取决于关系、与 tokenizer 无关。但该节正文所说的 "fixture tokenizer uses 99 of 256 L0 codes" 与 "真实 tokenizer 71 个码" 是**单次抽样**：fixture tokenizer 走的是冻结的 `init_emb` → `rqvae/models/layers.py:86` 的 `KMeans`（无 `random_state`），默认线程下**跨进程不可复现**。已确认的结论（A/C 与 B 可分离）依赖的是 **pair 轴逐位相同**，那一条依然成立；**码分布计数不得当作测量值引用**。

---

## 16 修订记录 R10 — C4 实现（Q-space collaborative constraint objective，objective half）

**日期**：2026-10-14 ｜ **Task**：C4 ｜ **判定**：**A-PASS（EXP-C4 87/87，exit 0；其中 E10 回归 64/64）** ｜ **既有 tracked 文件改动：0**

C4 只做一件事：**`ConstraintBatch` → Q-space 约束目标 → 标量 loss**。不接训练循环、不改 E7、不改 Tokenizer/Recommender stage、不改 RQ-VAE / T5 / SIA / PSA / C1 / C2 / C3、不加 projection / warmup / regularization / 新 loss 分支。**完成后停止，不进 C5。**

### 16.1 开工前 10 问（先从 `CSG4Rec_original` 实际代码确认，再写代码）

| # | 问题 | 答案（代码依据） |
|---|---|---|
| 1 | **query 是什么** | **anchor item 在该 RQ 层的码字** `W_l[Ia[:, l]]`。依据 `CSGT/models/vq.py:195` 的 `x_q = self.embedding(indices)` 与 `:84` 的 `sim = torch.matmul(x_q, emb.t())`——参考实现打分的就是这个"取到码字"的向量 |
| 2 | **candidate / target 是什么** | candidate = **该层自己的码本** `emb = self.embedding.weight`（`vq.py:69`），候选集 = 该层全部 `n_e` 个码字；target = **一个码字 id**（`y_true`，`:92`），C4 用协同邻居在该层的码 `Iy[:, l]` |
| 3 | **similarity logits 怎么算** | **裸点积** `S = W_l[Ia[:,l]] @ W_lᵀ`。依据 `:84`；**cosine 变体在 `:83` 被注释掉**；`temp = 1` 硬编码（`:70`，`:91` 除以它）⇒ 等价于无温度 |
| 4 | **y_true 怎么定义** | 一个**码字 id（int 类标签）**，与参考实现同类型。**不用相似度、不用软标签**（软标签是参考实现里没有的新机制） |
| 5 | **一个 anchor 有多个邻居** | **每个 (anchor, neighbour) 是 CE 的独立一行**，n 个邻居 = n 行；per-level 归约是 **pair 均值**。依据：参考实现一行一个正样本（`:72-81`），其"行轴"恰好在"每个 anchor 一个邻居"时等同 pair 轴；本实现是该情形的推广，**不是** multi-positive softmax（参考实现没有该形式） |
| 6 | **directed pair 怎么保留** | **不合并**：每对以自己的 anchor 作 query，`(a→b)` 与 `(b→a)` 是两行独立 CE，互不推断。实测 `L(0→1)=0.7201`、`L(1→0)=2.6657`、`L(both)=1.6929`（= 两者均值） |
| 7 | **C2 的 cosine weight 是否进 objective** | **不进**。依据 `vq.py:68`：`diversity_loss` 的签名是 `(x_q, indices, indices_cluster, indices_list)`，**根本没有 weight 参数** ⇒ 按 brief 措辞如实陈述为"**C2 weight 在 C3 construction 中保留，但 C4 objective 不使用**"，并用实测断言其不影响 loss（零 / 极大 / 取负 / 打乱四种扰动下逐位不变） |
| 8 | **reduction 怎么定义** | **两层**：层内 = `F.cross_entropy` 默认的 **行（pair）均值**（`:92`）；层间 = 对有 pair 的层取 `mean`（`rq.py:64` 的 `torch.stack(all_losses).mean()`）。**不是 anchor 均值**——参考实现的行轴之所以是 anchor 轴，只因为它每个 anchor 只有一个正样本；实测 anchor 均值 **1.8179** vs pair 均值 **1.5353**（拒绝前者） |
| 9 | **没有有效 pair 怎么办** | 返回 **0 维 `0.0`**（**不是**异常）。先例：`rqvae/trainer.py::_get_transition_loss:191-192,213-214`、`rqvae/trans_loss.py::ContrastiveLoss`。"开关关闭"是**另一回事**，那个**必须抛** `QSpaceDisabled`（复用 C3 的异常，禁止静默返回 0） |
| 10 | **同 anchor 多 target 的形态** | **独立 CE**（= 第 5 问），**不发明** multi-positive。实测 multi-positive 形式给 **1.3329**，与本实现 1.5353 不同，且与参考实现无对应 |

### 16.2 ★ 决定性事实 A：`emb` 到底是什么（brief §四的硬要求）

`emb = self.embedding.weight`（`vq.py:69`）⇒ **该 RQ 层自己的码本**。不是某个全局 embedding、不是别的层的码本、不是 CF embedding、不是 `x_q` 本身。

**推论（把 brief §五 的"不要默认加 level-wise loss"变成代码结论）**：`x_q` 是各层码字**之和**，而 `emb` 只有**单层**码本 ⇒ `sum(x_q) @ embᵀ` 在参考实现里**没有对应物**。因此 level-wise 不是我们"加的"，而是参考实现**唯一能表达**的形态；brief 的"优先保持完整 `x_q`"与代码证据在此**指向同一结论**（完整 `x_q` 上的约束在参考代码里不存在，不是被我们放弃的选项）。

### 16.3 ★ 决定性事实 B：可复原 / 不可复原的分界（**不伪造**）

| | 内容 |
|---|---|
| **可复原（有代码）** | `CSGT/models/vq.py:68-106` 的 `diversity_loss`：`x_q × 码本` 打分、`1e12` 自身屏蔽、`F.cross_entropy(sim, y_true)`、`temp=1`、逐层调用（`rq.py:51-61`）+ 层间平均（`:64`）+ `beta` 加权（`:209`） |
| **不可复原（无代码）** | **CSG4Rec 的 kNN loss 本体**：`CSGT/models/rqvae.py` 在快照中**不存在**（无归档、无 `.pyc`、本机无第二份副本）；`global_knn` / `use_knn` / `use_knn_argument` / `use_cf_loss` / `use_pairwise` / `use_kd` 的**唯一**读取点、`compute_loss`、`get_indices` 全在该文件里 |
| **本次裁决** | **不复刻、不猜、不补造**。C4 复现的是**存活约束的形态**（`vq.py:68-106`），正样本来源换成 **C2 的 item-to-item 关系**（照 C3 的交付）。`QSL_MISSING_REFERENCE` 把缺失文件**记录为缺失**，`provenance()` 的 `is_a_reproduction` 恒为 `False`。 |

### 16.4 最终数学形式（冻结）

对 batch 的 pair 轴 `p = 1..P`、RQ 层 `l = 0..L-1`（`L = len(num_emb_list)`，`W_l ∈ R^{n_e×e_dim}` 来自 `rqvae.rq.get_codebook()`，可微）：

```
keep[p]  = (Ia[p,l] != Iy[p,l])                  # 参考实现的正样本必为「不同码字」(vq.py:77-78)
X        = W_l[Ia[keep,l]]                       # (P_l, e_dim)          query
S        = X @ W_lᵀ                              # (P_l, n_e)            candidate 全集（裸点积）
S[p, Ia[p,l]] ← S[p, Ia[p,l]] - 1e12             # 自身屏蔽，vq.py:87-90 逐字（大有限常数，非 -inf）
CE_l     = F.cross_entropy(S, Iy[keep,l])        # 归约 = 行（pair）均值
L_csg    = mean{ CE_l : P_l > 0 }                # 层间均值，仅对有 pair 的层
```

**没有的东西**（brief 明令）：无 weight、无温度、无 L2 归一化、无 projection、无 warmup、无额外 regularization、无 detach、无新可训练参数。同码 pair（`Ia == Iy` 在该层）**丢弃并逐层计数**（参考实现由构造保证不发生，`:77-78`）。

### 16.5 与参考实现的差异（6 条，均写入 `_PORT_DIFFERENCES`）

| # | 项 | 参考实现 | 本实现 | 理由 |
|---|---|---|---|---|
| 1 | **loss 本体** | **未知**（kNN loss 在缺失文件里） | **不是复刻**：把存活约束的形态（`vq.py:68-106`）作用在 C2 的关系上 | 文件缺失；补造即伪造 |
| 2 | **正样本来源** | 同 constrained-kmeans 簇里的**另一个码字**（`vq.py:72-79`） | C2 协同邻居在该层的码字 | plan.md C3 的输入就是 item-to-item 关系；被约束的量、候选集、打分、target **类型**全部不变 |
| 3 | **query 的梯度路由** | ST **之前**的码本查表（`:195`）⇒ 梯度只到码本 | 同一次码本查表（`rqvae/models/vq.py:90`）⇒ **同样的路由** | 数值相同；C3 另交的 ST 之后张量其 encoder 通路**已实测存在但本 objective 不用**（见 16.6） |
| 4 | **正样本必与自身不同码** | 由构造保证（`:77-78`） | 不保证 ⇒ 同码 pair 在该层**丢弃并计数** | 参考实现没有"自身为目标"的算术；强行保留会让被屏蔽列赢下 softmax，loss ≈ 1e12 |
| 5 | **层** | 每层总贡献（正样本总存在） | 无 pair 的层**略去**，均值只取有定义的层 | 参考实现不构造"无正样本的层"；用制造出来的 0 摊薄 scale 更糟 |
| 6 | **开关** | `--use_knn` / `--knn_weight` / `--use_constrain_loss` / `--beta` | `lambda_csg`（默认 0，挂在 C4 自己的 mount 上，**不接入任何总 loss**） | plan.md 要求独立开关；C4 不得碰 trainer |

### 16.6 梯度（brief §六，**实测**，不是声称）

| 问题 | 实测结果 |
|---|---|
| `L_csg` 是否有梯度 | ✅ `requires_grad=True`、`grad_fn=MeanBackward0` |
| `backward()` 能否传播 | ✅ 无异常；`dL/dW_l ≠ 0`（查询侧 + 候选侧两项，与 `sim = x_q @ embᵀ` 同构） |
| 是否到达产生 `x_q` 的计算图 | ✅ 到达 **码本**（`nn.Embedding.weight`）；**不到 encoder**——indices 来自 `argmin/argmax`（`rqvae/models/vq.py:78,86`）无梯度。**这与参考实现的路由相同**（它打分 ST 前的张量 `:195`，`:213` 才重写 `x_q`） |
| ST 通路是否完好 | ✅ C3 交出的 `x_q` 直接 backward ⇒ **10 个 encoder 参数拿到梯度**（实测）。objective **没有**用这条路，但**也没有破坏它** |
| 是否 detach `x_q` | ❌ **没有**。AST 逐函数检查：仅有的两处 `.detach()` 都在 `float(...)` 统计转换里，另有 `verify_indices_match_qspace` 整体 `@torch.no_grad` |
| 是否新增 projection / 改 `requires_grad` | ❌ 没有。`[8]` 逐条断言：无 nn 层构造、0 参数 0 buffer、`torch.*` 只用 `{arange, matmul, stack, zeros, zeros_like, no_grad, allclose}`、唯一功能调用是 `cross_entropy` |
| **E7 是否被改** | ❌ **未改**（brief §六 明文："C4 只负责确认计算图正确，C5 才负责训练阶段如何处理这些梯度"） |

> **"约束反传到 encoder"是 C5 的决策，不是 C4 的遗漏**——已记入 `QSL_GRADIENT_POLICY` 与 16.10 第 1 条。

### 16.7 `lambda_csg` 语义（与 C3 的 disabled 语义严格一致）

| 取值 | 行为 |
|---|---|
| `== 0`（默认） | `enabled=False`，`forward()` **抛 `QSpaceDisabled`**——**不执行 objective、不静默返回人造 zero**（与 `SIAMount` / `PSAMount` 的 "guard on `enabled`" 一致） |
| `> 0` | 正常计算，返回 `lambda_csg * L_csg`。**仍未接入任何总训练 loss**（那是 C5） |
| 空 pair（与开关无关） | 返回 0 维 `0.0` + `stats["reason"]`。**这是数据结果，不是接线错误**——与上面那行严格区分，EXP-C4 两条都断言 |

### 16.8 实测结果（真实三数据集 + 真实已训 tokenizer；**全部可复现**，见 16.9）

| 数据集 | 关系边 | batch 内 pair | `L_csg` | per-level | 层内 same-code 丢弃 | `\|dL/dW\|` per level |
|---|---|---|---|---|---|---|
| Toys（fixture tok.） | 238,480 | 738 | **5.539878** | [5.5378, 5.5413, 5.5405] | 60 | [1.10883, 0.14682, 0.16637] |
| Sport（fixture tok.） | 367,140 | 118 | **5.556832** | [5.5875, 5.5418, 5.5412] | 10 | [1.46451, 0.22270, 0.35763] |
| Beauty（fixture tok.） | 242,020 | 333 | **5.559719** | [5.5967, 5.5410, 5.5415] | 29 | [1.09656, 0.15401, 0.22607] |
| Toys（**真实已训** ckpt） | 238,480 | 738 | **5.582809** | [5.6668, 5.5405, 5.5412] | — | [0.756085, 0.130800, 0.078298] |

- **尺度对照**：`log(255) = 5.5413` 是"码本完全无信息"时的值（256 类、自身已屏蔽）。四行都落在该值附近 ⇒ **不是退化数值，也不是一个已经"学会"的很小 loss**；这正是未训练码本应有的样子。
- **合成 fixture 手算锚点**：6 对 / 2 层，手算 per-row `[1.3132617, 0.3132617, 0.3132617, 1.3132617, 1.3132617]` 与 `[0.1269280, 4.0181499, 4.0181499, 0.1269280]`，层均值 `[0.9132616875, 2.0725389695]`，`L_csg = 1.492900328499`，丢弃 3。**由独立的 numpy float64 实现复算**（不是重跑同一段 torch 代码）。
- **被拒绝的归约**：anchor 均值 1.817900328、multi-positive 1.332852904（本实现 pair 均值 1.535298347）。
- **有向性**：`L(0→1)=0.720094860` vs `L(1→0)=2.665705681`；`L(both)=1.692900181` = 两者均值。
- **weight 不变性**：零 / 极大 / 取负 / 打乱四种扰动下 `L_csg = 0.765279412` 逐位不变（且 `batch.weight` 确实非空，"不变"不是空转）。
- **数值稳定**：零码本 ⇒ **恰好 `log(n_e-1)`**（实测 0.693147182465 vs log2 = 0.693147180560）；scale 1e3 / 1e-8 都有限且在解析界内；NaN 码本 ⇒ NaN（不粉饰）；越界 / 负 code index、层数不符、两侧形状不符、单码字层 **全部拒绝**。
- **已知的退化读数（如实记录）**：`[6]` 那个 64×32 随机 latent fixture 上，`|dL/dW|` 只有 level 0 非零（[3.617317, 0.0, 0.0]）——因为该 fixture 的 64 个残差被 256 个码字的最近邻完全吸收，level 1/2 退化。断言写成 `any(...)` 并把三个值全部打印；**真实内容上三层都非零**（上表四行）。不隐藏该读数。

### 16.9 ★ 顺带发现（两条，本次查实）

**① 冻结的 k-means 助手跨进程不可复现 → C4 的 fixture 换初始化**

| 项 | 内容 |
|---|---|
| 事实 | `rqvae/models/layers.py:86`：`KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)`——**没有 `random_state`**；它从全局 numpy 状态抽（seed 确实能设上），但默认多线程下 OpenBLAS/OpenMP 的归约顺序使结果**每次不同** |
| 实测（2026-10-14，3000×32 float32，固定 numpy seed） | 默认线程 **4/4 次 `cluster_centers_` 哈希互不相同**（inertia 仅第 5 位小数抖动）；`OMP_NUM_THREADS=1` 时 **3/3 完全相同**。运行时 `threadpool_limits(1)` **压不住**（运行库已初始化） |
| 污染源分离（真实 Toys content） | 已训 ckpt 路径 **3/3 稳定**；torch-seeded init + encoder forward **3/3 稳定**；**只有走 `init_emb` 的那条 3/3 不同** ⇒ k-means 是**唯一**污染源 |
| 裁决 | **不改冻结核心**（§九 禁止）。C4 的 fixture tokenizer 改为"**该层 residual 的 seeded 子样本**"（`FIXTURE_INIT_NOTE` 记录全部实测与命令），并在 `[6b]` 增加一条**同进程重建 ⇒ code index 逐位相同**的断言。**这不是掩盖：换掉的是一个每次都在变的初始化，换来的是一条可复现的测量**。 |
| **连带更正** | C3 的 fixture tokenizer 同样走 `init_emb` ⇒ **§15.8 的码分布计数（99 / 71）是一次抽样**。其**结论不受伤**（A/C 与 B 可分离依赖"pair 轴逐位相同 738"，与 tokenizer 无关）。C4 **未改** C3 的 harness。另注：这与 [[md5-is-not-a-data-fingerprint]]、C2 记录的 `WarpSampler n_workers` 属**同一类**"跨运行不可复现"陷阱 |

**② E10 的 CSG 守卫对 C4 假阳性 → 改 C4 的名字，不改 E10**

| 项 | 内容 |
|---|---|
| 守卫 | `integration/exp_e10_smoke.py:863`：`grep -rniE "csg_\|lightgcn\|graph_conv"`（**`-i`，大小写不敏感**）|
| 命中原因 | C4 的常量原用 `CSG_` 前缀（为避开 C2/C3 已导出的 `MISSING_REFERENCE` / `DIRECTION_POLICY` 而加）⇒ `CSG_WEIGHT_POLICY` 等**含有 `csg_`** ⇒ E10 **63/64** |
| 裁决 | **改 C4**（§九 禁止改 E1–E10）：13 个常量 `CSG_*` → **`QSL_*`**，函数 `qspace_csg_loss` → **`qspace_constraint_loss`** ⇒ E10 回到 **64/64** |
| 为什么不是"绕过守卫" | 守卫的意图是"CSG 线不得被**静默激活**在运行时里"。C4 与 C3 同性质：`tiger_ext/` 下**未接入任何运行时**的独立目标函数；`[8]` 逐条断言它不在任何运行时路径上、无参数、无 nn 层、不改内核。**意图成立，只是字面模式撞上了函数名** |

### 16.10 下游（C5 起）必须知道的边界

1. **ST 通路完好但 objective 未使用**：C5 若要让约束反传到 encoder，那是**新决策**（不是复刻，参考实现的路由只到码本）。
2. **C4 不算总 loss、不调 optimizer、不 backward 总 loss**；`lambda_csg` 只是 mount 上的乘子。
3. **空 pair ⇒ `0.0`（数据结果）；开关关闭 ⇒ 抛 `QSpaceDisabled`（接线错误）**。C5 必须先判 `enabled`。
4. **同码 pair 丢弃是适配**（参考实现保证正样本不同码）。
5. **层内是 pair 均值，不是 anchor 均值**：多邻居 anchor 会被按其邻居数加权（实测差 0.2826）。
6. **C2 的 weight 已确认不被消费**；C5 若要用它做加权，那是**新机制**，须另行记录。
7. `verify_indices_match_qspace` 是**校验路径**（`@torch.no_grad`），不是前向路径；`rebuild_quantized` 同理。
8. **C4 不改内核**：`git status --porcelain` 自 E6 结束起逐字未变（仍只有 `model/main.py`、`rqvae/models/rq.py`、`rqvae/models/vq.py` 三项 E1–E6 期间的既有改动）；EXP-C4 `[8]` 用 **6 个 sha256 pin** 证明 C1 / C2 / C3 / E7 / SIA / PSA 逐字节未变（`tiger_ext/` 是 untracked，git 看不见其中改动）。

**⚠️ 无效果结论**：C4 只证明"该目标函数被正确定义、梯度路径正确、且**不被任何运行时静默激活**"。**不含任何推荐效果结论**，也不含"该约束有效"的结论（`lambda_csg` 默认 0，样本 loss 仍在 `log(255)` 附近）。C5 / C6 与全部 B / C 类效果实验仍为 `PLANNED`。

---

## 17 修订记录 R11 — C5 实现（Q-space 约束接入训练，2026-10-14/15）

**任务**：把 C4 交付的 standalone 目标函数接进训练循环，使 `L_total = L_RQVAE + lambda_csg * L_csg` **真正进入 backward**，并证明这一点。**不做**最终推荐效果实验。

**判定**：**EXP-C5 = 99/99 checks passed（exit 0）**，`EXP_C5_REGRESSION=all` 下 7 份既有 harness 全部重跑，**每一条 FAIL 都是授权 guard**（`unexplained: none`）。

### 17.1 授权、唯一被改文件、"最小挂载"的机器核验

| 项 | 内容 |
|---|---|
| 唯一被修改的正式文件 | **`rqvae/trainer.py`**（plan.md §C5 明文授权；`git diff` = **7 hunks、+34/−5**；哈希两个都记，且**不可混用**：`sha256(git diff -- rqvae/trainer.py)[:16] = 9c1b73b4ffccbe3d`（4362 字节），EXP-C5 §[8] 打印的是 **Python `difflib.unified_diff(n=3, lineterm='')` 的哈希** `[:16] = a8da64e3291fa1c5`；改后文件 `sha256[:16] = d09ad776891dd3b2`） |
| 新增（全部在冻结内核之外） | `tiger_ext/qspace_train.py`（新驱动入口 + `CSGHook` + 常量）、`integration/exp_c5_csg_integration.py`（C5 harness）；`tiger_ext/__init__.py` 加 `CST_*` 导出 |
| 未动 | `clean_tiger/`、`reference_repos/`、`rqvae/main.py`、`rqvae/models/*`、`model/*`、T5、E7 契约、C1–C4 模块、全部既有 `exp_*.py` |

**用户 2026-10-14「方案 3」的 8 条**（不改任何既有 harness、不靠改 `PRE_*_MODIFIED` 掩盖）由 **EXP-C5 §[8] 逐条机械化**，全部实测通过：

1. **diff 只落在 `_train_epoch`（与 `__init__`）的授权挂载位置**——改动的 34 行**逐行**落在这两个方法的 AST 行带内（`outside: []`）；
2. **原有 `L_recon`/`L_trans`/`L_balance`/`L_align` 计算不变**——另 10 个方法体按 AST dump 逐一比对，逐字未变；
3. **只新增 `lambda_csg * L_csg`**——34 行扫描：无第二个 weight、无第二个 loss、无新增 arithmetic term；
4. **`lambda_csg = 0` 时不调用 CSG loss**——`n_calls = 0`、`n_disabled_calls = 0`；
5. **没有复制训练循环**——`backward`/`step`/`zero_grad`/`clip_grad_norm_`/`scheduler.step`/`def _train_epoch` 的出现次数改前改后逐项相同；
6. **没有修改 RQ-VAE 核心**——`rq.py`/`vq.py`/`rqvae.py`/`datasets.py`/`trans_loss.py`/`rqvae/main.py`/`model/main.py` 共 **7 个 pin 全 byte-identical**（C1/C2/C3/E7/SIA/PSA 的 6 个 pin 亦 6/6 命中，且这 6 个 pin 是**从 C4 记录它们的那份文件里重读**的，不是 C5 自己写的）；
7. **没有新增其他优化器 / warmup / regularization / 参数更新逻辑**——`__init__` 新增实例属性**只有 `csg`**，`_train_epoch` 新增实例状态**为零**，34 行里 0 处命中 warmup/weight_decay/regulariz/second optimiser 词表；
8. **harness 能机器核验上述条件**——`[8]` 25 项 + `[10]` 回归归因共同构成。

**两处条件分支的改法**：`is_track_b`（3 → 4 个 disjunct）与 5-tuple forward 分支（`Compare` → `Or`），由 **AST 断言**证明是"原表达式 + 恰好一个 disjunct"，而不是重写——即使重写后读起来一样也会失败。

**授权偏离（如实记录，不掩盖）**：`rqvae/trainer.py` 是第 4 个被修改的 tracked 文件，因此**每一份**既有 harness 的 "modified set" 守卫都会（**正确地**）判 FAIL。按用户裁决：**不改任何既有 harness**，由 EXP-C5 §[10] 单独报告该偏离并证明其可归因（见 17.7）。

### 17.2 total loss 的数学形式

```
L_total = L_RQVAE + lambda_csg * L_csg
L_RQVAE = loss_recon + quant_loss_weight * quant_loss      # 定义未改，rqvae/models/rqvae.py:87-97
```

- 挂载形状与既有三条可选加权项（`L_trans` / `L_balance` / `L_align`）**完全同构**，插在 `L_align` 之后、`loss.backward()` 之前。
- **权重只乘一次**：`CSGHook.forward` 返回的就是 `lambda_csg * L_csg`（与 `SIAMount`/`PSAMount` 同约定），trainer 侧只做 `loss = loss + loss_csg`。
- **入口资格守卫**：`if self.csg is not None and self.csg.enabled and batch_indices is not None`，且只有 `requires_grad and isfinite` 才真的加进 `loss`（C4 的空 pair 返回**无梯度**的 `0.0`，因此这一步是必要的而不是装饰）。
- **参考侧可复原到哪**：`reference_repos/CSG4Rec_original/CSGT/models/vq.py:209` = `codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss`；`CSGT/train_tokenizer.sh:8` 发布 `--beta 0.0001` ⇒ **参考对同形项的权重是 1e-4**（C4 实测 `L_csg ≈ log 255 = 5.5413`，故 1e-4 是**尺度匹配**而非照抄）。**不可复原**：`CSGT/models/rqvae.py` 在快照与 zip 中都不存在，它是 `use_knn/knn_weight/use_cf_loss/use_pairwise/use_kd` 的唯一读取点 ⇒ 参考的 `total` 公式与 `knn` 项**仍不可知**。C5 采用 plan.md 的 `L_RQVAE + λ·L_csg`，**不声称复刻参考**。

### 17.3 梯度策略（§二的四项，全部**实测**）

| 路由 | 判定 | 实测证据 |
|---|---|---|
| **P1 仅 codebook** | **采纳（默认）** | 三个 RQ level 的码本都拿到梯度（`Σ|dL_csg/dW_l|` = L0 7.99e-04 / L1 3.41e-05 / L2 2.62e-05）；**0 个 encoder 张量**从该目标拿到梯度。这也是**参考自身的路由**：`vq.py:195` 的约束 loss 在 straight-through **之前**，indices 出自 argmin/argmax，本无梯度 |
| **P2 ST → encoder** | **实测可达，不采纳** | 对同一个 `x_q` 做 `sum(x_q).backward()` 能触达 **10 个 encoder 张量**；`max|x_q − z| = 3.29e-02` 证明两条路由确实是不同量。不采纳的理由：要让 `L_csg` 走它，要么**替换目标里的 query 张量**（= 重新定义 `L_csg`，禁止），要么**从 `rqvae/models/{rq,vq}.py` 暴露逐层 ST 张量**（= 改冻结内核，禁止）⇒ 记为**边界冲突·待裁决** |
| **P3 baseline 不启用** | **采纳（默认）** | `lambda_csg = 0` ⇒ `enabled=False`、`n_calls=0`、不读码本、G0 逐位相等；且**禁用时 mount 抛 `QSpaceDisabled` 而不是返回人造零**（实测抛 1 次） |

**"真的进了 backward"的三重独立证明**（§六要求"不是只计算后未使用"）：

1. **同一初始状态、同一 batch**：`L_total(λ=1e-3) − L_total(λ=0) = 5.583858e-03`（给的确实是 backward 的那一个标量）；
2. **梯度可加性**：`max |g_total − g_base − g_objective| = 8.846e-13`（相对 `|g_base|` = 5.274e-08），逐码本张量成立——目标把梯度送到了可微叶子上；
3. **码本梯度真的动、encoder 梯度逐位不动**：3/3 码本张量随 λ 改变，**0/10** encoder 张量改变（同时印证 P1）。

> **测量陷阱（本次踩到，已修正）**：梯度可加性最初测在 **k-means 拟合的那个 batch** 上，那里 `|dL_RQVAE/dW| ≈ 2e-10`，相对误差被放大成 7.4e-03 而判 FAIL；改到 **batch 1**（k-means 已初始化完）后为 8.8e-13 / 5.3e-08。**先消耗 k-means 初始化再测梯度**是这次学到的规矩。

### 17.4 lambda 候选与选择依据

- 候选集合 = **`{1e-4, 1e-3, 1e-2}` + `λ=0` 控制**（不是搜索，每个一条短跑）。
- **相对贡献 `λ·E[L_csg] / E[L_RQVAE]`**：`1e-4 → 1.551%`、`1e-3 → 15.535%`、`1e-2 → 154.972%`（`E[L_RQVAE] = 0.035458`、`E[L_csg] = 5.508495`，20 步）。
- **选择依据**：`1e-4` 是**参考自己的同形项权重（尺度匹配锚点）**，在它上面 RQ-VAE loss 仍主导；`1e-2` 已让约束成为较大项 ⇒ 候选集合**横跨交叉点**而不是全在一侧。四个 arm 全部有限、无 NaN、无码本塌缩。
- **C5 不选工作点**：工作点属效果实验（EXP-C5/C6，明文禁止本次执行）。模块把 policy 写死、harness 把数字打印出来，**不宣布赢家**。

### 17.5 tokenizer sanity check（真实数据；Toys 前 2048 行 × 4096d，冻结配方逐字照搬）

配置：`num_emb_list=[256,256,256]`、`e_dim=32`、`layers=[512,256,128,64]`、`sk_epsilons=[0,0,0.003]`、`kmeans_init=True`、AdamW lr=1e-3、`quant_loss_weight=1.0`、batch 1024、10 epoch × 2 batch = **20 步**、`OMP_NUM_THREADS=1`。

| 检查 | 结果 |
|---|---|
| forward/backward 正常 | 20 步全部有限，`L_total ∈ [0.0232, 0.0834]`，无 NaN/Inf |
| reconstruction 正常下降 | epoch-1 均值 `0.0302` → epoch-10 `0.0180` |
| 冻结 trainer 的保存路径未被改坏 | 自写 `best_loss_model.pth` |
| code index 合法 | 2048 行索引全在 `[0, 255]` |
| **无码本塌缩** | L0 256/256 用、max share 0.047；L1 256/256、0.098；L2 137/256、0.106 |
| **无零码字**（改后判据） | 训练出的码本（**任何 forward 之前**读数，容差 1e-12）逐层最小行范数 = **6.903e-02 / 2.410e-02 / 6.368e-03** |
| L_csg 确实参与 total loss | 20/20 步都有 pair、都有三层 loss：累计 **109240 pair / 60 个 level-loss** |
| `λ=0` 保持原 TIGER 行为 | 20 步 `L_total` 逐位等于 `L_RQVAE`（且该控制跑确实在训练：20 个互不相同的 `L_RQVAE`） |
| disabled / empty-pair 语义 | disabled ⇒ 抛 `QSpaceDisabled`；空 pair ⇒ 无梯度的 `0.0` + `reason="no pairs"`，`L_total == L_RQVAE` 逐位 |
| 历史机制不受影响 | `contrastive_loss`/`balance_loss`/`_get_transition_loss`/`_get_align_loss` 在一次 λ>0 跑里**各 0 次**；负对照 `lambda_balance=1.0` 时同一探针**触发 20 次**（证明 0 是测出来的） |
| G0（关闭即等价） | 见 §17.6 末 |

### 17.6 顺带发现：冻结 k-means 初始化 × C3 provenance guard（**本次查实，必须记住**）

| 项 | 内容 |
|---|---|
| 现象 | 冻结配方 `kmeans_init=True` 下，`rqvae/models/layers.py:70-93` 的 k-means **拟合它看到的第一批点**。若该 batch 的不同点数 ≤ 码字数，则每个 RQ 残差都被精确复现 ⇒**`x_q == z` 逐位**、RQ-VAE 自己的码本梯度 ≈ 0 |
| 实测 | `kmeans_init=True`，**首 batch 1024 行 → `max|x_q − z| = 4.211e-02`**（guard-clean）；**首 batch 256 行 → `0.000e+00`**（落进 C3 的 1e-4 容差 ⇒ **guard 拒绝**）；`kmeans_init=False`，1024 行 → `1.101e+00`。拟合 batch 上 `|dL_RQVAE/dW| ≈ 2e-10`，下一 batch ≈ 3.0 |
| 裁决 | **不放宽 C3 的 guard、不改 C3**（§五）。C5 的 sanity check **用冻结的 batch size 1024**；harness `[6](c0)` 把该退化**端到端测出来**（"整个数据集只有一个 item"的跑在第一 step 就被 C3 拒绝，且拒绝语点明原因）；`[6](c)` 用 `kmeans_init=False` 跑那个 1-item 数据集，才能走到空 pair 路径 |
| 记录位置 | `tiger_ext/qspace_train.py` 的 `CST_PENDING_DECISIONS` 第 6 条（含全部实测数字）；EXP-C5 §[11] 三条 + §[6] (c0) |
| 与 C4 §16.9 的关系 | C4 记录的是"k-means 助手跨进程不可复现"；本项是"k-means 助手把**首 batch** 变成拟合集"。**同一枚硬币的另一面**：两者都要求"先明确 batch 几何，再谈数值" |

**G0（§C5 验收③，实测）**：用 `git show 5edfca0:rqvae/trainer.py` 取 pristine 源，同进程、同 loader、同种子，改前改后各跑一遍——**(a) 全默认配置**与 **(b) `lambda_balance=1.0`（专门触发被改的 `is_track_b` 分支）** 两种配置下，**逐步 loss 序列逐位相同**（8 个记录值）、**最终 `state_dict` 23/23 张量逐位相同**、初始状态 23/23 张量先证明可比。

### 17.7 回归结果（含**授权的** guard 偏离）

`EXP_C5_REGRESSION=all` 下重跑 7 份既有 harness（2026-10-15 实测）：

| harness | 结果 | 期望失败数 | 失败内容 |
|---|---|---|---|
| `exp_e8_sid_refresh` | **54/55** | 1 | 自身 modified-set guard |
| `exp_e9_alternating` | **80/81** | 1 | 自身 modified-set guard（`EXP_E9_REGRESSION=skip`，避免重复计 E1–E8 的 330 项） |
| `exp_e10_smoke` | **60/64** | 4 | 自身 guard + 它重跑的两个子 suite 的 guard + 二者 aggregate |
| `exp_c1_cf_embedding` | **130/132** | 2 | 自身 guard + 它内嵌的 E10 重跑 |
| `exp_c2_collab_relation` | **125/127** | 2 | 同上 |
| `exp_c3_qspace_constraint` | **96/99** | 3 | 两条自身 guard + 内嵌 E10 重跑（另外 4 条是它**回显**的 E10 输出，不计入） |
| `exp_c4_qspace_loss` | **84/87** | 3 | modified-set guard + **文本 guard**（"no frozen file mentions C4 or its switch"，挂载必然打破，因为开关名写在注释里）+ 内嵌 E10 重跑 |

**归因规则（机械，写进 harness）**：
- **R1** 该行点名一条 modified-set guard；
- **R2** 该行点名一份被重跑的 suite，而该 suite 的缺口**恰好等于**它的期望值、且它自己的失败也全被这些规则解释（R2 会核对行内引用的计数是否与实测一致，所以"写着 61/64 的行"不会被"实测 60/64"解释掉）；
- **R3** 该行是 E10 对两个受 guarded 子 suite 的 aggregate，其缺口 = 两者期望缺口之和（2 = 1 + 1）。

**结果**：7 份 harness 的**每一条** FAIL 行都被 R1/R2/R3 命中（`unexplained: none`），**且每条 harness 的"自身失败行数"恰好等于它的缺口**（回显行已被单独分离计数——`[FAIL]` 行数可以大于缺口，因为 C1–C4 会把它重跑的 E10 输出原样打出来；这一点被显式检查而不是被当作自己的结果）。

**C4 那条文本 guard 单独归因**：重跑 C4 自己的 grep 得到 **1 个** frozen 文件命中——正是被授权的 `rqvae/trainer.py`，命中的 3 行（119/121/366）**全部是注释**，冻结树里没有任何可执行行点名该开关。这既是偏离的证据，也是"偏离只有这一处"的证据。

**本次对 C5 自己 harness 的三处修正（如实记录，因为它改变了读数）**：
1. **计数行必须按 suite 自己的 `EXP-<id>` 读，且该行必须恰好出现一次**。第一版把 `COUNT_RE` 从 `(\d+)/(\d+) checks passed` 收紧到"必须带 `EXP-x:` 前缀"，**这还不够**：C3/C4 会把它们内嵌的 E10 重跑的**整段捕获输出——包括 E10 自己的 `EXP-E10: n/64 checks passed` 摘要行——打印在自己的摘要之前**，因此"匹配任意 `EXP-` 前缀"的第一条命中仍是 E10 的数字（实测 C3/C4 双双被读成 `60/64`）。最终修法是按 suite 自己的 id 建正则（`suite_summary_re(exp_id)`），并**新增一条检查**断言每份 suite 的自身摘要行**恰好出现 1 次**（否则读数为 `-1/-1` 而失败）。修后读数为真实的 **96/99**、**84/87**——**更严格，不是更宽松**（它同时暴露了"回显行会被算成自己的失败"这一独立问题，见下条）。
2. **`[FAIL]` 行要按 suite 分离"自身的"与"回显的"**。C1–C4 在重跑的 suite 失败时会把它捕获的输出原样打印，于是 E10 的 4 条失败行会**逐字出现在 C1–C4 的 stdout 里**。`_split_echoes()` 把两者分开，并**新增一条检查**断言"自身失败行数 == 该 suite 自己的缺口"。这一条与上一条互相独立：计数行来自摘要行，失败行来自 `[FAIL]` 行。
3. 码本"零行"判据由 `== 0` 改为 **1e-12 容差**，并**加一条配对对照**。原因：冻结的 `use_sk=True` forward 自己会替换死码，替换后 L2 第 0 行范数为 **6.628e-17**——"精确等于 0"的说法会把它判成健康。实测 **λ=0 的对照跑一模一样**（`5.854e-17`，同样 L2 一行），⇒ 该行是**冻结 forward 的死码替换**，不是约束造成的；训练出的码本（任何 forward 之前）逐层最小范数 6.9e-02 / 2.4e-02 / 6.4e-03，健康。

> **教训（写下来给后续 harness）**：当一份 harness 会**回显**它重跑的子套件输出时，"从 stdout 里读计数"必须**按被测量对象的身份**去读（自己的 id），不能按格式去读（任意 `EXP-` 前缀、任意 `N/M checks passed`）。同一个坑在本 port 已出现两次（第一次是 `$NF` 取 NDCG 列，见 [[eval-log-ndcg-column-trap]]），共同点是**日志格式是弱契约，位置/格式匹配会被嵌套输出骗到**。

### 17.8 下游（C6 起）必须知道的边界

1. **`L_csg` 只回传码本（P1）**；要让 ST 把梯度带给 encoder 是**新决策**，且当前边界下**做不到而不越界**（见 17.3 P2）。
2. **开关默认 0**；`lambda_csg = 0` 时 trainer **根本不进 hook**，`csg=None`（所有既有调用点）时所有新增表达式与改前**求值完全相同**。
3. **空 pair（数据结果，返回 0.0、无梯度）与开关关闭（接线错误，抛 `QSpaceDisabled`）严格分离**；trainer 因此在加进 `loss` 前检查 `requires_grad`。
4. **C5 新增的唯一训练入口是 `tiger_ext/qspace_train.py`**（默认 `lambda_csg=0`），`rqvae/main.py` 未动；E9 的 `tokenizer_phase` 仍以 `csg=None` 构造 trainer，因此 E9/E10 行为不变。
5. **CSG 只在 Tokenizer 阶段有意义**（它是 `L_RQVAE` 的一项，而 E7 契约已把 `L_RQVAE` 限定在 tokenizer 阶段）⇒ **E7 不需要任何改动**。
6. **batch 几何是硬约束**：首 batch 行数必须先让 k-means 初始化"花掉"（batch 1024 在 Toys 上 guard-clean；256 不 clean）。
7. **`rqvae/trainer.py` 现已是"被修改的 tracked 文件"** ⇒ 任何以 `PRE_*_MODIFIED` 为判据的既有 harness 都会报 1 条（或经 E10 传导的多条）guard FAIL。**这是授权状态，不是回归**；`integration/status.md` 的「已知问题」单独登记。

**⚠️ 无效果结论**：C5 只证明"该约束**真的接进了 backward**、梯度只到码本、关掉即与原 TIGER 逐位等价、且既有门全部只有授权 guard 一条偏离"。**不含任何推荐效果结论**，也不含"该约束有效"的结论（相对贡献 1.55%–155% 只是**尺度读数**）。**C6 已于 R12 通过；`lambda_csg` 工作点与全部 B/C 类效果实验仍为 `PLANNED`。**

---

## 18 修订记录 R12 — C6 实现（CSG smoke test，2026-10-15）

**判定**：**EXP-C6 63/63 checks passed，exit 0**（`EXP_C6_TOYS=all EXP_C6_REGRESSION=all`，2026-10-15；CPU）。**C6 = PASS**。

**定位（严格按用户设定的口径）**：**全链路 smoke test + 最小工作点验证**，**不是**最终论文效果实验。C6 只回答四个问题：① 四个 λ 能否跑通完整链；② CSG 是否真的进入训练；③ 完整 TIGER pipeline 是否被破坏；④ `kmeans_init` 哪个配置更安全。

**产物**：`tiger_ext/qspace_smoke.py`（新驱动：合成 fixture / 全链编排 / k-means 探针）+ `integration/exp_c6_csg_smoke.py`（新 harness，10 节）+ `tiger_ext/__init__.py`（一个 export block）。**C6 未新增任何 tracked 文件改动**——`git status --porcelain` 的 modified 集合仍是 C5 的四个文件，`sha256(difflib.unified_diff(..., n=3, lineterm=''))[:16]` 仍是 **`a8da64e3291fa1c5`**，改后文件 `sha16 = d09ad776891dd3b2`（与 C5 记录逐字符相同）。**`alt_pipeline.py` / `stage.py` / `sid_refresh.py` / `alternating.py` / `rqvae/*` / `model/*` / C1–C5 既有 harness / E7 / E10 全部未动。**

**组合而非改写**：tokenizer 段走 C5 的 `train_tokenizer(hook=…)`（它**不写** `phase_log`，只有 `alt_pipeline` 的三个 phase 写），其余三段走 `TigerAlternatingPipeline.refresh_phase → recommender_phase → eval_phase`。因此 `phase_log` 只有三条；tokenizer 段的证据是"ckpt + n_steps + 有限 best_loss"。**未伪造任何 `phase_log` 行。**

**四臂全链（Q1）**：A=0 / B=1e-4 / C=1e-3 / D=1e-2 全部跑完 `tokenizer → refresh → recommender → evaluation`，exit 0，四类产物（tokenizer ckpt / SID `.npy` / recommender ckpt / eval 指标）齐备。合成臂 3.3s / 2.2s / 2.2s / 2.2s；Toys（λ=0 与 λ=1e-3）65s / 72s。

**CSG 真的进入训练（Q2）**：λ>0 的三臂 `n_calls == n_steps`（合成 3/3，Toys 24/24），`n_nonempty == n_calls`，`n_zero = 0`，每步 `L_csg > 0` 且 `weighted == λ · L_csg`；同初始状态同 batch 下 `L_total(1e-3) − L_total(0) = 8.114338e-03`；梯度可加性 `max|g_total − g_base − g_objective| = 7.276e-12`（相对 9.245e-08）；**码本张量 3/3 随 λ 变、encoder 张量 0/6 变（P1 保持）**；全臂无 NaN/Inf。λ=0 臂 `n_calls = 0` **且 `n_disabled_calls = 0`**（关闭 ≠ 接线错误）。

**λ=0 逐位等价（Q3a）**：`hook=CSGHook.for_dataset(..., lambda_csg=0.0)` vs `hook=None`（Clean TIGER 的调用签名），**`torch.equal` / `np.array_equal`**：SID `.npy` sha256 相同、tokenizer `state_dict` 15/15 张量逐位相同、recommender `state_dict` 29/29 逐位相同、`recalls`/`ndcgs` 逐位相同、`best_loss` 同为 `3.148388147354126`。**这是 G0 从 trainer 级（C5 §[9]）扩到全链级。**

**pipeline 未被破坏（Q3b）**：recommender ckpt 带 E7 recommender-stage manifest；tokenizer 组 0/15 trainable、recommender 组 26/26 trainable；SID manifest 按哈希指向产出它的 tokenizer ckpt；handoff `mismatches = {sia: 0, psa: 0}`；`L_trans` / `L_balance` / `L_align` 在 λ>0 运行中各调用 **0 次**（负对照 `lambda_balance=1.0` 触发 2 次）；**SIA/PSA mount 的 forward 各调用 0 次**（负对照证明计数器活着，且 disabled 的 mount 会抛）。

**User SID 泄漏检查（Q3c）**：SID 形状 `(n_items, 4)`，**没有 user 轴**；tokenizer ckpt 记录的输入只有 `item_emb.parquet`；`run_chain` 的 85 个符号名 / 62 个字面量扫描中 **user 侧 0、交互文件 0**；受约束路径的 4 个模块全库扫描 **0 命中**；phase log 无任何 per-user 字段。

**kmeans_init 裁决（Q4）**：**维持冻结配方 `kmeans_init=True`**。batch 1024 下两个配置都过 C3 的两条首 batch guard（rebuild 与 latent），但**码本使用率差距很大**：`True` 在合成/Toys 上首 epoch distinct = `[128,128,128]` / `[256,256,240]`（max_share 0.041 / 0.146），`False` = `[22,20,65]` / `[42,46,256]`（max_share 0.321 / 0.523，**level 0 只用掉 42/256**）。**没有引入任何新初始化机制、没有改 RQ-VAE 核心、没有改 batch size**。C5 的退化情形（`x_q == z` 逐位）是 **batch 几何**性质而非 flag 性质，C6 继承的约束是"**每个首 batch 都要量**"，不是"换初始化器"。

**Toys 已知问题如实记录**：真实臂只能是 Toys（C1/C2 产物只有 Toys/Sport/Beauty，`Musical_Instruments` 两者皆无）。该臂 2-epoch 4096d tokenizer 的 **collision rate ≈ 0.9225（λ=0）/ 0.9233（λ=1e-3）**，**Recall/NDCG 全为 0**——这是已知的 4096d 欠训练 confound（status.md「已知问题」第 3 条），**不是 CSG 的效果读数**，C6 不据此对约束作任何判断。

**⚠️ 无效果结论**：C6 的 Recall/NDCG **只是"链路跑通"的证据**，**不得**作为 CSG 有效/无效的论据。`lambda_csg` 工作点、tokenizer 结构指标与 Recall/NDCG 的正式比较，属 C6 之后的效果实验。

---

## 19 修订记录 R13 — 效果实验设计（M8 / M9，2026-10-18）

**触发**：C6 smoke PASS 后，工程侧 E1–E10 / C1–C6 全部 A-PASS。按 §5.2 的阶段门禁，现在进入 **B 类机制验证** 与 **C 类正式实验**。本节把 §7 的"建议 5 组"收敛为**可执行的 2×2×2 设计**。

> **本节只写设计，不写任何效果结果。** 所有格子状态一律 `PLANNED`；任何数字出现在这里之前必须来自真实运行（[experiments.md §5](experiments.md) 规则 2）。

### 19.0 用户裁决（2026-10-18，四项，冻结）

| # | 决策点 | 裁决 |
|---|---|---|
| **D1** | content 管线口径 | **768d content-only**（放弃 §1.3 的 4096d） |
| **D2** | 矩阵结构 | **2×2×2 全因子，8 格**（含交互项） |
| **D3** | λ 工作点 | **先扫后定**（1 数据集 × 1 seed 标定，预注册选点规则） |
| **D4** | B 类规模 | **1 数据集 × 1 seed 全格筛选** |

> **D1 与 §1.3 冲突，以本节为准。** §1.3 的 `D_target` 表保留为 **4096d 历史口径**，R13 起本系列实验的 `D_target = 768`。

### 19.1 口径冻结（768d content-only）

**为什么改口径——三条实测证据，不是偏好**：

| 证据 | 4096d（§1.3） | 768d（本节） |
|---|---|---|
| Toys tokenizer 质量 | C6 实测 collision **0.9225**，Recall/NDCG 全 0（唯一权重是 2-epoch smoke） | **0.000419**，3000 ep 已训练 |
| Sport / Beauty 的 content-only RQ-VAE | **不存在**（只有融合表上的 `*_TA_SID_4096d`，EXP-C3 已记录） | **存在**（`{D}_Content768d_Baseline`） |
| Clean TIGER 基线 | 需新跑 3 个 RQ-VAE + 9 路 T5 | **9/9 已存在**（3 数据集 × seed 2025/2026/2027） |

⇒ 4096d 路线必须先把「已知问题」第 3 条（Toys 4096d 欠训练）从**阻塞项**解决掉才能开工；768d 路线**零新增基线成本、零 blocker**，且与 TA-SID 主实验同口径。

**固定资产（全部已存在，不重跑）**：

| 项 | 路径 |
|---|---|
| content 表 | `data/{D}/item_emb_768d_contentonly.parquet`（`in_dim = 768`） |
| RQ-VAE 基线 ckpt | `rqvae/ckpt/{D}_Content768d_Baseline/Jul-31-2026_*/best_collision_model.pth` |
| 基线 SID | `data/{D}/{D}_content768d_baseline_sid.npy` |
| T5 基线 ckpt | `model/ckpt/tiger_content768d_baseline_{ds}_seed{2025,2026,2027}.pth`（**9/9**） |
| 基线评测日志 | `model/logs/{D}_content768d_baseline_seed{S}_eval.log`（**3 列 = @5/@10/@20**） |

**实测基线读数（seed 2025；仅作口径锚点，不作结论）**：

| 数据集 | collision | N@5 | N@10 | N@20 | R@20 |
|---|---:|---:|---:|---:|---:|
| Toys | 0.000419 | 0.0210 | 0.0272 | **0.0341** | 0.0795 |
| Sport | 0.000545 | 0.0160 | 0.0211 | **0.0274** | 0.0664 |
| Beauty | 0.001653 | 0.0238 | 0.0301 | **0.0368** | 0.0822 |

**RQ-VAE 配方（从 `rqvae/logs/{D}_content768d_baseline_rqvae.log` 的 Namespace 逐项抄出，新臂必须逐字继承）**：

```
lr=1e-3, epochs=3000, batch_size=1024, num_workers=4, eval_step=50,
learner=AdamW, lr_scheduler_type=linear, warmup_epochs=50, weight_decay=1e-4,
dropout_prob=0.0, bn=False, loss_type=mse,
kmeans_init=True,          # C6 (R12) 裁决，不得改
kmeans_iters=100, sk_epsilons=[0,0,0.003], sk_iters=50,
num_emb_list=[256,256,256], e_dim=32, quant_loss_weight=1.0, beta=0.25,
layers=[512,256,128,64], tau=0.1,
lambda_trans=0.0, lambda_balance=0.0, trans_graph_path=None
```

- ⚠️ 该 Namespace **不含 `align_weight`**（产物早于该 flag）。新臂**必须显式传 `--align_weight 0.0`**，不得依赖默认值。开跑前用 `git show 5edfca0:rqvae/main.py` 复核默认值确为 `0.0`。
- ⚠️ 新臂的 RQ-VAE **不保证与基线逐位可复现**（`rqvae/models/layers.py:86` 的 `KMeans` 无 `random_state`，「已知问题」第 6 条）。比较是"**新臂 vs 固定基线产物**"，**不要求重跑基线**。

**维度契约**：`D_target = 768`（PSA adapter = `Linear(d_model=128, 768)`）。SIA 侧不变（3 层 × 256 code logits）。C1/C2 侧与 content 维度无关（CF 仍是 SASRec 50d）。

### 19.2 因子与 8 格

| 因子 | = 0 | = 1 |
|---|---|---|
| **C**（CSG） | tokenizer 训练**不含** `L_csg` | `L_csg` 进 tokenizer 训练（`CSGHook`） |
| **E**（ETE） | T5 训练**不含** `L_SIA` / `L_PSA` | 两者均进 T5 训练（`lambda_sia`, `lambda_psa` > 0） |
| **A**（ALT） | tokenizer 在 T5 训练期间**冻结**，SID 固定为 v0 | tokenizer 每 cycle **精化 1 ep** 并 refresh SID |

| # | C | E | A | 名称 | 与旧 ID 的关系 |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | **TIGER**（基线） | 已存在，**不跑** |
| 2 | 1 | 0 | 0 | **+CSG** | 旧 I1 |
| 3 | 0 | 1 | 0 | **+ETE** | 旧 I2 的固定调度版（**新增能力，P2**） |
| 4 | 0 | 0 | 1 | **+ALT** | 旧 I3 |
| 5 | 1 | 1 | 0 | **CSG+ETE** | **新增格** |
| 6 | 1 | 0 | 1 | **CSG+ALT** | **新增格（需 P1）** |
| 7 | 0 | 1 | 1 | **ETE+ALT** | 旧 I2 |
| 8 | 1 | 1 | 1 | **Full** | 旧 I4（需 P1） |

⇒ **`I1 = #2`、`I2 = #7`、`I3 = #4`、`I4 = #8`**。旧 ID 保留为别名，不废弃。

**为什么必须全因子**：CSG 作用在 **tokenizer 侧**，ETE 作用在 **推荐器侧**。若两者各自有效，论文只能写"两个模块都行"；只有测出 **`C×E` 交互项**才能回答"融合是协同、还是相互抵消"——这正是本文的卖点。§7 的 5 组并列设计**在结构上无法回答这个问题**。

### 19.3 预算等价规则（**开工前冻结；违反则整批实验作废**）

问题：`alt_pipeline` 每 phase **1 epoch**（`alt_pipeline.py:309`），而基线是 **RQ-VAE 3000 ep / T5 200 ep**。不先定规则，交替臂与基线的任何比较都无效。

1. **T5（被评测组件）预算严格相同**：8 格一律 **200 ep 上限 + `early_stop 10`**，判据 = valid NDCG@20，`beam 30`、`infer_size 96`、`topk_list = 5,10,20`。**不得为任何单格调整 T5 预算。**
2. **A=0 的 tokenizer** = 冻结基线 ckpt（C=0），或用同配方 + `L_csg` 重训 **3000 ep**（C=1）。
3. **A=1 的 tokenizer** 从该格对应的 ckpt 起步（C=1 从 CSG 版 3000-ep ckpt，C=0 从基线 ckpt），在 200 个 cycle 里每 cycle 精化 **1 ep**，每 cycle 后 refresh SID。
4. **epoch → cycle 映射**：A=1 的 `n_cycles = 200`（1 cycle = 1 T ep + 1 R ep），使 T5 有效 epoch 数与 A=0 相同。
5. **每 cycle 记录 SID diff rate**（E8 已有该产物）与 tokenizer ckpt 哈希。

> ⚠️ **这是"同 T5 预算 + 不同 tokenizer 调度"，不是"同 tokenizer 预算"。** A=1 的 tokenizer 额外获得 200 ep 精化；A=0 的 tokenizer 只有一次 3000 ep。**论文必须写成一条限制**，并同时报告两种范式的 tokenizer optimizer steps。

**为什么不让 A=1 从零交替（ETEGRec 原生做法）**：从零交替时 1 cycle 只给 1 tokenizer ep，200 cycle = 200 ep，相对 A=0 的 3000 ep 是**第二个混杂因子（预算）**，会把"调度效应"与"预算缩减效应"混在一起。**从收敛 ckpt 起步是唯一能把 A 归因干净的做法。**

**由此必然的预期（先说清楚，避免事后惊讶）**：

- A=1 且 C=0 时，tokenizer 的目标函数与起点都与 A=0 相同 ⇒ tokenizer 几乎不动 ⇒ SID diff ≈ 0 ⇒ **#4（+ALT）预期 ≈ #1（TIGER）**。
- 因此 **A 的独立主效应大概率 ≈ 0**；A 的增益只可能通过 **`A×C` 交互**出现（CSG 让 tokenizer 持续移动 ⇒ SID 真正刷新）。
- 若 A 主效应与 `A×C` 交互都 ≈ 0，**如实写"交替在本设置下无增益"**，并按 §7 收敛规则把 ALT 降级为附录。**不得**为保住 I3/I4 而改判据。
- **SID diff rate 是 A 是否"真有作用"的必要证据**：diff ≈ 0 却写"交替有效"，属造假。

### 19.4 Stage 0 — λ 标定（先扫后定）

**数据集 Toys**（成本最低 48.7 min/seed；768d 下 collision 0.000419 为三数据集最干净）× **seed 2025**。预算同 §19.3。

| 臂 | C | E | λ_csg | λ_sia | λ_psa |
|---|---|---|---|---|---|
| `λc-1e-4` | 1 | 0 | 1e-4 | 0 | 0 |
| `λc-1e-3` | 1 | 0 | 1e-3 | 0 | 0 |
| `λc-1e-2` | 1 | 0 | 1e-2 | 0 | 0 |
| `λe-0.1` | 0 | 1 | 0 | 0.1 | 0.1 |
| `λe-1.0` | 0 | 1 | 0 | 1.0 | 1.0 |

**候选域的依据**：λ_csg ∈ {1e-4, 1e-3, 1e-2} 来自 C5 的相对贡献实测（**1.551% / 15.535% / 154.972%**），C6 已证四档全链无 NaN/Inf/塌缩；1e-4 是参考实现 `--beta 0.0001` 的尺度锚点。
λ_sia/λ_psa 的 **0.1 是 `alt_pipeline.py:187` 的实现默认值，本文件从未记录其来源**（E4/E6 只写"默认 0"）⇒ **本次把它当待标定值，不当既定值**。1.0 是十倍方向的第二点。

**选点规则（预注册，事后不得改）**：

1. 主判据：`ΔNDCG@20 = 臂 − 基线(seed 2025)`。
2. 副判据（只用于**排除**，不用于选优）：tokenizer collision 不得高于基线 **2×**；三层码本不得塌缩（C3 的 `QSpaceBatch.validate` 会自动拒绝逐位复现）。
3. 选**在基线上不劣化（Δ ≥ −0.001 且不塌缩）的最大 λ**。
4. 若三档全部劣化 ⇒ 取尺度锚点 **λ_csg = 1e-4、λ_sia = λ_psa = 0.1**，并**如实记录"λ 标定未找到正向工作点"**。
5. 单 seed **不做显著性检验**，只作方向判据。

> ⚠️ 选点必须在 Stage 1 / 2 开始**之前**写入本节，**不得边跑边改**。

### 19.5 Stage 1 — B 类机制验证（1 数据集 × 1 seed 全格）

**设计**：Toys × seed 2025 × 8 格（#1 已有，**实跑 7 格**）+ **1 个 SIA-only 子臂**（`λ_sia > 0, λ_psa = 0`，与格 #3 只差 PSA 一项），λ 用 Stage 0 的选点。

> **为什么多这 1 路**：2×2×2 把 ETE 当作**一个因子**，无法区分 SIA 与 PSA 各自的贡献，而 EXP-E5/E6 的原始问题正是"PSA 的增量"。1 路 ≈ 0.8 h。**这是对裁决 D4 的 +1 路扩展，标注为待你追认**；若不采纳，则 ETE 只能整体归因，论文中不得写"SIA/PSA 各自的贡献"。

**门禁（预注册）**：

| 门禁 | 条件 | 不通过时 |
|---|---|---|
| **G-B1 链路** | 7 格全部 exit 0，四类产物齐备 | 停下修，不进 C 类 |
| **G-B2 不塌缩** | 每格 collision ≤ 2× 基线；码本 distinct 正常 | 该格判"机制有害"，进 C 类前须说明 |
| **G-B3 方向** | 无任何格出现 `ΔNDCG@20 < −0.002` | 该格降级为附录 |
| **G-B4 交替有效性** | A=1 各格的 SID diff rate **必须 > 0**；若 ≈ 0 ⇒ A 因子判为惰性 | ALT 降级为附录 |

**B 类不做的事**：不做显著性检验（单 seed）、不选最终模型、不进论文主表。唯一产出是"哪些格值得进 C 类"。

> ⚠️ **Toys 特例风险**：Toys 在本项目历史里多次是特例（E44「容量不足」；TA-SID 的 seed 2024 塌缩）。若 B 类给 NO-GO，**不得直接放弃**——先在 Sport 上复筛一次（7 格 ≈ 9.4 h）再决定。若 B 类给 GO，Toys 单点也**不足以**支撑结论，C 类必须覆盖三数据集。

### 19.6 Stage 2 — C 类正式实验

**规模**：通过 G-B 的格数 × 3 数据集（Toys / Sport / Beauty）× 3 seed（2025 / 2026 / 2027）。

- 最坏情形（8 格全过）：7 新格 × 9 = **63 路**。
- 典型情形（ALT 降级为附录）：5 格 × 9 = **45 路**。
- 基线格复用现有 9 路。

**统计口径（沿用本项目已确立的纪律）**：

- 主指标 **NDCG@20**；配对显著性用 `tools/paired_significance_ablation.py`（`sqrt(2+t²)` 兜底公式已于 2026-10-07 修正）。
- 同时报 Recall@20，但**主指标只报 NDCG@20 的配对 p**（2026-10-07 已实测 N20/R20 有 6 处判决翻转，不得混报）。
- std 一律 `ddof=1`。
- 交互分析：`C×E`、`C×A`、`E×A` 二阶 + 三阶，**用 2×2×2 的边际均值算，不用单格相减**。
- 取数纪律：NDCG 按 **Metric 表头映射**，**禁止 `$NF`**（5 列下会取到 @100）。

**结论写法（预注册）**：

- 某因子主效应不显著 ⇒ 写"无显著增益"，**不写**"有提升趋势"。
- 交互显著但主效应不显著 ⇒ 主结论写成"仅在 X 条件下有效"，**不得**写成"普遍有效"。
- **任何 `≈` 必须给区间**，不得只写"≈ 基线"（E50 的教训）。

### 19.7 需要新增的两条接线（P1 / P2）

| 编号 | 内容 | 改动面 | 影响 |
|---|---|---|---|
| **P1** | ✅ **ACCEPTED（2026-10-18）** `TigerAlternatingPipeline` 增加 `csg=None` 可选参数，`tokenizer_phase` 透传给 `RQVAETrainer` | `tiger_ext/alt_pipeline.py`（**未跟踪**） | 默认 `None` ⇒ 行为不变（实测逐位等价）；差分回归 **零新增失败** |
| **P2** | ✅ **IMPLEMENTED（2026-10-18），待验收** 新驱动 `tiger_ext/fixed_train.py`：冻结 tokenizer 下直接调 `model/main.py::train` 跑 N epoch，并构造 `SIAMount` / `PSAMount` | **新文件**，不改既有文件（`model/`、`rqvae/`、`clean_tiger/` 零改动，mtime 仍为 2026-10-14） | 无：无任何既有模块 import 它 |

**P1 的影响面已实测复核（2026-10-18）**：

- `alt_pipeline.py` **不在**任何 `PINNED_SHA16` 里——C4/C5 只钉 `cf_embed` / `collab_relation` / `qspace_constraint` / `stage` / `psa_loss` / `sia_loss` / `qspace_loss` 七个模块。
- `tiger_ext/` **未被 git 跟踪** ⇒ 不改变 `PRE_E9_MODIFIED` / `PRE_E10_MODIFIED` 的 tracked 集合。
- ⇒ P1 **不会**重开 E9 / E10 的守卫。**但改动后必须重跑 EXP-E9 与 EXP-E10，并证明逐项与基线一致**（见下「门禁口径」），不得外推。

> ⚠️ **更正一处口头判断**：早前在会话中说"`alt_pipeline.py` 被 sha256 pin 钉住"是**错的**——那些 sha256 是**产物**（ckpt / SID）的哈希，不是源码 pin。P1 的实际代价比预估小。

#### 门禁口径（**本节为准，适用于 P1、P2 及后续一切内核外改动**）

**判定用「差分回归」，不用绝对分数。**

| 项 | 规则 |
|---|---|
| 基线 | **C5 之后、本次改动之前**的同一套 suite 结果（即"改动前基线"），必须在同一 commit、同一 device、同一依赖环境下测得 |
| 判定 | 改动后 suite 的**检查项逐条与基线相同**（PASS/FAIL 判定 + detail；detail 中仅时间戳、临时目录名可不同）⇒ **无新增失败** |
| **禁止** | 不得以历史绝对分数（如 EXP-E9「90/90」、EXP-E10「64/64」）作为通过条件——那两个数是 **C5 之前**的读数，C5 授权改内核后**已不可达** |
| **禁止** | 不得写成"全部测试通过"。基线中既有的失败必须**原样列出**，并标注归属 |
| 既有失败归属 | 7 项 **tracked-file guard** 失败，全部来自 **C5 授权修改 `model/main.py` + `rqvae/models/{rq,vq}.py` + `rqvae/trainer.py`**，与 P1 无关。**不得**为让它们变绿而改 harness 或回滚 C5 |

#### P1 实现记录（2026-10-18，**ACCEPTED**）

| 项 | 值 |
|---|---|
| 生产文件 | `tiger_ext/alt_pipeline.py`（**未跟踪**）｜`sha256 = fc1b8652a72c04cec685192604841a3f8fcd64c8641e3f1c61e635cee3a86677`（`[:16] = fc1b8652a72c04ce`） |
| 规模 | **9 hunks，净 +76 行**（626 → 702 行） |
| 测试 harness | `integration/exp_p1_csg_passthrough.py`｜`sha256[:16] = dd267ea07c145356` |
| 新参数 | `TigerAlternatingPipeline(..., csg=None)`，**末位关键字**，既有位置参数完全不变 |
| 透传点 | `tokenizer_phase`：`mods["RQVAETrainer"](args, self.rqvae, len(loader), csg=csg)` |
| ⚠️ 配套改动 | 仅当 `csg.enabled` 时 `EmbDataset(..., return_index=True)`。**不这样改，trainer 的 `batch_indices is not None` 守卫会静默丢弃 L_csg**（C5 的 `build_dataset` 已对同一坑设防） |
| 记录面 | `result["csg"]` 与 `run_config()["csg"]` **只在挂载了 hook 时出现** ⇒ `csg=None` 的输出字典与 P1 之前逐键相同 |
| 新守卫 | ① 开启 hook 但数据集不返回行索引 ⇒ `RuntimeError`；② `trainer.csg is not csg` ⇒ `RuntimeError`；③ 非 hook 对象 ⇒ `TypeError`；④ 带参数的 hook ⇒ `ValueError`（C5 的 parameter-free 不变量） |

**EXP-P1 = 27/27**（CPU）：默认路径与 **pre-P1 调用点**（`EmbDataset(path)` / `Trainer(args, model, n)`）逐位等价 —— **23/23 张量 `torch.equal`**、`best_loss` 与 collision rate 完全相同、结果字典键集相同、`run_config` 无 `csg` 键；`lambda_csg=0` 的 hook 同样逐位惰性（`n_calls=0`、`n_disabled_calls=0`）；enabled hook 每 batch 命中一次（4/4）、经 C3/C4 各 4 次、4/4 步产出非零且可导的 `L_csg`，且 state_dict 与 `csg=None` **不同**；负向控制 3/3 抛错。

**差分回归结果（P1 前 vs P1 后）**：

| suite | 基线（P1 前） | P1 后 | 逐项比对 | 新增失败 |
|---|---|---|---|---|
| EXP-E9 | 87/90 | **87/90** | **90/90 检查行逐字节相同**（含 detail） | **0** |
| EXP-E10 | 60/64 | **60/64** | 64 行中 62 行逐字节相同；2 行仅差**临时目录名**（#4）与**进度条时间戳**（#62），判定与数值均不变 | **0** |

**基线中既有失败（原样列出，归属 C5，不属于 P1）**：

| suite | 失败项 | 归因 |
|---|---|---|
| E9 + E10 | `model/ and rqvae/ carry no E9/E10 change -- modified: model/main.py, rqvae/models/rq.py, rqvae/models/vq.py, rqvae/trainer.py` | C5 授权改内核 |
| E9 + E10 | `exp_e8_sid_refresh: 54/55` | 同上（E8 自己的同一守卫） |
| E10 | `exp_e9_alternating: 80/81` | 同上（E9 在同一守卫上） |
| E9 | `(12) the E1-E8 suites are unaffected by E9 (329/330)` | 同上 |
| E10 | `the E1-E9 suites are unaffected by E10 (409/411)` | 同上 |

> ⇒ **P1 的验收结论是「零新增失败」，不是「90/90、64/64 全绿」。** 上表 7 项在 P1 之前就存在，**本任务不修复**。

**已知限制**：① 上述差分回归在 **CPU** 上测得，CUDA 未复跑（历史 E9/E10 曾 CPU+CUDA 双跑，本任务的 A/B 只做 CPU）；② 基线文件是**逐字还原**的 pre-P1 版本（30227 字节，与改动前文件字节数一致），不是 git 存档（`tiger_ext/` 未被跟踪）；③ `result["csg"]` / `run_config()["csg"]` 的"仅挂载时出现"是不对称设计，下游需接受 `config.get("csg") is None`。

#### P2 实现记录（2026-10-18，**IMPLEMENTED，待验收**）

| 项 | 值 |
|---|---|
| 生产文件 | `tiger_ext/fixed_train.py`（**新增**，未跟踪）｜`sha256 = b3ec855be760bafc737e2398b36292a6140ab1b64dd293c56c95d863595b83af`（`[:16] = b3ec855be760bafc`）｜628 行 |
| 测试 harness | `integration/exp_p2_fixed_train.py`｜`sha256[:16] = 555ef348da5e0e24`｜675 行 |
| 不改既有文件 | `model/`、`rqvae/`、`clean_tiger/` 零改动（mtime 仍为 2026-10-14）；`grep -rn fixed_train model rqvae clean_tiger tools` = 空 ⇒ 无任何既有模块依赖它 |
| 入口（API） | `FixedSidTrainer(dataset, data_root, sid_path, tokenizer_ckpt, run_dir, ...)`；`run()` 返回完整 manifest 并写 `run.json` |
| 入口（CLI） | `python -m tiger_ext.fixed_train --sid_path … --tokenizer_ckpt … --run_dir …`；`--rqvae_config` 接受 JSON overrides（默认 `RQVAE_CONFIG`），非默认码本的 tokenizer 必须传 |
| 固定 tokenizer/SID | `StageFreeze.apply("recommender")`：RQ-VAE `requires_grad=False` + 码本 `initted=True` 严格封存；SID 只读一次，`validate_sid` 对 `n_items` 校验；**每个 epoch 后**复核 tokenizer `state_dict` digest、SID sha256、码本 `initted` 标志 |
| 多轮训练 | 单一 `Adam`（T5 ∪ 两 adapter，与 RQ-VAE 零交，构造后断言）驱动 `model/main.py::__main__` 的完整协议：每 epoch `train` → valid `evaluate` → `NDCG@20` 改进时 evaluate test + 存 best → `early_stop=10` |
| ETE 接入 | 仅 `λ>0` 时构造 `SIAMount`/`PSAMount`；`λ=0` 时向 `train()` 传 `sia=None, psa=None`（Clean TIGER 分支，不触发 `return_hidden`）；不新增/不改任何损失公式、温度或优化策略 |
| CSG | **拒绝** `csg=`（`FixedSidError`）：CSG 是 tokenizer 阶段损失，格 #5 只能经"CSG-trained ckpt + 该 ckpt 产出的 SID"进入 |
| checkpoint 布局 | `best.pth` / `last.pth` = **纯 T5 `state_dict`**（`tools/eval_trained_model.py` 可直接 strict load）；adapter 单独存 `{best,last}.{sia,psa}_adapter.pth` |

**EXP-P2 = 47/47（CPU）**，覆盖用户要求的最小正向测试：

| 节 | 证明 |
|---|---|
| [2][3] | ≥3 epoch 真跑（`train` 每 epoch 恰好一次，单 optimizer），T5 权重移动；SID 文件逐字节、tokenizer digest 逐位不变 |
| [4] | `λ=0`：两个 mount **根本不构造**且 `train()` 收到 `None`，两 adapter 逐位不动；SIA-only：SIA adapter 移动、**PSA adapter 逐位不动**（其损失从未计算）；both：两者都移动，SID/tokenizer 仍冻结 |
| [5] | `λ=0` 的一次运行与**手工重放 `model/main.py::__main__` 循环**（含每 epoch valid evaluate、改进时 test evaluate）**逐位相同**（50/50 张量 `torch.equal`，每 epoch `train_loss` 完全一致） |
| [6] | `recommender_phase` 被 monkeypatch 成抛错，训练仍跑完 ⇒ 未复用相变函数 |
| [7] | 改进分支**实际到达**（小词表 split：`NDCG@20` 由 0 → 0.6309）：test 评估次数 = 改进次数 = 4，`best.pth` 恰在改进时写且 ≠ `last.pth` |
| [8] | 格 #5 仅通过输入（ckpt + SID）支持并记录 provenance |
| [9] | `python -m tiger_ext.fixed_train` 端到端 exit 0、写 `run.json`、打印含 arm/λ 的 JSON 摘要 |

**已知限制**：① 全部验证在 **CPU** 上完成，CUDA 未复跑；② 小词表 split 用 `batch_size=16` 只是为了凑够梯度步（否则 256 batch 下每 epoch 仅 1 步、NDCG 恒 0），**不是**协议改动，正式臂仍用 `T5_BATCH=256`；③ `best.pth` 只在 valid `NDCG@20` 改进时写——若全程无改进则只有 `last.pth`（与 `model/main.py` 一致）；④ `tiger_ext/__init__.py` **未改**（不做 re-export，`from tiger_ext.fixed_train import FixedSidTrainer` 可直接用）；⑤ 本任务**未运行**任何真实效果实验（EXP-L0 / EXP-B1 未启动），也未调整 R13 设计；⑥ P2 是新增文件、未改既有代码，故**未重跑** EXP-E9/EXP-E10 差分回归——该门禁针对"改动内核外既有文件"，P1 已完成并记录。

**P2 为什么不能复用 `recommender_phase`**：该函数每调一次是一个**独立 Trainer**（每 phase 重建 optimizer + linear schedule，`PHASE_WARMUP_EPOCHS = 0`）。连调 200 次 ≠ 一次 200-epoch 训练，训练体制不同；直接复用会让固定调度的 ETE 臂与基线不可比。必须走 `model/main.py::train` 的单次多 epoch 调用（`train()` 已支持 `sia=` / `psa=` 关键字，`alt_pipeline.py:428` 即此用法）。

**ST→encoder 路由在本设计中不变**：仍为 **P1 路由（梯度只到码本）**。**不得**为"跑出效果"而改 objective——那是新授权，不是"补上 C5 漏掉的东西"。

### 19.8 必须写进论文的限制

1. **A 因子测的是"收敛后继续交替"，不是 ETEGRec 原生的"从零交替"**（§19.3 的预算理由）。若 `A×C` 显著，追加"从零交替"**附录臂**作稳健性，不进主表。
2. **768d 与 §1.3 的 4096d 不一致**：本系列全部在 768d content-only 口径下，与冻结 commit `5edfca0` 的 `clean_tiger` 默认入口（4096d）不同口径。**不得**把本系列数字与任何 4096d 数字并列。
3. **CSG 是"参考形态而非复刻"**（C4：`is_a_reproduction = False`，kNN loss 本体不可复原）。
4. **λ 单点 + 单 seed 标定**：工作点由 Toys × seed 2025 选出，未做跨数据集 λ 敏感性。
5. **交替的 tokenizer 额外预算**（§19.3 第 5 条）。

### 19.9 成本与排期

单 seed 单数据集 T5 200 ep 实测：**Toys 48.7 / Beauty 65.7 / Sport 80.3 min**。

| 阶段 | 规模 | 估算 |
|---|---|---|
| Stage 0 λ 标定 | 5 路 × Toys | ~4.1 h |
| Stage 0bis | C=1 需 **3 个** 3000-ep CSG tokenizer（λc 扫描每臂一个，见 §19.11.3）；每个 ~9 min | ~27 min |
| Stage 1 B 类 | **8 路** × Toys（7 新格 + 1 个 SIA-only 子臂） | ~6.5 h（+ refresh 开销） |
| **B 类合计** | | **~10.7 h**（单卡约一晚半） |
| Stage 2 C 类（典型 5 格） | 45 路 | ~48 h |
| Stage 2 C 类（最坏 8 格） | 63 路 | ~68 h |

> 交替臂每 cycle 一次 SID refresh（子进程）。200 cycle 的额外开销**按 Stage 1 实测口径回填**，此处不拍数。

### 19.10 开工门禁（缺一不开）

1. Stage 0 的选点已写入 §19.4，且未被事后修改。
2. P1 / P2 落地，且按 **§19.7「门禁口径」** 完成差分回归：改动后 EXP-E9 / EXP-E10 的检查项与**改动前基线**逐条相同（**不得**用历史 90/90、64/64 作条件，**不得**表述为"全部通过"）。✅ **P1 已满足**：E9 90/90 检查行逐字节相同、E10 62/64 行逐字节相同 + 2 行仅时间戳/临时目录差异，**零新增失败**；既有 7 项 guard 失败归属 C5，原样保留。✅ **P2 已实现（待验收）**：纯新增文件、无既有模块 import ⇒ 不改变 E9/E10 读到的任何代码路径，故差分回归沿用 P1 的结论；P2 自身的验证见 §19.7「P2 实现记录」（EXP-P2 47/47）。
3. 新臂的 RQ-VAE 配方与 §19.1 逐项核对通过（**含显式 `--align_weight 0.0`**）。**2026-10-18 更新**：原阻断的 B1（选错 embedding 表）与 B2（无法设 `warmup_epochs=50`）**已修复**（§19.11.6），CLI 现在支持 `--embedding-path` 与 `--warmup-epochs`。**2026-10-19 更新（本次已重做核对）**：λc 侧的 CLI→namespace 配方**逐项通过**（23/25 个共同字段逐值相同，2 项仅为 run dir 与路径写法），但**推荐器侧新发现硬阻塞 B6**：`FixedSidTrainer` 仍硬编码 `data_root/<dataset>/item_emb.parquet`（Toys 上为 4096d），无法加载任何 768d tokenizer ⇒ **本条仍不通过**，详见 §19.11.9。**2026-10-19 晚更新：B6 已修复**（`FixedSidTrainer.embedding_path` / `--embedding_path`，§19.11.10，EXP-B6 31/31）；推荐器侧改由显式 768d 表进入。**同日 B7 已修复**（`apply_seed` + `seeded_kmeans`，§19.11.11，EXP-B7 48/48）：`--seed` 现在真正驱动 python/numpy/torch/cuda 与每次 KMeans 调用 ⇒ 本条**配方与可复现性均通过**。`align_weight` 一项已实测安全：namespace 不带该键 ⇒ `trainer.py:57` 的 `getattr(args,'align_weight',0.0)` 取到 0.0，与 `git show 5edfca0:rqvae/main.py:63` 的默认值一致。
4. C1 / C2 产物在 Toys / Sport / Beauty 上均可加载（C6 已证 Musical_Instruments 不可用；本系列不涉及 MI）。
5. 每格 config 完整落盘（[experiments.md §5.1](experiments.md) 模板），**不得先跑后补**。

### 19.11 P2 后效果实验启动门禁（2026-10-18，**GATE = CONDITIONAL**）

本节的结论来自一次**只读**门禁检查：不跑 EXP-L0 / EXP-B1，不改模型、训练逻辑、损失公式。

#### 19.11.1 差分回归（EXP-E9 / EXP-E10）

按 §19.7「门禁口径」执行：与 **C5 后 / P1 前基线**逐项比对，**不以历史绝对分数为条件**。原始日志落盘于 `integration/artifacts/p2_gate/p2gate_{e9,e10}.log`。

| suite | 本次读数 | 基线读数 | 失败项身份 | 新增失败 |
|---|---|---|---|---|
| EXP-E9 | **87/90** | 87/90 | 3/3 相同 | **0** |
| EXP-E10 | **60/64** | 60/64 | 4/4 相同 | **0** |

失败项（**与 §19.7 记录的 7 项完全一致，归属 C5，不属于 P1/P2**）：

| suite | 失败项 |
|---|---|
| E9 + E10 | `model/ and rqvae/ carry no E9/E10 change -- modified: model/main.py, rqvae/models/rq.py, rqvae/models/vq.py, rqvae/trainer.py` |
| E9 + E10 | `exp_e8_sid_refresh: 54/55` |
| E10 | `exp_e9_alternating: 80/81` |
| E9 | `(12) the E1-E8 suites are unaffected by E9 (329/330)` |
| E10 | `the E1-E9 suites are unaffected by E10 (409/411)` |

**"P2 之后没有代码变化"的独立证据**：`tiger_ext/alt_pipeline.py` 的 `sha256[:16] = fc1b8652a72c04ce`，与 P1 验收记录**逐字符相同**；`model/` + `rqvae/` 四个被 C5 修改的文件 mtime 仍为 **2026-10-14**；`git status` 只列这 4 个 M 与 `integration/`、`tiger_ext/` 两个未跟踪目录。

> ⚠️ **口径限制（必须照抄进汇报）**：磁盘上没有保存 P1 前的基线日志文件，因此本次比对是**按总数 + 失败项身份 + 嵌套 suite 计数（329/330、409/411）+ provenance 哈希**逐项核对，**不是逐字节 diff**。P1 验收那次是逐字节比的（E9 90/90 行、E10 62/64 行）；本次是"同一套失败、同一套计数、同一份代码哈希"。

#### 19.11.2 格 #5 输入链路核对（CSG ckpt → SID → 固定 SID 的 ETE 训练）

**链路设计（已核对，成立）**：

```
qspace_train(λ_csg>0) ─► rqvae ckpt（args.data_path 为绝对路径，§19.11.4 B3）
      │
      ├─► sid_refresh.refresh_sid(dataset, ckpt, base_dir)  ── generate_code.py（cwd=rqvae/）
      │        └─► SID*.npy + sidecar manifest{ckpt_sha256, sid_sha256, sid_stats}
      │
      └─► fixed_train.FixedSidTrainer(sid_path=该 SID, tokenizer_ckpt=该 ckpt, λsia/λpsa>0)
               └─► 训练期冻结 tokenizer + SID，每 epoch 复核 digest / sha256
```

**实测证据（CPU）**：

| 检查 | 结果 |
|---|---|
| `refresh_sid` 从 768d 基线 ckpt（`Toys_Content768d_Baseline/…/best_collision_model.pth`）生成 SID | ✅ `sid_stats`: collision `0.000419`、`max_duplicate_group=1`、`token_max=255`、`per_column_max=[254,255,255,3]`、`ok=True` |
| sidecar manifest 绑定 | ✅ `verify_sid_manifest(manifest, ckpt, sid)` 通过；`ckpt_sha256=8069f998c1442dd1`、`sid_sha256=bb2693a6c71e8ac3` |
| `FixedSidTrainer` 在该 (ckpt, SID) 上跑 1 epoch（λsia=λpsa=0.1） | ✅ `routing` sia/psa 各 **0 mismatch / 11924 items**；`tokenizer_unchanged=True`、`sid_unchanged=True`；两 mount 均构造 |
| 用**另一个** ckpt（CSG smoke）+ 同一 SID 调 `FixedSidTrainer` | 修复前 ⚠️ **未被拒绝**（B4）；**§19.11.8 修复后在构造期抛 `SidProvenanceError`**（EXP-B4 [2]/[9] 复现同一形态） |
| 同一错配交给 `sid_refresh.verify_sid_manifest` | ✅ 抛 `SidManifestMismatch`（`405be3dead63 != 8069f998c144`） |
| CSG smoke 的 SID 被 `validate_sid` 拒绝（1-epoch 退化 tokenizer，collision ≈ 0.9999，duplicate 组 >256） | ✅ 守卫有效（`token out of range: max 311`）——**不是缺陷，是正确拒绝** |

#### 19.11.3 EXP-L0 运行清单（**只澄清既有方案，未改设计**）

**共同固定条件**（§19.1 / §19.3，五臂逐字相同）：

| 项 | 值 | 来源 |
|---|---|---|
| 数据集 / seed | **Toys** × T5 seed **2025**；RQ-VAE seed **2024** | §19.4；`SEED_T5` / `SEED_RQVAE` |
| content 表 | `data/Toys/item_emb_768d_contentonly.parquet`（`in_dim=768`） | §19.1 |
| RQ-VAE 配方（C=1 臂的新 tokenizer） | `lr=1e-3, epochs=3000, batch_size=1024, num_workers=4, eval_step=50, AdamW+linear, warmup_epochs=50, weight_decay=1e-4, dropout 0.0, bn=False, mse, kmeans_init=True, kmeans_iters=100, sk_epsilons=[0,0,0.003], sk_iters=50, num_emb_list=[256,256,256], e_dim=32, quant_loss_weight=1.0, beta=0.25, layers=[512,256,128,64], tau=0.1, lambda_trans=0, lambda_balance=0, align_weight=0.0` | §19.1（实测日志 Namespace） |
| T5 预算 | **200 ep 上限 + early_stop 10**；判据 valid **NDCG@20**；`lr=1e-4, batch=256, infer_size=96, beam=30, topk=5,10,20` | §19.3 第 1 条 |
| A 因子 | 五臂**全部 A=0**（tokenizer 冻结、SID 固定 = v0） | §19.4 表 |

**五臂**：

| # | 臂 | C | E | λ_csg | λ_sia | λ_psa | tokenizer 来源 | SID 来源 | 训练入口 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `λc-1e-4` | 1 | 0 | **1e-4** | 0 | 0 | 新训 3000 ep CSG tokenizer（λ_csg=1e-4） | 该 ckpt 经 `refresh_sid` 生成 | `fixed_train`（λsia=λpsa=0） |
| 2 | `λc-1e-3` | 1 | 0 | **1e-3** | 0 | 0 | 新训 3000 ep CSG tokenizer（λ_csg=1e-3） | 同上（同名 ckpt 的不同产物） | 同上 |
| 3 | `λc-1e-2` | 1 | 0 | **1e-2** | 0 | 0 | 新训 3000 ep CSG tokenizer（λ_csg=1e-2） | 同上 | 同上 |
| 4 | `λe-0.1` | 0 | 1 | 0 | **0.1** | **0.1** | **冻结** 768d 基线 ckpt（不重训） | `data/Toys/Toys_content768d_baseline_sid.npy`（或经 `refresh_sid` 复算，二者 collision 均为 0.000419） | `fixed_train`（λsia=λpsa=0.1） |
| 5 | `λe-1.0` | 0 | 1 | 0 | **1.0** | **1.0** | 同臂 4 | 同臂 4 | `fixed_train`（λsia=λpsa=1.0） |

**⇒ 对 §19.9 的一处澄清（不影响设计，只影响成本行）**：三个 λc 臂**各有自己的 tokenizer 与 SID**（λ_csg 不同 ⇒ 目标函数不同 ⇒ 不可能共用 ckpt），所以 Stage 0bis 是 **3 个 3000-ep CSG tokenizer + 3 次 refresh**，不是 1 个。已把 §19.9 的该行改为 ~27 min。

**选点规则（照抄 §19.4，未改）**：主判据 `ΔNDCG@20 = 臂 − 基线(Toys, seed2025, N@20=0.0341)`；副判据只用于排除（collision ≤ 2× 基线 0.000419、码本不塌缩）；选**不劣化（Δ ≥ −0.001 且不塌缩）的最大 λ**；三档全劣化 ⇒ 退回尺度锚点 `λ_csg=1e-4, λ_sia=λ_psa=0.1` 并如实记录"未找到正向工作点"；单 seed 不做显著性。

#### 19.11.4 阻塞项（**B1/B2/B3/B5 已于 2026-10-18 修复，见 §19.11.6 / §19.11.7；B4 于 2026-10-19 修复，见 §19.11.8**）

| # | 阻塞 | 证据 | 影响 | 最小修复建议 |
|---|---|---|---|---|
| **B1** ✅已修 | `qspace_train` 的 embedding 路径**硬编码** `data_root/{D}/item_emb.parquet` | `tiger_ext/qspace_train.py:470-471`（`CSGTrainerConfig.embedding_path`）；无 `--embedding-path` CLI（`:602-617`）。Toys 的 `item_emb.parquet` 是 **4096d** | λc 三臂会用错表（4096d），直接违反 §19.1 的 768d 冻结 | 加一个字段/CLI：`embedding_file` 或 `embedding_path`，默认保持 `item_emb.parquet` 不变 |
| **B2** ✅已修 | `qspace_train` **无法设 `warmup_epochs`**（默认 `PHASE_WARMUP_EPOCHS=0`，冻结配方要求 **50**） | `:464` 字段存在、`:602-617` 无对应 CLI | λc 三臂的 RQ-VAE 配方与 §19.1 不符 ⇒ 违反 §19.10 门禁第 3 条 | 加 `--warmup-epochs`（默认 0，避免改变既有行为） |
| **B3** ✅已修 | 历史 768d ckpt 的 `args.data_path` 是**仓库相对路径**，而 `generate_code.py` 以 `cwd=rqvae/` 运行 ⇒ `FileNotFoundError` | 实测：`refresh_sid(Toys_Content768d_Baseline/…best_collision_model.pth)` 报 `data/Toys/item_emb_768d_contentonly.parquet` 不存在；用临时 `rqvae/data -> ../data` 符号链接后**立即成功**（已删除该符号链接） | 只影响**复算历史 ckpt 的 SID**；`qspace_train` 产出的 ckpt 走 `os.path.abspath()`（`alt_pipeline.py:159`）故不受影响 | 在 `sid_refresh.run_generator` 里把相对的 `data_path` 按 `REPO` 解析（或给 `generate_code.py` 加 `--data_path` override） |
| **B4** ✅已修 | `FixedSidTrainer` **不校验** SID 的 sidecar manifest 与 `tokenizer_ckpt` 是否同一来源 | 修复前 `grep verify_sid_manifest tiger_ext/fixed_train.py` = 空；错配 ckpt **未抛错** | 格 #5 的"来源一致"曾只靠驱动脚本自律；现在训练器自身在构造期与 `run()` 起点复核 | 构造期调用 `verify_sid_manifest`；无 sidecar 时**默认硬失败**，显式 `allow_unregistered_sid` 才放行并记录（§19.11.8） |
| **B5** ✅已修 | `tiger_ext/qspace_train.py` **没有 `if __name__ == "__main__"` 守卫** | `grep -n "__main__" tiger_ext/*.py` 只有 `fixed_train.py:627` 与 `qspace_smoke.py:693`；实测 `python -m tiger_ext.qspace_train …` **exit 0、零输出**；改用 `qspace_train.main(argv)` 程序调用则正常（本次链路检查即此法） | 文档化的 CLI 入口是**惰性**的；即使修好 B1/B2，命令仍不会跑 | 文件末尾补 `if __name__ == "__main__": raise SystemExit(main())`（与 P2 同款修复） |

**B1/B2 是 Stage 0bis 的硬阻塞**（不修就跑不出符合 §19.1 的 λc 三臂）；**B5 让文档化 CLI 完全惰性**（B1/B2 修好后仍会踩）；**B3 只在复算历史 ckpt 时出现**；**B4 不阻塞实验，但削弱 lineage 保证**。**B1/B2/B5 已于 2026-10-18 修复（§19.11.6），B3 同日修复（§19.11.7），B4 于 2026-10-19 修复（§19.11.8）**。

#### 19.11.5 GATE 判定

- 门禁第 2 条（差分回归）→ ✅ **满足**（零新增失败）。
- 门禁第 3 条（新臂 RQ-VAE 配方逐项核对）→ **2026-10-19 已重做（§19.11.9）**：λc 侧 CLI→namespace 配方**通过**（共同字段逐值相同；warmup=50、epochs=3000、768d 表、batch 1024、eval_step 50、num_workers 4 均可由实际 CLI 产生，并已在 1-epoch smoke ckpt 的 `torch.load['args']` 中核实）。**但推荐器侧发现新的硬阻塞 B6**：`FixedSidTrainer` 硬编码 `item_emb.parquet`（Toys = 4096d），无法加载 768d tokenizer ⇒ **五臂全部无法开跑**；另有 B7（`qspace_train` 不播种 RNG，`--seed` 仅落盘）。
- **B3 已修（§19.11.7）**：历史相对路径 ckpt 现在能再生 SID（EXP-B3 30/30），不再阻塞 λe 两臂。
- **B4 已修（§19.11.8）**：`FixedSidTrainer` 在训练前强制校验 SID manifest ↔ ckpt（EXP-B4 38/38），格 #5 的 lineage 由训练器自身保证；λe 的冻结 SID 已于 2026-10-19 完成逐位复算并登记（§19.11.9）。
- **B6 已修（§19.11.10）**：`FixedSidTrainer` 现支持显式 `embedding_path` / `--embedding_path`；未传时严格保持 `data_root/<dataset>/item_emb.parquet`。构造期读取 checkpoint 的 `encoder.mlp_layers.1.weight` 宽度，与所读表列数不一致即**在建模前**报 `embedding dimension mismatch`（EXP-B6 31/31，含真实 768d Toys ckpt + `item_emb_768d_contentonly.parquet` 在**无 shadow data root**下构造成功）。回归：EXP-P2 47/47、EXP-B4 38/38。
- **B7 已修（§19.11.11）**：`qspace_train` 在数据集/模型/lazy k-means 之前 `apply_seed(config.seed)`（python/numpy/torch CPU/CUDA + cuDNN pin，与 `rqvae/main.py:77-84` 同构）；DataLoader shuffle 用独立 `torch.Generator`；冻结 `layers.kmeans` 的 `KMeans(...)` 由 `seeded_kmeans` **逐次注入显式 `random_state`**（不改 `rqvae/` 文件）。同 seed 跨进程逐位一致（init/final/codes/loss），异 seed 三项均变（EXP-B7 48/48）。
- ⇒ **GATE = CONDITIONAL（维持）**：EXP-L0 仍**暂不启动**。阻断项 = **R3**（驱动）；**B1–B7 均已修复且复核通过**，**R1 已于 §19.11.12 审计关闭**（缺键但 7/7 实际生效值 = 冻结值），**R2 已于 §19.11.13 关闭**（五条命令逐参数冻结，274/274）。

#### 19.11.6 B1/B2/B5 修复（2026-10-18，**待验收**）

**范围**：只改 `tiger_ext/qspace_train.py`（`sha256[:16]` 由 `ea224a1d56e8c5eb` → **`e10f24cf0378a816`**）。`rqvae/`、`model/`、`clean_tiger/`、`tiger_ext/fixed_train.py`、`tiger_ext/sid_refresh.py`、`tiger_ext/alt_pipeline.py` **零改动**（`alt_pipeline.py` 仍是 P1 的 `fc1b8652a72c04ce`）。`tiger_ext/` 未被 git 跟踪 ⇒ `git diff` 为空，改动以代码块形式记录在此。

**1) B1 — 显式 embedding 表**

```python
# CSGTrainerConfig 新增字段（原先是 @property embedding_path，恒返回拼接路径）
#: explicit embedding table. ``None`` keeps the frozen layout
#: (``data_root/<dataset>/item_emb.parquet``) ...
embedding_path: str = None

def resolved_embedding_path(self):
    """The embedding table this run will actually read (never a fallback)."""
    if self.embedding_path is not None:
        return os.path.abspath(self.embedding_path)
    return os.path.abspath(
        os.path.join(self.data_root, self.dataset, "item_emb.parquet"))

def validate_embedding_path(self):
    """Fail loudly on a missing table instead of silently reading another one."""
    path = self.resolved_embedding_path()
    if not os.path.isfile(path):
        source = ("--embedding-path" if self.embedding_path is not None
                  else "the frozen default layout data_root/<dataset>/"
                       "item_emb.parquet")
        raise QSpaceTrainError(
            f"the embedding table is missing: {path} (from {source}); "
            "refusing to fall back to a different embedding file")
    return path
```

`namespace()` / `build_dataset()` 都改为读 `config.resolved_embedding_path()`；`train_tokenizer()` 开头先 `config.validate_embedding_path()`（在任何 dataset 被构造之前）。CLI 新增 `--embedding-path`（默认 `None`）。**没有**"显式路径失效就退回默认表"的分支——显式路径必须是文件，否则报错。

**2) B2 — `--warmup-epochs`**

- `CSGTrainerConfig.warmup_epochs = PHASE_WARMUP_EPOCHS`（**0**，兼容旧调用方）；
- `namespace()` 现在把 `warmup_epochs=self.warmup_epochs` 真正透传给 `alt_pipeline.rqvae_namespace(..., warmup_epochs=)`；
- CLI 新增 `--warmup-epochs`（默认 0）。

**传参链路（逐跳已实测）**：`--warmup-epochs 50` → `CSGTrainerConfig.warmup_epochs` → `namespace()` → `rqvae_namespace(..., warmup_epochs=50)` → `args.warmup_epochs` → `rqvae/trainer.py:33` `self.warmup_steps = args.warmup_epochs * data_num` → `:82` 构造 `get_linear_schedule_with_warmup` → `:382` 每 batch `scheduler.step()`。**可观测证据**：合成数据 8 batch 时 `warmup_steps=400`（=50×8），末步 LR = `2.000e-05` vs warmup=0 的 `0.000e+00`（线性 warmup 阶段 LR = `step/warmup_steps × 1e-3`）⇒ 该值确实进了优化器，不只是日志。

**3) B5 — 真入口**

```python
if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
```

**验证（`integration/exp_p3_qspace_cli.py`，`sha256[:16]=2faddf50c189df48`）：EXP-P3 = 29/29 全通过**，全程 CPU + 合成数据 + Toys 1-epoch smoke，**未启动 EXP-L0**（持久日志：`integration/artifacts/p3_qspace_cli/exp_p3.log`）：

| 段 | 覆盖 | 结果 |
|---|---|---|
| [1] 静态 | AST 检查 `__main__` 守卫；`import tiger_ext.qspace_train` 子进程**不训练**（无输出、不建 ckpt）；parser 默认值逐项 = 修改前（dataset/λ/k/seed/epochs/batch/**warmup 0**/**embedding_path None**）；默认路径公式不变；`--help` 含两个新 flag | 9/9 |
| [2] 显式表 + warmup 50 | `resolved_embedding_path` 绝对化；`namespace.data_path`；**诱饵测试**：同目录放 8d `item_emb.parquet`，实跑读到 16d 自定义表（`in_dim=16`，诱饵未被读）；`trainer.warmup_steps=50×n_batches`；末步 LR 与公式一致；warmup 0 对照；ckpt `args` 记录 `data_path`+`warmup_epochs` | 8/8 |
| [3] 默认布局不变 | 不给 flag 时读 `data_root/<ds>/item_emb.parquet`，report 标 `default` | 2/2 |
| [4] 缺表拒绝 | 显式路径缺失 ⇒ `QSpaceTrainError` 含 "refusing to fall back"，且**在同目录存在诱饵表的情况下** dataset factory 调用次数 = 0（先校验后构造）；目录不算表 | 3/3 |
| [5] CLI 真跑 | `python -m tiger_ext.qspace_train` rc=0 且 stdout 摘要含 `warmup_epochs 50` / `embedding_path_source explicit`；ckpt 记录 768d 表 + warmup 50；无 flag 的 CLI 记录旧默认表 + warmup 0；显式缺表 / 默认缺表 两种 CLI 均 rc=1 | 7/7 |

**已知问题（均已登记）**：
- **B3 已于 2026-10-18 修复**（见 §19.11.7）。
- **B4 已修**（2026-10-19，见 §19.11.8）：`fixed_train.py` 现在构造期与 `run()` 起点都校验 SID manifest ↔ ckpt 同源。
- **`sid_refresh.ckpt_args` 会丢掉 `warmup_epochs`**：它按 `_CKPT_ARG_KEYS` 白名单过滤（不含 warmup），所以从 `.pth` 反读训练配置的既有读者看不到这个键。**ckpt 里本身是有的**（本次验证改用未过滤的 `torch.load(...)['args']`）。若有下游要审计 warmup，需扩白名单（**本次未改** `sid_refresh.py`）。
- `kmeans_init=True` + 小样本仍是老坑：合成 smoke 需要 `limit≥8192 / batch=1024`，否则首 batch distinct ≤ n_emb ⇒ `x_q==z` ⇒ C3 `QSpaceMismatch`。

#### 19.11.7 B3 修复：历史 checkpoint 的相对 `data_path`（2026-10-18，**待验收**）

**问题复现**：`generate_code.py:85` 用 `EmbDataset(args.data_path)` 原样读取，而 `run_generator` 以 `cwd=rqvae/` 启动 ⇒ 存 `data/Toys/fused_item_emb_768d.parquet` 的历史 ckpt 会去找 `rqvae/data/Toys/...`，实测 `FileNotFoundError`（rc=1）。

**语料实测**（`rqvae/ckpt/**/*.pth`，708 个）：**627 个**存 `data/...`（仓库根相对）、**61 个**存 `../data/...`（`rqvae/` 相对，即 `rqvae/main.py:27` 的默认值）、**20 个**已是绝对路径。⇒ 只按"仓库根相对"处理会漏掉 61 个。

**改动范围**：只改 `tiger_ext/sid_refresh.py`（`sha256[:16]` `6c54fab7c4ce6fa9` → **`983121cb3d9cd17b`**）。**未改** `rqvae/`（`generate_code.py` 属 tracked 文件，动它会新增 diff-regression 的 tracked-file guard 失败）、**未加符号链接**、**不依赖调用方 cwd**。

**新增/修改的函数**：

```python
def resolve_data_path(data_path, repo=REPO):
    # 绝对路径 → normpath 原样返回（新 ckpt 语义不变）
    # 相对路径 → 只试两个固定根：repo/<p> 与 repo/rqvae/<p>（后者对应 ../data/...）
    #   恰好一个存在 → 用它；两个都存在且不是同一文件 → 报 ambiguous
    #   两个都不存在 → 报 "refusing to guess"（绝不回退到别的表）

def ckpt_data_path(ckpt_path):          # -> (stored, resolved)
def ckpt_with_absolute_data_path(ckpt_path, workdir):
    # 只改 args.data_path 的一份临时副本（state_dict 逐位不变）；
    # 已是绝对路径时原文件直传，不复制
def run_generator(...):
    # 传入生成器的 ckpt 的 args.data_path 必为绝对；返回额外记录
    # ckpt_path/ckpt_for_generator/data_path/data_path_stored/data_path_rewritten
def n_items_of(...):                    # 改用同一个 resolver（此前依赖调用方 cwd）
```

`refresh_sid` 的 manifest 新增 `ckpt_data_path_stored`（ckpt 里原样存的值）、`generator_ckpt`（逻辑 ckpt 绝对路径）、`data_path_rewritten`；`data_path` 现在是**实际读取**的绝对表。另加一条守卫：若生成器要读的表与 `n_items_of` 统计的表不同，直接抛错，不登记混合产物。

**为什么用临时副本而不是新增 CLI**：`generate_code.py` 没有 data-path override，而它位于 `rqvae/`（tracked、冻结、且是 diff-regression 的 guard 对象）。临时副本只替换 `args.data_path` 一个字段，权重逐位相同 —— 实测 `state_dict` 23/23 张量 `torch.equal`，且**相对路径 ckpt 与其绝对孪生 ckpt 生成的 SID 逐位相同**。

**验证（`integration/exp_b3_data_path.py`，`sha256[:16]=44a7234aa5164789`）：EXP-B3 = 30/30**（日志 `integration/artifacts/b3_data_path/exp_b3.log`；CPU，未启动 EXP-L0/EXP-B1）：

| 段 | 覆盖 | 结果 |
|---|---|---|
| [1] resolver | `data/...`→仓库根、`../data/...`→`rqvae/`、绝对路径原样；缺失/空/歧义均抛 `SidRefreshError`（含 "refusing to guess"/"ambiguous"）；**从外部 cwd 解析结果不变**，`n_items_of` 亦然 | 8/8 |
| [2] 全语料 | 708 个 ckpt 逐个解析：绝对 20/20、`data/` 627/627、`../data/` 50/61 解析成功；**11 个拒绝**且全部是 `../data/Sports/item_emb.parquet`（数据集目录已改名 `Sport`，表已不存在）⇒ **响亮报错而非猜另一张表** | 4/4 |
| [3] 有牙 | 用**未解析**的原始 ckpt 直调冻结生成器：rc=1、`FileNotFoundError: 'data/Toys/fused_item_emb_768d.parquet'`（即从 `cwd=rqvae/` 解析） | 3/3 |
| [4] 历史 ckpt 端到端 | `refresh_sid(Toys_TA_SID_seed2025)` 无符号链接成功：11924 行、原始 collision 0.000587、2.5 s；manifest 记录 resolved/stored/rewritten 全部正确 | 2/2 |
| [5] 绝对 ckpt | 副本 `args.data_path` 绝对化后 **23/23 张量逐位相同**；直传不复制（`data_path_rewritten=False`）；**两形态 SID 逐位相同** | 4/4 |
| [6] 新 ckpt | `qspace_train` 现训 ckpt（绝对 data_path）：从**外部 cwd** 调 `run_generator` 成功、不复制、SID 覆盖 11924 行 | 5/5 |
| [7] 卫生 | 未创建 `rqvae/data` 符号链接；E8 的绝对 fixture 解析结果与记录值完全一致 | 3/3 |

**回归**：
- **EXP-E8 = 54/55**（CPU），唯一失败项仍是 tracked-file guard（列出 C5 授权改的 `rqvae/trainer.py`），**与基线一致，零新增失败**（`integration/artifacts/b3_data_path/regression_e8.log`）。
- **EXP-P3 = 29/29**（`regression_p3.log`）⇒ 上一轮的 B1/B2/B5 修复未被影响。
- E9/E10 未重跑：本次只改 `tiger_ext/sid_refresh.py`（未跟踪文件），不触碰 E9/E10 的 tracked-file guard 判据；E9 走的是**新 ckpt 绝对路径**分支，语义与 [5][6] 已覆盖。

**已知限制**：① 11 个 `../data/Sports/...` 历史 ckpt **无法再生 SID**（原表已随数据集改名消失），现在会在解析阶段响亮报错——这是有意的，不做跨表猜测；② 仍不改 `rqvae/`，因此每个需要重写的 run 会写一份临时 ckpt 副本（几 MB，`TemporaryDirectory` 自动清理）；`generator_cmd` 里记录的是该临时路径，`generator_ckpt` 才是逻辑上的原始 ckpt；③ B4（manifest ↔ ckpt 同源自检）已由 §19.11.8 同日修复，不属本节的遗留问题。

#### 19.11.8 B4 修复：SID 与 tokenizer checkpoint 的同源校验（2026-10-19，**待验收**）

**问题**：`FixedSidTrainer` 只记录 `tokenizer_ckpt_sha256` 与 `sid_sha256`，从不与 SID 的注册 manifest 比对。实测「另一个 ckpt + 原 SID」**未抛错**（见 §19.11.2 表），而同一错配交给 `sid_refresh.verify_sid_manifest` 会抛 `SidManifestMismatch` ⇒ 训练器自身不设防，格 #5 的"来源一致"只靠驱动脚本自律。

**范围**：只改 `tiger_ext/fixed_train.py`（`sha256[:16]` `b3ec855be760bafc` → **`8fde074584fe443f`**，628 → 808 行）。`model/`、`rqvae/`、`clean_tiger/`、`tiger_ext/sid_refresh.py`（仍 `983121cb3d9cd17b`）、`alt_pipeline.py`（仍 `fc1b8652a72c04ce`）**零改动**。`tiger_ext/` 未跟踪 ⇒ `git diff` 为空；`git status --porcelain` 与开工前逐字相同。对应测试文件：`integration/exp_p2_fixed_train.py`（fixture 现在写真实 manifest，`sha256[:16]=32a611fa8c78d6ba`）、新增 `integration/exp_b4_provenance.py`（`1e02a89004c508e4`）。

**1) 复用而非另造**

```python
def verify_sid_provenance(sid_path, ckpt_path, manifest_path=None,
                          allow_unregistered=False, dataset=None):
    side = resolve_sid_manifest(sid_path, manifest_path)   # 默认 <sid>.sid.json
    ...                                                    # 无 manifest 的分支见下
    record = json.load(open(side))                         # 显式 manifest 可任意文件名
    verify_sid_manifest(record, ckpt_path=ckpt_path, sid_path_=sid_path)
    # SidManifestMismatch -> SidProvenanceError（同时是 FixedSidError 子类）
```

哈希判据**完全**来自 `sid_refresh.verify_sid_manifest`：`ckpt_sha256` 与 `sid_sha256` 任一不符即拒绝。训练器不复制比较逻辑。

**2) 校验时机：开始训练之前**

- `FixedSidTrainer.__init__`：在文件存在性检查之后、`load_sid` / 模型 / dataset 构造**之前**调用一次；
- `run()` 起点：再调用一次，覆盖"构造后被换文件"的 TOCTOU（EXP-B4 [7] 实测在 epoch 0 前抛错，无 `last.pth` / `run.json`）。

**3) 缺失 manifest 的策略（不静默跳过）**

| 场景 | 默认行为 | 最小兼容方案 |
|---|---|---|
| 有 sidecar | 校验哈希 | — |
| 显式 `sid_manifest=` / `--sid_manifest` | 按该文件校验 | 用于 `publish_sid` 只复制 `.npy` 的场景 |
| 无 manifest（默认） | **`SidProvenanceError` 硬失败** | — |
| 无 manifest + `allow_unregistered_sid=True` | 放行，`run.json` 记 `mode="unregistered"` + warning | 仅限 legacy SID |
| 无 manifest 且文件名是 versioned 形式 | 仍校验 `{ds}_sid_v{ver}_{ckpt_sha8}.npy` 的 `ckpt_sha8` | — |

**登记在案的合法无 manifest 场景**：① **早于 `sid_refresh` 的 legacy SID**——R13 `λe-0.1 / λe-1.0` 两臂按 §19.11.3 读 `data/Toys/Toys_content768d_baseline_sid.npy`，该文件没有 sidecar；建议先用 `refresh_sid` 从同一 ckpt 复算（B3 已使历史 ckpt 可行；§19.11.3 记载二者 collision 同为 0.000419），否则必须显式 `--allow_unregistered_sid`；② **`publish_sid` 的目标路径**只得到 `.npy`（E8 [7] 明确 publish 是 opt-in 且只复制文件）⇒ 用 `--sid_manifest` 指向 versioned 目录里的 sidecar；③ **同一 SID 换数据集名复用**（EXP-P2 的 `P2SynSmall` 在相同 item 表上重放）⇒ manifest 的 `dataset` 只记录为 `dataset_matches`，硬判据仍是哈希。

**4) 记录**：`run_config()` / `run.json` / 阶段 manifest 新增 `sid_manifest`、`sid_provenance_mode`、`allow_unregistered_sid`、完整 `sid_provenance`（含 `mode`、`verified`、`manifest`、`ckpt_sha256`、`sid_sha256`、`dataset_matches`、`n_items`、`data_path`）。CLI 新增 `--sid_manifest`、`--allow_unregistered_sid`。

**验证（EXP-B4 = 38/38，日志 `integration/artifacts/b4_provenance/exp_b4.log`；CPU；未启动 EXP-L0/EXP-B1）**：

| 段 | 覆盖 | 结果 |
|---|---|---|
| [1] 正确三元组 | `refresh_sid` 真 manifest；λ=0 真跑 1 epoch；run.json 带 provenance | 6/6 |
| [2] 错 ckpt | 第二个真实 tokenizer；构造期 `SidProvenanceError`（"does not belong"），无 run 目录；底层确为 `SidManifestMismatch` | 5/5 |
| [3] 错 SID | 改一个 token 后仍配原 manifest ⇒ 拒绝；原 SID 仍通过 | 3/3 |
| [4] 缺 manifest | 默认硬失败；opt-out 记录 `unregistered`；versioned 名 `ckpt_sha8` 正/误两向 | 6/6 |
| [5] 两条训练路径 | λ=0（无 mount）与格 #5（λsia=λpsa=0.1，两 mount 均活）都跑通且 SID/tokenizer 不变；`csg=` 仍拒绝 | 3/3 |
| [6] 显式 manifest | publish 形态无 sidecar ⇒ 默认拒绝、`--sid_manifest` 接受；缺失显式文件报错；dataset 名不一致仅记录 | 6/6 |
| [7] TOCTOU | 构造后换 SID ⇒ `run()` 在 epoch 0 前拒绝，无产物 | 2/2 |
| [8] CLI | 正常 rc=0；缺 manifest rc=1；`--allow_unregistered_sid` rc=0 且 `mode=unregistered` | 3/3 |
| [9] 真实输入 | 历史 768d ckpt + 真 `refresh_sid` manifest（11924 行）接受；另一真实 tokenizer 拒绝 | 3/3 |

**回归**：
- **EXP-P2 = 47/47**（`integration/artifacts/b4_provenance/regression_p2.log`）：fixture 改为写真实 manifest 后，冻结契约、λ=0 逐位等价（50/50 张量）、格 #5、CLI 全部保持；
- **EXP-E8 = 54/55**（`regression_e8.log`）：唯一失败仍是 tracked-file guard（列出 4 个既有修改文件），与基线一致，**零新增失败**；
- **EXP-B3 = 30/30**（`regression_b3.log`）：`sid_refresh.py` 未被本次改动影响；
- E9/E10 未重跑：本次只改未跟踪的 `fixed_train.py`，不触碰其 guard 判据，且 E9/E10 不导入该模块。

**已知限制**：① `unregistered` 模式**不做内容级复算**（不重跑生成器核验 SID 是否真由该 ckpt 产生），lineage 依赖调用方声明——这是显式 opt-out 的代价，已写入 `run.json` warning；② manifest 的 `dataset` 字段不参与硬判据（同一 item 空间的合法复用），仅记录 `dataset_matches`；③ 训练器读取的 `data_root/<ds>/item_emb.parquet` 与 manifest 记录的训练表之间仍只由行数/维度间接约束（`validate_sid`），embedding 表级的哈希比对不在 B4 范围；④ 全部验证在 CPU 上完成；⑤ **不是效果证据**：只跑 1 个 epoch 的机制 smoke，不产生任何推荐指标结论。

#### 19.11.9 修后配方逐项核对（2026-10-19，**GATE = CONDITIONAL，B6 硬阻塞**）

**任务边界**：只做核对。**未启动 EXP-L0 / EXP-B1，未修改任何生产文件**（`tiger_ext/*.py` 哈希与 §19.11.6–§19.11.8 记录逐字符相同），未改 R13 设计。唯一新增的数据侧文件是冻结 SID 的 sidecar manifest（见下 §19.11.9.2），`.npy` 字节未变。

原始产物：`integration/artifacts/r13_gate/{regen_sid.log, fixed_train_construct.log, lc_smoke.log, lc_smoke_ckpt.log, shadow_construct.log, register_frozen_sid.log, default_discovery.log, fixed_train_defaults.log}`。

##### 19.11.9.1 逐项核对表（冻结要求 / 实际配置 / 证据 / 结论）

**A. CSG 三组（λc-1e-4 / 1e-3 / 1e-2）**

| 冻结要求 | 实际配置 | 证据 | 结论 |
|---|---|---|---|
| `lambda_csg ∈ {1e-4,1e-3,1e-2}` | CLI `--lambda-csg` 原样透传；默认 0 必须显式覆盖 | `qspace_train.py:632`、`CSGTrainerConfig.lambda_csg` | ✅ |
| 768d content-only 表 | `--embedding-path data/Toys/item_emb_768d_contentonly.parquet` ⇒ namespace `data_path` 为该文件的**绝对路径**；实际 parquet = 11924 行 × 768d（2 列） | `qspace_train.py:645`；`pyarrow` 实测 `dim=768`；1-epoch smoke ckpt 的 `args.data_path` 为绝对 768d | ✅ |
| `warmup_epochs=50` | 实际 CLI `--warmup-epochs 50` ⇒ `args.warmup_epochs=50` | **`torch.load(ckpt)['args']`（未过滤）**：smoke ckpt = 50；§19.11.6 已证 `warmup_steps=50×n_batches` 且末步 LR=`2.000e-05` | ✅ |
| 训练 3000 epochs | 正式 CLI `--epochs 3000` ⇒ namespace `epochs=3000`（smoke 用 `--epochs 1`，仅机制） | 解析后的 namespace = 3000；smoke ckpt 因显式 1 epoch 记为 1 | ✅（可产生，未跑正式 3000 ep） |
| 其余 RQ-VAE 配方逐字继承 §19.1 | 29 个基线 ckpt 字段中：**23 个逐值相同**；`ckpt_dir` 为 run-specific；`data_path` 仅绝对/相对写法不同（`os.path.samefile` 同一文件，B3 已证）；`tau=0.1 / lambda_trans=0 / lambda_balance=0 / trans_graph_path=None` 在 namespace 中**缺键**，由 `trainer.py:86-88` 的 `getattr` 默认取到冻结值；`align_weight` 同样缺键⇒`getattr` 取 0.0（§19.10 第 3 条已记载） | `fixed_train_defaults.log` / `lc_smoke_ckpt.log`；`rqvae/trainer.py:57,86-88`；**R1 已用真实 Trainer 逐项复核（§19.11.12）** | ✅（有效值一致；R1 已审计关闭） |
| 每组自己的 tokenizer ckpt | 三个 λ 各自 `--ckpt-dir`，互不复用 | §19.11.3 表；每条命令需显式 `--ckpt-dir` | ✅（命令层面） |
| 从**该 ckpt** refresh SID 并校验同源 | smoke：`refresh_sid(Toys, smoke_ckpt, ...)` ⇒ SID 11924 行、collision 0.00763、`max_duplicate_group=1`、`ok=True`；`verify_sid_manifest` = **True** | `lc_smoke_ckpt.log` | ✅（1-epoch smoke；正式 3000 ep 同一路径） |
| 默认值陷阱 | `--epochs/--batch-size/--eval-step/--warmup-epochs/--embedding-path/--num-workers/--device/--ckpt-dir` 的默认值**全部不等于** EXP-L0 要求（3/256/1/0/4096d/1/cpu/c5_sanity）⇒ 必须逐项显式传 | `qspace_train.py:628-651` | ⚠️ 操作项（已写入 §19.11.9.4 命令） |

**B. ETE 两组（λe-0.1 / λe-1.0）**

| 冻结要求 | 实际配置 | 证据 | 结论 |
|---|---|---|---|
| `lambda_sia=lambda_psa ∈ {0.1,1.0}` | CLI `--lambda_sia/--lambda_psa`，默认 0 必须显式传 | `fixed_train.py:748-749`；运行时默认抓取 = 0.0 | ✅ |
| C=0、A=0；冻结 768d baseline ckpt | `--tokenizer_ckpt .../Toys_Content768d_Baseline/.../best_collision_model.pth`；`StageFreeze.apply("recommender")` 冻结 23/23 tokenizer 张量 | 诊断构造（§19.11.9.3）：`freeze.groups.tokenizer.all_frozen=True` | ✅ |
| 输入 SID = `Toys_content768d_baseline_sid.npy` | 用**同一路径**；从同一 ckpt `refresh_sid` 复算 ⇒ **逐位一致** | `regen_sid.log`：`sid_sha256=bb2693a6c71e8ac3…` 前后相同；`rows_changed=0`、`per_column_rate=[0,0,0,0]`、`exact_match=True`；`sid_stats` 逐字段相同（collision 0.000419） | ✅ |
| 逐位一致 ⇒ 正式 manifest 登记并使用同一 SID | 已在 `data/Toys/Toys_content768d_baseline_sid.npy.sid.json` 写入 sidecar（`ckpt_sha256=8069f998…`、`sid_sha256=bb2693a6…`、`dataset=Toys`、`n_items=11924`，并记录复算来源与 `exact_match`）；`.npy` 哈希写入前后不变 | `register_frozen_sid.log`；`verify_sid_manifest`=True；`FixedSidTrainer` 默认发现 sidecar 后通过 B4 | ✅ |
| 不一致则停止 | **未触发**（逐位一致） | 同上 | ✅ |
| 冻结 SID 缺 manifest 时不得静默放行 | 登记前实测：无 sidecar ⇒ `SidProvenanceError`（默认硬失败），**未**使用 `--allow_unregistered_sid` | `fixed_train_construct.log` (a) | ✅ |

**C. 共同 T5 配置**

| 冻结要求 | 实际配置 | 证据 | 结论 |
|---|---|---|---|
| Toys / seed 2025 | 运行时 CLI 默认 `dataset='Toys'`、`seed=2025`；`__init__` 调 `tiger_main.set_seed(seed)` | `fixed_train_defaults.log`；`fixed_train.py:370` | ✅ |
| 200 ep + early_stop 10 | 默认 200 / 10 | 同上 | ✅ |
| lr=1e-4, batch=256 | 默认 1e-4 / 256 | 同上 | ✅ |
| infer 96, beam 30, topk 5,10,20 | 默认 96 / 30；`topk_list` **不经 CLI**，固定 `T5_TOPK=[5,10,20]` | `fixed_train.py:754-755`、`alt_pipeline.py` `T5_TOPK` | ✅ |
| 评估协议 = `model/main.py` | 每 epoch `train` → valid `evaluate` → valid NDCG@20 改进时 test `evaluate` + 存 best → `early_stop=10` | `fixed_train.py:600-646` 与 `model/main.py:307-331` 逐行对应；EXP-P2 [5] 已证 λ=0 与手工重放 **50/50 张量逐位相同** | ✅ |
| checkpoint 输出 | `run_dir/best.pth`、`last.pth`（纯 T5 `state_dict`）、`{best,last}.{sia,psa}_adapter.pth`、`run.json`、best 的 `.stage.json` | `fixed_train.py:582-583,632-654` | ✅ |
| 每组 SID 来源记录 | `run_config()` 记录 `sid_path/sid_sha256/tokenizer_ckpt/tokenizer_ckpt_sha256/sid_manifest/sid_provenance_mode/完整 sid_provenance` | `fixed_train.py:523-554` | ✅ |
| 结果记录一致 | `run.json = {config, result}`，含 `best_epoch/best_ndcg@20/best_ckpt/last_ckpt/n_test_evals/frozen per epoch` | `fixed_train.py:660-716` | ✅ |
| RQ-VAE 配置严格对应 §19.1 | `rqvae_config` 覆盖默认 `RQVAE_CONFIG`（256/32/layers…），与基线 ckpt 实测一致；λc 新 ckpt 由同一 namespace 产生 | `fixed_train_defaults.log`；`lc_smoke_ckpt.log` | ✅ |

##### 19.11.9.2 SID 复算与登记（λe 分支，已解决，不是阻塞）

- 原生成脚本 `rqvae/generate_code_Toys_contentbaseline.py:15` 使用 **`cuda:0`** ⇒ 复算也在 `cuda:0` 上进行（SID 与 device 相关，CPU/CUDA 有已知偏差，不能拿 CPU 结果做逐位结论）。
- 复算产物：`integration/artifacts/r13_gate/sid_regen/sid_versions/Toys_sid_v0001_8069f998.npy` + `.sid.json`；与冻结文件 **`sha256` 完全相同**。
- 登记方式：在冻结 SID 旁写 sidecar（默认发现约定 `<sid>.sid.json`），记录 `ckpt_sha256`、`sid_sha256`、`dataset`、`n_items`、复算来源 manifest、`comparison.exact_match=true`；**没有启用** `--allow_unregistered_sid`。
- 结论：**λe 两臂的 SID 来源校验不阻塞**，任务书要求的"若不一致立即停止"分支未触发。

##### 19.11.9.3 新发现的硬阻塞：B6（推荐器侧 embedding 表）

**问题**：`FixedSidTrainer.__init__` 仍写死

```python
self.embedding_path = os.path.join(self.data_dir, "item_emb.parquet")
```

而 Toys 的 `data/Toys/item_emb.parquet` 是 **4096d**（`item_emb_768d_contentonly.parquet` 才是 768d）。实测用真实基线 ckpt + 已登记的冻结 SID 构造训练器：

```
FixedSidError: the tokenizer checkpoint does not fit the RQ-VAE built from rqvae_config ...
size mismatch for encoder.mlp_layers.1.weight: checkpoint [512, 768] vs current model [512, 4096]
size mismatch for decoder.mlp_layers.13.weight/bias: [768,512]/[768] vs [4096,512]/[4096]
```

⇒ **五臂全部无法开跑**（λc 的 tokenizer 也是 768d；λe 的基线 ckpt 也是 768d）。这不是只有 ETE 分支的问题，而是 `FixedSidTrainer` 的通用输入路径缺陷（与已修的 B1 同型，但位于推荐器入口、此前未被覆盖）。

**归因证据**（把唯一变量隔离）：用诊断性 shadow data root（`integration/artifacts/r13_gate/shadow_data/Toys/`，其中 `item_emb.parquet` 是 768d 表的**符号链接**，train/valid/test 也链接到原文件）构造同一训练器 ⇒ **成功**：

```
in_dim=768, n_items=11924
provenance: mode=manifest, verified=True, dataset_matches=True
mounts: sia=True, psa=True
freeze: tokenizer 23/23 frozen; recommender 89 trainable; adapter 4 trainable
optimizer params: 4,694,048
```

⇒ 除 embedding 表路径外，B4 provenance、ckpt 加载、mounts、冻结契约、优化器作用域均正常。shadow root **仅用于诊断，不是批准的正式运行方式**（它会改变 `run.json` 的 `data_root` 记录）。

**最小修正建议（未实施，需批准）**：给 `FixedSidTrainer.__init__` 增加 `embedding_path=None`（默认保持 `data_root/<dataset>/item_emb.parquet`，不改变既有行为），CLI 增加 `--embedding_path`，与 B1 的 `resolved_embedding_path/validate_embedding_path` 同构；正式命令显式传 `data/Toys/item_emb_768d_contentonly.parquet`。这是**生产代码改动**，本次核对任务不越权实施。

##### 19.11.9.4 其他未决项（不改变冻结设计，但启动前必须处理）

| # | 项 | 实测 | 影响 | 最小处理 |
|---|---|---|---|---|
| ~~**B7**~~ ✅ **已修复（§19.11.11）** | `qspace_train` 曾**不播种 RNG** | 修复前 `grep manual_seed/np.random.seed/random.seed/setup_seed tiger_ext/qspace_train.py` = 空；`--seed 2024` 只写进 `args`，`RQVAETrainer.__init__` 不读 seed | 已由 `apply_seed` + `seeded_kmeans` 关闭：同 seed 跨进程逐位一致，异 seed 初始化/离散码均变（EXP-B7 48/48） | 无（已实施；R1 已由 §19.11.12 关闭，R2 已由 §19.11.13 关闭，留 R3） |
| ~~**R1**~~ ✅ **已审计关闭（§19.11.12）** | 新 ckpt 的 `args` 缺 `tau / lambda_trans / lambda_balance / trans_graph_path`（以及 `align_*`） | 实测缺 7 键（不是 4 键；`align_*` 两组 ckpt 都缺）；用**真实 `RQVAETrainer`** 从 ckpt args 构造后逐项取实际生效值 = 0.1/0/0/None/0/0.1/None，**21/21 等于冻结值** | 仅**事后可读性**缺口：`sid_refresh.ckpt_args` 白名单不含这 7 项（也不含 `lambda_csg`），但无任何消费者依赖它们 | **不改生产代码**（任务要求）；审计表落 `integration/artifacts/r1_ckpt_args/exp_r1_audit.json`，EXP-R1 44/44 |
| ~~**R2**~~ ✅ **已关闭（§19.11.13）** | 两套 CLI 的默认值全部不等于 EXP-L0 要求 | qspace_train：epochs 3 / batch 256 / eval 1 / warmup 0 / 4096d / workers 1 / cpu / `ckpt/c5_sanity`；fixed_train：λ 0 | 漏传任何一项都会静默偏离 §19.1 | 五条命令已逐参数冻结（EXP-R2 274/274，`CLOSED-BY-COMMAND-FREEZE`），见 §19.11.13；**不要**依赖默认值 |
| **R3** | 尚无 EXP-L0 驱动脚本 | `grep EXP_L0` 只命中文档 | 五臂的命令/输出目录靠手工拼装，易错 | 启动前落一份 driver（可复用 `rqvae/*.sh` 风格），把 **§19.11.13** 的五条冻结命令固定下来（R2 已给出唯一命令文本） |

##### 19.11.9.5 建议启动命令（B6 修复后；`{}` 为三选一）

λc 三臂（每臂独立 ckpt 目录）：

```bash
python -m tiger_ext.qspace_train --dataset Toys --lambda-csg {1e-4|1e-3|1e-2} \
  --epochs 3000 --batch-size 1024 --eval-step 50 --num-workers 4 --warmup-epochs 50 \
  --embedding-path data/Toys/item_emb_768d_contentonly.parquet \
  --device cuda:0 --seed 2024 --ckpt-dir rqvae/ckpt/EXP_L0_lc{...}
# 然后用该 run 的 best_collision_model.pth refresh SID（base_dir 指向实验产物目录）
python -c "from tiger_ext import sid_refresh as SR; SR.refresh_sid('Toys', '<run>/best_collision_model.pth', '<artifact>/EXP_L0_lc{...}', device='cuda:0')"
```

推荐器五臂（λc 用自己 refresh 出的 versioned SID；λe 用已登记的冻结 SID）：

```bash
python -m tiger_ext.fixed_train --dataset Toys \
  --sid_path <λc: versioned SID | λe: data/Toys/Toys_content768d_baseline_sid.npy> \
  --tokenizer_ckpt <λc: <run>/best_collision_model.pth | λe: rqvae/ckpt/Toys_Content768d_Baseline/Jul-31-2026_21-37-33/best_collision_model.pth> \
  --arm "<EXP-L0 λc-... | λe-0.1 | λe-1.0>" \
  --lambda_sia {0|0.1|1.0} --lambda_psa {0|0.1|1.0} \
  --seed 2025 --num_epochs 200 --early_stop 10 --batch_size 256 \
  --infer_size 96 --beam_size 30 --lr 1e-4 --device cuda:0 \
  --run_dir <artifact>/EXP_L0_<arm>/run
```

（B6 修复后追加 `--embedding_path data/Toys/item_emb_768d_contentonly.parquet`。λc 五臂的 `lambda_sia=lambda_psa=0`；λe 两臂分别为 0.1/0.1 与 1.0/1.0。）

##### 19.11.9.6 判定

- 门禁第 1 条（选点）→ 未涉及（本次不跑 Stage 0）。
- 门禁第 2 条（差分回归）→ ✅ 沿用 §19.11.1（零新增失败）。
- 门禁第 3 条（修后配方逐项核对）→ **不通过**：λc 侧 ✅，但推荐器侧 **B6 硬阻塞**；另有 B7/R1/R2/R3 待处理。（B6 已修 §19.11.10、B7 已修 §19.11.11；R1 已于 §19.11.12 审计关闭；R2 已于 §19.11.13 冻结关闭。）
- **GATE = CONDITIONAL**：**不得启动 EXP-L0**。待办顺序：①批准并修复 B6（推荐器 `embedding_path`）；②裁决 B7（播种或明确记为名义值）；③处理 R1/R2/R3（记录/命令/驱动）；④重跑本节核对表中 B6 相关行；⑤再判 GATE。（① ② 已完成；R1 已于 §19.11.12 审计关闭；R2 已于 §19.11.13 冻结关闭；剩 R3。）

> **证据性质声明**：本节所有运行均为 **1-epoch / 构造级机制 smoke**（`lc_smoke`、`shadow_construct`），**不是效果证据**，不产生任何推荐指标结论，也不能用于 Stage 0 选点。

#### 19.11.10 B6 修复：`FixedSidTrainer` 支持显式 embedding 表（2026-10-19）

**任务边界**：只改 `tiger_ext/fixed_train.py`（`sha256[:16]` 由 `8fde074584fe443f` → **`389e90854b5f6f13`**，808 → 924 行）并新增 harness `integration/exp_b6_embedding_path.py`。`model/`、`rqvae/`、`clean_tiger/`、`tiger_ext/sid_refresh.py`、`tiger_ext/qspace_train.py` **零改动**；未启动 EXP-L0 / EXP-B1；未处理 B7。`tiger_ext/` 未被 git 跟踪 ⇒ `git diff` 为空。

**接口**

```python
FixedSidTrainer(..., embedding_path=None)   # CLI: --embedding_path
```

- `embedding_path=None` ⇒ 与修复前**完全一致**：`data_root/<dataset>/item_emb.parquet`；
- 显式路径 ⇒ **按给定文件读取**（`os.path.abspath`），不回退、不与默认表合并；先 `isfile` 检查，缺表即 `FixedSidError`。

**构造期维度门禁**（新增 `tokenizer_input_dim(ckpt)`）：从 checkpoint `state_dict["encoder.mlp_layers.1.weight"]`（`MLPLayers` 第一个 `nn.Linear`，形状 `(layers[0], in_dim)`）读出 `in_dim`，与所读表的列数比较；不一致即抛

```
embedding dimension mismatch: <table> (<explicit --embedding_path | default data_root/<dataset>/item_emb.parquet>)
has <D> columns, but <ckpt> was built with in_dim=<E>
(encoder.mlp_layers.1.weight (512, E))[; pass embedding_path=...]
```

该检查发生在 `RQVAE(...)` 构造**之前**，因此不会退化成 `load_state_dict` 的裸 `size mismatch`。checkpoint 现在只 `torch.load` 一次（缓存后供 `_load_tokenizer` 使用），语义不变。

**记录**（`run.json` 的 `config` 与 `result`、以及 `best.pth.stage.json`）：`embedding_path`、`embedding_path_explicit`、`embedding_sha256`、`embedding_dim`、`embedding_n_items`、`tokenizer_input_dim`、`tokenizer_input_shape`；`result` 另有 `embedding_sha256_after` / `embedding_unchanged`，且 `_check_frozen` 每 epoch 复核该表未被改写（与 SID 同款）。

**B4 未放松**：provenance 校验仍在 embedding 读取之前、构造期与 `run()` 起点各一次，判据仍是 `verify_sid_manifest` 的 ckpt/sid 哈希；`allow_unregistered_sid` 语义未变。

**测试 `integration/exp_b6_embedding_path.py` = 31/31**（CPU，合成 32d fixture + 真实 Toys 768d）：

| 组 | 覆盖 |
|---|---|
| [1] 默认兼容 | 无参数 ⇒ 路径/张量/`run.json` 与历史一致，`embedding_path_explicit=False` |
| [2] 显式生效 | 读到的张量 == 显式文件（≠默认文件），路径/宽度/哈希进 `run.json` 与 stage manifest |
| [3] 维度错误 | 48d 表 + 32d ckpt ⇒ 构造期 `FixedSidError`，含两个宽度与两文件；**无 run 目录**；默认表错宽度亦报错并给 `embedding_path=` 提示 |
| [4] 无回退 | 数据集目录放 48d 诱饵默认表，显式 32d 表仍成功；manifest 的 `dataset_matches=False` 只记录不阻断 |
| [5] B4 未绕过 | 显式表下：错 ckpt 仍 `SidProvenanceError`；无 manifest 仍默认硬失败；opt-out 仍记 `mode=unregistered` |
| [6] CLI | `--embedding_path` 生效并记录；错宽度 rc≠0 且 stderr 含 `embedding dimension mismatch`；省略 flag 仍走默认 |
| [7] 真实阻塞场景 | `Toys_Content768d_Baseline/Jul-31-2026_21-37-33`（768d）+ `data/Toys/item_emb_768d_contentonly.parquet` 在 **`data_root=<repo>/data`（无 shadow root）** 下构造成功（`in_dim=768`、11924 行）；同 ckpt 走默认 4096d 表则报清晰 dimension mismatch |

> 陷阱：`data/Toys/fused_item_emb_768d.parquet` **实为 896d**（名字误导，见 [[toys-rqvae-fixture-dims]]）；768d 基线 ckpt 真正读的表是 `item_emb_768d_contentonly.parquet`。首版 part7 用错表，被本门禁当场拒绝（正是 B6 修复要防的情形）。

**回归**：EXP-P2 **47/47**、EXP-B4 **38/38**、EXP-E8 **54/55**（唯一失败仍是既有 tracked-file guard，零新增）。**不是效果证据**：所有训练均为 1 epoch 合成数据机制 smoke，真实输入组跑 0 epoch。

**剩余阻塞**：B7（`qspace_train` 不播种 RNG）待裁决（→ 已由 §19.11.11 关闭）；R1（新 ckpt `args` 缺 `tau/lambda_*` 键）（→ 已由 §19.11.12 审计关闭）、R2（CLI 默认值不等于 EXP-L0）（→ 已由 §19.11.13 关闭）、R3（无 EXP-L0 driver）。⇒ **GATE 仍 CONDITIONAL，EXP-L0 不启动。**

#### 19.11.11 B7 修复：seed 真正控制 CSG 训练的随机过程（2026-10-19）

**任务边界**：只改 `tiger_ext/qspace_train.py`（`sha256[:16]` 由 `e10f24cf0378a816` → **`305c50a7bf530f53`**，693 → 832 行）并新增 harness `integration/exp_b7_seed_control.py`。`fixed_train.py`（`389e9085…`）、`sid_refresh.py`（`983121cb…`）、`alt_pipeline.py`（`fc1b8652…`）以及 `model/`、`rqvae/`、`clean_tiger/` **逐字节未变**；未启动 EXP-L0 / EXP-B1；模型结构、loss、RQ-VAE 配方、epochs 与 R13 冻结定义均未改。

**修复前的事实（B7 的定义）**

| 随机源 | 修复前 | 后果 |
|---|---|---|
| python `random` / numpy / torch CPU / torch CUDA | **从未播种**（`--seed` 只写 `args` 与报告；`RQVAETrainer.__init__` 不读） | 同 `--seed` 重跑初始化不同 |
| 模型初始化（`RQVAE` 的 `nn.Linear`） | 继承进程全局 torch RNG | 不可复现 |
| DataLoader `shuffle=True` | 使用全局 torch RNG | batch 顺序不可控 |
| `layers.py:86` 的 `KMeans(...)` | **无 `random_state`** ⇒ 跟随 numpy 全局流；`n_init=10` 多次抽样 | 首 batch lazy k-means 中心不可复现；**仅设全局 seed 不足**（k-means 前任何 numpy 消费者都会移动流） |

**修复（三处，全部在 `qspace_train.py`）**

1. `apply_seed(seed)`：与 `rqvae/main.py:77-84` 同构——`random.seed` / `np.random.seed` / `torch.manual_seed` / `torch.cuda.manual_seed_all` + `cudnn.deterministic=True` / `benchmark=False`；返回写入报告的记录。`train_tokenizer` **第一行**（`validate_embedding_path` 之后、hook/dataset/model 之前）调用。
2. `torch.Generator().manual_seed(config.seed)` 传给 `DataLoader(generator=...)`，shuffle 顺序不再依赖全局 torch 流；报告记录 `shuffle_generator_seed` 与来源。
3. `seeded_kmeans(seed)` 上下文：把 `rqvae/models/layers.py` 的模块级 `KMeans` 名称在运行期换成一个**注入 `random_state`** 的工厂（值取自私有 `RandomState(seed)`，每次调用一抽），`finally` 还原。**不编辑 `rqvae/`**；`kmeans_init=True` 的 lazy 初始化发生在首 batch forward ⇒ 上下文覆盖整个 `fit`，不只 `build_model`。冻结 `layers.kmeans` 在 `num_clusters > B` 时使用的硬编码 `RandomState(0)` 子采样**保持不变**，在报告中如实记为 `kmeans_subsample_random_state: 0`（它不是 seed 控制项）。

**记录**：`report["seeding"]` = 15 个字段（seed、四条流、`torch_initial_seed`、cuDNN 两开关、shuffle 来源/seed、`kmeans_random_state_calls`、注入来源、子采样 seed、`cuda_nondeterminism`）。CLI 摘要行打印 seed / torch_initial_seed / shuffle seed / k-means draws。

**CUDA 明确不承诺逐位复现**（写入 `CST_CUDA_NONDETERMINISM` 与每份报告）：embedding backward 的 atomics、cuDNN kernel 选择、多流/多卡归约顺序、`num_workers>0` 时 worker 到达顺序。**未启用** `torch.use_deterministic_algorithms`——它可能在冻结循环依赖的算子上直接中止，且 B7 不改配方。CPU 路径是逐位验证路径。

**测试 `integration/exp_b7_seed_control.py` = 48/48**（CPU，1 epoch × 320 行合成表；三进程 + 进程内对照）：

| 组 | 覆盖 | 读数 |
|---|---|---|
| [1] 调用链 | `--seed` → parser → `CSGTrainerConfig.seed` → `apply_seed`；AST 顺序在 `build_dataset` / `build_model` 之前；CLI 不是死路 | 通过 |
| [2] 全局流 | python/numpy/torch 各自可复现；`torch.initial_seed()==seed`；cuDNN pin；**任意消耗 2000 个随机值后 re-seed 仍复位** | 通过 |
| [3] KMeans | 先**证明**冻结调用依赖 numpy 全局流（seed 1 vs 2 中心不同）；`seeded_kmeans` 下两段不同全局流中心**逐位相同**；draws == `RandomState(seed)` 前 n 抽；异 seed 抽值不同；spy 证明 `random_state` 非 None；异常路径还原 patch；`layers.py` 无 `random_state`、原调用完好 | 通过 |
| [4] 同 seed 同进程 | 两次运行间**烧掉 2000×3 个随机值**：init `aa17a452…`、final `be74cfe…`、codes（220 distinct）、`best_loss=0.982522`、3 个 k-means draws **全部相同**；dataset 处 `torch.initial_seed==1234`；shuffle 生成器同序 | 通过 |
| [5] 同 seed 跨进程 | 两个子进程 + 一个进程内运行（seed 777）`comparable()` **完全一致** | 通过 |
| [6] 异 seed | seed 4321：init `a51be790…`（≠）、k-means draws（≠）、final（≠）、codes（234 vs 220 distinct）；表 sha 与行数相同 ⇒ 差异只来自 seed | 通过 |
| [7] 范围 | 报告 15 字段齐全；`CST_SEED_CONTRACT` / `CST_CUDA_NONDETERMINISM` 导出且点名 k-means/atomics/num_workers；`torch.use_deterministic_algorithms` 未启用；配方 knob == `RQVAE_CONFIG`；1 epoch 仍是 1 epoch；种子相关代码只存在于 `qspace_train.py` | 通过 |

**回归**：

| suite | 结果 | 说明 |
|---|---|---|
| EXP-P3 | **29/29** | Stage 0bis CLI/表/warmup 不受影响 |
| EXP-C4 | **84/87** | 与 C5 之后记录的基线**逐条相同**：modified-set guard + 文本 guard（`rqvae/trainer.py:121` 注释提到 C4，C5 引入）+ 内嵌 E10 重跑；**零新增失败** |
| EXP-C6 | **63/63** | 全链 smoke；其内嵌 **EXP-C5 = 99/99, exit 0**，七份旧 suite 的 guard 归因仍为 `unexplained: none` |

**不是效果证据**：全部训练为 1 epoch 合成表机制 smoke（真实输入组 0 epoch），不产生 Recall/NDCG，不用于 λ 选点。

**剩余阻塞**：R1（新 ckpt `args` 缺 `tau/lambda_*` 键）（→ 已由 §19.11.12 审计关闭）、R2（CLI 默认值不等于 EXP-L0）（→ 已由 §19.11.13 关闭）、R3（无 EXP-L0 driver）⇒ **GATE 仍 CONDITIONAL，EXP-L0 不启动**；B1–B7 均已关闭。

#### 19.11.12 R1 审计：新 ckpt 缺失训练参数的实际生效值（2026-10-19）

**任务边界**：只做核实与审计证据。**未修改任何生产文件**（`tiger_ext/qspace_train.py` `305c50a7bf530f53…`、`fixed_train.py` `389e90854b5f6f13…`、`sid_refresh.py` `983121cb3d9cd17b…`、`alt_pipeline.py` `fc1b8652a72c04ce…` 均逐字节未变，EXP-R1 [8] 4/4）；未启动 EXP-L0 / EXP-B1；未改模型结构、损失函数或 R13 定义。新增 harness `integration/exp_r1_ckpt_args_audit.py` 与审计表 `integration/artifacts/r1_ckpt_args/exp_r1_audit.json`。

**R1 的原始定义**：新 ckpt（`qspace_train` 写出的）`args` 缺 `tau / lambda_trans / lambda_balance / trans_graph_path`，有效值靠 `getattr` 默认；基线 ckpt 有这 4 个键。审计后**缺键范围更正为 7 个**（`align_weight / align_tau / align_data_path` 同样缺，且基线 ckpt 也缺这 3 个）。

**链条（代码证据）**

| 环节 | 位置 | 事实 |
|---|---|---|
| namespace 构造 | `tiger_ext/alt_pipeline.py:153-163` | `rqvae_namespace` 只枚举 26 个键（`data_path / ckpt_dir / device / seed / epochs / eval_step / num_workers / warmup_epochs` + `RQVAE_CONFIG` 的 18 项），**不含** 7 个辅助损失键 |
| 落盘 | `rqvae/trainer.py:444` | `state = {"args": self.args, ...}` —— args **原样**写入，没有补键 |
| 读取 | `rqvae/trainer.py:86-88, 100, 57-58, 63` | `getattr(args, '<field>', <default>)`；默认值分别为 `0.0 / 0.0 / 0.1 / None / 0.0 / 0.1 / None` |
| 默认值出处 | `rqvae/main.py:52-68` **与 `git show 5edfca0:rqvae/main.py:52-68`** | 7 个 argparse 默认值在两个版本中**逐值相同**，且等于 R13 冻结值（AST 提取，非人工抄录） |
| 必需字段 | `rqvae/trainer.py` 的 `args.<field>` 直读 | 11 个：`ckpt_dir/device/epochs/eval_step/learner/lr/lr_scheduler_type/num_emb_list/save_limit/warmup_epochs/weight_decay` —— 新 namespace **全部存在** |

**逐字段核对表**（实际生效值 = 用 ckpt 的 args 构造**真实 `RQVAETrainer`** 后读出的属性，不是 `getattr` 推演）

| 字段 | 新 ckpt 有键 | 基线 ckpt 有键 | 默认出处 | 实际生效值 | R13 冻结要求 | 判定 |
|---|---|---|---|---|---|---|
| `tau` | ❌ | ✅ (0.1) | `trainer.py:88` | `0.1`（且 `contrastive_loss.tau=0.1`） | 0.1 | ✅ 安全 |
| `lambda_trans` | ❌ | ✅ (0.0) | `trainer.py:86` | `0.0`；`trans_dict=None` | 0.0 | ✅ 安全 |
| `lambda_balance` | ❌ | ✅ (0.0) | `trainer.py:87` | `0.0` | 0.0 | ✅ 安全 |
| `trans_graph_path` | ❌ | ✅ (None) | `trainer.py:100` | `None`（无图加载） | None | ✅ 安全 |
| `align_weight` | ❌ | ❌ | `trainer.py:57` | `0.0`；`align_proj=None` | 0.0 | ✅ 安全 |
| `align_tau` | ❌ | ❌ | `trainer.py:58` | `0.1` | 0.1 | ✅ 安全 |
| `align_data_path` | ❌ | ❌ | `trainer.py:63` | `None`；`align_table=None` | None | ✅ 安全 |

覆盖三份 ckpt：① 当前代码 1-epoch 现产（`fresh`）② R13 门禁产物 `lc_smoke`（warmup=50）③ 冻结 768d 基线（λe 两臂的 tokenizer）。**21/21 (ckpt, field) 对全部等于冻结值**。"缺键但安全使用冻结默认值"成立；**没有**任何字段落入"缺失导致实际配置不确定或不符合配方"一类。

**对照实验（证明审计不是空转）**

1. **有键就会被采用**：注入 `lambda_balance=0.7` → Trainer 取 0.7；注入 `tau=0.07` → `contrastive_loss.tau=0.07`。⇒ 默认值不是被硬编码忽略。
2. **present-but-broken 是另一类失败**：注入 `lambda_trans=0.7` + 不存在的 `trans_graph_path` → Trainer **静默**把 `lambda_trans` 改成 0.0 并告警。这与"缺键取默认"结果相同但成因不同；R13 不依赖此路径。
3. **必需/可选划分由代码强制**：删除 `warmup_epochs`（直读字段）→ `AttributeError`；删除 `lambda_trans`（可选）→ 正常。⇒ 我们报告的"必需字段"不是分类断言，而是代码行为。
4. **缺键 ≠ 有键取默认**：新 ckpt 上 `hasattr(args,'lambda_trans') is False`，而 `getattr(...,0.0)==0.0`。

**下游消费者**

| 消费者 | 是否读这 7 项 | 结论 |
|---|---|---|
| `rqvae/generate_code.py` | 否（只读模型结构键；新 ckpt 全有） | 安全 |
| `tiger_ext/fixed_train.py` | 否（`_read_tokenizer_checkpoint()` 只取 `state_dict` 做宽度/加载，从不读 `args`） | 安全 |
| `tiger_ext/sid_refresh.py::ckpt_args` | 白名单 11 键，**不含**这 7 项 | 仅 provenance **记录缺口**（不误报、不影响 refresh 行为） |
| 全仓扫描 | 34 个读取 checkpoint `["args"]` 的 .py 文件 | 0 个触碰这 7 项 |

**残余观察（不阻断，记录在案）**

1. `lambda_csg`（R13 真正的处理变量）**不在 ckpt args** 中，只在运行报告里。训练时由 CLI→`CSGHook` 传入，因此**不影响任何实际生效值**；但 ckpt 本身无法自证属于哪个 λc 臂。建议在 R3 driver 的 run 记录/manifest 中落 `lambda_csg`，**本次不改生产代码**。
2. `sid_refresh.ckpt_args` 的 11 键白名单同样不含 `warmup_epochs`；如需让 SID provenance 自证 warmup，可另行申请扩表（不属 R1）。
3. 基线 ckpt 缺 `seed`（`rqvae/main.py` 的 `--seed` 未进 namespace），但 `sid_refresh` 白名单只在 `hasattr` 为真时记录 —— 缺键不报错，只是不含该项。

**回归**

| suite | 结果 | 说明 |
|---|---|---|
| EXP-R1（新增） | **44/44**（verdict `CLOSED-BY-AUDIT`） | 见上 |
| EXP-P3 | **29/29** | CLI → namespace 配置链 |
| EXP-P2 | **47/47** | 冻结 Trainer 重放逐位等价 |
| EXP-B6 | **31/31** | tokenizer 宽度门禁与显式表路径 |

**不是效果证据**：唯一训练为 1 epoch × 320 行合成表（`fresh` ckpt 的来源），无 Recall/NDCG，不用于选点。

**结论**：**R1 = CLOSED-BY-AUDIT（关闭）**。不需要修改生产代码（符合任务第 4 条）。**GATE 仍 CONDITIONAL**：剩余阻断 = **R2**（CLI 默认值 ≠ EXP-L0）（→ 已由 §19.11.13 关闭）、**R3**（无 EXP-L0 driver）⇒ EXP-L0 不启动。

#### 19.11.13 R2 关闭：EXP-L0 五条冻结命令（2026-10-19）

**任务边界**：只做命令冻结与核对。**未修改任何生产文件**（`qspace_train.py 305c50a7…`、`fixed_train.py 389e9085…`、`sid_refresh.py 983121cb…`、`alt_pipeline.py fc1b8652…` 逐字节未变，EXP-R2 [6] 4/4）；**未新增 EXP-L0 driver**（R3 仍开）；**未启动 EXP-L0 / EXP-B1**；未改 R13 配方（λ、epochs、warmup、seed、SID/ckpt 指定方式全部照抄 §19.1 / §19.3 / §19.11.3）。新增 harness `integration/exp_r2_exp_l0_cli.py` 与审计表 `integration/artifacts/r2_exp_l0_cli/exp_r2_audit.json`。

**五条冻结命令（仓库根目录执行；三组 CSG + 两组 ETE）**

λc 三臂（`{1e-4|1e-3|1e-2}` 三选一，`--ckpt-dir` 与 λ 同步）：

```bash
python -m tiger_ext.qspace_train \
  --dataset Toys --data-root data \
  --artifact-dir integration/artifacts/collab_relation --k 20 \
  --lambda-csg {1e-4|1e-3|1e-2} \
  --epochs 3000 --batch-size 1024 --eval-step 50 \
  --num-workers 4 --warmup-epochs 50 \
  --embedding-path data/Toys/item_emb_768d_contentonly.parquet \
  --device cuda:0 --seed 2024 \
  --ckpt-dir rqvae/ckpt/EXP_L0_lc{1e-4|1e-3|1e-2}
```

λe 两臂（`{le0.1|le1.0}` ↔ `{0.1|1.0}`）：

```bash
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

**λc 臂的完成步骤（不在这五条之列：SID 路径是运行时产物，由 R3 driver 拼接）**

```bash
RUN=$(ls -dt rqvae/ckpt/EXP_L0_lc{λ}/*/ | head -1)          # Trainer 自动加时间戳子目录
python -c "from tiger_ext import sid_refresh as SR; SR.refresh_sid('Toys', '$RUN/best_collision_model.pth', 'integration/artifacts/exp_l0/lc{λ}/sid', device='cuda:0')"
SID=$(ls -t integration/artifacts/exp_l0/lc{λ}/sid/sid_versions/*.npy | head -1)
python -m tiger_ext.fixed_train ... \                       # 其余参数与上面 λe 命令逐字相同
  --sid_path "$SID" --sid_manifest "$SID.sid.json" \
  --tokenizer_ckpt "$RUN/best_collision_model.pth" \
  --arm "EXP-L0 λc-{1e-4|1e-3|1e-2}" \
  --lambda_sia 0 --lambda_psa 0 \
  --run_dir integration/artifacts/exp_l0/lc{λ}/run
```

**逐参数核对（EXP-R2 [1]–[5]；全部由**真实解析器**解析，不执行训练）**

| 项 | CSG 三臂 | 冻结要求（出处） | 结论 |
|---|---|---|---|
| `--dataset` | Toys | Toys（§19.1） | ✅ |
| `--data-root` / `--data_root` | `data` | 仓库 `data/`（§19.1） | ✅（拼写按各自 CLI） |
| `--lambda-csg` | 1e-4 / 1e-3 / 1e-2 | §19.4 三档 | ✅ |
| `--epochs` | 3000 | §19.3 第 2 条 | ✅ |
| `--batch-size` | 1024 | §19.1 | ✅ |
| `--eval-step` | 50 | §19.1 | ✅ |
| `--num-workers`（CSG） | 4 | §19.1 | ✅ |
| `--warmup-epochs` | 50 | §19.1 | ✅ |
| `--embedding-path`（CSG） | `data/Toys/item_emb_768d_contentonly.parquet` | §19.1（768d） | ✅ |
| `--device`（CSG） | cuda:0 | 门禁运行环境 | ✅ |
| `--seed`（CSG） | 2024 | `SEED_RQVAE`（§19.4） | ✅ |
| `--ckpt-dir` | 每臂独立 `rqvae/ckpt/EXP_L0_lc{λ}` | 每臂独立 tokenizer（§19.11.3） | ✅ |
| `--k` / `--artifact-dir` | 20 / `integration/artifacts/collab_relation` | `RELATION_K=20`；默认产物目录 | ✅ 路径存在 |
| `--lambda_sia` / `--lambda_psa` | λe：0.1/0.1、1.0/1.0；λc：0/0 | §19.4 / §19.11.3 | ✅ |
| `--seed`（ETE） | 2025 | `SEED_T5` | ✅ |
| `--num_epochs` / `--early_stop` | 200 / 10 | §19.3 第 1 条 | ✅ |
| `--batch_size` / `--infer_size` / `--beam_size` / `--lr` | 256 / 96 / 30 / 1e-4 | §19.3 第 1 条 | ✅ |
| `--num_workers`（ETE） | 1 | `alt_pipeline.NUM_WORKERS=1`（见观察 3） | ✅ |
| `--device`（ETE） | cuda:0 | 门禁运行环境 | ✅ |
| `--sid_path`（λe） | `data/Toys/Toys_content768d_baseline_sid.npy` | §19.11.3 臂 4/5 | ✅ |
| `--sid_manifest`（λe） | `<sid>.sid.json`（显式） | 已登记 manifest | ✅ 校验通过 |
| `--tokenizer_ckpt`（λe） | `Toys_Content768d_Baseline/Jul-31-2026_21-37-33/best_collision_model.pth` | §19.1 冻结基线 | ✅ 唯一 run，`in_dim=768` |
| `--allow_unregistered_sid` | **不传**（False） | B4 门禁 | ✅ |
| `--run_dir` | 每臂独立 `integration/artifacts/exp_l0/<arm>/run` | 每臂独立输出 | ✅ |

**参数/路径核对中的具体发现**

1. **CLI 拼写不能照任务描述的 `--embedding_path` 字面写进 CSG 命令**：`qspace_train` 的 flag 是**短横线** `--embedding-path`（`qspace_train.py:778`），argparse 不做 `-`/`_` 别名；测试实测 `python -m tiger_ext.qspace_train --embedding_path x` 以 **exit 2** 拒绝。冻结命令因此写 `--embedding-path`（CSG）与 `--embedding_path`（ETE，`fixed_train.py:845`）。**这是拼写校正，不是配方变更。**
2. **`align_weight 0.0` 无法由 `qspace_train` CLI 显式传入**：该入口没有 `--align_weight`，`rqvae_namespace` 也不带该键。§19.1 的"必须显式传 `--align_weight 0.0`"在**效果上**由 `trainer.py:57` 的 `getattr` 默认满足（R1 §19.11.12 已用真实 Trainer 复核为 0.0）。记录为**观察项**，不新增 CLI、不改配方。
3. **`--num_workers 1`（ETE）与旧 `model/main.py` 的 dataloader 默认 4 不同**：§19.3 未钉该项；R13 管线常量 `alt_pipeline.NUM_WORKERS=1` 是项目可复现性裁决（"never leave it at 4"）。`shuffle=True` 的 `RandomSampler` 在主进程生成索引，worker 数不改变 batch 顺序，故不构成语义差异。冻结命令显式写 1，**不依赖 CLI 默认**。
4. **896d 诱饵表已排除**：`data/Toys/fused_item_emb_768d.parquet` 实测 **896d**；五条命令中 0 次出现。被选中的 `item_emb_768d_contentonly.parquet` 实测 **768d / 11924 行**，与冻结 ckpt 的 `encoder.mlp_layers.1.weight` 宽度 768 一致。默认表 `data/Toys/item_emb.parquet` 实测 **4096d**（R2 的原始风险）。
5. **λe 的 R2 风险是 λ 默认 0 与 embedding 默认 None**：`fixed_train` 不传 λ 时静默等于 C/E=0 的基线；不传 `--embedding_path` 时会读 4096d 表并被 B6 宽度门禁拒绝（拒绝而非静默）。冻结命令两者都显式给出。
6. **λc 的 SID 是运行时产物**：版本号 `{dataset}_sid_v{NNNN}_{ckpt_sha12}.npy` 依赖 `base_dir` 已有版本；全新 base_dir 下为 `v0001`。driver 必须在 refresh 后从 `sid_versions/` 解析，**不得硬编码 `v0001`**（R3 负责）。
7. **trainer 的 `--ckpt-dir` 是父目录**：冻结 `Trainer` 在其下再建 `get_local_time()` 时间戳子目录（`rqvae/trainer.py:42-45`），实际 ckpt 在 `rqvae/ckpt/EXP_L0_lc{λ}/<timestamp>/best_collision_model.pth`。命令中的父目录按臂唯一，Δ 不会跨臂覆盖。

**验证与回归**

| suite | 结果 | 说明 |
|---|---|---|
| **EXP-R2（新增）** | **274/274**，verdict `CLOSED-BY-COMMAND-FREEZE` | 五条命令逐参数解析 + 路径/宽度 + 默认值反证 + 作用域 |
| EXP-P3 | 29/29 | CSG CLI → namespace |
| EXP-P2 | 47/47 | 冻结 Trainer 重放 |
| EXP-B6 | 31/31 | tokenizer 宽度与显式表 |

持久日志 `integration/artifacts/r2_exp_l0_cli/exp_r2.log`；审计 JSON `…/exp_r2_audit.json`（含五条命令原文与 274 项判定）。**全部为解析/构造级核对，未训练任何模型，不是效果证据。**

**结论**：**R2 = CLOSED-BY-COMMAND-FREEZE（关闭）**。五条命令参数显式、路径存在、不依赖任何不匹配的 CLI 默认值，且与冻结配方逐项一致。**GATE 仍 CONDITIONAL，唯一剩余阻断 = R3（无 EXP-L0 driver）**；EXP-L0 未启动。

#### 19.11.14 R3 关闭：EXP-L0 自动执行 driver（2026-10-19）

**任务边界**：只在 `integration/` 下新增 driver、机制测试与文档。**未修改任何生产文件**（`qspace_train.py 305c50a7…`、`fixed_train.py 389e9085…`、`sid_refresh.py 983121cb…`、`alt_pipeline.py fc1b8652…` 逐字节未变，EXP-R3 [9] 4/4）；**未启动 EXP-L0 / EXP-B1**（所有测试均为 mock/spy + 合成 fixture；正式运行的唯一入口已就绪但未调用）；**未改 R13 配方**（λ、epochs、warmup、seed、embedding、输入 SID/ckpt 全部照抄 §19.11.13）。新增 `integration/exp_l0_driver.py`（driver）、`integration/exp_r3_l0_driver.py`（机制测试）与 `integration/artifacts/r3_l0_driver/{exp_r3.log,exp_r3_audit.json}`。删除 R3 期间一度存在的 shell 草稿，交付物只有 Python driver 一个入口。

**交付物**

| 项 | 内容 |
|---|---|
| driver | `integration/exp_l0_driver.py`；`--dry-run` / `--arms` / `--run-id` / `--run-root` / `--record` / `--device` / `--python` / `--quiet` |
| 五臂 | λc = 1e-4 / 1e-3 / 1e-2（tokenizer → refresh → recommender，λsia=λpsa=0）；λe = 0.1 / 1.0（冻结 baseline tokenizer + 已登记 baseline SID，λsia=λpsa=λ） |
| 执行顺序 | `lc1e-4 → lc1e-3 → lc1e-2 → le0.1 → le1.0`；任一阶段非 0 退出即停止该臂及所有后续臂 |
| 输出隔离 | 每次调用一个全新 run root：`integration/artifacts/exp_l0/runs/<run_id>/<arm>/{ckpt,sid,run,logs}`；`<arm>` 非空或 `sid_versions/` 已存在 ⇒ **preflight 失败，不启动任何子进程** |
| 产物识别 | checkpoint = 训练前后 `ckpt_parent` 子目录差集中的**唯一新目录**；SID = refresh 前后 `sid_refresh.list_versions` 差集中的**唯一新版本**；均不硬编码时间戳或版本号 |
| provenance | refresh 后校验 `sid_path` / SID sha256 / ckpt sha256 / 文件名版本与 manifest 一致，并要求 `.sid.json` sidecar 存在；不一致 ⇒ refresh 阶段失败 |
| 总记录 | `integration/artifacts/exp_l0/driver_run.json`（最新一次）+ `runs/<run_id>/driver_run.json`（该次不可变副本）；含五臂冻结参数、逐阶段命令/退出码/开始结束时间/日志路径、输入输出路径、ckpt/SID/manifest 哈希与验证结果、失败阶段与原因 |
| R1 残余 | 每条臂记录 `lambda_csg`（λc 为 1e-4/1e-3/1e-2，λe 为 `null`）——关闭 §19.11.12 的事后可读性缺口 |
| dry-run | 输出五臂计划、依赖关系、完整 argv 与命令文本；**不执行子进程、不创建目录、不刷新 SID、不写记录** |

**与 §19.11.13 冻结命令的关系**：五臂的**配方参数与输入路径逐 token 一致**（EXP-R3 [2]）。唯一差异是**输出位置按运行隔离**：`--ckpt-dir` 由 `rqvae/ckpt/EXP_L0_lc{λ}` 改为 `<run_root>/<arm>/ckpt`，`--run_dir` 由 `integration/artifacts/exp_l0/<arm>/run` 改为 `<run_root>/<arm>/run`。这是 R3 任务第 2 条（禁止复用旧产物、每次运行独立输出目录）的直接要求，属**输出位置**而非实验配方；λc 的 refresh `base_dir` 同理改为 `<run_root>/<arm>/sid`。命令 diff 由测试逐臂钉死，确保除上述输出路径外无任何参数漂移。

**机制测试（EXP-R3，105/105，`CLOSED-BY-MECHANISM-TESTS`）**

| 覆盖面 | 结果 |
|---|---|
| 五臂配置完整性（kind / λc / λsia / λpsa / label / 顺序） | 8/8 |
| 与 R2 冻结命令逐 token 对比（3 个 λc tokenizer、2 个 λe recommender、3 个 λc recommender 模板） | 8/8 |
| flag 拼写、768d 表、无 `--allow_unregistered_sid`、λc 不引用 baseline SID | 5/5 |
| dry-run 计划（11 个阶段、依赖链、无目录/无记录/无 runner 调用） | 10/10 |
| 合成 fixture 全链路：阶段顺序、动态 ckpt/SID 识别、manifest 校验、哈希、记录完整性 | 24/24 |
| 动态版本解析（v0008 新版本；非匹配文件名忽略；时间戳动态） | 3/3 |
| 失败注入：tokenizer/refresh/recommender 失败即停 + 后续臂未启动 | 12/12 |
| provenance 篡改：SID sha / ckpt sha / 版本 / 缺 sidecar / 两个新 SID / 无新 SID | 18/18 |
| 旧产物不复用：非空 arm dir / 已有 SID 版本 / 旧 run root 的 ckpt 与 SID 零引用 | 7/7 |
| CLI dry-run（子进程）可解析、无 run root、λ 值正确 | 4/4 |
| 生产文件哈希 4/4 与 git 状态不变 | 6/6 |

**回归**：EXP-R2 **274/274**、EXP-P3 **29/29**、EXP-P1 **27/27**、EXP-P2 **47/47**、EXP-B6 **31/31** —— 与各自历史结果逐项相同，**零新增失败**。（EXP-R2 的 “no EXP-L0 driver added” 断言只匹配 `integration/*l0*.sh`，Python driver 不触发；该断言是 R2 当时的任务边界，R3 落地后仍如实通过。）

**dry-run 示例**：`python -I integration/exp_l0_driver.py --dry-run --arms lc1e-3,le1.0 --run-root /tmp/exp_l0_dry` 输出两臂计划与完整 argv；实测 `/tmp/exp_l0_dry` 与 `integration/artifacts/exp_l0` 均未被创建。

**结论**：**R3 = CLOSED-BY-MECHANISM-TESTS（代码与机制测试通过）**。按任务要求，**GATE 仍 CONDITIONAL**：下一步是**独立门禁复核**（重跑 §19.11.9 相关行并裁决 R3），复核通过前 **EXP-L0 不得启动**。本次未产生任何训练、Recall/NDCG 或效果证据。

#### 19.11.15 独立门禁复核（2026-10-19，**GATE = CONDITIONAL：2 个必须修复项**）

**任务边界**：只做独立复核。**未修改任何生产文件**（四个 `tiger_ext/*.py` 哈希与 §19.11.14 记录逐字符相同），**未修改 driver / 机制测试**，**未启动 EXP-L0 / EXP-B1**，未改 R13 配方。复核产物新增 `integration/exp_r3_gate_review.py`（独立复核 harness，46 项，`REVIEW 44/46`）与 `integration/artifacts/r3_gate_review/{review.log,exp_r3_gate_review.json}`。复核中所有探测均在 `/tmp` 临时目录内完成并已删除；`integration/artifacts/exp_l0/` 至复核结束时仍不存在。

**复核方法（不依赖 driver 自带的测试报告）**

1. 从 §19.11.13 的命令模板**重新独立拼写**五臂 11 条命令，与 driver 生成的 argv **逐 token 全等比较**（不是 diff-only）。
2. 用 `qspace_train.main` / `fixed_train.main` 的**真实解析器**解析 driver 生成的每一条 argv（训练入口以 spy 替换，不建模型）。
3. 直接检查数据/权重/manifest 前置条件。
4. 对 driver **注入失败与异常**（不是只看既有 harness 的断言）。

##### 19.11.15.1 六项必查结论

| # | 核查项 | 结论 | 证据 |
|---|---|---|---|
| 1 | 重跑冻结门禁测试，区分历史失败与新增失败 | **PASS** | 见 §19.11.15.2；R1/R2/R3、P1/P2/P3、B4/B6/B7 全部与历史计数相同；E9 87/90、E10 60/64 的失败行与历史**逐条相同** |
| 2 | 从实际执行代码独立核对五臂参数/输入/顺序/输出 | **PASS** | 11 条命令与从 §19.11.13 独立重写的模板**逐 token 全等**（无任何允许差异）；真实解析器接受，且 config/构造 kwargs 的 15 项配方字段逐值等于冻结值；执行顺序 `lc1e-4 → lc1e-3 → lc1e-2 → le0.1 → le1.0` |
| 3 | 真实执行安全性（差集识别/版本识别/manifest/hash/旧产物/失败即停/日志） | **FAIL** | 主体安全：动态识别只接受"恰好 1 个新产物"、provenance 四项交叉校验、旧产物 preflight 零子进程、失败即停且不回退——均实测通过。但发现 **M1 / M2** 两个缺口（见 §19.11.15.3） |
| 4 | 训练参数与 R2 冻结配方一致（run-scoped 仅改输出位置） | **PASS** | 唯一输出差异为 `--ckpt-dir` / `--run_dir` / λc `base_dir` 的 run-scoped 前缀；模型输入、超参、λ、seed、embedding、baseline SID/ckpt 零漂移；`--allow_unregistered_sid` 未传 |
| 5 | 启动前置条件（768d 表 / baseline SID+manifest / tokenizer ckpt / CUDA / 数据） | **PASS** | 768d 表 11924×768；baseline ckpt `in_dim=768` 且 sha `8069f998…`；baseline SID `(11924,4)`、sidecar 对 ckpt 与 SID 双向校验通过、`sid_stats.ok=true`；Toys 关系产物 k=20/11924 项；`rqvae/generate_code.py` 存在；torch 2.6.0+cu124、CUDA 可用、1 卡；`train/valid/test.parquet`、`item_mapping.npy`、`item_emb.parquet` 齐备；磁盘 1.0 TB 空闲 |
| 6 | 不改生产代码、不启动正式实验 | **PASS** | 四文件哈希不变；git 状态与开工基线逐字符相同；`integration/artifacts/exp_l0/` 不存在；`data/Toys/Toys_content768d_baseline_sid.npy` 与 baseline ckpt 哈希复核不变 |

##### 19.11.15.2 回归结果与历史对照

| suite | 历史 | 本次 | 失败行是否相同 |
|---|---|---|---|
| EXP-R1 | 44/44 | **44/44** | — |
| EXP-R2 | 274/274 | **274/274** | — |
| EXP-R3 | 105/105 | **105/105** | — |
| EXP-P1 | 27/27 | **27/27** | — |
| EXP-P2 | 47/47 | **47/47** | — |
| EXP-P3 | 29/29 | **29/29** | — |
| EXP-B4 | 38/38 | **38/38** | — |
| EXP-B6 | 31/31 | **31/31** | — |
| EXP-B7 | 48/48 | **48/48** | — |
| EXP-E9 | 87/90 | **87/90** | 相同 3 行：tracked-file guard（C5 的 `rqvae/trainer.py`）、`exp_e8_sid_refresh 54/55`、汇总行 329/330 |
| EXP-E10 | 60/64 | **60/64** | 相同 4 行：tracked-file guard、`e8 54/55`、内嵌 `e9 80/81`（进度条时间戳）、汇总行 409/411 |

⇒ **零新增失败**；E9/E10 的 rc=1 是既有已知失败导致，与本次复核无关。

##### 19.11.15.3 发现：两个必须修复项

**M1（假成功路径）空臂选择被记为 completed。** 证据：`python -I integration/exp_l0_driver.py --arms "" --run-root /tmp/empty_arms/run --record /tmp/empty_arms/record.json` → 退出码 **0**、`status="completed"`、`arms=[]`、`arms_completed=[]`、`arms_not_started=[]`，并写出 `driver_run.json`。根因：`select_specs("")` 返回空列表而不报错，`execute_run` 的空循环仍置 `completed`。最小修复（`integration/` 内，2 行）：

```python
def select_specs(value: str) -> List[ArmSpec]:
    ...
    if not tags:
        raise DriverError("no arms selected; pass --arms all or a subset")
    return [ARM_BY_TAG[t] for t in tags]
```

**M2（异常丢失整轮记录）**。证据：复核 harness 让 runner 抛 `RuntimeError("boom")` → 异常穿出 `execute_run`，**`driver_run.json` 完全未写**（`REVIEW 44/46` 的两个失败项之一，另一个是 M1）。影响面：`subprocess.Popen` 失败（python 路径错/权限/资源耗尽）、`run.json` 损坏导致 `json.JSONDecodeError`、Ctrl-C 之外的未捕获异常。前后阶段不再执行（fail-fast 未被破坏），也**不会**把失败记成成功；缺的是整轮记录本身（已完成的臂与阶段状态只在内存里）。最小修复（`integration/` 内，约 8 行）：

```python
try:
    for index, spec in enumerate(specs):
        ...
    record["status"] = "completed" if ok else "failed"
except BaseException as exc:                       # noqa: BLE001 - record first
    record["status"] = "failed"
    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
    raise
finally:
    record["finished_at"] = now_iso()
    record["arms_completed"] = [...]
    record["arms_not_started"] = [...]
    write_json_atomic(record_path, record)
    ... run copy ...
```

按任务第 6 条，**本次未实施上述修改**，只报告证据与最小修复建议。

##### 19.11.15.4 残余风险（不阻断，但应在正式运行记录中注明）

| # | 风险 | 观察 |
|---|---|---|
| R-a | 记录只在整轮结束时落盘 | 进程被 SIGKILL/断电 ⇒ 无 `driver_run.json`；每阶段日志仍在（`.log` 追加写），产物目录可人工重建证据 |
| R-b | driver 不反查 `fixed_train` 的 `run.json` provenance | `run.json` 已含 `sid_path/sid_sha256/tokenizer_ckpt_sha256`；固定训练器自己也做 manifest↔ckpt 校验，故不构成错误 SID 接受，但可加一项"训练后回读比对" |
| R-c | `run_id` 取自 run-root 目录名 | 同时给 `--run-id X --run-root .../Y` 时记录里是 `Y`；只影响书签，不影响产物 |
| R-d | `sid_stats` 缺键视为通过 | `refresh_sid` 必写 `sid_stats.ok`，只有伪造 manifest 才可达 |
| R-e | 文档 typo | §19.11.13 观察 6 写 `{ckpt_sha12}`，实际 `sid_refresh.sid_name` 用 **sha8**；driver 与代码一致 |
| R-f | refresh 无超时 | `refresh_sid(timeout=None)`；generator 卡死会挂住 driver（日志有阶段性输出） |
| R-g | `topk_list` 记录为冻结配方字段但无 CLI 直传 | `fixed_train` 无 `--topk_list`；实际生效值 `alt_pipeline.T5_TOPK=[5,10,20]` 与记录一致；`best_ndcg@20` 从 `run.json` 读取，**不解析日志列**，规避了 topk 列数陷阱 |

##### 19.11.15.5 判定

- 六项中 **1/2/4/5/6 = PASS，3 = FAIL（M1、M2）**。
- **GATE = CONDITIONAL**：M1/M2 均为 `integration/` 内的小修复，不涉及生产代码与配方；修复后须重跑 EXP-R3 与本节复核相关行，再判 PASS。
- **EXP-L0 / EXP-B1 仍未启动**；本次复核不产生任何训练或效果证据。

#### 19.11.16 M1/M2 最小修复与回归验收（2026-10-19，**修复完成，等待独立复核；GATE 未判**）

**范围**：只改 `integration/exp_l0_driver.py` 与 `integration/exp_r3_l0_driver.py`（新增回归）。生产代码、R2 冻结配方、五臂参数、执行顺序、输出路径**零改动**；**未启动 EXP-L0 / EXP-B1**。

**M1 修复（空臂选择）**

- `select_specs()`：`tags` 为空（`--arms ""`、`","`、`" , "`）⇒ `DriverError`，`main()` 在任何目录创建前返回 **rc=2**，不写记录、不建 run root。
- `execute_run()`：`specs` 为空（绕过 CLI 的程序化调用）⇒ `raise DriverError`，不再可能产出 `status="completed"` 的零臂记录。

**M2 修复（异常收敛 + 整轮记录）**

- `execute_run()` 的臂循环包在 `try/except BaseException` 中；`finalize(status)` 统一计算 `arms_completed` / `arms_not_started`、写 `record_path` 与 `<run_root>/driver_run.json`。
- 异常路径先落盘 **`status="failed"`** 再 `raise`（异常仍向上传播，fail-fast 与"不把失败写成成功"同时成立）；记录新增顶层 `failure = {arm, stage, type, message, traceback}`。
- `_mark_raised_stage()` 把在飞阶段从 `"running"` 改为 `"failed"` 并写入异常原因，记录里不再残留 `"running"`。
- 异常归属阶段取"最后一个 `status=="running"` 的阶段"，无在飞阶段时回退 `"driver"`。
- `main()` 对其他未预期异常打印 `ERROR: arm execution aborted: ...` 并返回 **rc=1**（`BaseException` 如 KeyboardInterrupt 仍向上传播）。
- 顺带把 preflight 失败也写入顶层 `failure`；正常路径记录 `failure = null`。

**diff 规模**：`integration/exp_l0_driver.py` +113 / −41 行（4 个 hunk）；`integration/exp_r3_l0_driver.py` 新增 §[10]/[10b] 25 项检查。

**回归验收（全部重跑）**

| 套件 | 本次 | 历史 | 差异 |
|---|---|---|---|
| EXP-R1 | 44/44 | 44/44 | — |
| EXP-R2 | 274/274 | 274/274 | — |
| EXP-R3 | **130/130** | 105/105 | +25（M1/M2 新增） |
| EXP-P1 / P2 / P3 | 27/27, 47/47, 29/29 | 同 | — |
| EXP-B4 / B6 / B7 | 38/38, 31/31, 48/48 | 同 | — |
| EXP-E9 | 87/90 | 87/90 | 相同 3 行（tracked-file guard、e8 54/55、汇总 329/330） |
| EXP-E10 | 60/64 | 60/64 | 相同 4 行（tracked-file guard、e8 54/55、内嵌 e9 80/81、汇总 409/411） |
| 独立复核 harness | **REVIEW 46/46** | 44/46 | +2（M1/M2 转绿） |

⇒ **零新增失败**。

**失败注入证据**（`integration/artifacts/r3_m1m2/`）

| 探针 | 异常 | 记录 | 状态 / 失败阶段 | 后续臂 |
|---|---|---|---|---|
| 空臂 CLI | — | 未写 | rc=2 | 无 |
| Popen 失败 | `OSError: cannot spawn python` | 已写 | `failed` / `tokenizer` | `lc1e-3,lc1e-2,le0.1,le1.0` 未启动，runner 只调用 1 次 |
| `run.json` 损坏 | `JSONDecodeError` | 已写 | `failed` / `recommender` | `le1.0` 未启动，runner 只调用 1 次 |
| 正常臂（对照） | — | 已写 | `completed`，`failure=null` | — |

**未做/未变**：未改 `run_process`、命令构造器、产物识别、provenance 校验；未改任何 `tiger_ext/*.py`（四哈希见 §19.11.15）；`git status` 与基线一致；`integration/artifacts/exp_l0/` 仍不存在。

**判定**：修复完成，**本任务不宣布 GATE PASS**；等待独立门禁复核（重跑 §19.11.9 与 `exp_r3_gate_review.py`）。**EXP-L0 / EXP-B1 仍未启动。**

#### 19.11.17 M1/M2 修复的独立复核与最终门禁裁决（2026-10-19，**GATE = PASS**）

**任务边界**：只复核，**未修改任何代码**（生产四文件与 `integration/exp_l0_driver.py` 哈希均与 §19.11.16 一致）；**未启动 EXP-L0 / EXP-B1**（`integration/artifacts/exp_l0/`、`exp_b1/` 不存在；无相关进程）。

##### 19.11.17.1 六项核查

| # | 核查项 | 结论 | 独立证据 |
|---|---|---|---|
| 1 | M1/M2 修复逻辑与写入顺序 | **PASS** | `select_specs` 空选择 `raise` 发生在 `main` 建目录之前；`execute_run` 在 `makedirs` 前再守一道。M2 顺序为：`_mark_raised_stage`（stage `running→failed`）→ arm `failed` → 顶层 `failure` → `raise` → 外层 `except` → `finalize("failed")` **先落盘** → 再向调用方传播。独立探针在捕获异常后立刻读盘，记录已存在；`finalize` 的 `status` 只在 `ok=True` 时为 `completed`，失败臂不计入 `arms_completed` |
| 2 | `exp_r3_gate_review.py` / `exp_r3_l0_driver.py` | **PASS** | **REVIEW 46/46**（rc=0）；**EXP-R3 130/130**（rc=0） |
| 3 | §19.11.9 相关门禁测试 | **PASS** | 见 §19.11.17.2；历史失败 8 行逐条复现，**零新增** |
| 4 | 失败注入 JSON | **PASS** | 见 §19.11.17.3；并用独立脚本在全新临时目录**复现**一遍，结果逐字段一致 |
| 5 | R2 命令/输入/SID provenance/哈希/Git | **PASS** | 见 §19.11.17.4 |
| 6 | 不改代码、不启动实验 | **PASS** | 四生产哈希未变；`git status` 与基线逐行相同；无 EXP-L0/B1 产物或进程 |

##### 19.11.17.2 回归结果与历史对照

| 套件 | 本次 | 历史 | 结论 |
|---|---|---|---|
| EXP-R1 | 44/44 | 44/44 | 无差异 |
| EXP-R2 | 274/274 | 274/274 | 无差异 |
| EXP-R3 | 130/130 | 105/105（+25 新增） | 无新增失败 |
| EXP-P1 / P2 / P3 | 27/27、47/47、29/29 | 同 | 无差异 |
| EXP-B3 / B4 / B6 / B7 | 30/30、38/38、31/31、48/48 | 同 | 无差异 |
| **EXP-E8** | **54/55** | 54/55 | **历史失败 1 行**：`model/ and rqvae/ carry no E8 change`（C5 授权修改 `rqvae/trainer.py`） |
| **EXP-E9** | **87/90** | 87/90 | **历史失败 3 行**：同上 guard、`exp_e8 54/55`、汇总 329/330 |
| **EXP-E10** | **60/64** | 60/64 | **历史失败 4 行**：同上 guard、`e8 54/55`、内嵌 `e9 80/81`（进度条时间戳）、汇总 409/411 |

⇒ **零新增失败**。E8/E9/E10 的非零退出码由上述**既有、已知、已归因**的 guard/汇总行导致，**不是全部通过**，也与本次修复无关。

##### 19.11.17.3 失败注入证据核对

独立读取 `integration/artifacts/r3_m1m2/` 并在全新临时目录重放：

| 项 | 空臂选择 | `Popen` 异常 | `JSONDecodeError` |
|---|---|---|---|
| 退出/异常 | rc=2 | `OSError` 向调用方传播 | `JSONDecodeError` 向调用方传播 |
| 记录是否写出 | **否** | **是**（含 `<run_root>/driver_run.json` 副本） | **是** |
| 顶层 `status` | —（不产生记录） | `failed` | `failed` |
| `failure.{arm,stage,type,message,traceback}` | — | `lc1e-4 / tokenizer / OSError / cannot spawn python / 非空` | `le0.1 / recommender / JSONDecodeError / 非空 / 非空` |
| `arms_completed` | — | `[]` | `[]` |
| `arms_not_started` | — | `lc1e-3, lc1e-2, le0.1, le1.0` | `le1.0` |
| 残留 `"running"` 阶段 | — | **无** | **无** |
| 后续臂是否启动 | 无 | 否（runner 只调用 **1** 次） | 否（runner 只调用 **1** 次） |
| run root 是否被创建 | **否** | 是（预期产物目录） | 是 |
| 成功对照 | — | — | `completed`，`failure=null`，`arms_completed=[le0.1]` |

**观察（不阻断）**：`JSONDecodeError` 属于"阶段命令 rc=0、之后解析产物失败"，故该阶段自身的 `status` 记为 `completed`（其语义只等于**子进程退出码**），失败由臂级 `failed_stage=recommender` 与顶层 `failure` 表达——三层信息一致且无歧义。

##### 19.11.17.4 配方、输入与 provenance

- **R2 五条冻结命令**：`--dry-run` 输出的 8 条 argv 与 §19.11.13 逐 token 相同（仅 run-scoped 的 `--ckpt-dir/--run_dir/base_dir` 不同）；`--embedding-path`（CSG，短横线）与 `--embedding_path`（ETE，下划线）各自正确；`--allow_unregistered_sid` **从未出现**。
- **输入**：`data/Toys/item_emb_768d_contentonly.parquet` = **11924 行 / 768 维**（`first vector length=768`）；baseline ckpt `encoder.mlp_layers.1.weight` = **(512, 768)**；`Toys_content768d_baseline_sid.npy` = **(11924, 4)**；`train/valid/test.parquet`、`item_mapping.npy`、`Toys_k20.*` 关系产物均在。
- **SID provenance**：sidecar `Toys_content768d_baseline_sid.npy.sid.json` 的 `ckpt_sha256` = `8069f998…`、`sid_sha256` = `bb2693a6…`，与**当前文件实测哈希逐一相等**；`sid_stats.ok=true`、`n_items=11924`、`raw_duplicate_rate=0.000419322…`。
- **内存/进程**：`--dry-run` 未创建任何目录或记录。
- **哈希与 Git**：`qspace_train 305c50a7…`、`fixed_train 389e9085…`、`sid_refresh 983121cb…`、`alt_pipeline fc1b8652…`、`exp_l0_driver 244b71b6…` 全部未变；`git status --porcelain` = 基线 6 行。

##### 19.11.17.5 残余风险（不阻断 PASS）

| # | 风险 | 处理 |
|---|---|---|
| R-a | 记录仍只在整轮/异常点落盘，SIGKILL 丢 JSON | 阶段日志仍逐行落盘；正式运行后应核对 `driver_run.json` 存在 |
| R-b | 阶段 `status` 只反映子进程退出码 | 产物级失败看臂级 `failed_stage` + 顶层 `failure`（本次已验证） |
| R-c | `Popen` 的 `OSError` 经 `except OSError` 报为 rc=2 且提示 "filesystem error" | 非零退出 + 记录完整；措辞可后续改进 |
| R-d | `--run-id` 被 `--run-root` 目录名覆盖 | 仅影响书签 |
| R-e | `refresh_sid` 无超时 | 日志有阶段输出 |
| R-f | driver 不回读 `fixed_train` 的 `run.json` provenance | 训练器自身已做 manifest↔ckpt 校验 |
| R-g | §19.11.13 文档 `{ckpt_sha12}` 应为 `sha8` | 文档 typo，代码与 driver 均为 sha8 |

##### 19.11.17.6 判定

- 六项核查 **6/6 PASS**；两个必须修复项（M1/M2）证据完整、独立复现一致。
- **GATE = PASS**：可以进入 **Stage 0（EXP-L0）** 正式训练。
- **本次仍未启动 EXP-L0 / EXP-B1**；等待用户明确启动指令。

#### 19.11.18 Stage 0（EXP-L0）启动与完成（2026-10-19，**DONE / COMPLETED**）

**授权**：用户在 GATE = PASS 后明确授权启动。**本步骤未修改任何代码**；启动前生产四哈希 `qspace_train 305c50a7…` / `fixed_train 389e9085…` / `sid_refresh 983121cb…` / `alt_pipeline fc1b8652…`、driver `244b71b6…`，git HEAD `5edfca0`。

**命令**：`python -I integration/exp_l0_driver.py --quiet`（nohup 后台）。
**run root**：`integration/artifacts/exp_l0/runs/20261018T202823Z`（UTC 2026-10-18T20:28:23Z / CST 2026-10-19 04:28）。
**顺序**：`lc1e-4 → lc1e-3 → lc1e-2 → le0.1 → le1.0`；每条命令 = §19.11.13 冻结文本，仅 `--ckpt-dir / --run_dir / refresh base_dir` 为 run-scoped。
**记录**：整轮 `integration/artifacts/exp_l0/driver_run.json`；每臂 `<run_root>/<arm>/`（`ckpt/`、`sid/`、`run/`、`logs/`）。
**环境**：RTX 4090（启动前空闲）、`/data` 余 972G、torch 2.6.0+cu124。

**阶段日志**：`logs/tokenizer.log`、`logs/refresh.log`、`logs/recommender.log`；首臂实测 ~4.4 epoch/s（Toys，12 batch/epoch），3000 epoch ≈ 11–13 min。

**整轮结果（2026-10-19 10:38:23 CST 完成，wall clock 6 h 10 m）**：`driver_run.json` **`status=completed`**、`failure=null`、`arms_not_started=[]`、`arms_completed` = 五臂全列；15 个 stage 全部 `status=completed` / `exit_code=0`（CSG 臂各 3 个，ETE 臂各 1 个）。五个 `run.json` 的实际 sha256 与 driver 记录**逐一相等**。

| arm | 类型 | λ | best epoch | 执行 epochs | early stop | valid NDCG@20 | valid Recall@20 | test NDCG@20（仅记录） | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| lc1e-4 | CSG | λc=1e-4 | 96 | 107 | 是 | **0.040444** | 0.095454 | 0.033515 | 4020.8 s |
| lc1e-3 | CSG | λc=1e-3 | 94 | 105 | 是 | 0.039014 | 0.092334 | 0.032033 | 4081.6 s |
| lc1e-2 | CSG | λc=1e-2 | 66 | 77 | 是 | 0.037674 | 0.090261 | 0.031311 | 3016.4 s |
| le0.1 | ETE | λsia=λpsa=0.1 | 112 | 123 | 是 | 0.039269 | 0.093042 | 0.031531 | 4571.9 s |
| le1.0 | ETE | λsia=λpsa=1.0 | 113 | 124 | 是 | 0.033796 | 0.079700 | 0.028671 | 4667.8 s |

**过程核验**：CSG 三臂 tokenizer 日志的 `--lambda-csg` 分别为 `1e-4 / 1e-3 / 1e-2`，hook `n_calls=36000`、`n_nonempty=36000`、`n_disabled_calls=0` ⇒ CSG 真实进入训练；三条 SID 均为 `(11924, 4) int64`、`sid_stats.ok=true`，`raw_duplicate_rate` = 0.000587 / 0.000587 / 0.001761，sha 互不相同 ⇒ 三臂走的是不同 tokenizer 与不同 SID。ETE 两臂 `mounts.sia/psa=true`，λ 分别为 0.1 / 1.0，SID 为冻结 baseline（`sid_unchanged=true`）。

**结论边界（必须随记录携带）**：① 单 seed（CSG 侧 tokenizer seed 2024、T5 seed 2025；ETE 侧 seed 2025），**无方差、无显著性**；② CSG 与 ETE 是两条不同 pipeline（前者自训 tokenizer 并刷新 SID，后者用冻结 baseline tokenizer+SID），二者差异**不是单变量对照**；③ 本轮五臂**不含 λ=0 对照臂**；④ 表面趋势为 λ 越大越低（λc: 1e-4 > 1e-3 > 1e-2；λe: 0.1 > 1.0），best epoch 与执行轮数同步左移，仅可记为**筛选阶段观察**，不足以支撑效果结论。

#### 19.11.19 指标口径审计：EXP-L0 × Clean TIGER（2026-10-19，**只读 / 完成**）

**任务边界**：只读审计。**未修改任何代码/配置/产物，未重跑训练**；只做一次不落盘的 GPU 复算（历史评测路径，约 2 分钟）。权威文档 [doc/exp_l0_baseline_metric_audit.md](../doc/exp_l0_baseline_metric_audit.md)。

**核心发现**：**基线的 0.0341 / 0.0795 是 test 指标，而 EXP-L0 `run.json` 的 `best_ndcg@20` 是 valid 指标。** 因此 Stage 0 此前把 arm 的 valid 与 baseline 的 test 相比，属**口径错配**；由此得到的 `lc1e-4 +18.6%` **已撤回**。

**基线来源（本次新找到的一级证据）**：[model/logs/tiger_content768d_baseline_toys.log](../model/logs/tiger_content768d_baseline_toys.log) 含逐 epoch valid/test 记录。最佳 epoch = **76**（1-based，`early_stop=10`，共 86 epoch）：

| 口径 | NDCG@20 | Recall@20 |
|---|---|---|
| **valid**（选择用，此前从未记录） | **0.03980784884796207** | **0.09360632170083488** |
| **test**（报告用） | **0.03408397088203524** | **0.07945402289552642** |

单 seed 2025、单次运行。独立复算（`tools/eval_trained_model.py` 路径）与训练日志、eval 日志三方逐位一致。

**同口径差（test 对 test）**：lc1e-4 **−0.000569**、lc1e-3 −0.002051、lc1e-2 −0.002773、le0.1 −0.002553、le1.0 −0.005413。**没有任何一臂在 test 上超过基线**；lc1e-4 的差小于基线自身 3-seed 极差（0.0328–0.0341，跨度 0.0013）⇒ 只能判「无差异」。
**同口径差（valid 对 valid）**：lc1e-4 **+0.000636**、lc1e-3 −0.000794、lc1e-2 −0.002134、le0.1 −0.000539、le1.0 −0.006012。

**流程核对（代码证实）**：模型选择只用 valid（[tiger_ext/fixed_train.py:705-741](../tiger_ext/fixed_train.py#L705-L741)）；test **仅在 valid 改善时**评估并记录，`n_test_evals == n_best_saves`（46/47/41/46/50），**不参与任何控制流** ⇒ 无 test 调参泄漏。五臂共享 split、eval code、beam 30 / infer 96 / max_len 20 / topk [5,10,20]、early stop 10。

**可比性差异（不逐位等价，但不改变「无增益」判定）**：
- **T5 初始化**：基线 [model/main.py:260](../model/main.py#L260) 先建模型、[L265](../model/main.py#L265) 才 `set_seed` ⇒ 初始化未定种；EXP-L0 在构造前定种（[fixed_train.py:423](../tiger_ext/fixed_train.py#L423)）。
- **worker 数**：4（基线）vs 1（EXP-L0），shuffle 序列未逐位验证。
- **adapters**：始终构造并进 optimizer（103,200 参数），但 λ=0 时未挂载、无梯度、不入 `best.pth`（92 keys 与基线一致）⇒ λ=0 时 T5 计算图等价。
- **SID 空间**：CSG 三臂各自重训 tokenizer（seed 2024，3000 ep）并刷新 SID（sha 各异）；ETE 两臂用基线 tokenizer/SID（`8069f998…` / `bb2693a6…`）⇒ CSG 与 ETE 不是单变量对照。
- **valid/test 是不同任务**（valid 目标 `seq[-2]`、test 目标 `seq[-1]`），系统性 gap ≈ 0.006（−14%）⇒ valid 不能与 test 混用。

**Stage-0 规则影响**：在正确口径（test，Δ ≥ −0.001）下，CSG 仅 `lc1e-4` 合格 ⇒ **λc = 1e-4**；ETE 无合格臂 ⇒ **λe 回退锚点 0.1，并记录「未找到正向工作点」**。若改用 valid 对 valid，合格集为 {lc1e-4, lc1e-3, le0.1}，但 `lc1e-3` 的 L0 distinct 118/148 ≈ 80% 触发塌缩判据，排除后仍落在 λc=1e-4、λe=0.1。**「合格」只表示未劣化超过 0.001，不表示提升。**

**未确认清单**：① 基线启动脚本是否在外部设置过全局随机状态（`main.py` 自身没有）；② `..._toys.pth → ..._toys_seed2025.pth` 只有 mtime + 命名约定，无哈希记录；③ eval 日志不记录 `--code_path`（SID 归属由训练日志 config + 复现结果反推）；④ 多 seed 与配对显著性：EXP-L0 每臂仅 seed 2025 一次。

**对后续分析的含义**：CSG/ETE 的机制确实生效（码本/SID 变化、adapter 进入前向），但**效果层面没有可声称的增益**。若要写「各解决什么问题」，需要同 seed、同初始化口径的对照臂（当前**无 λ=0 的 fixed_train 臂**），不能直接对历史基线。
