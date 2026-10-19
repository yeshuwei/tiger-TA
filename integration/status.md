# CSG4Rec + ETEGRec → TIGER 移植状态

- **更新时间**：2026-10-19（**R13 门禁：修后配方逐项核对完成 → CONDITIONAL（B6）；同日晚 B6 已修复（EXP-B6 31/31）、B7 亦已修复（EXP-B7 48/48）、R1 已审计关闭（EXP-R1 44/44）、R2 已冻结关闭（EXP-R2 274/274）、R3 driver 已落（EXP-R3 105/105）；独立门禁复核完成 → CONDITIONAL（M1 空臂假成功 / M2 异常丢整轮记录，均在 integration/ 内，EXP-L0 未启动）；同日晚 M1/M2 已最小修复 → EXP-R3 **130/130**、独立 harness **REVIEW 46/46**；**最终独立复核 6/6 PASS ⇒ GATE = PASS，可进入 Stage 0（EXP-L0）**；**EXP-L0 已获用户授权并于 2026-10-19 04:28 CST 启动、10:38:23 CST 全部完成（`driver_run.json` status=completed，五臂 5/5，15 stage 全 exit 0）**（run root `integration/artifacts/exp_l0/runs/20261018T202823Z`）；2026-10-18 = R13 效果实验设计：口径改 768d、矩阵改 2×2×2、λ 先扫后定；此前 R12 = C6 CSG smoke test PASS（63/63）、R11 = C5、R10 = C4、R9 = C3、R8 = C2、R7 = C1、R6 = E9/E10、R5 = E8、R4 = E7、R3 = E1–E6 源码核验）
- **配套文档**：[plan.md](plan.md)（移植计划，见 §9 R3 / §10 R4 / §11 R5 / §12 R6 / §13 R7 / §14 R8 / §15 R9 / §16 R10 / §17 R11 / §18 R12 修订记录）、[experiments.md](experiments.md)（实验总表，见 §7–§16 修订记录）

---

## 当前状态

```
Phase 0        : DONE

Current Task   : C6（已完成，A-PASS）
M0.5           : DONE / CLOSED
E1-E10         : DONE / A-PASS
C1             : DONE / A-PASS
C2             : DONE / A-PASS
C3             : DONE / A-PASS
C4             : DONE / A-PASS
C5             : DONE / A-PASS
C6             : DONE / A-PASS（smoke，无效果结论）
P1             : DONE / ACCEPTED（差分回归零新增失败，见下「P1（ACCEPTED）」）
P2             : DONE / IMPLEMENTED，待验收（EXP-P2 47/47，见下「P2（IMPLEMENTED）」）
P2 后门禁      : **CONDITIONAL（维持）** —— 差分回归零新增失败（E9 87/90、E10 60/64）；
                 B1/B2/B3/B5 已于 2026-10-18 **修复**（EXP-P3 29/29、EXP-B3 30/30），
                 B4 同日修复（EXP-B4 38/38）；**修后配方核对 2026-10-19 已完成**：
                 λc 侧通过、λe 冻结 SID 逐位复算一致；**B6 已于 2026-10-19 晚修复**
                 （`embedding_path`/`--embedding_path`，EXP-B6 31/31），**B7 亦于同日修复**
                 （`apply_seed` + `seeded_kmeans`，EXP-B7 48/48）⇒ seed 真正控制随机过程。
                 R1 已于同日**审计关闭**（EXP-R1 44/44：7 个缺键的实际生效值全等于冻结值，
                 不改生产代码）；**R2 亦已于同日冻结关闭**（EXP-R2 274/274：五条命令逐参数
                 核对，不依赖任何 CLI 默认值）；**R3 driver 亦已落成**（EXP-R3 105/105：
                 五臂执行、独立 run root、动态产物识别、失败即停、driver_run.json）。
                 **独立门禁复核已于 2026-10-19 完成 → CONDITIONAL**：六项中 5 项 PASS，
                 安全项 FAIL 于两个**必须修复项**（M1 空臂假成功、M2 异常丢整轮记录），
                 均在 `integration/` 内、不涉生产代码与配方。
                 **M1/M2 已于 2026-10-19 修复**（EXP-R3 **130/130**、独立 harness **REVIEW 46/46**，
                 零新增失败；生产四哈希与 git 状态不变）；**同日晚独立复核 6/6 PASS ⇒ GATE = PASS**，
                 可以进入 Stage 0（EXP-L0）。**用户已于 2026-10-19 授权启动，
                 EXP-L0 五臂以 `exp_l0_driver.py` 在独立 run root 依次执行**。
                 见下「修后配方逐项核对」「B6 修复」「B7 修复」「R1 审计」「R2 冻结」
                 「R3 driver」「独立门禁复核」「M1/M2 最小修复」「最终门禁裁决」
B1/B2/B5 修复  : DONE / 待验收（只改 tiger_ext/qspace_train.py，EXP-P3 29/29）
B3 修复        : DONE / 待验收（只改 tiger_ext/sid_refresh.py，EXP-B3 30/30；E8 54/55 零新增失败）
B4 修复        : DONE / 待验收（只改 tiger_ext/fixed_train.py，EXP-B4 38/38；P2 47/47）
B6 修复        : DONE / 待验收（tiger_ext/fixed_train.py `389e9085…`；EXP-B6 31/31，P2 47/47，B4 38/38，E8 54/55）
B7 修复        : DONE / 待验收（tiger_ext/qspace_train.py `305c50a7…`；EXP-B7 48/48，P3 29/29，C4 84/87，C6 63/63）
R1 审计        : DONE / 待验收（只审计不改生产代码；EXP-R1 44/44，7 键实际生效值 21/21 = 冻结值；P3 29/29，P2 47/47，B6 31/31）
R2 冻结        : DONE / 待验收（五条命令逐参数冻结，不改生产代码；EXP-R2 274/274，P3 29/29，P2 47/47，B6 31/31）
R3 driver      : DONE / 待验收（`integration/exp_l0_driver.py` + EXP-R3 105/105；五臂独立 run root、动态产物识别、失败即停、driver_run.json；未启动 EXP-L0）
独立门禁复核   : DONE / **CONDITIONAL**（`integration/exp_r3_gate_review.py` 44/46；六项 5 PASS / 安全项 FAIL；M1 空臂假成功、M2 异常丢整轮记录；未启动 EXP-L0）
M1/M2 修复     : DONE / 验收通过（`integration/exp_l0_driver.py` +113/−41；EXP-R3 **130/130**、harness **REVIEW 46/46**、其余套件零新增失败；生产四哈希与 git 状态不变；未启动 EXP-L0）
最终门禁裁决   : **PASS**（2026-10-19 独立复核 6/6 PASS；历史失败 8 行逐条复现、零新增；失败注入独立重放一致；配方/输入/SID provenance/哈希/Git 全部核对）
Stage 0 (EXP-L0): **DONE / COMPLETED**（2026-10-19 04:28 CST 启动 → 10:38:23 CST 完成，
                 6 h 10 m；`driver_run.json` `status=completed`、`failure=null`、
                 `arms_not_started=[]`、五臂全列、15 个 stage 全部 exit 0；run root
                 `integration/artifacts/exp_l0/runs/20261018T202823Z`；执行顺序
                 lc1e-4 → lc1e-3 → lc1e-2 → le0.1 → le1.0；启动前 GPU 空闲、
                 972G 磁盘可用、生产四哈希与 git 状态已存档 `/tmp/exp_l0_pre_*.txt`；
                 进度 5/5：lc1e-4/lc1e-3/lc1e-2/le0.1/le1.0 全部完成；
                 五臂 `run_summary.json` 已用只读脚本 `integration/summarize_l0_run.py`
                 全部生成，五个原文件均未改动）
EXP-B1         : NOT STARTED

Next Task      : EXP-L0 五臂结果待用户审查（valid NDCG@20：lc1e-4 0.040444 / le0.1
                 0.039269 / lc1e-3 0.039014 / lc1e-2 0.037674 / le1.0 0.033796；
                 单 seed、无显著性、无 λ=0 对照臂，只能作筛选观察）；EXP-B1 需另获授权；
                 E8/E9/E10 的历史 guard 失败仍须如实随记录归档，不得描述为全部通过

Model modification : YES  — model/main.py      (E1/E4/E6 埋点，默认关闭)
                           rqvae/models/rq.py  (E3/E4 惰性开关，默认关闭)
                           rqvae/models/vq.py  (E3/E4 惰性开关，默认关闭)
                           rqvae/trainer.py    (C5 授权挂载，csg=None 时逐位等价)
                           + 新增 tiger_ext/   (SIA/PSA + E7 阶段控制层 + E8 SID 版本层
                                                + E9 交替调度层 + C1 CF embedding 接口
                                                + C2 collaborative relation
                                                + C3 Q-space 约束构造层
                                                + C4 Q-space 约束目标层
                                                + C5 Q-space 训练接入层
                                                + C6 全链 smoke 驱动层，
                                                均在冻结内核之外；
                                                E7–E10/C1–C5 未改内核，C6 未新增 tracked 改动)
Experiment         : A 类工程/机制验证 16/16 通过（含 EXP-C6 63/63）
                     推荐效果实验 0 执行（EXP-L0 / EXP-B1 / 8 格全部 PLANNED）

Baseline       : 5edfca08b22f23a2e965f5e21062adb53e1214b

Known issues   : 4 项（Toys 4096d RQ-VAE；SID 设备相关；**C5 授权修改内核文件后
                 既有 harness 的 tracked-file guard 必然失败**；**k-means 初始化 ×
                 batch 几何会吃掉 RQ-VAE 自己的码本梯度**——第 4 项已由 C6 裁决
                 为沿用 `kmeans_init=True` + 每个首 batch 都要量），见下方
                 「已知问题 / 待决」（另有 2 条**已查实但非阻塞**的复现性事实：
                 冻结 k-means 助手跨进程不可复现（第 6 条）；E10 的 CSG 守卫对 C4
                 假阳性、已靠改名关闭（第 7 条）；以及 C6 新登记的 2 条限制：
                 MI 不可作真实臂（第 11 条）、Toys 欠训练的具体读数（第 12 条））
```

> **E1–E10 与 C1 / C2 / C3 / C4 / C5 / C6 已完成实现，并已通过 ETEGRec / CSG4Rec 官方源码核验（R3–R10）+ 全部 A 类回归验证（含 E7 阶段契约、E8 SID 刷新验收、E9 交替调度、E10 集成 smoke、C1 CF embedding、C2 collaborative relation、C3 Q-space 约束构造、C4 约束目标函数、C5 训练接入、C6 全链 smoke）。**
> **工程侧已收口。下一步是效果实验，设计已冻结在 R13**（[plan.md §19](plan.md)）：768d content-only 口径、2×2×2 全因子 8 格、λ 先扫后定、B 类 1 数据集 × 1 seed 筛选。
> **推荐效果仍未验证**——C6 的 Recall/NDCG **只是"链路跑通"的证据**（Toys 臂因 4096d 欠训练全为 0）；**没有一个可信的效果读数**，全部 B / C 类效果实验仍为 `PLANNED`（见 [experiments.md §3](experiments.md)）。

---

## 字段说明

| 字段 | 含义 | 当前值 |
|---|---|---|
| **Phase 0** | Clean TIGER 的隔离、验证、冻结 | `DONE` — commit `5edfca0` |
| **Current Task** | 本次正在做的 Task | `C5` — 已完成（`A-PASS`） |
| **M0.5** | 管理文档修订（R1 接口加固 → R2 解环并冻结） | `DONE / CLOSED` |
| **E1–E10** | ETEGRec SIA / PSA / 阶段冻结 / SID refresh / 交替调度 / 集成 smoke 移植实现 | ✅ `DONE / A-PASS`（484 项断言） |
| **C1** | CSG4Rec CF embedding 接口 | ✅ `DONE / A-PASS`（132 项断言） |
| **C2** | CSG4Rec collaborative relation（cosine top-k kNN） | ✅ `DONE / A-PASS`（127 项断言） |
| **C3** | CSG4Rec Q-space collaborative constraint（**construction half**） | ✅ `DONE / A-PASS`（99 项断言；`lambda_csg` 默认 0，不构造目标函数） |
| **C4** | CSG4Rec Q-space collaborative constraint objective（**objective half**） | ✅ `DONE / A-PASS`（87 项断言；**standalone**，未接入训练循环） |
| **C5** | CSG trainer integration（`L_total = L_RQVAE + λ·L_csg` 接入冻结训练循环） | ✅ `DONE / A-PASS`（`lambda_csg` 默认 0；`csg=None` 时逐位等价；**唯一被改的正式文件 = `rqvae/trainer.py`**） |
| **Next Task** | 下一个要开工的 Task | `C6`（CSG smoke test）或效果实验 **EXP-C5/C6**（B + C 类） |
| **Model modification** | 是否已改动 `model/` / `rqvae/` | `YES` — `model/main.py`、`rqvae/models/rq.py`、`rqvae/models/vq.py`（均只加**默认关闭**的可选分支）、**`rqvae/trainer.py`（C5 授权的受守卫挂载，+34/−5，7 hunks）**；**E7 / E8 / E9 / E10 / C1 / C2 / C3 / C4 均未新增任何内核改动** |
| **Experiment** | 是否已跑过任何实验 | `A 类 14/14 通过`；**无推荐效果训练 / 评估运行**（E9/E10/EXP-C5 内部跑的是 fixture / sanity 规模的 tokenizer 训练，**不是**效果实验），无 Recall / NDCG |
| **Baseline** | 冻结基线 commit | `5edfca08b22f23a2e965f5e21062adb53e1214b` |
| **Known issues** | 已知阻塞问题 / 待决 | 4 项，见「已知问题 / 待决」（第 3 / 5 / 8 / 9 条） |

---

## Task 进度总览

| Task | 名称 | 状态 |
|---|---|---|
| M0 | Clean TIGER 隔离 / 验证 / 冻结 | ✅ DONE |
| **M0.5** | **接口移植计划 + 阶段性实验验证计划**（R1 接口加固 → R2 解环并冻结） | ✅ **DONE / CLOSED** |
| E1 | T5 hidden-state interface | ✅ DONE（A-PASS 15/15） |
| E2 | SIA adapter | ✅ DONE（A-PASS 29/29 + 29/29） |
| E3 | SIA code-distribution extraction | ✅ DONE（A-PASS 39/39）｜B 部分 PLANNED |
| E4 | SIA loss | ✅ DONE（机制 A-PASS 43/43）｜**推荐效果 PLANNED** |
| E5 | PSA decoder interface | ✅ DONE（机制 A-PASS 54/54）｜**推荐效果 PLANNED** |
| E6 | PSA loss | ✅ DONE（机制 A-PASS 54/54）｜**推荐效果 PLANNED** |
| E7 | Tokenizer / recommender freeze interface | ✅ DONE（A-PASS 66/66，CPU + CUDA）｜**EXP-E7-E9 效果 PLANNED** |
| E8 | SID refresh | ✅ DONE（A-PASS 55/55，CPU + CUDA；含真实生成）｜**EXP-E7-E9 效果 PLANNED** |
| E9 | Alternating cycle | ✅ DONE（A-PASS 90/90，CPU + CUDA，含 2 真实 cycle）｜**EXP-E7-E9 效果 PLANNED** |
| E10 | ETEGRec integration smoke test | ✅ DONE（A-PASS 64/64，CPU + CUDA；全链 2 cycle + G0/G1 门禁） |
| C1 | CF embedding interface | ✅ DONE（A-PASS **132/132**，CPU + CUDA；正式源 = **SASRec item embedding 50d**） |
| C2 | Collaborative relation | ✅ DONE（A-PASS **127/127**，CPU + CUDA；k=20 有向 cosine top-k）｜**关系无效果结论** |
| C3 | Q-space collaborative constraint（construction half） | ✅ DONE（A-PASS **99/99**，CPU + CUDA；`lambda_csg` 默认 0）｜**约束构造不产生 loss** |
| C4 | Q-space collaborative constraint objective（objective half） | ✅ DONE（A-PASS **87/87**；**standalone**，无 trainer 接入）｜**loss 尺度不构成效果结论** |
| C5 | CSG trainer integration | ✅ DONE（A-PASS；`L_total = L_RQVAE + λ·L_csg` 实测进入 backward；G0 逐位等价）｜**唯一内核改动 = `rqvae/trainer.py` 的受守卫挂载**｜**无效果结论** |
| C6 | CSG smoke test | ✅ DONE（A-PASS **63/63**，CPU；四臂 λ=0/1e-4/1e-3/1e-2 + Toys λ=0/1e-3 全链跑通；λ=0 与 Clean TIGER 全链逐位等价；`kmeans_init` 裁决 = 维持冻结 `True`）｜**未新增 tracked 改动**｜**Recall/NDCG 仅 smoke，无效果结论** |
| I1–I4 | 组合实验 | ⬜ 未开始 |

图例：✅ 完成 ｜ 🔄 进行中 ｜ ⬜ 未开始 ｜ ⚠️ 阻塞

---

## 边界状态（E1–E10 + C1 + C2 + C3 + C4 + C5 + C6 结束时）

| 目录 | 本阶段是否被修改 | 说明 |
|---|---|---|
| `model/` | ✅ 已改（**加法**） | `main.py`：`forward(return_hidden=False)`；`train(..., sia=None, psa=None)`。默认关闭时逐位等价 |
| `rqvae/` | ✅ 已改（**加法**） | `models/vq.py`、`models/rq.py`：`return_logits=False` / `logit_temperature` 惰性开关。默认关闭时逐位等价；**`trainer.py`：C5 授权的受守卫挂载（`Trainer(..., csg=None)`，`+34/−5`，7 hunks，`sha256(diff)[:16] = a8da64e3291fa1c5`），`csg=None` 时逐位等价** |
| `clean_tiger/` | ❌ NONE（只读） | 仅读取其 smoke 产物作为验证 fixture，未写入 |
| `tools/` | ❌ NONE | — |
| `doc/` | ❌ NONE | — |
| `integration/` | ✅ 管理文档 + 验证脚本 + `artifacts/cf_embed/` + `artifacts/collab_relation/` | C4 只新增 `exp_c4_qspace_loss.py`，未新增 artifact 目录（C4 不产出数据）；**C5 只新增 `exp_c5_csg_integration.py`**，同样不产出 artifact；**C6 只新增 `exp_c6_csg_smoke.py`**，不产出 repo 内 artifact（运行期产物落在 `tempfile.mkdtemp()`） |
| `tiger_ext/` | ✅ 新增 | 移植模块（SIA / PSA）+ E7 阶段控制层 + E8 SID 版本层 + E9 交替调度层 + **C1 CF embedding 接口** + **C2 collaborative relation** + **C3 Q-space 约束构造层** + **C4 Q-space 约束目标层** + **C5 Q-space 训练接入层（`qspace_train.py`）** + **C6 全链 smoke 驱动层（`qspace_smoke.py`）**，均不在冻结内核内 |

> `model/` 与 `rqvae/` 的改动面已由 R1 修正过口径（E1 不是唯一改 `model/` 的 Task）。
> **E7 至 C4 的 8 个 Task 全部零内核改动**：`git status --porcelain` 自 E6 结束起逐字未变（仍只有 `model/main.py`、`rqvae/models/rq.py`、`rqvae/models/vq.py` 三项 E1–E6 期间的既有改动）。C4 额外用 **6 个 sha256 pin** 证明 `cf_embed.py` / `collab_relation.py` / `qspace_constraint.py` / `stage.py` / `psa_loss.py` / `sia_loss.py` 逐字节未变（`tiger_ext/` 是 untracked，git 看不见其中改动）。
> **⚠️ C5 是第一次打破"零内核改动"**：`rqvae/trainer.py` 加入第 4 个被修改的 tracked 文件。这是 plan.md §C5 的明文授权（"仅受守卫的最小挂载"），并由 EXP-C5 §[8] 的 25 项机器核验钉死；**不是**回归，但会让所有以 `PRE_*_MODIFIED` 为判据的既有 harness 报 guard FAIL（见「已知问题」第 8 条）。
> **C6 未再扩大改动面**：`git status --porcelain` 与 C5 结束时逐字相同，`sha256(difflib.unified_diff(..., lineterm=''))[:16]` 仍是 `a8da64e3291fa1c5`、改后文件 `sha16 = d09ad776891dd3b2`。EXP-C6 §[1] 直接核验这一点，并重放 E10 的守卫 grep（`tiger_ext/ model/ rqvae/` 零命中）。

| 动作 | 是否执行 |
|---|---|
| E1（T5 hidden 改造） | ✅ 已执行 |
| C1（CF embedding 接入） | ✅ 已执行（正式源 = SASRec item embedding 50d；ItemCF-128 保留为备用源） |
| C2（collaborative relation） | ✅ 已执行（k=20 有向 cosine top-k；**numpy 叶子，不接入任何人的输入**） |
| C3（Q-space 约束构造） | ✅ 已执行（**construction half**：capture → pair → gather → `ConstraintBatch`；**无 loss、无 backward、无参数**） |
| C4（Q-space 约束目标） | ✅ 已执行（**objective half**：`ConstraintBatch` → `L_csg` **标量** + `QSpaceLossMount`；**standalone，未接训练循环 / 未改 E7 / 0 新参数**） |
| C5（Q-space 训练接入） | ✅ 已执行（`L_total = L_RQVAE + λ·L_csg` 实测进入 backward；**唯一内核改动 = `rqvae/trainer.py` 的受守卫挂载**；`csg=None`/`λ=0` 逐位等价；**未改 E7、未改 C1–C4、未新增优化器/warmup/正则**） |
| C6（全链 smoke） | ✅ 已执行（四臂合成 + 两臂 Toys 走完 `tokenizer → refresh → recommender → evaluation`；**未新增任何 tracked 改动、未改 alt_pipeline / E7 / E10 / 既有 harness**） |
| 模型训练 | ⚠️ **仅 fixture / sanity / smoke 规模**（E9/E10 内部是合成 fixture；EXP-C5 §[4]–[5] 与 EXP-C6 在 Toys 上分别跑 20 / 24 步 tokenizer + `limit_train=64`）。**推荐效果训练 / 评估仍未执行** |
| CSG4Rec 移植 | ✅ C1 / C2 / C3 / C4 / C5 / C6 完成（C6 = smoke，无效果结论） |
| ETEGRec 移植 | ✅ E1–E10 全部完成 |
| 复制源项目代码 | ❌ 无（仅**只读**参考 `reference_repos/ETEGRec_original`、`reference_repos/CSG4Rec_original`） |

---

## E1–E10 + C1 + C2 + C3 + C4 实现与源码核验（R3–R10，2026-10-14）

**实现产物**

| Task | 产物 |
|---|---|
| E1 | `model/main.py::TIGER.forward(return_hidden=)`；验证 `integration/exp_e1_consistency.py` |
| E2 | `tiger_ext/sia.py`（masked mean pool + `Linear(128,32)`）；验证 `exp_e2_encoder_interface.py`、`exp_e2_sia_adapter.py` |
| E3 | `rqvae/models/{rq,vq}.py` 可选 logits 输出；`tiger_ext/sia_codes.py`；验证 `exp_e3_code_distribution.py` |
| E4 | `tiger_ext/sia_loss.py`（symmetric KL）；`model/main.py::train(sia=)`；验证 `exp_e4_sia_loss.py` |
| E5 | `tiger_ext/psa.py`（位置 0 + `Linear(d_model, D_target)`） |
| E6 | `tiger_ext/psa_loss.py`（symmetric contrastive + `PSAMount`）；`model/main.py::train(psa=)`；E5/E6 验证 `exp_e5_psa.py` |
| E7 | `tiger_ext/stage.py`（`StageFreeze` 阶段控制层 + 码本封存 + 阶段清单）；验证 `exp_e7_stage_freeze.py`。**未改动 `model/` / `rqvae/`** |
| E8 | `tiger_ext/sid_refresh.py`（SID 版本化命名 + provenance manifest + 原子提交，**只驱动** `rqvae/generate_code.py` 的既有 CLI）；验证 `exp_e8_sid_refresh.py`。**未改动 `clean_tiger/` / `model/` / `rqvae/`** |
| E9 | `tiger_ext/alternating.py`（纯调度状态机，**不 import torch/numpy**）+ `tiger_ext/alt_pipeline.py`（4 个 phase 体，全部调用冻结代码）+ `exp_e9_alternating.py`。**未改动 `clean_tiger/` / `model/` / `rqvae/`** |
| E10 | `integration/exp_e10_smoke.py`（**唯一产物，0 个既有文件被改**） |
| C1 | `tiger_ext/cf_embed.py`（`CFEmbedding` 接口 **+** SASRec / ItemCF 两个加载器）+ `integration/exp_c1_cf_embedding.py` + `integration/artifacts/cf_embed/*.json`。**既有 tracked 文件零改动** |
| C2 | `tiger_ext/collab_relation.py`（`CollabRelation` 接口 **+** top-k cosine 构建器；复用 C1 的 `canonical_item_ids` / `item_ids_sha256` 以免指纹漂移）+ `integration/exp_c2_collab_relation.py` + `integration/artifacts/collab_relation/<D>_k20.*`。**既有 tracked 文件零改动** |
| C3 | `tiger_ext/qspace_constraint.py`（`QSpaceBatch` capture + `QSpaceConstraint` 挂载 + `ConstraintBatch` 交付物；**无 loss、无 backward、无参数**）+ `integration/exp_c3_qspace_constraint.py`。**既有 tracked 文件零改动**，不新增 artifact 目录 |
| C4 | `tiger_ext/qspace_loss.py`（`QSpaceLoss` / `QSpaceLossMount` / `qspace_constraint_loss` / `codebook_stack` / `verify_indices_match_qspace` + 14 个 `QSL_*` 记录常量；**standalone，未接训练循环**）+ `integration/exp_c4_qspace_loss.py`。**既有 tracked 文件零改动**，不新增 artifact 目录 |

**验证结果（A 类工程 / 机制验证）**

| 脚本 | CPU | CUDA |
|---|---|---|
| `exp_e1_consistency.py` | 15/15 | — |
| `exp_e2_encoder_interface.py` | 29/29 | — |
| `exp_e2_sia_adapter.py` | 29/29 | — |
| `exp_e3_code_distribution.py` | 39/39 | 39/39 |
| `exp_e4_sia_loss.py` | 43/43 | 43/43 |
| `exp_e5_psa.py` | 54/54 | 54/54 |
| `exp_e7_stage_freeze.py` | 66/66 | 66/66 |
| `exp_e8_sid_refresh.py` | 55/55 | 55/55 |
| `exp_e9_alternating.py` | 90/90 | 90/90 |
| `exp_e10_smoke.py` | 64/64 | 64/64 |
| **E 线合计** | **484/484 通过** | |
| `exp_c1_cf_embedding.py` | 132/132 | 132/132 |
| **C 线合计（C1）** | **132/132 通过** | |
| `exp_c2_collab_relation.py` | 127/127 | 127/127 |
| `exp_c3_qspace_constraint.py` | **99/99** | 99/99（含 4 项 CUDA 专项） |
| `exp_c4_qspace_loss.py` | **87/87** | —（C4 未做 CUDA 专项；§[6b] 用真实三数据集 + 真实已训 ckpt） |
| **总计（E + C）** | **929/929 通过** | |

> 每个 Task 落地后按工程惯例**重跑此前所有门**：E7 落地后重跑 E1–E6（209 项），E8 后重跑 E1–E7（275 项），E9 后重跑 E1–E8，E10 后重跑 E1–E9（411 项），**C1 后又重跑 E1–E10**（`exp_c1_cf_embedding.py` 的 §8 直接驱动 `exp_e10_smoke.py`，64/64），**C2 后再次重跑 E1–E10**（`exp_c2_collab_relation.py` 的 §8 同样直接驱动 `exp_e10_smoke.py`，64/64），**C3 后第三次重跑 E1–E10**（`exp_c3_qspace_constraint.py` 的 §9 以子进程驱动 `exp_e10_smoke.py`，64/64，exit 0），**C4 后第四次重跑 E1–E10**（`exp_c4_qspace_loss.py` 的 `[10]` 同样以子进程驱动 `exp_e10_smoke.py`，64/64，exit 0）——均再次全部通过（上表为 C4 之后的复跑结果）。
> ⚠️ E10 内层给 E9 设 `EXP_E9_REGRESSION=skip`，故 E10 里 E9 报 **81/81** 而非独立跑的 **90/90**；两者相加**无重复计数**。

**原始 TIGER 回归**：`loss` / `logits` / `generate()` 对 `5edfca0` **逐位相同**（`torch.equal`，不接受 allclose）；`lambda_sia = 0` 与 `lambda_psa = 0` 时 epoch loss 与全部参数梯度对 Clean TIGER 逐位相同，且挂载点 0 次调用。**通过。**

**源码核验修正（R3，详见 [plan.md §9](plan.md)）**

| 编号 | 位置 | 修正 |
|---|---|---|
| R3-1 | `tiger_ext/psa_loss.py` | PSA similarity：裸点积 → **cosine** |
| R3-2 | `tiger_ext/psa_loss.py` | PSA temperature 默认 1.0 → **0.07** |
| R3-3 | `tiger_ext/psa_loss.py` | PSA 对称项 `0.5*(CE1+CE2)` → **`CE1+CE2`** |
| R3-4 | `rqvae/models/vq.py` | SIA code logits `-raw_d` → **`+raw_d`** |

核验同时确认：E6 的"靶 = RQ-VAE 重建"与 E5 的"decoder 最后一层序列位置 0"**均与官方实现一致**。

> **⚠️ 上表全部为"实现正确 / 机制忠实 / 未破坏原始行为"的工程结论。**
> **SIA 与 PSA 的推荐效果（Recall / NDCG）尚未验证，本项目目前没有任何效果数据。**

**未复刻的官方行为（R3 记录时 3 处）**：batch 内按 target 去重、`warm_epoch=10` 预热、rec/id 相位 `requires_grad` 交替切换。
→ **R4 更新**：其中第 3 处（rec/id 相位的 `requires_grad` 切换）已由 **E7** 实现（`StageFreeze`）；剩 **2 处**仍不复刻（去重、`warm_epoch`），相位**调度**（`epoch_idx % cycle`）归 E8 / E9。

---

## E7 实现（R4，2026-10-14）

**阶段契约**（[plan.md §1.4](plan.md) / E7 表，= 实现契约）：

| 组件 / loss | Tokenizer 阶段 | Recommender 阶段 |
|---|---|---|
| RQ-VAE | **trainable** | **frozen** |
| T5 | **frozen** | **trainable** |
| adapter（E2 / E5） | frozen | **trainable** |
| `L_RQ-VAE` | **生效** | 不生效 |
| `L_T5` / `L_SIA` / `L_PSA` | 不生效 | **生效** |

**实现方式**：控制层 `tiger_ext/stage.py::StageFreeze` 只做两件事——① 按契约显式重置各组 `requires_grad`；② `mounts_for_stage()` 在**调用方**重置 loss 开关（Tokenizer 阶段把 `sia` / `psa` 置 `None`，Recommender 阶段原样透传）。**不重写任何训练循环**，`model/` 与 `rqvae/` **零改动**（两侧训练循环不导入控制层，控制层也不导入它们）。

**实测关键判定（EXP-E7，66/66，CPU + CUDA）**：

| 判定 | 结果 |
|---|---|
| Tokenizer 阶段：T5 / adapter 逐位不动 | T5 **0 张量**、adapter **0 张量**移动；RQ-VAE **23/23** 移动 |
| Recommender 阶段：RQ-VAE 逐位不动 | **0/23** 张量移动，epoch 末梯度**全为 `None`**（修复前：λ>0 时 SIA 10 个 encoder 张量 + 3 码本、PSA 20/23 张量会被更新） |
| "frozen ≠ `eval()`" | RQ-VAE **前向仍在执行**（encoder 8 次 / decoder 4 次）——PSA 目标侧需要重建；`L_RQ-VAE` 作为**训练目标**关闭与其前向被调用不矛盾 |
| 验收 ⑤（参数状态 × loss 状态一致） | `assert_consistent()` 双重检查：冻结组内无 trainable 参数，且生效 loss 所需的组未被冻结（`LOSS_REQUIRES`） |
| `vq.initted` 坑 | **负对照成立**：裸 `load_state_dict` + train 模式前向**确实**被惰性 k-means 覆盖码本；`load_tokenizer_state_dict` 后同一前向**逐位不变**；`seal_codebooks(strict=True)` **拒绝**封存全零（从未初始化）码本 |
| 优化器覆盖 | 一个 Adam 实例跨阶段复用即可（覆盖两组参数，冻结参数被跳过、不建 state），**无需重建优化器** |
| checkpoint / resume | `<ckpt>.stage.json` 记录冻结切分 + loss 开关 + 码本 `initted` 状态；`verify_stage_manifest()` **拒绝跨阶段**或冻结切分不一致的 resume |
| 回归（E7 未启用 / λ=0） | loss / logits / `generate()` 对 `5edfca0` **逐位相同**（`torch.equal`） |

**两条集成事实（E9 / E10 必须遵守）**：

1. **adapter 不在 `TIGER.parameters()` 内** ⇒ Recommender 阶段优化器必须**显式**把 adapter 加进去；冻结的 `model/main.py::__main__` 入口 `optim.Adam(model.parameters())` **不会**训练 adapter。
2. Recommender 阶段必须是"**冻结但执行**"，不得把 RQ-VAE 切到 `eval()` 或跳过其前向——PSA 的目标侧依赖它。

**与 ETEGRec 官方的差异（以本计划为准，见 [plan.md §10](plan.md)）**：官方 `_train_epoch_id` 在 Tokenizer 阶段**仍计算** `kl_loss` / `dec_cl_loss`（用 `id_*` 权重）；[plan.md §1.4](plan.md) 规定 `L_SIA` / `L_PSA` **仅** Recommender 阶段生效，本实现**从计划**。另：相位调度（`epoch_idx % cycle`、`warm_epoch=10`）属 E8 / E9，E7 只提供阶段机。

> **⚠️ 全部为工程结论。**E7 只验证阶段契约（冻结范围、梯度流、loss 生效、码本状态、checkpoint 恢复、回归），**不含任何推荐效果结论**。

---

## E8 实现（R5，2026-10-14）

**做法**：`tiger_ext/sid_refresh.py` **不自带任何编码机制**——它通过**既有 CLI** 驱动冻结的 `rqvae/generate_code.py`，在其上加两样东西（正是 [plan.md](plan.md) E8 表「需要新增什么」要求的）：**版本化命名** `<D>_sid_v<NNNN>_<ckpt_sha8>.npy` 与 **provenance manifest** `<sid>.sid.json`（沿用 E7 的 sidecar 约定）。`clean_tiger/`、`model/`、`rqvae/` **零改动**。

**验收结果（EXP-E8，55/55，CPU + CUDA；含真实生成 11924 item / 4096-d / 真实 ckpt）**：

| 验收 | 结果 |
|---|---|
| ① 覆盖全部 item | 11924 行 == embedding 行数（从 ckpt 的 `data_path` **独立**读出，不信任生成器自报） |
| ② 保留原有 collision handling | `generate_code.py` 逐字未改；`use_sk` 去重循环后**零重复行**（11924/11924 唯一；前置 3 列重复率 3.92%，最大组 21 → 第 4 列最大值 20） |
| ③ token ∈ [0,256) | min 0 / max 255；**第 4 列是 duplicate 计数器**，其上界是"最大重复组大小"（非结构性约束，已在 manifest 记录实测值） |
| ④ 与上一轮差异率 | 同设备 **= 0**；跨设备 **3.25%**（见下） |
| ⑤ ckpt ↔ SID 可回溯 | manifest 记 ckpt sha256、generator sha256、repo commit+dirty、时间、**device**、ckpt 训练参数、生成命令；`verify_sid_manifest()` 拒绝 ckpt 字节不符 / SID 被改动 / SID 缺失 |

**实测关键事实（E9 必须知道）**：

| 事实 | 内容 |
|---|---|
| **复用无偏差** | CUDA 上重新生成与 clean_tiger 基线产出的 SID **逐位相同**（11924/11924）——证明"复用既有流程"确实没有引入新编码 |
| **SID 与设备相关** | 同设备逐位确定性；**CPU vs CUDA 差 3.25%（388/11924）**，差异**只落在第 2、3 列**（最后一层 + collision 列，SK 路径对边界 item 的判定）⇒ **diff rate 必须对照 3.25% 这个噪声底**，低于它的差异不能解读为码本变化；manifest 因此记录 device |
| **`generate_code.py` 不能在 `python -I` 下运行** | 它 `from datasets import EmbDataset`，隔离模式会命中 HuggingFace 的 `datasets` 包（且 `-I` 丢弃 `PYTHONPATH`）；唯一可靠调用是纯解释器 + `cwd=rqvae/` 或 `PYTHONPATH=rqvae` |
| **成本** | 单次全量 SID 生成 ≈ **18 s**（Toys 11924 item，CPU 与 CUDA 相近） |

**覆盖写入 / 失败态 / 交接**：版本化路径**永不原地覆盖**（版本号自动递增；同版本冲突一律拒绝）；生成失败 → 抛 `SidRefreshError`，**不留 SID、不留 sidecar、不留临时文件**，且不触碰上一轮的好版本；`publish=...` 把新版本**原子**交付到 T5 读的规范路径，**默认关闭**——**谁在什么时机发布**是 E9 的调度职责。

> **⚠️ 全部为工程结论。**E8 只验证"SID 能被正确、可追溯地重新生成"，**不含任何推荐效果结论**。

---

## E9 实现（R6，2026-10-14）

**做法**：`tiger_ext/alternating.py` 是**纯编排层**（状态机 + 周期计划 + handoff 校验，**不 import torch/numpy**），4 个 phase 体由调用方注入 ⇒ 分支 / 失败 / 恢复 / handoff 逻辑可在**零训练**下用 fake phase 测；真实链路走 `tiger_ext/alt_pipeline.py`，它只**调用** E7 阶段机 + E8 refresh + 两个**冻结**训练循环。`model/` / `rqvae/` / `clean_tiger/` **零改动**。

**周期规则（关闭「已知问题」原第 4(b) 条）**：1 phase = 1 epoch，1 cycle = `1 Tokenizer + 1 Recommender`；相位判定用 **0-based** `epoch_idx % cycle == 0`（与 ETEGRec 官方 `trainer.py:513/542`、`config/scientific.yaml` 的 `cycle: 2` 一致）。**1-based 读法会把 T/R 整体反相**，脚本内有对照断言钉住这一点。

**验收结果（EXP-E9，90/90，CPU + CUDA，含 2 个真实交替周期）**：

| 验收 | 结果 |
|---|---|
| ① ≥2 个完整交替周期 | ✅ 2 cycle 端到端跑通（`Tokenizer → Refresh → Recommender`）；产物链含不可变 SID 版本 v1/v2/v3 |
| ② ckpt / SID / 指标链条可追溯 | ✅ 每 cycle 的 `.stage.json` 阶段正确；SID sidecar 的 `ckpt_sha256` 指向**本 cycle** 的 tokenizer；evaluation 只挂在 recommender epoch |
| ③ 中途中断可续跑 | ✅ 四状态机：`started`→重跑 Tokenizer；`tokenizer_done`→Refresh+Recommender；`refresh_done`→**直接 Recommender（Refresh 不重复）**；`recommender_done`→下一 cycle。恢复前 `_verify_unchanged` 校验 ckpt/SID 的 sha256，被篡改即抛 `ResumeError` |
| ④ 随机性有种子控制 | ✅ 种子按 phase 派生；**同 seed 重跑 SID 逐位一致** |

**实测关键判定**：

| 判定 | 结果 |
|---|---|
| 阶段冻结（与 E7 契约一致） | Tokenizer 阶段 RQ-VAE **15/15** 张量动、T5 / 双 adapter **0 张量**动；Recommender 阶段 RQ-VAE **0/15** 动且 epoch 末**0 个冻结参数带梯度**（前向仍执行，是 PSA 靶侧），T5 26/26 + adapter 4 张量动 |
| SID handoff 证法 | `GenRecDataset` 把 item `i+1` 映射成 `code + level*256 + 1`，两个 mount 的 `_lookup` 用**同一套 offset 键** ⇒ `verify_sid_consistency` 逐 item 验 dataset↔SIA↔PSA 三方路由一致 + 与 manifest 的 version/sha256 相符（用别的 SID 建的 dataset/mount **立刻失败**，负对照已实测） |
| refresh 失败 | 抛 `RefreshAborted`，state 停在 `tokenizer_done`，**Recommender 绝不运行**；不留残留 |
| 从不 `publish` | canonical SID 是冻结基线产物；交接靠每 cycle 的 `--code_path` 指向**不可变版本文件**（E8 的 `publish` 始终保持默认关闭） |

**三处必须上报的偏离**：① `PHASE_WARMUP_EPOCHS = 0`（冻结 `Trainer` 每 phase 重建 optimizer + linear schedule，50-epoch warmup 会超过单 phase 的 step 预算；实际值已记入 `run.json`）；② Tokenizer 优化器**归 `rqvae/trainer.py` 所有**（该文件冻结），E9 不改它而是**断言** `trainer.optimizer` 的参数 id 集合 == `freeze.trainable_parameters('tokenizer')`；Recommender 优化器才由 E9 用 `trainable_parameters('recommender')` 显式建（含 T5 ∪ adapter）；③ **initial SID refresh 在本 port 没有训练消费者**（Tokenizer 阶段训 RQ-VAE 于固定 content embedding，从不读 SID）——仍按规格执行，作为**版本链根 + cycle-0 的 diff 基线**。

> **⚠️ 全部为工程结论。**E9 只验证"交替调度能正确、可续、可追溯地跑"，**不含任何推荐效果结论**（EXP-E7-E9 仍 `PLANNED`）。

---

## E10 实现（R6，2026-10-14）

**做法**：`integration/exp_e10_smoke.py` 是**唯一产物**，**0 个既有文件被改**（`git status` 与 E9 结束时逐字相同）。5 个部分：探针装置 + 阴性对照 / G0 非同源逐位对比 / 全链 2 cycle + G0/G1 / 边界 / 回归。

**验收结果（EXP-E10，64/64，CPU + CUDA，exit 0）**：

| 验收 | 结果 |
|---|---|
| ① 全链 exit 0 | ✅ `E10Syn` fixture（1024 item × 32d，8 簇，24 train / 8 valid）跑真实 2 cycle，CPU 8s / CUDA 8s |
| ② 每个新增模块调用次数 > 0 | ✅ E1–E9 共 **23 个入口**全部计数 > 0（E1=18、E2=18、E3=18、E4=18/18/54、E5=18、E6=18/18/18、E7=6/8/2、E8=3、E9=1/2/3/3/2/2） |
| ③ 关闭即逐位等价（G0） | ✅ **非同源**对比：从 `5edfca0` 用 `git show` 抽原始源码并排导入，同权重同输入下 T5 单 epoch mean loss / **89 个梯度** / **92 个 state 张量**、`get_indices`（全量 11924 item × `use_sk∈{T,F}`）、`RQVAE.forward` 4-tuple、`generate()` token、eval 指标 **全部逐位相同** |
| ④ 无历史机制被激活（G1） | ✅ tripwire **0 次**触发；链上读的 23 个文件无一命中禁用子串；链上 3 个 `Trainer` 的 `lambda_trans/lambda_balance/align_weight` **全为 0** 且 `trans_dict/align_table/align_proj` 全 `None` |

**探针必须有牙齿**：每个探针逐一做**阴性对照**（自检计数 +1；选择性探针在 2 次调用中只记 1 次；`ReadSpy` 拦到人为打开的 `behavior_emb_probe.npy`；三个历史损失**真的被打开**后 tripwire 各自触发 1 次）⇒ 开关关闭时的 **0 是"测出来的"，不是"没测"**。

**另外两项**：**幂等 / resume**（完成后再 `run()` 是 no-op；refresh 失败 ⇒ `RefreshAborted` + state 停在 `tokenizer_done`、recommender 未跑，解封后恢复只补 cycle 1 的 tokenizer）；**关闭对照**（同一 E9 机器换 `lambda_sia=lambda_psa=0` ⇒ SIA/PSA 链 284→**0** 次调用、`TIGER.forward(return_hidden=True)` **0** 次、mount 以 `enabled=False` 到达冻结循环；单 batch 逐位复现 current 与 pristine **两条** clean TIGER 路径）。

> **⚠️ 全部为工程结论。**E10 是**门禁**，不含任何推荐效果结论。

---

## C1 实现 + CSG CF 来源裁定（R7，2026-10-14）

**来源裁定（用户裁决，推翻首轮的 ItemCF 路线）**

| 项 | 内容 |
|---|---|
| **正式 CSG 来源** | **SASRec item embedding** — `sasrec_output/<D>/seed2025_best.pth` 的 `item_emb.weight`，**50d** |
| 依据 | CSG4Rec 的 `--cf_emb` 默认值就是 SASRec 的 item embedding 产物（`CSGT/main.py:54` → `Instruments-32d-sasrec.pt`；`CSGT/datasets.py:24-29` 直接 `np.load` → `torch.from_numpy` → 与 `text_emb_np` **行对齐**）⇒ **其 CF 向量 = SASRec item embedding 表，不是 ItemCF 矩阵** |
| 为何不复刻生成端 | 参考快照**不含 SASRec 训练器**（`CSGT/models/` 仅 `layers.py`/`rq.py`/`vq.py`）⇒ 取本仓库**同语义**的 SASRec 基线 `tools/run_sasrec.py`；seed 选 **2025**（v2 对齐协议；v1 的 `sasrec_<D>_best.pth` 有 off-by-one eval bug） |
| ItemCF-128 的处置 | **保留、不删除**。它**不是**正式 CSG 来源（口径 = 本项目 ItemCF cosine + SVD，与 CSG4Rec 语义不同）；接口按 `source="itemcf"` 显式加载，`is_default_source=False` |

**接口与产物**：`tiger_ext/cf_embed.py` = **纯 numpy 叶子**（加载 / 校验 / 建 id 映射 / 回存 / 留 provenance），**不自算任何表征**，不碰 RQ-VAE 输入路径与 T5。`CFEmbedding` 核心接口**未重写**，新增的只是**两个加载器**（`load_sasrec_item_embedding` / `load_cf_embedding`）。**既有 tracked 文件零改动**。

**验收结果（EXP-C1，132/132，CPU + CUDA）**：① item 数量 == `item_emb.parquet` 行数 ✅ ② 维度明确且被记录（50d）✅ ③ **无错误 item mapping** ✅ ④ 来源可追溯 ✅

**抽取契约（实测，三数据集）**：ckpt 的 `item_emb.weight` = **`(N+1, 50)` float32**（Toys 11925 / Sport 18358 / Beauty 12102）；**第 0 行 = `padding_idx`，恒为全零**；ItemID 空间恰为 **1..N**，无 id 落在 padding 上 ⇒ 规范表 = **`W[canonical ItemID]`**，**不是 `W[1:]`**。

| 数据集 | n_items | d_cf | ckpt 行数 | ItemID == 行+1 ? | 全零行（SASRec） | 全零行（ItemCF 备用源） |
|---|---|---|---|---|---|---|
| Toys | 11924 | 50 | 11925 | ✅ 是 | 0 | 24 |
| Sport | 18357 | 50 | 18358 | ✅ 是 | 0 | 28 |
| Beauty | 12101 | 50 | 12102 | ❌ **否，4158 行错位** | 0 | 33 |

**负对照（脚本内实测断言）**：Beauty 上朴素 `W[1:]` 会错配 **4158 行**（例：第 8 行 ItemID=10 应取 `W[10]`，`W[1:]` 给的是 `W[9]`）。另外三条 guard 也都有负对照：非全零 padding 行 / id 空间含 0 / item 数与 ckpt 行数不符，**一律拒绝而不是静默取值**。

**两种来源的冷启动语义不同（如实记录，不粉饰）**：ItemCF 对"训练交互里没出现过的 item"写**全零行**（24/28/33）；SASRec 对"从未训练到的 item"**保留随机初始化** ⇒ **无全零行**（0/0/0，但这类行不含学到的信号）。**未知 id 与冷启动是两回事**：未知 id 默认抛 `MissingItemError`（`missing="zero"` 才给零向量）。

**明确不做**（用户硬约束）：**不因 128→50 加 projection / normalization**、不改 `CFEmbedding` 核心接口、不实现 C2 的 cosine/kNN、不实现 C3 约束、不实现 C4 loss（EXP-C1 用 AST 断言模块顶层无 `cosine/knn/loss` 类符号、无任何 `nn.*` / `F.*` 层构造）、不引入 User SID。

> **⚠️ 全部为工程结论。**C1 只验证"CF 向量能被正确、可追溯地加载且无错映射"，**不含任何推荐效果结论**。

---

## C2 实现 + 参考实现差异（R8，2026-10-14）

**产物**：`tiger_ext/collab_relation.py`（**纯 numpy 叶子**：加载 C1 的 `CFEmbedding` → L2 归一化 → 分块归一化 matmul → 逐行 `stable argsort` 取 top-k → 回存 + provenance）；`integration/exp_c2_collab_relation.py`（127 项断言）；`integration/artifacts/collab_relation/<D>_k20.{neighbors,similarity,pairs,pairs_itemid,item_ids}.npy` + `<D>_k20.json`。**既有 tracked 文件零改动**（`git status --porcelain` 与 C1 结束时逐字相同）。

**关系的确切定义（plan.md C2 规格为准）**

| 项 | 取值 |
|---|---|
| 相似度 | `cos(a,b) = dot(a,b) / (‖a‖₂·‖b‖₂)`，float32；表**只归一化一次**，kNN 走分块归一化 matmul |
| k | **20**，**继承 CSG4Rec `--knn_k` 默认值**（`CSGT/main.py:39`），**未重新调参**；`k_eff = min(k, n-1)` |
| 方向 | **有向**：每一行独立选自己的 top-k，**不做对称化**（实测互反比例 0.652 / 0.612 / 0.694） |
| 自环 | **显式屏蔽对角线**排除自身（**不是**参考实现的"按位丢第 0 列"） |
| 零范数行 | `‖v‖ == 0` ⇒ 不选邻居（`-1` / `NaN` 哨兵）**且**永不被人选中 |
| 并列 | similarity 降序、再按**行号升序**（`kind="stable"`，完全确定） |

**与 CSG4Rec 原实现的差异（5 条，实测支撑，不扩大设计）**

| # | 参考实现 | 本实现 | 实测依据 |
|---|---|---|---|
| 1 | sklearn `NearestNeighbors(metric='cosine')` | 手写 numpy（float32 分块 matmul） | 三数据集 top-k 集合与顺序与 sklearn 完全一致 |
| 2 | 按位 `indices[:, 1:k+1]` 去自身 | **屏蔽对角线** | 参考写法在**零范数行**上会把自身留在 top-k（ItemCF 备用源实测 Toys 6/24、Beauty 11/33）；**SASRec 正式源 0/0/0 ⇒ 正式源上看不出差别，极易被静默继承** |
| 3 | 零范数行无特殊处理 | 明确排除 + 哨兵 | 零行点积恒为 0.0（有限），不加 mask 会得到"cos=0 的假邻居" |
| 4 | 关系只在内存中随 batch 使用（`trainer.py:155,192`） | 落盘为**独立 artifact**（含 ItemID 伴随件 + 摘要） | 供 C3 跨进程消费 |
| 5 | 由 `--use_knn` 等开关驱动训练时构造 | 独立接口，**不复用历史 `trans_graph_path` 的开关语义** | 用户硬约束 |

**正确性证据（三重独立交叉验证）**：① 手算 fixture（6×3 表，逐格 cosine/顺序/并列/自环/零行/k 截断全部手推）；② **float64 朴素暴力**逐行复算，**集合与顺序都相同**；③ **sklearn `NearestNeighbors(metric='cosine')`** 在三张 SASRec 表上完全一致（即参考实现自己用的库）。另有**负对照**：篡改 artifact 的任意字段、换 ItemID 顺序加载、`pairs` 与 `neighbors` 不一致 —— 一律被 `validate()` / `load()` 拒绝。

**行空间护栏**：行 = C1 的 `CFEmbedding` 行 = `data/<D>/item_emb.parquet` 的 `ItemID` 列顺序；artifact 携带 `item_ids` 与其**顺序敏感 sha256**，错序加载直接拒绝；伴随件 `pairs_as_item_ids` 以 ItemID 空间导出。**Beauty 的 4158 行错位由该 sha256 拦下**（Beauty 的 `row+1` 在共现探针上只有 0.3472，显式映射是 0.2010；Toys / Sport 因 `ItemID == row+1` 分辨不出——所以负对照必须落在 Beauty 上）。

**B 类共现探针及其限制**：`valid` / `test` basket 的 held-out 共现富集 ≈ **12.7× / 5.5× / 3.9×**（Toys / Sport / Beauty），随机置换映射后塌到 ≈1.0×（证明探针确实对映射敏感）。**⚠️ 有限制**：本项目的 `valid`/`test` history 是 SASRec 训练序列的**前缀**（`valid.history ⊃ train.history + train.target`）⇒ 该探针是**映射 / 相关性诊断**，**不是泛化证据**，**更不能当作"CSG 有效"的效果证据**。

> **⚠️ 全部为工程结论。**C2 只验证"关系被正确、可复现、可追溯地构建与存取"，**不含任何推荐效果结论**。

---

## C3 实现 + 参考实现差异（R9，2026-10-14）

**产物**：`tiger_ext/qspace_constraint.py`（A 关系 / B 表示 / C 约束三者**分开声明**：`QSpaceBatch` = 用 RQ-VAE 自己的 `forward(..., return_xq=True)` 取回 Q-space 张量并做**来源断言**；`QSpaceConstraint` = 唯一挂载点，把 C2 关系翻译成 batch 内的 pair；`ConstraintBatch` = 交付给 C4 的输入）+ `integration/exp_c3_qspace_constraint.py`（99 项断言）。**既有 tracked 文件零改动**（`git status --porcelain` 与 C2 结束时逐字相同：仍只有 `model/main.py`、`rqvae/models/rq.py`、`rqvae/models/vq.py` 三项 E1–E6 期间的既有改动）。

**Q-space 的确切定义（B）**：`x_q` = RQ-VAE 自身 forward 在**码本查询之后、straight-through 之前**的张量 = `Σ_l codebook_l[indices_l]`（`rqvae/models/rqvae.py:61-79`，`return_xq=True`）。**不取 E3 的 code logits**——代码依据：`rqvae/models/vq.py:76` `raw_d = d`、`:107` `logits = (raw_d / logit_temperature)...`，logits 由**量化前的 encoder 输出**算出 ⇒ 约束 logits 等于约束 encoder 向量，正是 plan.md C3 **禁止**的 latent-space alignment。

**约束作用于哪个张量（C）**：`ConstraintBatch` 的 `x_q_anchor` / `x_q_target`（以及两侧的 `indices`）。**不含任何标量 loss**；`forward()` 只做 gather + 过滤 + 计数并**返回 pair 轴**，`lambda_csg` 默认 **0**、`lambda_csg == 0` 时 `forward()` **抛 `QSpaceDisabled`**（不是静默返回 0）。

**数学形式（无标量输出，故没有 `loss = ...` 这一行）**：

```
P = {(a,b) | b ∈ N(a), b ≠ a, sim(a,b) 有限, a、b 同在 batch 内}   # 有向，不对称化
X = x_q[anchor_pos]     Y = x_q[target_pos]
Ix = indices[anchor_pos]  Iy = indices[target_pos]
w = similarity[anchor_pos, slot]          # 原样透传 C2 的 cosine，不做变换
ConstraintBatch(x_q_anchor=X, x_q_target=Y, indices_anchor=Ix, indices_target=Iy, weight=w, ...)
```

**CSG4Rec 原实现 vs 当前适配（5 条差异，不扩大设计）**

| # | 参考实现 | 本实现 | 依据 |
|---|---|---|---|
| 1 | **CSG4Rec 的 kNN loss 本体在 `CSGT/models/rqvae.py`——该文件在快照中缺失**（无归档、无 `.pyc`、无第二份副本） | **不复刻 kNN loss**；C3 只交付其输入 | 缺失文件中包含 `global_knn` / `use_knn` / `use_cf_loss` / `use_pairwise` / `use_kd` 的**唯一**读取点与 `compute_loss` / `get_indices`；与 C1（缺 SASRec 训练器）、C2（缺 RQVAE 模型）同类 |
| 2 | 可复原的约束（`CSGT/models/vq.py:68-106` `diversity_loss`）监督信号 = **码字 id**，目标是同簇的**另一个码字** | 监督信号 = **batch 内真实 item 的 pair**，靶是**另一个 item 的量化向量** | 参考代码的"关系"是 `constrained_km` 对**码本**聚类（码-码），不是 CF kNN（item-item）；C3 的任务是把 **CF 关系**送进 Q-space |
| 3 | 打分 `sim = x_q @ embᵀ` 再 `F.cross_entropy(sim, y_true)` | **只 gather，不打分**（打分是 C4 的 `objective half`） | 用户硬约束：本次不实现 CSG loss |
| 4 | 近邻可来自全库（`global_knn`） | **仅 batch 内** pair，越界者丢弃并计数 | 本仓库既有先例 `rqvae/trainer.py::_get_transition_loss:213-219`（按 batch 内位置去重）；参考实现的 out-of-batch 策略**不可复原** |
| 5 | `--use_knn true --knn_weight 0.01 --beta 0.0001 --use_constrain_loss true`（`train_tokenizer.sh`） | 单一开关 `lambda_csg`，**默认 0**、默认关闭 | 用户硬约束；且历史开关语义（`trans_graph_path`）已被 C2 明确不复用 |

**同：受约束的**量**与参考实现一致——都是**码本查询之后的 `x_q`**，而不是 encoder latent、不是 logits、不是重建输出。

**来源断言（plan.md C3 验收 ① 的机械化）**：`QSpaceBatch.validate()` 每次 capture 都验证 ① `x_q ≈ Σ_l codebook_l[indices_l]`（`atol=rtol=1e-4`，实测偏差 0.0 / 1.49e-08 / 2.98e-08 / 5.96e-08，仅 float32 straight-through 舍入）② `‖x_q − z‖` **大于**容差（实测 0.131–0.229 / fixture 3.000）⇒ 检查**不是空转**。4 条负对照实测会咬：把 latent 当 `x_q` 交出 / 篡改 code index / RQ 层数不符 / `rows` 与张量条数不符。

**哨兵与映射**：`neighbor == -1` 与 `~isfinite(similarity)` **必须先一致**（不一致直接 `QSpaceMismatch`，两个方向都实测会咬），过滤后才进入张量运算；恒等式 `slots == pairs + sentinel + out_of_batch` 每次实测。行空间一律走 C2 的 `expected_item_ids` + `row_of()`（**禁止 `a-1`**；fixture 自身就能证伪 `a-1`，Beauty 实测错位 4158 行）。关系只通过 `CollabRelation.load(..., expected_item_ids=..., expected_k=20)` 读取，**从不直接打开 `.npy`**。

**真实三数据集最小验证（CPU + CUDA）**

| 数据集 | 关系 | batch 内 pair | 重建偏差 | `‖x_q − z‖` | 哨兵 |
|---|---|---|---|---|---|
| Toys | 11924 × k20（238,480 边） | 738（2.88 / item） | 2.98e-08 | 0.154 | 0 |
| Sport | 18357 × k20（367,140 边） | 118（0.46 / item） | 5.96e-08 | 0.229 | 0 |
| Beauty | 12101 × k20（242,020 边） | 333（1.30 / item） | 2.98e-08 | 0.131 | 0 |

- **A / B / C 可分离的实证**：同一数据集（Toys）用**真实已训 tokenizer** 与 fixture tokenizer，**pair 轴逐位相同（均 738）**，但码分布不同（L0 用到 71 vs 99 个码）⇒ 配对只取决于关系（A + C），与表示（B）解耦。
- **⚠️ batch 内 pair 数是 batch 几何量，不是效果指标**：256 item 的 batch 里 Sport 的 0.46/item 与"均匀随机期望 `256/18357 × 20 ≈ 0.28–0.43`"同量级，Toys 的 2.88 反映其邻居更集中在列表头部。**不得把 pair 数当作"约束强度"或"信号强度"引用。**
- **⚠️ 三数据集用统一 fixture tokenizer**（冻结架构 + 该数据集真实 latent 上的 `init_emb` k-means 初始化，seed 2025，**不训练**），因为仓库里只有 Toys 有在**规范 4096d content 表**上训过的 RQ-VAE；Sport / Beauty 只有 fused 表的 `*_TA_SID_4096d`。Toys 上用真实已训 tokenizer 复跑通过。

> **⚠️ 全部为工程结论。**C3 只验证"关系能被正确、可追溯地翻译成 Q-space 约束的输入张量"，**不含任何推荐效果结论**，也不含任何关于约束是否有效的结论（`lambda_csg` 默认为 0）。

---

## C4 实现 + 参考实现差异（R10，2026-10-14）

**产物**：`tiger_ext/qspace_loss.py`（`qspace_constraint_loss` = 目标函数本体；`QSpaceLoss` = 无状态 `nn.Module` 包装；`QSpaceLossMount` = 带 `lambda_csg` 的挂载；`codebook_stack` / `verify_indices_match_qspace` = 取码本与校验；14 个 `QSL_*` 常量把参考实现、缺失文件、6 条差异、10 条设计决策**随代码冻结**）+ `integration/exp_c4_qspace_loss.py`（10 节 + `[6b]`，87 项断言）。**既有 tracked 文件零改动**（`git status --porcelain` 与 C3 结束时逐字相同：仍只有 `model/main.py`、`rqvae/models/rq.py`、`rqvae/models/vq.py` 三项 E1–E6 期间的既有改动）。

**编码前确认的 10 问（全部有代码依据，详见 [plan.md §16.1](plan.md)）**：query = anchor 在该层的**码字** `W_l[Ia[:,l]]`（`CSGT/models/vq.py:195`）；candidate = **该层自己的码本** `emb = self.embedding.weight`（`:69`）；打分 = **裸点积**（cosine 在 `:83` 被注释，`temp=1` 在 `:70` 硬编码）；target = **码字 id**（`:92`）；多邻居 = **每对独立一行**；有向 = **不合并**；**C2 weight 不使用**（`diversity_loss` 签名里没有 weight 参数，`:68`）；归约 = 层内 CE 默认**行（pair）均值** + 层间对有 pair 的层取均值（`:92` + `rq.py:64`）；空 pair = **返回 0.0**（先例 `rqvae/trainer.py::_get_transition_loss` / `rqvae/trans_loss.py::ContrastiveLoss`）；多 target = **独立 CE**，**不发明 multi-positive**。

**数学形式（最终冻结）**

```
keep[p] = (Ia[p,l] != Iy[p,l])          # 参考的正样本必为「不同码字」(vq.py:77-78)
X = W_l[Ia[keep,l]]                     # query；W_l = rqvae.rq.get_codebook()（可微）
S = X @ W_lᵀ                            # candidate = 该层全部 n_e 个码字，裸点积
S[p, Ia[p,l]] -= 1e12                   # 自身屏蔽，vq.py:87-90 逐字（大有限常数，非 -inf）
CE_l = F.cross_entropy(S, Iy[keep,l])   # 行（pair）均值
L_csg = mean{ CE_l : P_l > 0 }          # 层间均值，仅对有 pair 的层
```

**level-wise 不是我们加的**：`emb` 是该层**自己的**码本（`vq.py:69`），而 `x_q` 是各层码字**之和** ⇒ `sum(x_q) @ embᵀ` 在参考实现里**无对应物**。brief 的"优先保持完整 `x_q`"与代码证据在此**指向同一结论**。

**CSG4Rec 原实现 vs 当前适配（6 条差异，不扩大设计）**

| # | 参考实现 | 本实现 | 依据 |
|---|---|---|---|
| 1 | **kNN loss 本体在 `CSGT/models/rqvae.py`——该文件在快照中缺失** | **不复刻**；复现的是存活约束（`vq.py:68-106`）的**形态**，正样本换成 C2 的关系 | 文件缺失（无归档 / 无 `.pyc` / 无第二副本）；补造即伪造。`provenance()["is_a_reproduction"]` 恒 `False` |
| 2 | 正样本 = 同 constrained-kmeans 簇的**另一个码字**（码-码） | 正样本 = **C2 协同邻居在该层的码字**（item-item） | plan.md C3 的输入就是 item-to-item；被约束量 / 候选集 / 打分 / target **类型**全不变 |
| 3 | query 用 ST **之前**的码本查表（`:195`）⇒ 梯度只到码本 | 同一次查表（`rqvae/models/vq.py:90`）⇒ **同样路由** | 数值相同；C3 另交的 ST 后张量其 encoder 通路**实测存在但本 objective 不用**（C5 决策点） |
| 4 | 正样本必与自身不同码（由构造保证 `:77-78`） | **同码 pair 在该层丢弃并计数** | 参考没有"自身为目标"的算术；强行保留会让被屏蔽列赢下 softmax，loss ≈ 1e12 |
| 5 | 每层总贡献（正样本总存在） | **无 pair 的层略去**，均值只取有定义的层 | 参考不构造"无正样本的层"；用制造的 0 摊薄 scale 更糟 |
| 6 | `--use_knn` / `--knn_weight` / `--use_constrain_loss` / `--beta` | `lambda_csg` 默认 0，**不接入任何总 loss** | plan.md 要求独立开关；C4 不得碰 trainer |

**梯度（brief §六，实测）**：`L_csg` 可微（`grad_fn=MeanBackward0`）；`dL/dW_l ≠ 0`（查询侧 + 候选侧，与 `sim = x_q @ embᵀ` 同构）；**不到 encoder**（indices 来自 `argmin/argmax`，`rqvae/models/vq.py:78,86`）——**与参考实现同路由**（它打分 ST 前张量 `:195`，`:213` 才重写 `x_q`）。**ST 通路本身完好**：C3 的 `x_q` 直接 backward ⇒ **10 个 encoder 参数拿到梯度**（实测）；objective 不用它，但**没破坏它**。**未 detach**（AST 逐函数检查，仅有的两处 `.detach()` 都在 `float(...)` 统计里）、**未改 `requires_grad`**、**未加 projection**、**未改 E7**。

**`lambda_csg` 语义（与 C3 严格一致）**：`== 0`（默认）⇒ **抛 `QSpaceDisabled`**，不执行、**不静默返回人造 zero**；`> 0` ⇒ `lambda_csg * L_csg`，**仍未接入任何总训练 loss**（C5 的事）。**空 pair 与开关关闭严格区分**：前者返回 0 维 `0.0` + `reason`（数据结果），后者抛异常（接线错误）——两条都实测。

**实测（真实三数据集 + 真实已训 tokenizer，全部可复现）**

| 数据集 | 关系边 | batch 内 pair | `L_csg` | per-level | 层内 same-code 丢弃 | `\|dL/dW\|` per level |
|---|---|---|---|---|---|---|
| Toys（fixture tok.） | 238,480 | 738 | 5.539878 | [5.5378, 5.5413, 5.5405] | 60 | [1.10883, 0.14682, 0.16637] |
| Sport（fixture tok.） | 367,140 | 118 | 5.556832 | [5.5875, 5.5418, 5.5412] | 10 | [1.46451, 0.22270, 0.35763] |
| Beauty（fixture tok.） | 242,020 | 333 | 5.559719 | [5.5967, 5.5410, 5.5415] | 29 | [1.09656, 0.15401, 0.22607] |
| Toys（**真实已训** ckpt） | 238,480 | 738 | 5.582809 | [5.6668, 5.5405, 5.5412] | — | [0.756085, 0.130800, 0.078298] |

- **尺度不是结论**：`log(255) = 5.5413` 是"码本完全无信息"时的值，四行都在它附近 ⇒ **未训练码本应有的样子**。**不得**把 `L_csg` 的大小读成"约束有效 / 无效"，也**不得**与其他方法比 loss 数值。
- **合成 fixture 由独立 numpy float64 复算**：`L_csg = 1.492900328499`（手算），层均值 `[0.9132616875, 2.0725389695]`，丢弃 3。**被拒绝的归约**：anchor 均值 1.817900328、multi-positive 1.332852904。**weight 不变性**：零/极大/取负/打乱下 `0.765279412` 逐位不变。**数值稳定**：零码本 ⇒ 恰好 `log(n_e-1)`；scale 1e3/1e-8 有限；NaN 码本 ⇒ NaN（不粉饰）；越界/负 index、层数不符、形状不符、单码字层 **全拒**。
- **如实记录的退化读数**：64×32 随机 fixture 上 `|dL/dW|` 只有 level 0 非零（[3.617317, 0.0, 0.0]，因 64 个残差被 256 个码字完全吸收）；断言写成 `any(...)` 并打印全部三个值。**真实内容上三层全非零**（上表）。
- **C2 weight 已确认不被消费**：`batch.weight` 确实非空（`[0.9, 0.4, 0.7]`），但目标函数不读它，且四种扰动下逐位不变。

**闸门（EXP-C4 = 87/87，exit 0）**：`[1]` 手算 fixture｜`[2]` 多邻居/pair 均值（并拒绝 anchor 均值与 multi-positive）｜`[3]` 有向｜`[4]` 空 pair 与"某层全丢"｜`[5]` weight 不读｜`[6]` 梯度 + ST 通路独立测量｜`[6b]` 真实三数据集 + 真实 ckpt + fixture 可复现｜`[7]` 数值稳定与拒绝面｜`[8]` 禁项扫描 + **6 个 sha256 pin**｜`[9]` `lambda_csg=0` 干净路径｜`[10]` **`exp_e10_smoke` 64/64**。

> **⚠️ 全部为工程结论。**C4 只证明"该目标函数被正确定义、梯度路径正确、且**不被任何运行时静默激活**"，**不含任何推荐效果结论**。`lambda_csg` 默认 0，**未接入任何训练循环**；样本 loss 仍在 `log(255)` 附近，**这不构成"约束无效"的结论**（码本从未被该目标函数更新过）。

---

## C5 实现（R11，2026-10-15）

**判定**：**EXP-C5 99/99 checks passed，exit 0**（`EXP_C5_REGRESSION=all`，2026-10-15；CPU）。**C5 = PASS**。

**产物**：`rqvae/trainer.py`（**唯一被改的正式文件**：`Trainer.__init__(..., csg=None)` + `_train_epoch` 里一处受守卫的挂载；`git diff` = **7 hunks / +34/−5**；两个哈希都记，因为二者不同：**`sha256(git diff -- rqvae/trainer.py)[:16] = 9c1b73b4ffccbe3d`**（4362 字节）、**EXP-C5 §[8] 的 Python 侧 `difflib.unified_diff(..., n=3, lineterm='')` 哈希 `[:16] = a8da64e3291fa1c5`**，改后文件 `sha256[:16] = d09ad776891dd3b2`）+ `tiger_ext/qspace_train.py`（新驱动入口、`CSGHook`、`CST_*` 常量）+ `integration/exp_c5_csg_integration.py`（11 节）+ `tiger_ext/__init__.py` 再导出。

**total loss**：`L_total = L_RQVAE + lambda_csg * L_csg`，`L_RQVAE = loss_recon + quant_loss_weight * quant_loss`（**定义未改**）。挂载形状与既有 `L_trans` / `L_balance` / `L_align` 同构，插在 `L_align` 之后、`backward()` 之前；权重只乘一次；**只有 `requires_grad and isfinite` 才加进 `loss`**。

**梯度策略**：**P1 仅 codebook = 采纳（也是参考自身的路由）**，实测三层码本都有梯度（`Σ|dL/dW_l|` = 7.99e-04 / 3.41e-05 / 2.62e-05）、**0 个 encoder 张量**；**P2 straight-through → encoder = 实测可达（10 个 encoder 张量）但不采纳**——要让 `L_csg` 走它必须重定义 `L_csg` 或改冻结内核，记为**边界冲突·待裁决**；**P3 不启用 = 默认**（`λ=0` ⇒ `n_calls=0`、G0 逐位等价）。

**"真的进了 backward"三重证明**：① 同初始状态同 batch 下 `L_total(λ=1e-3) − L_total(λ=0) = 5.583858e-03`；② 梯度可加性 `max|g_total − g_base − g_objective| = 8.846e-13`（相对 5.274e-08）；③ 码本张量 **3/3** 随 λ 变、encoder 张量 **0/10** 变。

**λ 候选（报告，不排名）**：`λ·E[L_csg]/E[L_RQVAE]` = **1.551% / 15.535% / 154.972%**（λ = 1e-4/1e-3/1e-2；`E[L_RQVAE]=0.035458`、`E[L_csg]=5.508495`）。`1e-4` 是**参考自己的同形项权重（`--beta 0.0001`，尺度匹配）**，`1e-2` 已让约束成为较大项 ⇒ 候选集合横跨交叉点。**工作点不由 C5 选择**（属效果实验）。

**tokenizer sanity check（真实数据，非效果实验）**：Toys 前 2048 行、冻结配方、batch 1024、20 步。无 NaN；重建 `0.0302 → 0.0180`；码本使用 L0 256/256 / L1 256/256 / L2 137/256（无塌缩）；训练出的码本（任何 forward 之前）逐层最小行范数 **6.903e-02 / 2.410e-02 / 6.368e-03**；`L_csg` 每步参与（累计 109240 pair / 60 level-loss）；`λ=0` 时 20 步逐位等于 `L_RQVAE`；历史机制 `L_trans`/`L_balance`/`L_align` 各 **0 次**（负对照触发 20 次）。

**G0（验收③）**：pristine 源取自 `git show 5edfca0:rqvae/trainer.py`，同进程同 loader 同种子，(a) 全默认与 (b) `lambda_balance=1.0` 两种配置下**逐步 loss 逐位相同、最终 `state_dict` 23/23 张量逐位相同**。

**授权偏离（见「已知问题」第 8 条）**：7 份既有 harness 重跑，每一条 FAIL 都是本次授权修改引起的 tracked-file guard，已由 EXP-C5 §[10] 用 **R1/R2/R3** 三条规则**逐行**机械归因（`unexplained: none`），并分离了 C1–C4 回显的重跑输出。

**⚠️ 全部为工程结论**：C5 只证明"该约束**真的接进了 backward**、梯度只到码本、关掉即与原 TIGER 逐位等价"。**不含任何推荐效果结论**——相对贡献 1.55%–155% 只是**尺度读数**。

---

## C6 实现（R12，2026-10-15）

**判定**：**EXP-C6 63/63 checks passed，exit 0**（`EXP_C6_TOYS=all EXP_C6_REGRESSION=all`，CPU）。**C6 = PASS**。

**定位**：**全链路 smoke test + 最小工作点验证**，**不是**最终论文效果实验。回答四个问题：① 四个 λ 能否跑通完整链；② CSG 是否真的进入训练；③ 完整 TIGER pipeline 是否被破坏；④ `kmeans_init` 哪个配置更安全。**C6 不选工作点、不做超参搜索。**

**产物**：`tiger_ext/qspace_smoke.py`（新驱动）+ `integration/exp_c6_csg_smoke.py`（新 harness，10 节）+ `tiger_ext/__init__.py`（一个 export block）。**C6 未新增任何 tracked 文件改动**：modified 集合仍是 C5 的四个文件，`sha256(difflib.unified_diff(..., n=3, lineterm=''))[:16] = a8da64e3291fa1c5`、改后文件 `sha16 = d09ad776891dd3b2`，**与 C5 记录逐字符相同**。`alt_pipeline.py` / `stage.py` / `sid_refresh.py` / `alternating.py` / `rqvae/*` / `model/*` / C1–C5 harness / E7 / E10 **全部未动**。

**Q1 四臂全链**：A=0 / B=1e-4 / C=1e-3 / D=1e-2 全部跑完 `tokenizer → refresh → recommender → evaluation`，exit 0，产物齐备（合成 3.3/2.2/2.2/2.2s）。真实臂 Toys（λ=0 / 1e-3）各 65s / 72s。

**Q2 真的进入训练**：λ>0 各臂 `n_calls == n_steps`（合成 3/3、Toys 24/24）、`n_nonempty == n_calls`、`n_zero = 0`、逐步 `weighted == λ · L_csg`；`L_total(1e-3) − L_total(0) = 8.114338e-03`；梯度可加性 `max|g_total − g_base − g_objective| = 7.276e-12`（相对 9.245e-08）；**码本 3/3 随 λ 变、encoder 0/6 变（P1 保持）**；无 NaN/Inf。λ=0 臂 `n_calls = 0` **且 `n_disabled_calls = 0`**。

**Q3 未被破坏**：(a) **λ=0 与 `hook=None`（Clean TIGER 调用签名）逐位等价** —— SID sha256 相同、tokenizer `state_dict` 15/15、recommender `state_dict` 29/29、`recalls`/`ndcgs` 全逐位相同（`torch.equal`），`best_loss` 同为 `3.148388147354126`；(b) E7 stage manifest 齐备、tokenizer 组 0/15 trainable / recommender 组 26/26 trainable、handoff `{sia: 0, psa: 0}`、`L_trans`/`L_balance`/`L_align` 各 **0 次**（负对照触发）、**SIA/PSA mount forward 各 0 次**（负对照证明计数器活着）；(c) SID `(n_items, 4)` 无 user 轴、tokenizer 输入只有 `item_emb.parquet`、85 个符号名 / 62 个字面量扫描 **user 侧 0 / 交互文件 0**、受约束路径 4 模块全库 **0 命中**。

**Q4 kmeans_init 裁决 = 维持冻结配方 `True`**：batch 1024 下 True/False 都过 C3 的两条首 batch guard，但码本覆盖率差距大（首 epoch distinct：合成 `128/128/128` vs `22/20/65`；Toys `256/256/240` vs `42/46/256`；max_share 0.041/0.146 vs 0.321/0.523）。**未引入新初始化机制、未改 RQ-VAE 核心、未改 batch size。**C5 的退化情形（`x_q == z` 逐位）是 **batch 几何**性质而非 flag 性质 ⇒ 继承的约束是"**每个首 batch 都要量**"。

**regression**：`[10]` 在 C6 内重跑 `exp_c5_csg_integration.py` → **99/99，exit 0，0 FAIL**；该一次重跑传递覆盖七份更早 suite 的 R1/R2/R3 逐行归因。

**⚠️ C6 无任何效果结论**：Recall/NDCG 只证明"链路跑通"。Toys 臂 2-epoch 4096d 的 collision rate ≈ 0.9225 / 0.9233、Recall/NDCG 全 0，是「已知问题」第 3 条的欠训练 confound，**与 CSG 无关**。`lambda_csg` 工作点选择属 C6 之后的效果实验。

---

## 效果实验设计（R13，2026-10-18）

**权威设计 = [plan.md §19](plan.md)。本节只记状态，不重复设计。**

**四项用户裁决**：① content 口径 = **768d content-only**（`D_target = 768`，放弃 §1.3 的 4096d）；② 矩阵 = **2×2×2 全因子 8 格**（C×E×A）；③ λ = **先扫后定**；④ B 类 = **1 数据集 × 1 seed 全格筛选**。

**口径变更的实测依据**（不是偏好）：768d 三数据集 collision = **0.000419 / 0.000545 / 0.001653**，对照 C6 的 4096d Toys **0.9225**；且 768d 的 clean baseline **9/9 已训好**（`tiger_content768d_baseline_*_seed{2025,2026,2027}.pth`），实测 N@20(seed2025) = **0.0341 / 0.0274 / 0.0368**。⇒ 4096d 的「已知问题第 3 条」在本系列中**不再构成阻塞**（该条仍对任何 4096d 实验有效）。

**开工前必须完成的两条接线**：

| 编号 | 内容 | 状态 |
|---|---|---|
| **P1** | `alt_pipeline` 增加可选 `csg=` 透传（格 #6 / #8 需要） | ✅ **ACCEPTED（2026-10-18）** —— 见下「P1（ACCEPTED）」 |
| **P2** | 新驱动 `tiger_ext/fixed_train.py`（冻结 tokenizer + 单次多 epoch `model/main.py::train` + `SIA/PSA` mounts；格 #3 / #5 需要） | ✅ **IMPLEMENTED（2026-10-18），待验收** —— 见下「P2（IMPLEMENTED）」 |

**已核实的影响面**：`alt_pipeline.py` **不在**任何 `PINNED_SHA16` 中（C4/C5 只钉 7 个模块），`tiger_ext/` **未被 git 跟踪** ⇒ P1 **不会**改变既有 tracked-file guard 的 FAIL 集合。⚠️ 设计讨论中"它被 sha256 pin 钉住"的口头判断**已更正为错误**。

**预注册的负结果**：A=1 且 C=0 时 tokenizer 目标与起点均与 A=0 相同 ⇒ **格 #4（+ALT）预期 ≈ 格 #1（TIGER）**；A 的增益只可能经 `A×C` 交互出现。若每 cycle 的 **SID diff rate ≈ 0**，A 判为惰性、ALT 降级为附录，**不得改判据**。

**当前进度**：Stage 0 / Stage 1 / Stage 2 **全部 PLANNED，0 路已跑**。

### P1（ACCEPTED，2026-10-18）

**改动面**：只动 `tiger_ext/alt_pipeline.py`（**未跟踪**），9 hunks / 净 +76 行（626 → 702），`sha256[:16] = fc1b8652a72c04ce`。`rqvae/`、`model/`、`clean_tiger/` **零改动**；**RQ-VAE 核心未触碰**；P2 未实施。

| 项 | 内容 |
|---|---|
| 生产文件 | `tiger_ext/alt_pipeline.py`（未跟踪）｜`sha256[:16] = fc1b8652a72c04ce` |
| 测试 harness | `integration/exp_p1_csg_passthrough.py`｜`sha256[:16] = dd267ea07c145356` |
| 新参数 | `TigerAlternatingPipeline(..., csg=None)`（末位关键字） |
| 透传点 | `tokenizer_phase` 的 `RQVAETrainer(args, rqvae, len(loader), csg=csg)` |
| 配套（必需） | 仅当 `csg.enabled` 时 `EmbDataset(..., return_index=True)`；否则 trainer 的 `batch_indices is not None` 守卫会**静默丢弃** `L_csg` |
| 记录 | `result["csg"]` / `run_config()["csg"]`，**只在挂载 hook 时出现** |
| 新守卫 | 启用但数据无行索引 / hook 未到达 trainer / 非 hook 对象 / 带参数的 hook —— 四种都抛错 |

**EXP-P1 = 27/27**（CPU）：

- **默认路径逐位不变**：`csg=None` 与 **pre-P1 调用点**（`EmbDataset(path)`、3 参 `Trainer`）对比 —— 23/23 张量 `torch.equal`、`best_loss` 相同、collision 相同、结果字典键集相同、`run_config` 无 `csg` 键。
- **禁用 hook 惰性**：`lambda_csg=0` 与 `csg=None` 逐位相同，且 `n_calls=0`、`n_disabled_calls=0`。
- **启用 hook 确实到达 trainer**：`attached=True`、`return_index=True`、`n_calls=4`（= batch 数）、C3/C4 各 4 次、4/4 步产出非零且可导的 `L_csg`，且 state_dict 与 `csg=None` **不同**。

#### 门禁口径（差分回归；权威表述见 [plan.md §19.7](plan.md)）

**判定用「差分回归」，不用绝对分数。**

| 项 | 规则 |
|---|---|
| 基线 | **C5 之后、本次改动之前**的同一套 suite 结果（同 commit / 同 device / 同依赖） |
| 判定 | 改动后检查项**逐条与基线相同**（判定 + detail；仅时间戳、临时目录名可不同）⇒ 无新增失败 |
| 禁止 | 以历史绝对分数（EXP-E9 90/90、EXP-E10 64/64）作通过条件 —— 那是 **C5 之前**的读数，已不可达 |
| 禁止 | 表述为"全部测试通过"；基线中既有失败必须**原样列出**并标注归属 |

**差分回归结果**：

| suite | 基线（P1 前） | P1 后 | 逐项比对 | 新增失败 |
|---|---|---|---|---|
| EXP-E9 | 87/90 | **87/90** | **90/90 检查行逐字节相同**（含 detail） | **0** |
| EXP-E10 | 60/64 | **60/64** | 62/64 行逐字节相同；2 行仅差临时目录名（#4，仍 PASS）与进度条时间戳（#62，仍 FAIL、同为 80/81） | **0** |

**基线中既有的 7 项失败（原样列出；归属 C5，**不属于 P1**，本任务不修复）**：

| suite | 失败项 |
|---|---|
| E9 + E10 | `model/ and rqvae/ carry no E9/E10 change -- modified: model/main.py, rqvae/models/rq.py, rqvae/models/vq.py, rqvae/trainer.py` |
| E9 + E10 | `exp_e8_sid_refresh: 54/55` |
| E10 | `exp_e9_alternating: 80/81` |
| E9 | `(12) the E1-E8 suites are unaffected by E9 (329/330)` |
| E10 | `the E1-E9 suites are unaffected by E10 (409/411)` |

> ⇒ **P1 的验收结论是「零新增失败」，不是「90/90、64/64 全绿」。**

**已知限制**：① 差分回归只在 **CPU** 复跑，CUDA 未复跑；② 基线是**逐字还原**的 pre-P1 文件（30227 字节，与改动前一致），不是 git 存档（`tiger_ext/` 未被跟踪）；③ `csg` 记录字段"仅挂载时出现"是不对称设计，下游需接受 `config.get("csg") is None`；④ ~~P2 未实施，格 #3 / #5 仍无执行路径~~ → **2026-10-18 P2 已实现（待验收）**；⑤ 既有 7 项 guard 失败保留原状，不修、不回滚 C5。

### P2（IMPLEMENTED，2026-10-18，待验收）

**改动面**：**纯新增** `tiger_ext/fixed_train.py`（未跟踪，628 行，`sha256[:16] = b3ec855be760bafc`）+ 新增 harness `integration/exp_p2_fixed_train.py`（675 行，`sha256[:16] = 555ef348da5e0e24`）。`model/`、`rqvae/`、`clean_tiger/`、既有 harness **零改动**（mtime 仍 2026-10-14）；`grep -rn fixed_train model rqvae clean_tiger tools` 为空 ⇒ 无既有模块 import 它。

| 项 | 内容 |
|---|---|
| 入口 | `FixedSidTrainer(dataset, data_root, sid_path, tokenizer_ckpt, run_dir, ...)` → `run()`；CLI `python -m tiger_ext.fixed_train --sid_path … --tokenizer_ckpt … --run_dir …`（`--rqvae_config` 接 JSON overrides） |
| 固定 tokenizer/SID | `StageFreeze.apply("recommender")`（RQ-VAE `requires_grad=False` + 码本 `initted=True` 严格封存）；SID 只读一次并 `validate_sid`；**每 epoch 后**复核 tokenizer digest / SID sha256 / 码本 `initted` |
| 多轮训练 | 单一 `Adam`（T5 ∪ 两 adapter，与 RQ-VAE 零交，构造后断言）+ 完整 `model/main.py::__main__` 协议（每 epoch `train` → valid `evaluate` → 改进时 test + `best.pth` → `early_stop=10`）。**不调用** `recommender_phase` |
| ETE | `λ>0` 才构造 `SIAMount`/`PSAMount`；`λ=0` 时传 `sia=None, psa=None`（Clean TIGER 分支）；不改任何损失公式 / 温度 / 优化策略 |
| CSG | 构造期**拒绝** `csg=`（`FixedSidError`）——它是 tokenizer 阶段损失；格 #5 只经 CSG-trained ckpt + 其 SID 进入 |
| 落盘 | `best.pth` / `last.pth` = 纯 T5 `state_dict`（`tools/eval_trained_model.py` strict load 可用）；adapter 存 `{best,last}.{sia,psa}_adapter.pth`；`run.json` = config + result |

**EXP-P2 = 47/47（CPU）**：SID 逐字节不变、tokenizer digest 逐位不变（每 epoch 复核）；`train` 每 epoch 恰一次、T5 权重确实移动；`λ=0` 两 mount 不构造且 `train()` 收到 `None`、adapter 逐位不动；SIA-only 时 **PSA adapter 逐位不动**（梯度恒 `None`）、both 时两者都动；`λ=0` 一次运行与**手工重放 `model/main.py::__main__` 循环**逐位相同（50/50 张量 + 每 epoch `train_loss` 完全一致）；`recommender_phase` 被替换成抛错仍跑完；**改进分支实跑到达**（小词表 split：NDCG@20 0 → 0.6309，test 评估 4 次 = 改进 4 次，`best.pth` ≠ `last.pth`）；CLI 端到端 exit 0、写 `run.json`、打印含 arm/λ 的 JSON 摘要。

**已知限制**：① 全部验证在 **CPU**，CUDA 未复跑；② 小词表 split 的 `batch_size=16` 只为凑梯度步（256 batch 下每 epoch 仅 1 步、NDCG 恒 0），**不是**协议改动，正式臂仍 `T5_BATCH=256`；③ 全程无改进时只写 `last.pth`（与 `model/main.py` 一致）；④ `tiger_ext/__init__.py` 未改（不 re-export）；⑤ 未启动任何真实效果实验（EXP-L0 / EXP-B1 未跑），R13 设计未调整；⑥ 纯新增文件 ⇒ 不触发 P1 那套 E9/E10 差分回归（**门禁任务仍按要求跑了一遍**，见下）。

### P2 后启动门禁（2026-10-18，**CONDITIONAL**）

只读检查：**未跑 EXP-L0 / EXP-B1，未改模型、训练逻辑、损失公式**。权威记录见 [plan.md §19.11](plan.md)，实验侧见 [experiments.md §17.6c](experiments.md)。

**① 差分回归**（按 §19.7 口径，基线 = C5 后 / P1 前）：

| suite | 本次 | 基线 | 失败项 | 新增失败 |
|---|---|---|---|---|
| EXP-E9 | **87/90** | 87/90 | 3/3 相同 | **0** |
| EXP-E10 | **60/64** | 60/64 | 4/4 相同 | **0** |

7 项失败全部是 **C5 授权改内核**导致的 tracked-file guard，与 P1/P2 无关。原始日志：`integration/artifacts/p2_gate/p2gate_{e9,e10}.log`。`alt_pipeline.py` 的 `sha256[:16]` 仍为 `fc1b8652a72c04ce`；`model/`+`rqvae/` 四文件 mtime 仍 2026-10-14。
⚠️ **诚实标注**：磁盘无 P1 前基线日志 ⇒ 本次按**总数 + 失败项身份 + 嵌套计数 + 哈希**核对，**非逐字节 diff**（P1 验收那次才是逐字节）。

**② 格 #5 链路**：`qspace_train(λ>0)` → ckpt → `refresh_sid` → SID + sidecar manifest → `FixedSidTrainer(sid, ckpt, λsia/λpsa>0)`。实测（768d 基线 ckpt）：refresh 成功（collision 0.000419、`ok=True`）、manifest `verify_sid_manifest` 通过（`ckpt 8069f998` / `sid bb2693a6`）、固定 SID 的 ETE 跑 1 epoch 时 **routing 0/11924 mismatch**、tokenizer 与 SID 均未变。该链路当时靠驱动自律；**B4 修复后训练器自身在构造期与 `run()` 起跑前都会复核 manifest ↔ ckpt**（见下「B4 修复」）。

**③ EXP-L0 五臂**（澄清，未改设计）：`λc-1e-4 / λc-1e-3 / λc-1e-2`（C=1,E=0，各自新训 3000 ep CSG tokenizer + 各自的 SID）+ `λe-0.1 / λe-1.0`（C=0,E=1，冻结 768d 基线 ckpt 与 `Toys_content768d_baseline_sid.npy`，λsia=λpsa=0.1/1.0）；共同固定条件 = Toys × T5 seed 2025、A=0、T5 200 ep + early_stop 10、RQ-VAE 配方逐字继承 plan §19.1。**Stage 0bis 实为 3 个 tokenizer（非 1 个）**，plan §19.9 已订正。

**④ 阻塞（B1/B2/B3/B5/B4 均已修，见下各节）**：

| # | 阻塞 | 影响 | 状态 |
|---|---|---|---|
| B1 | `qspace_train` 硬编码 `item_emb.parquet`（Toys 为 4096d），无 override | λc 三臂用错表 | ✅ **已修**（`--embedding-path`） |
| B2 | `qspace_train` 无法设 `warmup_epochs`（0 vs 配方 50） | 配方不符 | ✅ **已修**（`--warmup-epochs`，默认 0） |
| B3 | 历史 ckpt 的 `data_path` 为仓库相对路径，生成器 `cwd=rqvae/` | 复算历史 SID 失败（新 ckpt 不受影响） | ✅ **已修**（`resolve_data_path` + 临时绝对化副本） |
| B4 | `FixedSidTrainer` 不校验 SID manifest ↔ ckpt | #5 lineage 靠驱动自律 | ✅ **已修**（构造期 + `run()` 起点复核；缺 manifest 默认硬失败，显式 opt-out 记录在案） |
| B5 | `qspace_train.py` 无 `__main__` 守卫 | 文档化 CLI 惰性（exit 0 零输出） | ✅ **已修** |

**判定**：门禁第 2 条 ✅ / 第 3 条原 ❌ 的直接理由（B1/B2）与 lineage 缺口（B3/B4）均已消除。**修后逐项核对已于 2026-10-19 重做**（见下「修后配方逐项核对」）：λc 配方与 λe 冻结 SID 均通过，但推荐器入口发现 **B6 硬阻塞**（`FixedSidTrainer` 写死 4096d 表）⇒ **CONDITIONAL（维持）**，EXP-L0 不得启动。**同日晚 B6 已修复（见下「B6 修复」，EXP-B6 31/31）、B7 亦已修复（见下「B7 修复」，EXP-B7 48/48）、R1 已审计关闭（见下「R1 审计」，EXP-R1 44/44）、R2 已冻结关闭（见下「R2 冻结」，EXP-R2 274/274）**，剩余阻塞仅 R3。

### B1/B2/B5 修复（2026-10-18，**待验收**）

**范围**：只改 `tiger_ext/qspace_train.py`（`sha256[:16]`：`ea224a1d56e8c5eb` → **`e10f24cf0378a816`**）。`rqvae/`、`model/`、`clean_tiger/`、`tiger_ext/fixed_train.py`、`sid_refresh.py`、`alt_pipeline.py` **零改动**（`alt_pipeline.py` 仍 `fc1b8652a72c04ce`）。`tiger_ext/` 未被 git 跟踪 ⇒ `git diff` 为空。

1. **B1**：`CSGTrainerConfig.embedding_path: str = None` + `resolved_embedding_path()`（默认仍是 `data_root/<ds>/item_emb.parquet`）+ `validate_embedding_path()`（缺文件即抛 `QSpaceTrainError`，**无 fallback**）+ CLI `--embedding-path`；`namespace()`/`build_dataset()` 改读解析结果，`train_tokenizer()` 开头先校验。
2. **B2**：`CSGTrainerConfig.warmup_epochs`（默认 `PHASE_WARMUP_EPOCHS=0`）+ CLI `--warmup-epochs`，并真正透传 `rqvae_namespace(..., warmup_epochs=)`。
3. **B5**：文件末尾 `if __name__ == "__main__": raise SystemExit(main())`。

**验证**：harness `integration/exp_p3_qspace_cli.py`（`sha256[:16]=2faddf50c189df48`）**EXP-P3 = 29/29 全通过**（CPU + 合成数据 + Toys 1-epoch smoke，**未启动 EXP-L0**）。关键证据：

- parser 默认值逐项 = 修改前（warmup 0 / embedding_path None）；`import` 模块不启动训练；`--help` 含两个新 flag。
- **诱饵测试**：同目录放 8d `item_emb.parquet`，指定 16d 自定义表实跑 ⇒ `in_dim=16`，诱饵未被读；缺表时报错**先于** dataset 构造（factory 调用次数 = 0）。
- **warmup 真进优化器**：`trainer.warmup_steps = 50 × n_batches`（8 batch → 400），末步 LR `2.000e-05`；warmup=0 时 `0.000e+00`。
- **CLI 真跑**：`python -m tiger_ext.qspace_train`（768d + warmup 50）rc=0，ckpt 记录 768d 表 + warmup 50；无 flag 时记录旧默认表 + warmup 0；显式缺表 / 默认缺表两种 CLI 均 rc=1。
- **新登记小坑**：`sid_refresh.ckpt_args` 按 `_CKPT_ARG_KEYS` 白名单过滤、**不含 `warmup_epochs`**（ckpt 内有），审计须用未过滤的 `torch.load(...)['args']`；本次未改 `sid_refresh.py`。

**已知未决**：~~「修后配方逐项核对」尚未执行~~ → **2026-10-19 已执行**，结论见下「修后配方逐项核对」：B6 硬阻塞、B7 待裁决。**B6 已于同日修复**（见下「B6 修复」）；**B7 亦于同日修复**（见下「B7 修复」）；**R1 已审计关闭**（见下「R1 审计」）；**R2 已冻结关闭**（见下「R2 冻结」）；R3 仍未决。

### B3 修复（2026-10-18，**待验收**）

**只改** `tiger_ext/sid_refresh.py`（`sha256[:16]`：`6c54fab7c4ce6fa9` → **`983121cb3d9cd17b`**）；**未改 `rqvae/`**（`generate_code.py` 是 tracked 文件，动它会新增 diff-regression 的 tracked-file guard 失败）、**未加符号链接**、**不依赖调用方 cwd**。

- **语料实测**（708 个 RQ-VAE ckpt）：`data/...` 627（仓库根相对）、`../data/...` 61（`rqvae/` 相对）、绝对 20 ⇒ 两种相对约定都要处理，否则漏 61 个。
- `resolve_data_path(data_path, repo=REPO)`：绝对路径 `normpath` 原样；相对路径只试 `repo/<p>` 与 `repo/rqvae/<p>`，恰一个存在就用，两个都存在且不同 ⇒ ambiguous 报错，都不存在 ⇒ `refusing to guess`（**绝不回退到别的表**）。
- `run_generator` 传给冻结生成器的是**只改 `args.data_path` 的临时副本**（`state_dict` 23/23 张量逐位相同；相对 ckpt 与其绝对孪生生成的 SID **逐位相同**）；已是绝对路径时原文件直传、不复制。`n_items_of` 改用同一 resolver（此前随调用方 cwd 漂移）。
- manifest 新增 `ckpt_data_path_stored` / `generator_ckpt` / `data_path_rewritten`；并加守卫：生成器读的表与 `n_items_of` 统计的表不一致即抛错，不登记混合产物。

**验证**：harness `integration/exp_b3_data_path.py`（`44a7234aa5164789`）**EXP-B3 = 30/30**，日志 `integration/artifacts/b3_data_path/exp_b3.log`。覆盖：resolver 矩阵与 cwd 无关性；708 个 ckpt 全语料（绝对 20/20、`data/` 627/627、`../data/` 50/61 解析 + 11 个**响亮拒绝**）；原始 ckpt 直调冻结生成器的**有牙证据**（rc=1、`FileNotFoundError: 'data/Toys/fused_item_emb_768d.parquet'`）；历史 ckpt 端到端 `refresh_sid` 成功（11924 行、collision 0.000587、2.5 s）；绝对 ckpt 直传；`qspace_train` 新 ckpt 从外部 cwd 可用；无符号链接残留。

**回归**：**EXP-E8 = 54/55**（唯一失败仍是 tracked-file guard 列出 C5 改的 `rqvae/trainer.py`，与基线一致，**零新增失败**；`regression_e8.log`）、**EXP-P3 = 29/29**（`regression_p3.log`）。E9/E10 未重跑：本次只改未跟踪文件、不触碰其 guard，且 E9 走的是新 ckpt 绝对路径分支（已由 [5][6] 覆盖）。

**已知限制**：11 个 `../data/Sports/...` ckpt 的原表已随数据集改名（`Sport`）消失，现于解析阶段**响亮报错**（不跨表猜），无法再生 SID。

### B4 修复（2026-10-19，**待验收**）

**只改** `tiger_ext/fixed_train.py`（`sha256[:16]`：`b3ec855be760bafc` → **`8fde074584fe443f`**）；`sid_refresh.py`（仍 `983121cb3d9cd17b`）、`model/`、`rqvae/`、`clean_tiger/` **零改动**。对应测试文件：`integration/exp_p2_fixed_train.py` 的 fixture 现在注册真实 manifest；新增 `integration/exp_b4_provenance.py`。

- **复用**：`verify_sid_manifest(record, ckpt_path, sid_path_)` 是唯一哈希判据；训练器只负责找到 manifest、调用它、并把 `SidManifestMismatch` 包装成 `SidProvenanceError`（同时是 `FixedSidError` 子类）。
- **何时校验**：`FixedSidTrainer.__init__`（任何模型/数据构造之前）与 `run()` 起点各一次 ⇒ 训练开始前必然拦下；构造后换文件的 TOCTOU 也覆盖。
- **manifest 定位**：默认 `<sid>.sid.json`；`sid_manifest=` / `--sid_manifest` 指向显式文件（覆盖 `publish_sid` 只复制 `.npy` 的场景）。
- **缺失 manifest**：默认**硬失败**（不再静默通过）；只有显式 `allow_unregistered_sid=True`（CLI `--allow_unregistered_sid`）才放行，`run.json` 记录 `mode="unregistered"` + warning；若文件名是 versioned 形式（`{ds}_sid_v{ver}_{ckpt_sha8}.npy`）仍校验 `ckpt_sha8`。
- **记录的合法无 manifest 场景**：① 早于 `sid_refresh` 的 legacy SID（R13 λe 两臂用的 `Toys_content768d_baseline_sid.npy`）——建议先用 `refresh_sid` 复算（B3 已使其可行），否则显式 opt-out；② `publish_sid` 只复制 `.npy` ⇒ 用 `--sid_manifest`；③ 测试 fixture 用同一 SID 换数据集名（`P2SynSmall`）⇒ 数据集名只记录 `dataset_matches`，判据仍是哈希。

**验证**：harness `integration/exp_b4_provenance.py`（`1e02a89004c508e4`）**EXP-B4 = 38/38**，日志 `integration/artifacts/b4_provenance/exp_b4.log`。覆盖：正确三元组训练通过（λ=0 真跑 1 epoch）；错 ckpt 构造期拒绝且无 run 目录；错 SID 拒绝；缺 manifest 默认拒绝 + opt-out 记录 + versioned 名双向检查；λ=0 与格 #5 两条路径；显式 manifest / 缺失 manifest / 数据集名不一致；构造后换 SID 在 epoch 0 前被拦；CLI 三种策略；**真实历史 768d ckpt + `refresh_sid` 真 manifest（11924 行）接受，另一真实 tokenizer 拒绝**。

**回归**：**EXP-P2 = 47/47**（fixture 改为带 manifest；逐位等价与新契约同时满足）、**EXP-E8 = 54/55**（唯一失败仍是 tracked-file guard，零新增失败）、**EXP-B3 = 30/30**。

**已知限制**：① 无 manifest 的 SID 在 opt-out 模式下**不做内容级复算**（不重跑生成器），lineage 只能依赖调用方声明；② 数据集名不参与硬判据（仅记录）；③ embedding 表与 SID 的同源性仍只由行数/维度间接约束（`validate_sid`），不在 B4 范围。

### 修后配方逐项核对（2026-10-19，**GATE = CONDITIONAL，B6**）

只核对：**未启动 EXP-L0 / EXP-B1，未改生产文件**（`fixed_train.py` 仍 B4 后的 `8fde074584fe443f`，`qspace_train.py e10f24cf…`、`sid_refresh.py 983121cb…`、`alt_pipeline.py fc1b8652…` 均未变）。权威表见 [plan.md §19.11.9](plan.md)；原始产物 `integration/artifacts/r13_gate/`。**所有运行均为 1-epoch / 构造级机制 smoke，不是效果证据。**

**通过项**

| 项 | 证据 |
|---|---|
| λc 配方（三组） | 正式 CLI 的 namespace 与基线 ckpt `args`：**23/25 个共同字段逐值相同**（2 项为 run dir / 路径写法）；`torch.load(ckpt)['args']` 实测 `warmup_epochs=50`、`batch=1024`、`eval_step=50`、`workers=4`、`data_path`=绝对 768d；正式解析 `epochs=3000` |
| λc → SID 同源 | 1-epoch CSG smoke（λ=1e-4，hook 12/12）→ `refresh_sid`（11924 行、collision 0.00763、`ok=True`）→ `verify_sid_manifest=True` |
| λe 冻结 SID | 同一 ckpt 在 **cuda:0** 复算 ⇒ 与 `Toys_content768d_baseline_sid.npy` **sha256 完全相同**（`bb2693a6…`，`rows_changed=0`，`exact_match=True`）；已写 sidecar 登记，`.npy` 字节未变；**未用** `--allow_unregistered_sid` |
| B4 默认策略 | 登记前无 sidecar ⇒ `SidProvenanceError`（硬失败） |
| T5 共同配置 | 运行时默认：Toys / seed 2025 / 200 ep / early_stop 10 / lr 1e-4 / batch 256 / infer 96 / beam 30 / topk [5,10,20]；协议与 `model/main.py` 一致（EXP-P2 的 50/50 张量逐位等价） |
| 产物 / 记录 | `run_dir/{best,last}.pth` + 两 adapter + `run.json`（config+result）+ best 的 `.stage.json`；SID/ckpt 哈希与 provenance 全记录 |

**新阻塞 B6（硬阻塞，五臂全挡）**：`FixedSidTrainer` 仍写死 `data_root/<dataset>/item_emb.parquet`（Toys = **4096d**），用真实 768d 基线 ckpt 构造即

```
FixedSidError: ... size mismatch encoder.mlp_layers.1.weight: checkpoint [512,768] vs model [512,4096]
```

λc 的 tokenizer 也是 768d ⇒ **五个 EXP-L0 臂全部无法开跑**。隔离实验：把 `item_emb.parquet` 换成 768d 表的诊断 shadow root 后构造成功（`in_dim=768`、provenance `verified=True`、两 mount 均活、tokenizer 23/23 冻结、optimizer 4,694,048 参数）⇒ 唯一变量就是 embedding 表路径。**最小修正（未实施，需批准）**：给 `FixedSidTrainer` 加 `embedding_path`/`--embedding_path`（默认不变），与 B1 同构。shadow root 仅诊断，不是批准运行方式。

**其他未决**：~~**B7** `qspace_train` 不播种 RNG（`--seed 2024` 仅落盘；基线由 `rqvae/main.py::setup_seed` 播种）~~ → ✅ **已修复（见下「B7 修复」）**；~~**R1** 新 ckpt `args` 缺 `tau/lambda_trans/lambda_balance/trans_graph_path`（有效值由 `getattr` 取冻结默认）~~ → ✅ **已审计关闭（见下「R1 审计」，7 键实际生效值全等于冻结值，不改生产代码）**；~~**R2** 两套 CLI 默认值全部不等于 EXP-L0 要求~~ → ✅ **已冻结关闭（见下「R2 冻结」；只用 plan §19.11.13 的五条完整命令）**；**R3** 尚无 EXP-L0 driver ⇒ 启动前落 driver。

**判定**：第 2 条 ✅；第 3 条 λc 侧 ✅、推荐器侧 **B6 ❌** ⇒ **GATE = CONDITIONAL，不得启动 EXP-L0**。

> **后续（2026-10-19 晚）**：B6 已按最小方案修复，见下节「B6 修复」；同日 B7 亦已修复，见「B7 修复」。本节 B6/B7 相关行由两节取代；R1 亦已审计关闭，R2 亦已冻结关闭。GATE 仍 CONDITIONAL（仅 R3）。

### B6 修复（2026-10-19，**EXP-B6 31/31，待验收**）

只改 `tiger_ext/fixed_train.py`（`8fde074584fe443f` → **`389e90854b5f6f13`**，808 → 924 行）＋新增 `integration/exp_b6_embedding_path.py`；`model/`、`rqvae/`、`clean_tiger/`、`sid_refresh.py`、`qspace_train.py` 零改动；未启动 EXP-L0/EXP-B1；未处理 B7。权威记录 [plan.md §19.11.10](plan.md) / [experiments.md §17.6e](experiments.md)。

| 项 | 实现 |
|---|---|
| 接口 | `FixedSidTrainer(..., embedding_path=None)`、CLI `--embedding_path`；`None` 严格保持旧默认，显式路径按给定文件读取、无回退 |
| 维度门禁 | 构造期从 ckpt `encoder.mlp_layers.1.weight` 读 `in_dim`，与所读表列数不符 ⇒ **建模前** `FixedSidError: embedding dimension mismatch ... has D columns ... in_dim=E`（不再是裸 `size mismatch`） |
| 记录 | `run.json` config/result 与 `best.pth.stage.json` 记 `embedding_path / *_explicit / *_sha256 / embedding_dim / embedding_n_items / tokenizer_input_dim / tokenizer_input_shape`；`_check_frozen` 每 epoch 复核表未改写，result 增 `embedding_unchanged` |
| B4 | 未放松：provenance 仍在 embedding 读取前、构造期 + `run()` 起点两次，判据仍是 `verify_sid_manifest` 哈希 |

**测试**：EXP-B6 **31/31**（默认兼容 / 显式被真实读取 / 错宽度建模前失败且无 run 目录 / 48d 诱饵默认表不回退 / 错 ckpt 与无 manifest 仍拒 / CLI 三态 / 真实 768d Toys 对在**无 shadow data root** 下构造成功）；回归 EXP-P2 **47/47**、EXP-B4 **38/38**、EXP-E8 **54/55**（唯一失败仍为既有 tracked-file guard，零新增）。**不是效果证据**（1-epoch 合成 fixture 机制 smoke；真实输入组 0 epoch）。

**剩余阻塞**：B7 + R1/R2/R3（B7 / R1 / R2 随后已于同日关闭）⇒ **GATE = CONDITIONAL，EXP-L0 不启动**。

### B7 修复（2026-10-19，EXP-B7 48/48，待验收）

只改 `tiger_ext/qspace_train.py`（`e10f24cf0378a816` → **`305c50a7bf530f53`**，693 → 832 行）＋新增 `integration/exp_b7_seed_control.py`；`fixed_train.py`、`sid_refresh.py`、`alt_pipeline.py`、`model/`、`rqvae/`、`clean_tiger/` 零改动；未启动 EXP-L0/EXP-B1；配方/epochs/R13 定义未改。权威记录 [plan.md §19.11.11](plan.md) / [experiments.md §17.6f](experiments.md)。

- **修复前**：`--seed` 只写 `args` 与报告，`RQVAETrainer` 不读；模型初始化、DataLoader shuffle、`layers.py:86 KMeans`（无 `random_state`）全部跟随全局流 ⇒ 同 seed 不可复现。
- **修复**：① `apply_seed`（python/numpy/torch CPU/CUDA + cuDNN pin，与 `rqvae/main.py:77-84` 同构）在 dataset/model **之前**；② DataLoader 用独立 `torch.Generator`；③ `seeded_kmeans` 在运行期给冻结 `KMeans` **逐次注入 `random_state`**（私有 `RandomState(seed)`；不改 `rqvae/` 文件；`finally` 还原），覆盖整个 `fit`（k-means 首 batch lazy 初始化）；④ 报告新增 15 字段 `seeding`。
- **不承诺项**：CUDA atomics / cuDNN kernel / 多流归约 / `num_workers>0` worker 顺序；未启用 `torch.use_deterministic_algorithms`（可能中止冻结循环）。CPU 为逐位验证路径。
- **证据**：同 seed 同进程（间隔烧 6000 随机值）init `aa17a452…`、final `be74cfe…`、codes 220 distinct、`best_loss=0.982522` 全同；同 seed 跨三进程完全一致；异 seed 4321 init `a51be790…`、draws、final、codes（234 distinct）全变而表 sha 相同。KMeans 组先证明冻结调用依赖全局 numpy（seed 1 vs 2 中心不同），再证明注入后不同全局流中心逐位相同。
- **回归**：EXP-P3 **29/29**；EXP-C4 **84/87**（与 C5 后基线逐条相同：modified-set guard + `trainer.py:121` 文本 guard + 内嵌 E10；零新增）；EXP-C6 **63/63**（内嵌 EXP-C5 **99/99 exit 0**）。
- **不是效果证据**：1 epoch × 320 行合成表机制 smoke，无 Recall/NDCG，不用于选点。

**剩余阻塞**：R1/R2/R3 ⇒ **GATE = CONDITIONAL，EXP-L0 不启动**；B1–B7 已全部关闭（R1 / R2 随后已于同日关闭，见「R1 审计」「R2 冻结」）。


### R1 审计（2026-10-19，EXP-R1 44/44，待验收）

只做核实，**未修改任何生产文件**（`qspace_train.py` `305c50a7…`、`fixed_train.py` `389e9085…`、`sid_refresh.py` `983121cb…`、`alt_pipeline.py` `fc1b8652…` 逐字节未变）；新增 `integration/exp_r1_ckpt_args_audit.py` 与审计表 `integration/artifacts/r1_ckpt_args/exp_r1_audit.json`；未启动 EXP-L0/EXP-B1。权威记录 [plan.md §19.11.12](plan.md) / [experiments.md §17.6g](experiments.md)。

- **缺键范围更正为 7 个**（不是 4 个）：`tau / lambda_trans / lambda_balance / trans_graph_path` 新 ckpt 缺、基线 ckpt 有；`align_weight / align_tau / align_data_path` **两类 ckpt 都缺**。来源是 `rqvae_namespace` 只枚举 26 键 + `trainer.py:444` 原样落盘。
- **实际生效值用真实 `RQVAETrainer` 读出**（不是 `getattr` 推演）：三份 ckpt（当前代码现产 / `lc_smoke` warmup=50 / 冻结 768d 基线）× 7 字段 = **21/21 等于 R13 冻结值**（0.1 / 0 / 0 / None / 0 / 0.1 / None）；`contrastive_loss.tau=0.1`、`trans_dict=None`、`align_proj/align_table=None`。
- **默认出处钉死**：`trainer.py:57,58,63,86,87,88,100`；其默认值与 `git show 5edfca0:rqvae/main.py:52-68` 的 argparse 默认 **7/7 逐值相同**（AST 提取）；11 个直读字段在新 namespace 中全部存在。
- **判别证据**：有键必用（注入 0.7/0.07 → Trainer 取到）；坏路径 `trans_graph_path` 会静默把 `lambda_trans` 降为 0（另一类失败，R13 不走）；删除直读字段 `warmup_epochs` → `AttributeError`（必需/可选由代码强制）；新 ckpt 上 `hasattr=False` 但 `getattr` 得 0.0。
- **下游**：`generate_code` 不读这 7 项；`fixed_train` 从不读 tokenizer ckpt 的 `args`；`sid_refresh.ckpt_args` 白名单不含（**仅 provenance 记录缺口**）；全仓 34 个 ckpt-args 读取文件 0 个触碰这 7 项。
- **不阻断的残余**：`lambda_csg`（处理变量）不在 ckpt args、只在运行报告 ⇒ 建议 R3 driver 落记录；`sid_refresh` 白名单亦不含 `warmup_epochs`；基线 ckpt 缺 `seed`。
- **回归**：EXP-R1 **44/44**（`CLOSED-BY-AUDIT`）、EXP-P3 **29/29**、EXP-P2 **47/47**、EXP-B6 **31/31**。**不是效果证据**（1 epoch × 320 行合成表机制 smoke）。

**结论**：**R1 关闭**（按任务第 4 条，审计即可，不改生产代码）。**GATE 仍 CONDITIONAL**：剩 R2（CLI 默认值 ≠ EXP-L0）（→ 已由「R2 冻结」关闭）、R3（无 EXP-L0 driver）⇒ EXP-L0 不启动。

### R2 冻结（2026-10-19，EXP-R2 274/274，待验收）

只做命令冻结与核对：**未修改任何生产文件**（四个 `tiger_ext` 文件哈希逐字节未变）；**未新增 driver**（R3 仍开）；未启动 EXP-L0 / EXP-B1；未改 R13 配方。新增 `integration/exp_r2_exp_l0_cli.py` 与审计表 `integration/artifacts/r2_exp_l0_cli/exp_r2_audit.json`。权威记录 [plan.md §19.11.13](plan.md) / [experiments.md §17.6h](experiments.md)。

- **五条命令已冻结**（3 组 CSG + 2 组 ETE），逐参数显式：CSG 的 λ/3000 ep/batch 1024/eval 50/workers 4/warmup 50/seed 2024/768d 表/独立 ckpt-dir；ETE 的 λ_sia=λ_psa=0.1/1.0、seed 2025、200 ep + early_stop 10、batch/infer/beam/lr、冻结 tokenizer、已登记 baseline SID + manifest、无 `--allow_unregistered_sid`、独立 run_dir。
- **不依赖默认值**：实测 qspace 默认 3 ep / 256 batch / eval 1 / warmup 0 / workers 1 / cpu / 4096d；fixed 默认 λ=0、embedding=None（→4096d）；五条命令逐项覆盖。
- **路径核实**：content-only 表实测 **768d / 11924 行**，与冻结 ckpt `in_dim=768` 一致；`fused_item_emb_768d.parquet` 实测 **896d**，五条命令 0 次引用；默认 `item_emb.parquet` 为 4096d。baseline SID manifest（`sid_sha=bb2693…`、`ckpt_sha=8069f9…`）校验通过，唯一 baseline run `Jul-31-2026_21-37-33`。
- **两项记录项（不改配方）**：① 两套 CLI 的 flag 拼写不同 —— `qspace_train` 是 `--embedding-path`（短横线），`fixed_train` 是 `--embedding_path`（下划线），argparse 不互为别名，混用会 exit 2；② `qspace_train` 无 `--align_weight`，§19.1 的"显式传 0.0"只能由 `getattr` 默认满足（R1 已用真实 Trainer 复核）。
- **λc 的 SID 是运行时产物**：`refresh_sid` 后版本号由 `base_dir` 内容决定（全新目录 `v0001`），driver 必须解析 `sid_versions/` 而不得硬编码；`--ckpt-dir` 是父目录，Trainer 自动追加时间戳子目录。
- **回归**：EXP-R2 **274/274**（`CLOSED-BY-COMMAND-FREEZE`）、EXP-P3 **29/29**、EXP-P2 **47/47**、EXP-B6 **31/31**。**不是效果证据**（全部为解析/构造级核对，未训练任何模型）。

**结论**：**R2 关闭（命令已冻结，不改生产代码）**。**GATE 仍 CONDITIONAL，唯一剩余阻断 = R3（无 EXP-L0 driver）**；EXP-L0 不启动。

### R3 driver（2026-10-19，EXP-R3 105/105，待验收）

只新增 `integration/` 下的 driver、机制测试与文档：**未修改任何生产文件**（四个 `tiger_ext` 文件哈希逐字节未变，EXP-R3 [9] 4/4）；**未启动 EXP-L0 / EXP-B1**；未改 R13 配方。新增 `integration/exp_l0_driver.py`、`integration/exp_r3_l0_driver.py` 与审计表 `integration/artifacts/r3_l0_driver/exp_r3_audit.json`。权威记录 [plan.md §19.11.14](plan.md) / [experiments.md §17.6i](experiments.md)。

- **五臂**：`lc1e-4/lc1e-3/lc1e-2` 各走 tokenizer → refresh → recommender（λsia=λpsa=0）；`le0.1/le1.0` 用冻结 baseline tokenizer + 已登记 baseline SID（λsia=λpsa=λ）。任一阶段非 0 退出即停止该臂及所有后续臂。
- **产物隔离**：每次调用一个全新 `integration/artifacts/exp_l0/runs/<run_id>/<arm>/{ckpt,sid,run,logs}`；非空 arm dir 或已有 SID 版本 ⇒ preflight 失败且零子进程。旧 ckpt / 旧 SID 永不回退。
- **动态识别**：ckpt = 训练前后子目录差集中的唯一新目录；SID = refresh 前后 `list_versions` 差集中的唯一新版本；时间戳与版本号均不硬编码。
- **provenance**：refresh 后校验 `sid_path`、SID sha256、ckpt sha256、文件名版本与 manifest 一致，并要求 `.sid.json` sidecar 存在。
- **总记录**：`integration/artifacts/exp_l0/driver_run.json` + `runs/<run_id>/driver_run.json`（不可变副本），含五臂 `lambda_csg/lambda_sia/lambda_psa`、逐阶段命令/退出码/时间/日志、输入输出路径、ckpt/SID/manifest 哈希与验证结果、失败原因；关闭「已知问题」第 15 条 ① 的 `lambda_csg` 缺口。
- **dry-run**：`python -I integration/exp_l0_driver.py --dry-run` 输出五臂计划与完整 argv，不执行子进程、不建目录、不刷新 SID、不写记录。
- **与冻结命令**：配方与输入路径逐 token 一致；唯一差异是输出位置按运行隔离（`--ckpt-dir` / `--run_dir` / λc `base_dir`），属 R3 任务第 2 条的直接要求，非配方变更。
- **回归**：EXP-R2 **274/274**、EXP-P1 **27/27**、EXP-P2 **47/47**、EXP-P3 **29/29**、EXP-B6 **31/31** —— 零新增失败。
- **不是效果证据**：全部为 mock/spy、合成 fixture 与静态核对，未训练任何模型。

**结论**：**R3 的代码与机制测试通过（`CLOSED-BY-MECHANISM-TESTS`）**。按任务要求，**GATE 仍 CONDITIONAL**：下一步是**独立门禁复核**，复核通过前 **EXP-L0 不得启动**。

### 独立门禁复核（2026-10-19，**GATE = CONDITIONAL，M1/M2**）

**只复核，未改生产代码、未改 driver、未启动 EXP-L0 / EXP-B1。** 新增复核 harness `integration/exp_r3_gate_review.py`（46 项，`REVIEW 44/46`，两个失败即 M1/M2），证据 `integration/artifacts/r3_gate_review/{review.log,exp_r3_gate_review.json}`。权威记录 [plan.md §19.11.15](plan.md) / [experiments.md §17.6j](experiments.md)。

| 必查项 | 结论 | 证据 |
|---|---|---|
| 1 重跑冻结门禁测试 | **PASS** | R1 44/44、R2 274/274、R3 105/105、P1 27/27、P2 47/47、P3 29/29、B4 38/38、B6 31/31、B7 48/48；E9 87/90、E10 60/64，失败行与历史逐条相同 ⇒ **零新增失败** |
| 2 独立核对五臂配置 | **PASS** | 11 条 argv 与从 §19.11.13 独立重写的模板**逐 token 全等**；真实解析器接受且 15 项配方字段逐值相等 |
| 3 真实执行安全性 | **FAIL** | 动态识别/provenance/旧产物/失败即停均实测通过；但发现 M1、M2 |
| 4 配方一致 | **PASS** | 仅输出位置 run-scoped |
| 5 启动前置条件 | **PASS** | 768d 表 11924×768、baseline ckpt in_dim=768、baseline SID+manifest 双向校验、关系产物 k=20、CUDA 可用、数据齐备、磁盘 1.0 TB |
| 6 不改生产代码/不启动 | **PASS** | 四文件哈希与 git 状态不变；`integration/artifacts/exp_l0/` 不存在 |

- **M1（假成功）**：`--arms ""` ⇒ 退出 0、`status=completed`、0 臂、仍写 `driver_run.json`。最小修复：`select_specs` 对空选择抛 `DriverError`。
- **M2（异常丢记录）**：stage 抛异常 ⇒ 异常穿出 `execute_run`，`driver_run.json` 完全未写（fail-fast 与"不误记成功"未被破坏）。最小修复：`try/except BaseException/finally` 保证记录落盘并标 failed。
- 两项均在 `integration/exp_l0_driver.py` 内；**本次按任务第 6 条未实施**，只报告证据与补丁。
- **残余风险（不阻断）**：记录非增量落盘、driver 不回读 `fixed_train` 的 `run.json` provenance、`--run-id`/`--run-root` 同时给出时记录取目录名、`sid_stats` 缺键视为通过、§19.11.13 文档 `{ckpt_sha12}` 应为 `sha8`、refresh 无超时、`topk_list` 无 CLI 直传（实际生效值与记录一致，NDCG 取自 `run.json`）。

**结论**：**GATE = CONDITIONAL**；修复 M1/M2 并重跑 EXP-R3 + 复核 harness 后才可判 PASS。**EXP-L0 / EXP-B1 未启动**，本次无任何训练或效果证据。

### M1/M2 最小修复（2026-10-19，EXP-R3 130/130，**待独立复核**）

**只改** `integration/exp_l0_driver.py`（+113 / −41，4 个 hunk）与 `integration/exp_r3_l0_driver.py`（新增 §[10]/[10b] 共 25 项）。生产代码、R2 冻结配方、五臂命令与执行顺序**零改动**；**未启动 EXP-L0 / EXP-B1**。

| 修复 | 实现 | 验收 |
|---|---|---|
| **M1** 空臂假成功 | `select_specs` 空选择抛 `DriverError`（`main` rc=2、不建 run root、不写记录）；`execute_run([])` 亦抛 | `--arms ""`/`","`/`" , "` 全部 rc=2 且零副作用；不再存在 0 臂 `completed` 记录 |
| **M2** 异常丢记录 | 臂循环 `try/except BaseException` + `finalize()`；异常先以 `status="failed"` 落盘（`failure={arm,stage,type,message,traceback}`、`arms_completed`、`arms_not_started`）再 `raise`；在飞阶段 `"running"→"failed"`；`main` 未预期异常 rc=1 | Popen `OSError` ⇒ `failed/tokenizer`；`run.json` 损坏 ⇒ `failed/recommender`；两者均写 `driver_run.json`、后续臂未启动（runner 各只调用 1 次）；成功对照 `failure=null` |

- **证据**：`integration/artifacts/r3_m1m2/{m1_m2_evidence.json,record_popen_failure.json,record_corrupt_run_json.json,record_success.json}`。
- **回归**：R1 44/44、R2 274/274、**R3 130/130（+25）**、P1 27/27、P2 47/47、P3 29/29、B4 38/38、B6 31/31、B7 48/48、E9 87/90、E10 60/64（与历史失败逐条相同，**零新增**）；独立 harness **REVIEW 46/46**（原 44/46）。
- **哈希**：`qspace_train 305c50a7…`、`fixed_train 389e9085…`、`sid_refresh 983121cb…`、`alt_pipeline fc1b8652…` 全部未变；`git status` 与基线一致；`integration/artifacts/exp_l0/` 仍不存在。
- **判定**：修复完成，**本任务不宣布 GATE PASS**，等待独立门禁复核。权威记录 [plan.md §19.11.16](plan.md) / [experiments.md §17.6k](experiments.md)。

### 最终门禁裁决（2026-10-19，**GATE = PASS**）

**只复核、未改代码、未启动实验。** 六项核查 **6/6 PASS**，权威记录 [plan.md §19.11.17](plan.md) / [experiments.md §17.6l](experiments.md)。

| 核查项 | 结论 | 证据 |
|---|---|---|
| 1 M1/M2 修复逻辑与写入顺序 | PASS | 空选择在建模目录前抛错；异常路径 `stage→arm→failure→raise→finalize(failed)` **先落盘再传播**，失败臂永不进入 `arms_completed` |
| 2 两套 harness | PASS | EXP-R3 **130/130**、REVIEW **46/46** |
| 3 §19.11.9 回归 | PASS | R1 44/44、R2 274/274、P1/P2/P3 27·47·29、B3/B4/B6/B7 30·38·31·48 全同；**E8 54/55（1 行）、E9 87/90（3 行）、E10 60/64（4 行）为历史已知失败，逐条复现，零新增** |
| 4 失败注入 JSON | PASS | 空臂 rc=2 零副作用；`OSError`→`failed/tokenizer`；`JSONDecodeError`→`failed/recommender`；均无残留 `running`、后续臂未启动（runner 各 1 次）；已独立重放一致 |
| 5 配方/输入/provenance/哈希/Git | PASS | 8 条 argv 与 §19.11.13 逐 token 相同；768d 表 11924×768、ckpt in_dim 768、SID (11924,4) 且 sidecar 双哈希实测相等、`sid_stats.ok=true`；生产四哈希未变；`git status` = 基线 6 行 |
| 6 不改代码/不启动 | PASS | 无 `artifacts/exp_l0`、`exp_b1`，无相关进程 |

**残余风险（不阻断）**：记录非增量落盘；阶段 `status` 只反映子进程退出码（产物级失败看臂级 `failed_stage`）；`Popen` 的 `OSError` 报为 rc=2 且措辞为 "filesystem error"；`--run-id` 被 run-root 目录名覆盖；refresh 无超时；§19.11.13 文档 `{ckpt_sha12}` 应为 `sha8`。

**结论**：**GATE = PASS**，可进入 Stage 0（EXP-L0）。**正式实验未启动，等待用户明确指令。**

---

## 已知问题 / 待决

1. ~~**阶段冻结未实现（属 E7）**~~ → ✅ **已由 E7 关闭（R4）**：`tiger_ext/stage.py::StageFreeze` 在 Recommender 阶段冻结 RQ-VAE；实测一次 λ>0 的 epoch 中 **0/23** RQ-VAE 张量移动、梯度全为 `None`（修复前 SIA 10 个 encoder 张量 + 3 码本、PSA 20/23 张量会被更新）。
2. ~~**`RQVAE.load_state_dict` 的 `vq.initted` 坑**~~ → ✅ **已由 E7 关闭（R4）**：`load_tokenizer_state_dict()` 加载后恢复 `initted=True`；`seal_codebooks(strict=True)` 拒绝封存全零码本；负对照已实测证明该坑真实存在。
3. **Toys 缺一个训练充分的 4096d RQ-VAE**：`plan.md §1.3` 规定 Toys `D_target = 4096`（content-only），但仓库内唯一的 4096d Toys RQ-VAE 是 `clean_tiger` 的 **2-epoch smoke** 产物（重建 R² ≈ 0.0016，batch 内 pairwise cos ≈ 0.90）⇒ 默认 fixture 下 E6 的"重建靶"近乎常数。**任何依赖真实重建质量的 Toys 效果实验（EXP-E4 / EXP-E5-E6）开工前须先补训一个 4096d Toys RQ-VAE。**（⚠️ 注意与 [[toys-rqvae-fixture-dims]] 记录的权重×维度配对关系）
4. ~~**⚠️ 阻塞：E9 的交替调度周期与边界规则在冻结计划中留白**~~ → ✅ **已由 E9 关闭（R6）**：1 phase = 1 epoch，1 cycle = 1T + 1R，相位判定用 **0-based** `epoch_idx % cycle == 0`（与 ETEGRec 官方 `trainer.py:513` 的 `cycle: 2` 一致）。**(a) 编号错位**（任务把「E8」说成调度）已由 **R5** 裁决按计划的 E8 执行；**(b) 周期规则**由 R6 落地并如实记录（1-based 读法会整体反相，脚本内有对照断言）。
5. **SID 与设备相关（R5 实测）**：同一 ckpt 在 CPU 与 CUDA 上生成的 SID 相差 **3.25%**（仅最后一层 / collision 列）⇒ **diff rate 有 3.25% 的噪声底**；且实验对比时必须固定 device，否则"差异率"会被设备噪声污染。已在 E8 manifest 记录 device。
6. **冻结的 k-means 助手跨进程不可复现（R10 查实，非阻塞）**：`rqvae/models/layers.py:86` 的 `KMeans(n_clusters=..., max_iter=...)` **没有 `random_state`**。实测 2026-10-14（3000×32 float32、固定 numpy seed）：默认线程下 **4/4 次 `cluster_centers_` 哈希互不相同**（inertia 仅第 5 位小数抖动），`OMP_NUM_THREADS=1` 时 **3/3 相同**（运行时 `threadpool_limits(1)` 压不住，运行库已初始化）。污染源分离（真实 Toys content）：已训 ckpt 路径 **3/3 稳定**、torch-seeded init + encoder forward **3/3 稳定**、**只有走 `init_emb` 的那条不稳** ⇒ k-means 是**唯一**污染源。**裁决：不改冻结核心**（§九 禁止改 RQ-VAE 核心实现）；C4 的 fixture tokenizer 改为"该层 residual 的 **seeded 子样本**"并加同进程重建断言。**连带更正 C3**：C3 的 fixture tokenizer 同样走 `init_emb` ⇒ [plan.md §15.8](plan.md) / [experiments.md §13.6](experiments.md) 里的**码分布计数（99 / 71）是一次抽样**，其结论（A/C 与 B 可分离）依据的"pair 轴逐位相同"**不受影响**。与 [[md5-is-not-a-data-fingerprint]] 记录的 parquet md5 漂移、C2 记录的 `WarpSampler n_workers` 属**同一类**"跨运行不可复现"陷阱。
7. **E10 的 CSG 守卫对 C4 假阳性（R10 已处理，非待决）**：`exp_e10_smoke.py:863` 是 `grep -rniE "csg_|lightgcn|graph_conv"`（**`-i`，大小写不敏感**）。C4 的常量原用 `CSG_` 前缀（为避开 C2/C3 已导出的 `MISSING_REFERENCE` / `DIRECTION_POLICY`），**恰好含 `csg_`** ⇒ E10 掉到 **63/64**。**裁决：改 C4 的名字（13 个 `CSG_*` → `QSL_*`、`qspace_csg_loss` → `qspace_constraint_loss`），不改 E10**（§九 禁止改 E1–E10）⇒ E10 回到 **64/64**。这不是绕过守卫：守卫意图是"CSG 线不得被**静默激活**"，而 C4 与 C3 同性质——`tiger_ext/` 下**未接入任何运行时**的独立目标函数（EXP-C4 `[8]` 逐条断言其不在任何运行时路径上）。

8. **C5 授权修改内核文件 ⇒ 既有 harness 的 tracked-file guard 必然失败（R11，已按用户裁定的「方案 3」处理，非缺陷、无需修）**：`rqvae/trainer.py` 成为**第 4 个被修改的 tracked 文件**，这是 [plan.md §C5](plan.md) 「仅受守卫的最小挂载」的**明文授权**。既有 7 份 harness 各自都有 `modified ⊆ PRE_*_MODIFIED` 的守卫断言，因此**每份至少掉 1 项**；**实测（2026-10-15，`EXP_C5_REGRESSION=all`）**：`exp_e8_sid_refresh` **54/55（1 条）**、`exp_e9_alternating` **80/81（1 条）**、`exp_e10_smoke` **60/64（4 条 = 自身 guard + 它重跑的两个子 suite 的 guard + 二者 aggregate）**、`exp_c1_cf_embedding` **130/132（2 条 = 自身 guard + 它内嵌的 E10 重跑）**、`exp_c2_collab_relation` **125/127（2 条，同上）**、`exp_c3_qspace_constraint` **96/99（3 条 = 两条自身 guard + 内嵌 E10 重跑）**、`exp_c4_qspace_loss` **84/87（3 条 = modified-set guard + 文本 guard + 内嵌 E10 重跑）**。**裁决（用户，2026-10-14）：不修改任何既有 harness，不把 `rqvae/trainer.py` 加进 `PRE_*_MODIFIED` 去掩盖**——那等于改测试定义。改为：① 这些 FAIL 在 C5 报告里**如实列出并逐行归因**；② **C5 自己的 harness 新增机器核验**（EXP-C5 §[8]，25 项）证明 `rqvae/trainer.py` 是**唯一**被改的正式文件、diff 只落在 `_train_epoch` 的授权挂载位、`L_recon` / `L_trans` / `L_balance` / `L_align` 计算逐字节未变、只新增 `lambda_csg * L_csg`、`λ=0` 不调用 CSG、**没有复制训练循环**、没有改 RQ-VAE 核心、没有新增优化器 / warmup / regularization；③ C5 的 PASS/FAIL **与既有 regression 分开报告**（[experiments.md §15.7](experiments.md)）。EXP-C5 §[10] 用 **R1/R2/R3** 三条规则把 7 份 harness 的**每一条** FAIL 机械归因（`unexplained: none` 已实测）。**这不是"绕过守卫"**：守卫意图是"新机制不得不经授权地悄悄接进内核"，而 C5 的接入是**授权 + 自证**的，且 `csg=None` 时逐位等价（EXP-C5 §[9] G0）。
9. **k-means 初始化与 batch 几何的交互会静默吃掉 RQ-VAE 自己的码本梯度（R11 新发现，非阻塞，但**任何 C5 之后的训练都必须知道**）**：`kmeans_init=True` 时 `rqvae/models/layers.py::kmeans` **用第一个 batch 拟合每层码本**。若该 batch 的 distinct 点数 ≤ `n_emb`，码本会把残差**逐位复现**⇒ `x_q == z`、`L_RQVAE` 自己对码本的梯度 ≈ 0。**实测**：batch=1024 ⇒ `max|x_q − z| = 4.211e-02`（干净）；batch=256 ⇒ `0.000e+00`（**被 C3 的 1e-4 重建守卫拒绝**）；`kmeans_init=False` batch=1024 ⇒ `1.101e+00`。**同一效应也污染梯度测量**：拟合 batch 上 `|dL_RQVAE/dW| ≈ 2e-10`，下一个 batch ≈ 3.0 ⇒ **任何梯度读数必须在"耗尽 k-means 初始化那个 batch"之后取**。EXP-C5 的 sanity 配置因此固定 batch=1024，且 §[4]/§[5] 的读数全部取自初始化之后。**待裁决（留给效果实验）**：正式训练是否沿用 `kmeans_init=True`（继承冻结配方）——本次不裁决，只记录。

10. ~~**`kmeans_init` 待裁决**~~ → ✅ **已由 C6 裁决（R12）**：**维持冻结配方 `kmeans_init=True`**。batch 1024 下 True/False 都过 C3 的两条首 batch guard（rebuild 2.384e-07 / 6.072e-07 合成、2.980e-08 / 6.566e-08 Toys，均 < 1e-4；latent 均 > 1e-4），但**码本覆盖率差距大**（首 epoch distinct：合成 `128/128/128` vs `22/20/65`；Toys `256/256/240` vs `42/46/256`；max_share 0.041/0.146 vs 0.321/0.523）⇒ True 明显更健康。**未引入新初始化机制、未改 RQ-VAE 核心、未改 batch size。**C5 的退化情形（`x_q == z` 逐位）是 **batch 几何**性质而非 flag 性质 ⇒ 继承的约束是"**每个首 batch 都要量**"（`kmeans_init=True` 时 `step0|dW|` 只有 1.184e-04 / 1.105e-04，比 False 小 1–2 个数量级，量级本身就依赖 batch 几何）。

11. **Mi 不能作为真实数据臂（R12 查实，非阻塞）**：CSG hook 走 `CSGHook.for_dataset` → `load_constraint_for_dataset`，必须加载 **C1 的 CF 表 + C2 的关系**。`integration/artifacts/` 下 `cf_embed/` 与 `collab_relation/` **只有 Toys / Sport / Beauty**；`Musical_Instruments` 既无 `behavior_emb_itemcf.parquet` 也无 SASRec checkpoint ⇒ **在 MI 上跑 CSG 需要新生成 C1/C2 产物，属于新范围**。C6 的真实数据臂因此是 **Toys**，该限制**如实记录而非绕过**。

12. **Toys 4096d 欠训练在 C6 里的具体读数（R12，属「已知问题」第 3 条的实例，非新问题）**：C6 的 Toys 臂（2 epoch、4096d、`limit_train=64`）训练出的 SID **collision rate ≈ 0.9225（λ=0）/ 0.9233（λ=1e-3）**，**Recall@5/10/20 与 NDCG@5/10/20 全为 0.0**。**这是欠训练的 confound，与 CSG 无关**——C6 不据此对约束作任何判断。**任何 Toys 效果实验开工前仍须先补训一个充分的 4096d Toys RQ-VAE**（第 3 条）。另：合成臂的 eval 在 1024 item / 1-epoch T5 下同样退化（全 0），其 λ=0 逐位等价的意义是"两边同为 0 的逐位相同"，**不是"评估有效"**。
13. ~~**B6（2026-10-19 新发现，硬阻塞）**~~ → ✅ **已修复（2026-10-19 晚，EXP-B6 31/31）**：`FixedSidTrainer` 新增 `embedding_path` / CLI `--embedding_path`（`None` 严格保持 `data_root/<dataset>/item_emb.parquet`）；显式表按给定文件读取、无回退；构造期比对 ckpt `encoder.mlp_layers.1.weight` 的 `in_dim` 与表列数，不符即**建模前**报 `embedding dimension mismatch`。真实 768d Toys 对（`Toys_Content768d_Baseline/Jul-31-2026_21-37-33` + `data/Toys/item_emb_768d_contentonly.parquet`）在 **`data_root=<repo>/data`（无 shadow root）** 下构造成功。⚠️ `data/Toys/fused_item_emb_768d.parquet` **实为 896d**（名字误导，见 [[toys-rqvae-fixture-dims]]）。详见 [plan.md §19.11.10](plan.md)。
14. ~~**B7（2026-10-19 查实，待裁决）**~~ → ✅ **已修复（2026-10-19 晚，EXP-B7 48/48）**：`qspace_train` 在 dataset/model/lazy-k-means 之前 `apply_seed(config.seed)`（python/numpy/torch CPU/CUDA + cuDNN pin，与 `rqvae/main.py:77-84` 同构）；DataLoader shuffle 用独立 `torch.Generator`；冻结 `layers.py:86 KMeans` 无 `random_state` 的问题由 `seeded_kmeans` **逐次注入显式 `random_state`** 解决（不改 `rqvae/` 文件；`num_clusters > B` 的硬编码 `RandomState(0)` 子采样保持不变并记录）。同 seed 跨进程 init/final/codes/loss 逐位一致；异 seed 三者均变。CUDA 仍不承诺逐位复现（atomics/cuDNN/worker 顺序，已写入 `CST_CUDA_NONDETERMINISM` 与每份报告）。详见 [plan.md §19.11.11](plan.md)。
15. **R1 审计的残余可读性缺口（2026-10-19，非阻塞）**：新 ckpt 的 `args` 缺 7 个可选辅助损失键，但经真实 `RQVAETrainer` 复核，**21/21 实际生效值 = R13 冻结值**（[plan.md §19.11.12](plan.md)）⇒ 不构成配置不确定性，不修改生产代码。仍有两项**事后可读性**缺口：① R13 的处理变量 `lambda_csg` **不在 ckpt 的 `args` 里**（只在运行报告），ckpt 本身无法自证属于哪个 λc 臂 ⇒ ✅ **R3 driver 已落记录**：每条臂在 `driver_run.json` 与逐臂阶段记录中写 `lambda_csg`（EXP-R3 105/105）；② `sid_refresh.ckpt_args` 白名单（11 键）既不含这 7 项也不含 `warmup_epochs` ⇒ SID provenance 无法自证这些值（行为不受影响）。下游消费者实测安全：`generate_code` 不读、`fixed_train` 不读 ckpt args、全仓 34 个 ckpt-args 读取文件 0 个触碰这 7 项。
16. ~~**R3 driver 的两个必须修复项（2026-10-19 独立复核发现，阻塞 EXP-L0）**~~ → ✅ **已于 2026-10-19 修复并通过独立复核（GATE = PASS）**：① **M1** `--arms ""`（空选择）⇒ 现由 `select_specs` 抛 `DriverError`，`main` rc=2、不建 run root、不写记录；`execute_run([])` 亦抛，杜绝零臂 `completed`。② **M2** 阶段抛异常（`Popen` 失败、`run.json` 损坏等）⇒ 现在先以 `status="failed"` 落盘 `driver_run.json`（`failure={arm,stage,type,message,traceback}`、`arms_completed`、`arms_not_started`）再 `raise`，在飞阶段由 `"running"` 改 `"failed"`，后续臂不启动。EXP-R3 **130/130**（+25 项）、独立 harness **REVIEW 46/46**、其余套件零新增失败；生产四哈希与 git 状态不变，**未启动 EXP-L0**。见 [plan.md §19.11.16](plan.md)。

### C6 记录的事实（不是待决项，是效果实验起必须知道的边界）

- **C6 是组合，不是新管线**：tokenizer 段走 C5 的 `train_tokenizer(hook=…)`（**不写 `phase_log`**），其余三段走 `TigerAlternatingPipeline`。因此**只有三条 `phase_log` 行是正常的**，不是遗漏；tokenizer 段的证据是 ckpt + `n_steps` + 有限 `best_loss`。**`alt_pipeline.py` 不得为改这一点而编辑**（会重开 E9 并破坏 C1–C4 的 sha256 pin）。
- **C3 的 guard 在训练每一步都被强制执行**：`CSGHook.forward` 每步调用 `QSpaceBatch.validate(rqvae)`。⇒ **"臂跑完了"本身就是 C3 两条首 batch guard 通过的证据**（不通过会抛）。测量首 batch 的 rebuild/latent 时**必须在同一次 forward 上取值**——优化器从第一步起就在移动码本，事后重算 `Σ_l W_l[I_l]` 会比出 2.9e-02 的假偏差。
- **`alt_pipeline` 的 `mounts` 字段不是开关**：`alt_pipeline.py:455` 是 `{"sia": mounts["sia"] is not None, ...}`，即"adapter 是否被构造"，**任何 recommender phase 都为 True，与 λ 无关**。判断 SIA/PSA 是否真的关掉，只能测 `SIAMount.forward` / `PSAMount.forward` 的进入次数（`main.py:162` 用 `sia is not None and sia.enabled` 分流；两者的 `forward` 在 disabled 时会**抛**）。
- **C6 的 harness 不修改任何既有 harness / E7 / E10，也不通过改守卫凑 PASS**；C5 授权修改导致的既有 harness tracked-file guard FAIL，**归因由 C5 §[10] 负责，C6 不重述**（C6 只机器核验 tracked 改动集合没有扩大）。
- **C6 无任何效果结论**：Recall/NDCG 只证明链路跑通。

### C1 记录的事实（不是待决项，是下游必须知道的边界）

- **ItemID 映射**：Toys / Sport 的 `item_emb.parquet` ItemID 就是 `1..N`；**Beauty 是 `1..N` 的排列，4158 行 `ItemID != 行号+1`** ⇒ **任何按行取值的代码在 Beauty 上都会静默取错**。C1 一律走显式映射表；C2 起必须沿用同一张表。
- **正式 CF 源 = SASRec item embedding（50d）**，不是 ItemCF-128。`sasrec_output/<D>/seed2025_best.pth`；`item_emb.weight` 是 `(N+1, 50)`，第 0 行 padding 全零。
- **两种来源冷启动语义不同**：SASRec **无全零行**（未训练 item 保留随机初始化）；ItemCF **有全零行**（24/28/33）。下游若用"零向量 = 冷启动"的判据，**只对 ItemCF 成立**。
- **维度不是契约**：`CFEmbedding` 不假定固定维度（50 与 128 同走一个类），**C2 起不得硬编码 50**。
- **`torch.from_numpy` 共享 buffer**：`to_tensor()` 返回的是**副本**；下游若拿到原始 numpy 表做原地归一化，会改写所有消费者看到的表。

### C2 记录的事实（不是待决项，是 C3 起必须知道的边界）

- **行空间 = C1 的行空间**：关系第 i 行就是 C1 表第 i 行 = `item_emb.parquet` 的 `ItemID` 列第 i 项。**禁止 `item_id - 1`**；Beauty 上错位 4158 行。artifact 自带 `item_ids` + 顺序敏感 sha256，**必须用它校验后再消费**。
- **关系是有向的**：`neighbors[i]` 是 i 选的人，**不等于**"选 i 的人"（互反率仅 ~0.6–0.7）。C3 若需要无向邻接，**必须自己显式对称化并记录该决定**，不能假定已对称。
- **`-1` / `NaN` 是合法值**：出现在 `neighbors` / `similarity`，含义是"该行没有第 j 个邻居"（零范数行整行为哨兵、普通行也可能因 `k_eff` 不足补哨兵）。**下游取邻居前必须先过滤哨兵**，`rel.neighbors_of()` 已代劳。
- **k = 20 是继承值不是搜索值**：直接来自 CSG4Rec `--knn_k` 默认；**C3 若想改 k 属新的设计决定**，须另行记录，不得静默改。
- **共现探针（≈12.7× / 5.5× / 3.9×）是映射诊断不是效果证据**（前缀序列 ⇒ 有循环性）。**C3 起不得把它当作"CSG 有效"的论据引用**。
- **C2 是旁支产物**：不接入任何人的输入，不改 `model/` / `rqvae/` / `clean_tiger/`，不新增 projection / loss / warmup。

### C3 记录的事实（不是待决项，是 C4 起必须知道的边界）

- **Q-space = `x_q` = 码本查询之后、straight-through 之前**（`Σ_l codebook_l[indices_l]`）。**不是 encoder latent**（C3 禁止 latent alignment），**不是 E3 code logits**（logits 由量化前的 latent 算出，约束它 == 约束 encoder 向量），**不是重建输出**。
- **公开接口只有两个**：`capture_qspace(rqvae, x, rows)` → `QSpaceBatch`（含 `x_q` / `indices` / `rows` / `diagnostics`），与 `QSpaceConstraint.forward(quant)` → `ConstraintBatch`。**`rebuild_quantized` 是验证专用路径**（定义 1 次、调用 1 次，且调用点在 `validate()` 内），**C4 不得复用它做前向重建**。
- **`x_q` 携带 straight-through 梯度**（指向 encoder）。C4 若对它求 loss 并 backward，梯度会**穿过量化器回到 encoder** —— 这是 RQ-VAE 的正常行为，但**与"冻结 RQ-VAE"的阶段契约（E7）叠加时必须显式处理**（C4 决策点）。
- **pair 是 batch 内产物，不是全局表**：`ConstraintBatch` 每个 forward 重新算，**不落盘、不缓存**。任何依赖"全库 pair"的设计都不成立。
- **pair 有向、不对称**；需要无向须**由 C4 显式对称化并记录该决定**（C2 已明确留下这条）。
- **batch 内 pair 数是几何量**，与数据集规模强相关（Sport 0.46/item vs Toys 2.88/item）；**不得作为约束强度或效果指标引用**。
- **`lambda_csg` 默认 0**；`lambda_csg == 0` 时 `forward()` **抛 `QSpaceDisabled`**（与 `SIAMount` / `PSAMount` 的"guard on enabled"一致，**不是**返回零 loss）。C4 接训练循环时**必须先判 `enabled`**。
- **C3 不改内核**：`git status --porcelain` 自 E6 结束起逐字未变。

### C4 记录的事实（不是待决项，是 C5 起必须知道的边界）

- **ST 通路完好但 objective 未使用**：C4 的 `L_csg` 只把梯度送到**码本**（indices 来自 `argmin/argmax`，无梯度）——**这与参考实现的路由相同**。C5 若要让约束反传到 encoder，那是**新决策**（不是复刻），参考实现的路由只到码本。
- **C4 不算总 loss、不调 optimizer、不 backward 总 loss**；`lambda_csg` 只是 `QSpaceLossMount` 上的乘子，**默认 0**。
- **空 pair ⇒ `0.0`（数据结果）；开关关闭 ⇒ 抛 `QSpaceDisabled`（接线错误）**。C5 必须先判 `enabled`（与 `SIAMount` / `PSAMount` 一致）。
- **同码 pair 在该层丢弃并计数**是**适配**（参考实现由构造保证正样本不同码，`vq.py:77-78`）。
- **层内归约是 pair 均值，不是 anchor 均值**：多邻居 anchor 按其邻居数加权（实测两者相差 0.2826）。**不是 multi-positive**（实测 1.3329 ≠ 1.5353）。
- **C2 的 weight 已确认不被消费**（`diversity_loss` 签名无 weight 参数，`vq.py:68`）；C5 若要用它做加权，那是**新机制**，须另行记录。
- **`verify_indices_match_qspace` / `rebuild_quantized` 是校验路径**（`@torch.no_grad`），不是前向路径——C5 若要前向重算 `x_q`，仍必须用 `RQVAE.forward(..., return_xq=True)` 返回的那个张量（`rqvae/models/rqvae.py:64-67` 的警告）。
- **C4 的常量前缀是 `QSL_*`，不是 `CSG_*`**：`tiger_ext/` 里任何含 `csg_` 的标识符都会触发 E10 的守卫（见「已知问题」第 7 条）。
- **C4 不改内核**：`git status --porcelain` 自 E6 结束起逐字未变；EXP-C4 用 **6 个 sha256 pin** 证明 C1 / C2 / C3 / E7 / SIA / PSA 逐字节未变。

### C5 记录的事实（不是待决项，是 C6 / 效果实验起必须知道的边界）

- **`rqvae/trainer.py` 已被改动，且是唯一被改的正式文件**（`+34/−5`，7 hunks，`sha256(diff)[:16] = a8da64e3291fa1c5`）。改动只在两处：`Trainer.__init__` 多一个 `csg=None` 关键字（存 `self.csg`）+ `_train_epoch` 里 `L_align` 之后的一段受守卫挂载。**`L_recon` / `L_trans` / `L_balance` / `L_align` 的表达式逐字节未变**；`clip_params` 未变（`CSGHook` 无参数，已断言）。
- **`csg=None`（默认）时逐位等价**：EXP-C5 §[9] 用 `git show 5edfca0:rqvae/trainer.py` 做 pristine 源，同进程同 loader 同 seed，(a) 全默认与 (b) `lambda_balance=1.0` 两种配置下**逐步 loss 逐位相同、最终 `state_dict` 23/23 张量逐位相同**。**⇒ 任何沿用默认构造的既有调用方（含 E9 的 `alt_pipeline.tokenizer_phase`）行为不变。**
- **总 loss 的确切形式**：`L_total = L_RQVAE + lambda_csg * L_csg`，其中 `L_RQVAE = loss_recon + quant_loss_weight * quant_loss`（**定义未动**）。`lambda_csg` **只乘一次**（在 `QSpaceLossMount` 内），且**只有 `requires_grad and isfinite` 才加进 `loss`**。
- **梯度只到码本，不到 encoder**：`Σ|dL_csg/dW_l|` = 7.99e-04 / 3.41e-05 / 2.62e-05（三层），**0 个 encoder 张量**。这是**参考实现自己的路由**（`CSGT/models/vq.py:195` loss 在 `:213` 的 straight-through **之前**）。**走 ST 到 encoder 是可达的（实测 10 个 encoder 张量）但未采纳**——要采纳必须重定义 `L_csg` 或改 RQ-VAE 核心，两者都在边界外。**记录为边界冲突·待裁决，不是遗漏。**
- **`λ=0` ⇒ `n_calls == 0`**：trainer 先判 `self.csg.enabled`，所以关闭时 `CSGHook.forward` **根本不被调用**。**`QSpaceDisabled` 只在接线错误时才可能抛出**（C4 语义：关闭即抛，不静默返回人造零）。
- **权重是尺度匹配的，不是从论文抄的**：参考自己的同形项（`diversity_loss`）权重是 `--beta 0.0001`（`train_tokenizer.sh:8`）⇒ `1e-4` 是**尺度匹配锚点**；`λ·E[L_csg]/E[L_RQVAE]` = **1.551% / 15.535% / 154.972%**（λ=1e-4/1e-3/1e-2）。**C5 只报告不选择工作点**——选点是效果实验的事。
- **C5 的 sanity 配置与冻结配方有一处刻意差异**：`batch_size` 固定 **1024**（见「已知问题」第 9 条，batch=256 在 Toys 前 2048 行上会让 k-means 首 batch 逐位复现残差、码本梯度归零）。**这不是改配方，是为了让测量有意义**；正式训练是否沿用 `kmeans_init=True` 仍待裁决。
- **历史机制在 CSG 运行时保持惰性**：`L_trans` / `L_balance` / `L_align` 在 λ>0 的 20 步里各调用 **0 次**，且**阴性对照**（`lambda_balance=1.0`）证明探针能触发（20 次）。
- **C5 的机器核验不覆盖既有 harness 的守卫**：既有 7 份 harness 各掉 1 项 tracked-file guard，**逐行归因后 `unexplained: none`**（见「已知问题」第 8 条）。**C6 起若再动 `rqvae/trainer.py`，这些 FAIL 的集合会变，必须重测而不是外推。**
- **C5 是新驱动入口，不是新机制**：`tiger_ext/qspace_train.py` 提供 `CSGHook` / `CSGTrainerConfig` / `train_tokenizer` / `build_parser`；**它不改 `rqvae/main.py`**，也不复制训练循环（EXP-C5 §[8] 有 AST 核验）。
- **⚠️ C5 无任何效果结论**：全部数字都是**工程读数**（进没进 backward、梯度去哪、关掉是否等价、尺度多大）。**没有 Recall / NDCG，没有"约束有效"的论据。**

---

## 下一步（效果实验开工前须确认）

1. **C5（CSG trainer integration）**：✅ **已完成（R11）**——`L_total = L_RQVAE + lambda_csg * L_csg` 已实测进入 backward，梯度只到码本，`csg=None` 逐位等价，历史机制惰性，既有 harness 的 1 项 guard FAIL 已逐行归因。**三个前置决策点全部落地**：① 挂载路径 = `rqvae/trainer.py::_train_epoch`（tokenizer 阶段；CSG 作为 `L_RQVAE` 的一项自动继承 E7 的 tokenizer-stage-only 契约，**E7 未改**）；② 梯度路由 = **仅 codebook（P1，采纳）**，ST→encoder **实测可达但不采纳**，记为**边界冲突·待裁决**；③ E7 接口 = **不需要改**——`tiger_ext/qspace_train.py` 是纯旁路驱动，E9 的 `alt_pipeline.tokenizer_phase` 仍以 `csg=None` 构造 trainer。详见 [plan.md §17](plan.md) / [experiments.md §15](experiments.md)。
2. **C6（CSG smoke test）**：✅ **已完成（R12）**——`EXP-C6 63/63, exit 0`。四臂（λ=0/1e-4/1e-3/1e-2）+ Toys（λ=0/1e-3）全链跑通；CSG 真的进入训练且梯度只到码本；λ=0 与 Clean TIGER 全链逐位等价；E7/handoff/历史机制/SID 泄漏检查全绿；`kmeans_init` 已裁决为维持冻结的 `True`；regression 内重跑 C5 得 99/99。**无任何效果结论。**
3. ~~**效果实验开工前须裁决的第一件事：`lambda_csg` 工作点**~~ → **已由 R13 定为「先扫后定」**（EXP-L0，5 路 × Toys × seed 2025，预注册选点规则见 [plan.md §19.4](plan.md)）。候选域不变：`{1e-4, 1e-3, 1e-2}` 相对贡献 = **1.551% / 15.535% / 154.972%**（`E[L_RQVAE]=0.035458`、`E[L_csg]=5.508495`）。**同时新增一个未登记决策点**：`lambda_sia = lambda_psa = 0.1` 只是 `alt_pipeline.py:187` 的实现默认值，plan.md 从未记录其来源 ⇒ 一并进 EXP-L0 标定（`{0.1, 1.0}`）。
4. ~~**效果实验开工前须裁决的第二件事：`kmeans_init`**~~ → ✅ **已由 C6 裁决（R12，见「已知问题」第 10 条）**：维持冻结配方 **`kmeans_init=True`**。C5 的退化情形是 batch 几何性质，继承的约束是"每个首 batch 都要量"。
5. **效果实验开工前须裁决的第三件事：ST→encoder 路由（边界冲突）**。要让 `L_csg` 反传到 encoder，必须**重定义 `L_csg`**（换掉 objective 的 query 张量）或**改 RQ-VAE 核心**（暴露逐层 ST 值）——两者都被 §九 / C4 继承边界禁止。**参考实现本身也只到码本**，所以采纳 P1 不是妥协。若后续确要走 encoder，**这是一次显式的新授权，不是"补上 C5 漏掉的东西"**。C6 实测在 smoke 配置下 encoder 张量 **0/6** 受影响（P1 保持）。
6. **效果实验起须遵守 C2 留下的 6 条事实**：① 行空间同 C1（禁止 `a-1`，Beauty 错位 4158）；② 关系**有向**，需要无向须显式对称化并记录；③ `-1` / `NaN` 是合法哨兵，取值前必须过滤；④ **k=20 是继承值**，改 k 属新设计决定；⑤ 共现探针**不是效果证据**，不得援引；⑥ 不改 `CollabRelation` 核心接口。
7. **效果实验起仍须遵守 C1 留下的 5 条事实**：① 维度不得硬编码（50 与 128 都合法）；② 一律走显式 `ItemID → 行` 映射（禁止 `a-1`）；③ "全零行 = 冷启动"**只对 ItemCF 成立**（SASRec 源无全零行）；④ 未知 id 默认抛 `MissingItemError`，要零向量须显式 `missing="zero"`；⑤ 不改 `CFEmbedding` 核心接口。
8. **效果实验起须遵守 C5 留下的 10 条事实**（见上文「C5 记录的事实」）与 **C6 留下的 5 条事实**（见上文「C6 记录的事实」），其中与效果实验直接相关的四条：① **再动 `rqvae/trainer.py` 就会改变既有 harness 的 FAIL 集合** ⇒ 必须重测不能外推；② **`QSpaceDisabled` 只在接线错误时抛出**，空 pair 才是 `0.0`，两者不可混同；③ **C5 的 batch=1024 是测量条件不是配方**，且 C6 已裁决沿用 `kmeans_init=True` + 每个首 batch 都要量；④ **`alt_pipeline` 的 `mounts` 字段不是开关**，判断 SIA/PSA 是否真的关掉必须测 forward 进入次数。
9. **E 线已收口**：E1–E10 实现部分全部完成，可执行 **EXP-E7-E9**（含"阶段生效统一验证项"：逐阶段打印参数状态 × loss 状态矩阵作为审计依据）。**EXP-E7-E9 的调度机器已由 E9 实现并验证**。
10. **B / C 类效果实验已解锁，但被 R13 的开工门禁前置**：先落 **P1 / P2** 两条接线（P1 已按差分回归证明默认路径不变：E9 90/90、E10 62/64 行逐字节相同、零新增失败；P2 为纯新增文件，不改变任何既有代码路径），再跑 **Stage 0（EXP-L0）→ Stage 1（EXP-B1）→ Stage 2**。~~开工前须先解决「已知问题」第 3 条~~ → **R13 改走 768d content-only 绕开该阻塞**（768d Toys collision 0.000419；4096d 的 ≈0.92 只影响 4096d 实验）。门禁清单见 [plan.md §19.10](plan.md)。
    ⚠️ **2026-10-18 P2 后门禁 = CONDITIONAL**：差分回归 ✅（E9 87/90、E10 60/64，零新增失败）。B1（选错 embedding 表）/ B2（无 warmup 50）**已修复**（EXP-P3 29/29），**B3（历史 ckpt 相对路径）已修复**（EXP-B3 30/30），**B4（lineage 不自检）已修复**（EXP-B4 38/38）。**2026-10-19 修后逐项核对已完成**（见上「修后配方逐项核对」）：λc 配方通过、λe 冻结 SID 逐位一致并已登记。**2026-10-19 晚：B6 已修复**（`embedding_path`/`--embedding_path`；EXP-B6 31/31，P2 47/47，B4 38/38，E8 54/55 零新增），推荐器侧不再阻塞；**B7 亦已修复**（`apply_seed` + `seeded_kmeans`；EXP-B7 48/48，P3 29/29，C4 84/87 零新增，C6 63/63）；**R1 已审计关闭**（EXP-R1 44/44，7 键实际生效值 21/21 = 冻结值，不改生产代码）；**R2 亦已冻结关闭**（EXP-R2 274/274，五条命令逐参数核对）；**R3 driver 亦已落成**（EXP-R3 105/105，五臂独立 run root、动态产物识别、失败即停、`driver_run.json`）⇒ **独立门禁复核已于 2026-10-19 完成：CONDITIONAL**（M1/M2 两个必须修复项，均在 `integration/` 内）；修复并重跑复核前 **GATE 仍 CONDITIONAL，EXP-L0 不得启动**。见 [plan.md §19.11.9](plan.md)、[§19.11.10](plan.md)、[§19.11.11](plan.md)、[§19.11.12](plan.md)、[§19.11.13](plan.md)、[§19.11.14](plan.md)。
11. ~~**C1 决策点**~~ → ✅ **已裁决（R7）**：正式来源 = **SASRec item embedding（50d）**；ItemCF-128 保留为备用源。详见 [plan.md §3 C1](plan.md) 与 [experiments.md §11](experiments.md)。
12. ~~**C2 决策点**~~ → ✅ **已裁决（R8）**：k = 20（继承 CSG4Rec 默认，未搜索）、**有向**（不对称化）、**显式屏蔽对角线**去自身（不复刻参考实现的按位丢列）、零范数行**排除 + 哨兵**、并列按行号升序、pair 文件用**行索引空间 + ItemID 伴随件**（不复用历史 `trans_graph_path` 开关语义）。详见 [plan.md §14](plan.md) 与 [experiments.md §12](experiments.md)。

### 已闭环问题

- **prerequisite 环（R1 提出，R2 已解除）**：E4 / E6 曾把"loss 在正确阶段生效"列为验收项，而该验证需要 E7 的阶段机，与 `E7 前置 = E6` 形成环。**已按方案 A 解除**——E4 / E6 只验证 loss 机制本身；阶段生效统一由 **E7–E9** 验证（见 [plan.md](plan.md) E7 阶段生效统一验证项）。prerequisite 链保持 `E1 → E2 → E3 → E4 → E5 → E6 → E7 → E8 → E9`，**无环**。
- **status.md 的 `model/` 改动面断言**：R1 已修正（原句声称 E1 是唯一改 `model/` 的 Task，与 E4/E6 挂载到 Recommender 训练循环冲突）。
- **E5 期发现的 plan.md 冲突（R3 已解决）**：E5 实现时 PSA 的 similarity / temperature 在冻结文档中未钉死，当时按"最少发明"选了裸点积 / T=1.0 并上报为待追认项。**R3 通过源码核验裁决为严格复刻官方**（cosine / 0.07 / 双向求和），冲突关闭。
- **SIA logit 符号（R3 已解决）**：plan.md E3 原写"由 `-distance` 得到"，与 ETEGRec 官方 `dist='l2'` 分支的 `+d` 不符。R3 裁决**字面保真**改为 `+d`，plan.md 该处已同步修正。
- **阶段冻结未实现（R4 已解决）**：见「已知问题」第 1 条，由 E7 的 `StageFreeze` 关闭。
- **`vq.initted` 坑（R4 已解决）**：见「已知问题」第 2 条，由 E7 的 `load_tokenizer_state_dict` / `seal_codebooks` 关闭。
- **E7 与 ETEGRec 官方在 Tokenizer 阶段 loss 上的分歧（R4 已裁决）**：官方 `_train_epoch_id` 在该阶段**仍计算** `kl_loss` / `dec_cl_loss`；[plan.md §1.4](plan.md) 规定 `L_SIA` / `L_PSA` 仅 Recommender 阶段生效。裁决**从冻结计划**（计划是 E7 的显式契约；`warm_epoch` 与官方 trainer 全量复刻已在"未复刻"清单），差异已记入 [plan.md §10](plan.md)。
- **「已知问题」第 4(a) 条的编号错位（R5 已裁决）**：任务把 E8 说成"调度"，与计划的 E8 = SID refresh 错位。用户裁定**按计划的 E8 执行**，已交付；调度归 E9。
- **E9 周期与边界规则（R6 已解决）**：见「已知问题」第 4 条。1 phase = 1 epoch、1 cycle = 1T + 1R、**0-based** `epoch_idx % cycle`，与官方 `cycle: 2` 一致；1-based 会整体反相（脚本内有对照断言）。原先"待用户裁决"的留白由 R6 的 E9 实现 + EXP-E9（90/90）关闭。
- **C1 决策点（R7 已裁决）**：CF 向量原计划二选一（复用 ItemCF 文件 vs 按 CSG4Rec 原方法重新生成）。**用户裁定改用 SASRec item embedding（50d）**——因为 CSG4Rec 的 `--cf_emb` 本就是 SASRec 的 item embedding 产物，而 ItemCF-128 的语义不同。ItemCF 文件保留为备用源，不作正式来源。详见 [plan.md §13](plan.md) / [experiments.md §11](experiments.md)。
- **C2 决策点（R8 已裁决）**：k = **20**（继承 CSG4Rec `--knn_k` 默认，**未重新搜索**）；**有向**（余弦对称但 top-k 截断不对称，不做对称化以避免扩大设计）；自环用**显式对角线屏蔽**去除（**不复刻**参考实现的按位 `indices[:, 1:]`——已实测该写法在零范数行上仍会留下自身，而正式 SASRec 源上恰好看不出，属易被静默继承的缺陷）；零范数行**排除 + `-1`/`NaN` 哨兵**（源与靶双向）；并列按"相似度降序、行号升序"完全确定（`kind="stable"`，代价已实测并接受为离线产物）；pair 文件为 `(M,2)` **行索引**空间 + ItemID 伴随件 + `item_ids` sha256（**不复用**历史 `trans_graph_path` 的开关语义）。详见 [plan.md §14](plan.md) / [experiments.md §12](experiments.md)。
- **C3 决策点（R9 已裁决，非用户决策点，系源码核验裁决）**：① **Q-space 取 `x_q`**（码本查询后、straight-through 前）——依据是参考实现 `CSGT/models/vq.py:195` 与 `:68-106` 实际消费的就是这个量；**排除 E3 code logits** 的结构性理由 = logits 由量化前的 latent 算出（`vq.py:76,107`），约束它等于 latent alignment，是 plan.md C3 明文禁止项。② **CSG4Rec 的 kNN loss 本体不可复原**（`CSGT/models/rqvae.py` 在快照中缺失，无归档 / 无 `.pyc` / 无第二副本），因此 **C3 不实现 loss**，只交付其输入——这与用户"不实现 C4 最终 CSG loss"的边界正好一致，并把差异**显式记录**而非静默补造。③ **batch 内配对**取代参考实现不可复原的 out-of-batch 策略，先例取本仓库 `rqvae/trainer.py::_get_transition_loss:213-219`。④ **`lambda_csg` 默认 0 + 关闭即抛**，与 `SIAMount` / `PSAMount` 一致。详见 [plan.md §15](plan.md) / [experiments.md §13](experiments.md)。
- **R1（2026-10-19 审计关闭）**：新 ckpt `args` 缺 7 个可选辅助损失键（不是 4 个），但用**真实 `RQVAETrainer`** 从 ckpt args 构造后读出实际生效值，三份 ckpt × 7 字段 **21/21 等于 R13 冻结值**；默认值与 `5edfca0` 的 argparse 默认 7/7 相同，且下游 34 个 ckpt-args 读取文件均不触碰这些键 ⇒ **缺键安全、无需改生产代码**。残余可读性缺口（`lambda_csg` 不在 ckpt、`sid_refresh` 白名单不含）见「已知问题」第 15 条。EXP-R1 44/44。详见 [plan.md §19.11.12](plan.md)。
- **R2（2026-10-19 冻结关闭）**：两套 CLI 的默认值均不等于 EXP-L0 配方，故把五条命令逐参数冻结（3 CSG + 2 ETE），并用**真实解析器**核对 λ/epochs/batch/eval/warmup/seed/device/embedding/ckpt-dir、λ_sia=λ_psa、冻结 tokenizer、已登记 baseline SID + manifest、无 `--allow_unregistered_sid` ⇒ 不依赖任何默认值。同时记录两项非阻塞事实：两套 CLI 的 flag 拼写不同（`--embedding-path` vs `--embedding_path`），以及 `align_weight 0.0` 无 CLI 只能由 `getattr` 默认满足（R1 已复核）。EXP-R2 **274/274**（`CLOSED-BY-COMMAND-FREEZE`），P3 29/29、P2 47/47、B6 31/31。详见 [plan.md §19.11.13](plan.md)。
- **R3（2026-10-19 driver 落地，待独立门禁复核）**：新增 `integration/exp_l0_driver.py`，把 §19.11.13 的五臂命令与"tokenizer → refresh → 解析 SID → recommender"串接固定为唯一可执行入口；每次运行独立 `runs/<run_id>/<arm>/{ckpt,sid,run,logs}`，产物（checkpoint 时间戳目录、SID 版本）按前后差集动态识别，refresh 后交叉校验 manifest/sha/sidecar，任一失败即停且不回退旧产物；总记录 `driver_run.json` 落五臂 `lambda_csg/lambda_sia/lambda_psa`、命令、退出码、路径、哈希与 provenance。EXP-R3 **105/105**（`CLOSED-BY-MECHANISM-TESTS`），回归 R2 274/274、P1 27/27、P2 47/47、P3 29/29、B6 31/31 零新增失败。**未启动 EXP-L0**。详见 [plan.md §19.11.14](plan.md) / [experiments.md §17.6i](experiments.md)。
- **C4 决策点（R10 已裁决，系源码核验裁决）**：① **目标函数形式 = `CSGT/models/vq.py:68-106` 存活约束的形态**——`x_q × 码本` 裸点积 + `1e12` 自身屏蔽 + `F.cross_entropy(sim, 码字 id)`；**CSG4Rec 的 kNN loss 本体不可复原**（`CSGT/models/rqvae.py` 缺失），因此**显式记录为"参考形态而非复刻"**（`provenance()["is_a_reproduction"] = False`）。② **`emb` 经确认为"该 RQ 层自己的码本"**（`vq.py:69`），⇒ **level-wise 是被代码证据强制的**，不是我们添加的层级。③ **C2 的 cosine weight 不使用**——参考实现的 `diversity_loss` 签名里**没有** weight 参数，并**实测**目标函数对该 weight 逐位不变。④ **归约 = 层内 pair 均值 + 层间对有 pair 的层取均值**（`:92` + `rq.py:64`），**并实测拒绝** anchor 均值与 multi-positive。⑤ **梯度采用参考实现自己的路由**（只到码本），ST 到 encoder 的通路**实测完好但本 objective 不用**——是否使用**留给 C5**。⑥ **`lambda_csg=0` ⇒ 抛 `QSpaceDisabled`**（不静默返回人造零），与 `SIAMount` / `PSAMount` 一致；空 pair ⇒ `0.0`（两者严格区分）。⑦ **两条附带发现**：冻结 k-means 助手跨进程不可复现（见「已知问题」第 6 条，含对 C3 码分布计数的更正）；E10 的 CSG 守卫对 `CSG_` 前缀假阳性 ⇒ C4 常量改名 `QSL_*`（见第 7 条）。详见 [plan.md §16](plan.md) / [experiments.md §14](experiments.md)。
- **EXP-L0 指标口径错配（2026-10-19 只读审计关闭）**：基线 `NDCG@20=0.0341 / Recall@20=0.0795` 是 **test** 指标（seed 2025 单次，来源 [model/logs/tiger_content768d_baseline_toys.log](../model/logs/tiger_content768d_baseline_toys.log) 的 epoch 76，同 epoch valid = **0.039808 / 0.093606**，此前从未入表）；EXP-L0 `run.json.best_ndcg@20` 是 **valid**。把两者相减会凭空制造约 14% 的「提升」，`lc1e-4 +18.6%` 已撤回。同口径复核：test 对 test **−0.000569**，valid 对 valid **+0.000636**；**五臂在 test 上均未超过基线**，lc1e-4 的差距小于基线 3-seed 极差 0.0013 ⇒ 判「无差异」。模型选择只用 valid、test 不参与控制流（`n_test_evals == n_best_saves`），**无 test 调参泄漏**。Stage-0 规则按 test 口径重放：λc = 1e-4（唯一 Δ ≥ −0.001），λe = 0.1（无合格臂，回退锚点，记「未找到正向工作点」）。**CSG/ETE 的机制生效但没有可声称的效果增益；当前无 λ=0 的 fixed_train 对照臂，不能把「各解决什么问题」写成效果结论。** 权威：[doc/exp_l0_baseline_metric_audit.md](../doc/exp_l0_baseline_metric_audit.md)、[plan.md §19.11.19](plan.md)、[experiments.md §17.6n](experiments.md)。**未改代码/配置/产物，未重跑训练，未启动 EXP-B1。**
- **集成代码发布（2026-10-19）**：已推送独立分支 `integration-review-20261019` 到 `github.com:yeshuwei/tiger-TA`，提交 `7a578bac…`，父提交 = 基线 `5edfca08…`。**未改算法实现、未跑实验、未启动 EXP-B1、未删除服务器产物**；`main` / `experiments-2026-09-25` 远程仍为 `bc0e9ee5…`。上传 56 文件（tiger_ext 17 + integration 33 + doc 1 + 4 个可选钩子 + .gitignore）；排除 data/权重/SID/embedding/`integration/artifacts/`/缓存/日志。审查：无密钥、无 >10 MB 新文件、无二进制或符号链接。详见 [plan.md §19.11.20](plan.md) / [experiments.md §17.6o](experiments.md)。
