# integration/ — CSG / SIA / PSA 集成与审计代码

本目录存放 TIGER 推荐器与 CSG4Rec（Q-space 约束）、ETEGRec（SIA / PSA）集成的
**验证 harness、驱动脚本与过程文档**。所有 harness 都是只读或写临时目录的验证程序，
不改变 `model/`、`rqvae/`、`tiger_ext/` 的生产行为。

## 文档

| 文件 | 内容 |
|---|---|
| `plan.md` | 冻结的集成计划、接口契约、各项裁决与门禁记录（权威） |
| `experiments.md` | 每个 EXP-* 验证项的方法、结果与结论边界（权威） |
| `status.md` | 当前状态、已闭环问题与已知问题 |
| `README.md` | 本文件 |

## 代码布局

- `tiger_ext/`（仓库根目录）：集成实现本体
  - `stage.py` — 阶段冻结（Tokenizer / Recommender）
  - `sia.py` / `sia_loss.py` / `sia_codes.py` — SIA adapter 与 ETEGRec 语义 ID 损失
  - `psa.py` / `psa_loss.py` — PSA adapter 与其损失
  - `qspace_constraint.py` / `qspace_loss.py` / `qspace_train.py` — CSG4Rec Q-space 约束、目标与 tokenizer 训练入口
  - `sid_refresh.py` — 用冻结 tokenizer 重新生成 SID
  - `fixed_train.py` — 推荐器阶段的冻结配方训练/评测循环
  - `alternating.py` / `alt_pipeline.py` — 交替训练与流水线
  - `cf_embed.py` / `collab_relation.py` — 协同过滤向量与 kNN 关系构造
- `integration/exp_*.py` — 逐项验证 harness（EXP-E*, EXP-C*, EXP-B*, EXP-P*, EXP-R*）
- `integration/exp_l0_driver.py` — EXP-L0 五臂的唯一执行入口
- `integration/summarize_l0_run.py` — 只读的 `run.json` 摘要工具

## 运行方式

从仓库根目录运行；大多数 harness 自带 `--help` 或模块 docstring 中的示例：

```bash
python integration/exp_e1_consistency.py            # 单文件 harness
python -I integration/exp_l0_driver.py --dry-run    # 只打印计划，不写盘
```

`exp_l0_driver.py` 通过冻结的 CLI 调用：

```bash
python -m tiger_ext.qspace_train   # kebab-case 参数
python -m tiger_ext.fixed_train    # snake_case 参数
```

## 环境

以下版本为集成实验实际使用的环境（仓库根的 `requirements.txt` 是早期 TIGER 的历史快照，
不适用于本目录）：

- Python 3.10.12
- torch 2.6.0+cu124（CUDA 可用时自动使用 `cuda:0`）
- numpy 1.26.4, pandas 1.5.3, pyarrow
- scikit-learn 1.3.2（`rqvae/models/layers.py` 的 KMeans）
- scipy 1.11.4, transformers 4.57.6, tqdm 4.67.1

## 不在本仓库中的数据与产物

以下内容按设计不纳入版本控制（见仓库根 `.gitignore`）：

- `data/` 下的用户交互序列、物品 embedding、parquet 划分；
- `*.pth` / `*.npy` / `*.pt` / `*.parquet` / `*.pkl`；
- `integration/artifacts/`（每次运行的 checkpoint、SID、日志与结果记录）；
- `clean_tiger/_smoke/`（Clean TIGER 的 smoke 数据与 checkpoint）。

因此部分 harness 在全新 clone 上需要先准备本地数据（例如 `clean_tiger/_smoke/Toys`、
`data/Toys/{train,valid,test}.parquet` 与 768d content-only embedding）才能运行；
它们的**断言逻辑**仍可独立阅读与审计。

## 约定

- `integration/artifacts/` 永远是**运行产物**，不是源码；不要提交。
- 机制 smoke / harness 通过**不是效果证据**，不能用于选点或声称增益。
- 任何「未确认」事项在对应 `experiments.md` / `status.md` 条目中显式标注。
