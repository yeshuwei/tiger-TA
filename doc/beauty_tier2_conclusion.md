# Beauty Tier-2 结论 — 6 变体 × Behavior AUC × T5 NDCG@20 齐备

**日期**：2026-09-24（运行 08:28:10 → 22:08:18）　**汇总**：2026-09-24 22:08:27
**驱动**：`rqvae/run_beauty_tier2_t5.sh`（PID 2911129）· 接力 `rqvae/run_beauty_tier2_chain.sh`（PID 2911653）
**汇总脚本**：`tools/beauty_tier2_summary.py`（`SUMMARY_EXIT=0`）
**原始汇总日志**：`logs/beauty_tier2_summary.log` ← **权威原始输出，本文件是它的成文版**

> ⚠️ 本文件由人写成文，**会被 `tools/beauty_tier2_summary.py` 重跑时覆盖**（脚本默认 `--report` 即指向此路径）。
> 需要重跑汇总时请追加 `--report doc/beauty_tier2_conclusion_rerun.md`。

---

## 0. 一句话结论

> Beauty 的 4 个缺失变体 T5 NDCG@20 已补齐（4 变体 × 3 seed = 12 次评测，**12/12 成功**）。
> **Beauty 上 Behavior AUC 排序与 NDCG@20 排序完全一致（Spearman ρ = +1.0000）**，
> 且分层结构（`TA-SID` 第 1 / `Shuffled` 第 6 / 行为变体 > 内容变体）与 Sport、Toys 同构。

---

## 0.5 实验配置（自包含 —— 不依赖 preflight 文档即可单读）

**数据集**：Beauty。items **12101**；交互序列 22363；行为 E 图 111649 边（window=3、仅相邻对、无向、`min_cooccur=1`）。
物品 id 空间与 `data/Beauty/item_emb_768d.parquet` 行序一致。

**输入 SID codes**（本次**只读、未生成**）：

| Variant | codes 文件 | shape / dtype / 取值 | 本次 |
|:--|:--|:--|:--|
| TA-SID | `data/Beauty/Beauty_ta_sid_768d_sid.npy` | (12101, 4) int64 [0,255] | 未重跑 |
| Content-only | `data/Beauty/Beauty_content768d_baseline_sid.npy` | (12101, 4) int64 [0,255] | 未重跑 |
| Raw-CoOcc | `data/Beauty/Beauty_rawcooc_768d_sid.npy` | (12101, 4) int64 [0,255] | 本次使用 |
| ItemCF | `data/Beauty/Beauty_itemcf_768d_sid.npy` | (12101, 4) int64 [0,255] | 本次使用 |
| Shuffled | `data/Beauty/Beauty_shuffled_768d_sid.npy` | (12101, 4) int64 [0,255] | 本次使用 |
| ZeroPad | `data/Beauty/Beauty_zeropad_768d_sid.npy` | (12101, 4) int64 [0,255] | 本次使用 |

**Generator（T5）架构与训练超参**（`model/main.py`，**本次未修改**）：

| 项 | 值 |
|:--|:--|
| backbone | T5 **from scratch**（非预训练），encoder 4 层 + decoder 4 层 |
| `d_model` / `d_ff` | 128 / 1024 |
| `num_heads` / `d_kv` | 6 / 64 |
| tokenizer `vocab_size` | **1025**（4 个 SID 位置 × 256 码 + 1 特殊符） |
| `max_len` | 20 |
| **可训练参数总量** | **4 590 848** ← 全流程架构指纹 |
| `num_epochs` | 200 |
| `lr`（optimizer） | 1e-4 |
| `batch_size` | 256 |
| `infer_size` | 96 |
| `early_stop` | 10 |
| seeds | **2025 / 2026 / 2027**（未改 seed 列表） |

**评测配置**（`tools/eval_trained_model.py`，**本次未修改**）：`--beam_size 30`；`--topk_list 5,10,20,50,100`
（**唯一与已有锚点的差异** —— 已有锚点为 `5,10,20`；已读源码核实只决定 `evaluate()` 打印哪几列，**不影响 NDCG@20**，见 §7-3）。

**本次未执行的上游配置**（声明边界，参数在此备案以便复现 Tier-1）：
RQ-VAE `num_emb_list=[256,256,256]`、`e_dim=32`、`layers=[512,256,128,64]`、
`sk_epsilons=[0.0,0.0,0.003]`、`epochs=3000`、`batch_size=1024`、`kmeans_init=True`、`seed=2024`。

**实际驱动命令**（逐字，4 变体 × 各 3 seed）：

```bash
bash rqvae/run_behavior_ablation.sh Beauty t5-only \
     data/Beauty/Beauty_<variant>_768d_sid.npy <variant>_768d 2025 2026 2027
```

其中 `<variant>` ∈ {`rawcooc`, `itemcf`, `shuffled`, `zeropad`}。`t5-only` 分支在 `run_t5` 之前即 `exit 0`，
A/B/C 段（behavior embedding / fuse / RQ-VAE / codegen）**物理上不可达**。
完整预检（含 154 份日志架构指纹、argparse 逐字段比对）见 [beauty_tier2_preflight.md](beauty_tier2_preflight.md)。

---

## 1. §六 逐 seed 结果（NDCG@20）

| Variant | seed2025 | seed2026 | seed2027 | mean | std |
|:--|--:|--:|--:|--:|--:|
| **TA-SID** | 0.0418 | 0.0425 | 0.0403 | **0.0415** | 0.0011 |
| **ItemCF** | 0.0383 | 0.0389 | 0.0402 | **0.0391** | 0.0010 |
| **Raw-CoOcc** | 0.0380 | 0.0400 | 0.0382 | **0.0387** | 0.0011 |
| **ZeroPad** | 0.0372 | 0.0371 | 0.0379 | **0.0374** | 0.0004 |
| **Content-only** | 0.0368 | 0.0352 | 0.0354 | **0.0358** | 0.0009 |
| **Shuffled** | 0.0348 | 0.0352 | 0.0337 | **0.0346** | 0.0008 |

- 前 4 行为 **Tier-2 新补**（2026-09-24 当天跑完）；后 2 行为 **已有锚点、未重跑**
  （TA-SID / Content-only 的 eval 日志 mtime 为 2026-08-02~08-03）。
- `std` 为样本标准差（`ddof=1`），n=3。
- 产出：`doc/Beauty_Tier2_NDCG_results.csv`（即本表格式）。

## 2. §七 合并表（Behavior AUC 读自 Tier-1，未重算）

| Variant | Behavior AUC | NDCG@20 |
|:--|--:|--:|
| TA-SID | 0.9560 | 0.0415 |
| ItemCF | 0.9463 | 0.0391 |
| Raw-CoOcc | 0.9377 | 0.0387 |
| ZeroPad | 0.8456 | 0.0374 |
| Content-only | 0.8365 | 0.0358 |
| Shuffled | 0.7974 | 0.0346 |

- Behavior AUC 来源：`doc/beauty_tier1_results.csv`，口径 `space=xq[full]`、`pos_regime=topk`、
  `auc_cos_degmatch`。**直接读取，未重新计算**。
- 产出：`doc/Beauty_Tier2_merged_auc_ndcg.csv`（`Variant | Behavior AUC | NDCG@20`）。

## 3. §八 Beauty 内排序分析（n = 6）

```
Spearman ρ = +1.0000   (p = 0)
Pearson  r = +0.9302   (p = 0.007135)
n = 6
AUC  降序: TA-SID > ItemCF > Raw-CoOcc > ZeroPad > Content-only > Shuffled
NDCG 降序: TA-SID > ItemCF > Raw-CoOcc > ZeroPad > Content-only > Shuffled
```

**两个排序逐位相同，无任何倒置。** Pearson 明显低于 Spearman，是量纲造成的：
AUC 只跨 0.16 个点（0.797→0.956），而 NDCG 跨 0.007 个点，且 `TA-SID` 在 AUC 上略高、
在 NDCG 上高得多 → 单调但非线性的映射。

> ⚠️ **照规格 §八 不过度解释**：n=6，两个序列还共用同一对端点（TA-SID 首 / Shuffled 尾），
> ρ=+1.000 的可用信息量远小于它的字面冲击力。**不给显著性断言**，只作方向性证据。

## 4. 补充观察（**超出 §八 要求**，供后续分析参考）

把同一口径套到 Sport / Toys 上（数据来自 `doc/beauty_tier2_three_dataset_table.csv`，未做任何
跨数据集归一化）：

| 数据集 | ρ(AUC, NDCG) | Pearson r (p) | AUC 降序 vs NDCG 降序 |
|:--|--:|:--|:--|
| Sport | **+0.9429** | +0.9775 (p=0.00076) | 倒置 1 处：`Content-only ↔ ZeroPad`（相邻） |
| Toys | **+0.9429** | +0.8933 (p=0.0165) | 倒置 1 处：`Raw-CoOcc ↔ ItemCF`（相邻） |
| **Beauty** | **+1.0000** | +0.9302 (p=0.00713) | **无倒置** |

**三个数据集共同的、稳定的部分**（AUC 与 NDCG **两个口径都成立**）：

```
TA-SID  >  {ItemCF, Raw-CoOcc}  >  {ZeroPad, Content-only}  >  Shuffled
```

**不稳定的部分**：花括号内部两对的先后次序在数据集间漂移，且漂移幅度都很小
（`{ZeroPad, Content-only}` 的 AUC 差：Sport 0.0011 / Toys 0.0035 / Beauty 0.0090）。

**★Beauty 的特别之处★**：Beauty 的 AUC 把 `ItemCF` 排在 `Raw-CoOcc` **之前**，而 Sport/Toys
的 AUC 是 `Raw-CoOcc` 在前（见下表）。也就是说——

- Sport：AUC `Raw-CoOcc > ItemCF`，NDCG 也 `Raw-CoOcc > ItemCF` → 自洽
- Toys：AUC `Raw-CoOcc > ItemCF`，NDCG 却 `ItemCF > Raw-CoOcc` → **不一致**（这是 Toys 那处倒置）
- Beauty：AUC `ItemCF > Raw-CoOcc`，NDCG 也 `ItemCF > Raw-CoOcc` → **自洽**

⇒ Beauty 的 NDCG 跟的是 **Beauty 自己的 AUC**，而不是 Sport/Toys 的排序。
**跨数据集不变的是「分层 + 两端」，不是「具体名次」。**

## 5. §九 三数据集整理表（**只整理，未算跨数据集相关性**）

`doc/beauty_tier2_three_dataset_table.csv`，18 行 = 3 数据集 × 6 变体，列为
`dataset, variant, behavior_auc_xq_full_topk, ndcg20`：

| dataset | TA-SID | ItemCF | Raw-CoOcc | ZeroPad | Content-only | Shuffled |
|:--|--:|--:|--:|--:|--:|--:|
| **Beauty** AUC | 0.9560 | 0.9463 | 0.9377 | 0.8456 | 0.8365 | 0.7974 |
| **Beauty** NDCG | 0.0415 | 0.0391 | 0.0387 | 0.0374 | 0.0358 | 0.0346 |
| **Sport** AUC | 0.9647 | 0.9397 | 0.9469 | 0.9069 | 0.9080 | 0.8763 |
| **Sport** NDCG | 0.0322 | 0.0295 | 0.0305 | 0.0280 | 0.0275 | 0.0264 |
| **Toys** AUC | 0.9651 | 0.9455 | 0.9543 | 0.8716 | 0.8751 | 0.8127 |
| **Toys** NDCG | 0.0377 | 0.0355 | 0.0340 | 0.0334 | 0.0336 | 0.0284 |

- NDCG 覆盖率 **18/18**（Beauty 补齐后无缺口）。
- ⚠️ 按规格 §九：**本步未对原始值做任何跨数据集相关性计算**。三个数据集的 NDCG 绝对水平
  不同（Beauty 0.035 / Sport 0.028 / Toys 0.034），跨数据集合并前需先做 dataset-wise
  z-score normalization —— 留待下一步指令。

## 6. §十 交付物：日志与 checkpoint 路径

逐 run 资源清单（含 `eval_finished_at` = 该 seed 评测完成时刻）：`doc/Beauty_Tier2_manifest.csv`（19 行）。

| Variant | train log（3 seed） | eval log（3 seed） | ckpt |
|:--|:--|:--|:--|
| Raw-CoOcc | `model/logs/Beauty_rawcooc_768d_seed{2025,2026,2027}_t5_train.log` | `model/logs/Beauty_rawcooc_768d_seed*_eval.log` | `model/ckpt/tiger_rawcooc_768d_beauty_seed*.pth` |
| ItemCF | `model/logs/Beauty_itemcf_768d_seed*_t5_train.log` | `model/logs/Beauty_itemcf_768d_seed*_eval.log` | `model/ckpt/tiger_itemcf_768d_beauty_seed*.pth` |
| Shuffled | `model/logs/Beauty_shuffled_768d_seed*_t5_train.log` | `model/logs/Beauty_shuffled_768d_seed*_eval.log` | `model/ckpt/tiger_shuffled_768d_beauty_seed*.pth` |
| ZeroPad | `model/logs/Beauty_zeropad_768d_seed*_t5_train.log` | `model/logs/Beauty_zeropad_768d_seed*_eval.log` | `model/ckpt/tiger_zeropad_768d_beauty_seed*.pth` |

**逐 seed 真实完成时刻**（唯一可信的进度时间轴；driver 日志里同一变体的 3 行是批量打印的）：

| 顺序 | eval 日志 | 完成时刻 | 耗时 |
|--:|:--|:--|--:|
| 1 | rawcooc seed2025 | 2026-09-24 09:23:29 | 55.3 min |
| 2 | rawcooc seed2026 | 10:23:01 | 59.5 min |
| 3 | rawcooc seed2027 | 11:35:39 | 72.6 min |
| 4 | itemcf seed2025 | 12:47:46 | 72.1 min |
| 5 | itemcf seed2026 | 13:58:29 | 70.7 min |
| 6 | itemcf seed2027 | 15:26:50 | 88.3 min |
| 7 | shuffled seed2025 | 16:40:22 | 73.5 min |
| 8 | shuffled seed2026 | 17:55:53 | 75.5 min |
| 9 | shuffled seed2027 | 18:57:32 | 61.7 min |
| 10 | zeropad seed2025 | 20:00:09 | 62.6 min |
| 11 | zeropad seed2026 | 21:03:38 | 63.5 min |
| 12 | zeropad seed2027 | **22:08:18** | 64.7 min |

总墙钟 **13 h 40 min**（平均 ~69.8 min/seed，与 §四-5 预估的 ≈1 h/seed 吻合）。

## 7. §十-5 三条声明（附可核验证据）

**1. 未修改 Tier-1 结果** ——
`doc/beauty_tier1_results.csv` sha256 = `ebc01ae471738824…`，mtime = **2026-09-23 23:59:05**，
**早于 Tier-2 启动时刻 2026-09-24 08:28:10**。Behavior AUC 全部从该文件读取，脚本中无任何重算路径。

**2. 未重新训练 RQ-VAE、未重新生成 SID** ——
- 4 份 codes（`data/Beauty/Beauty_{rawcooc,itemcf,shuffled,zeropad}_768d_sid.npy`）的 mtime 均早于 Tier-2 启动。
- 结构性保证：driver **只调用 `rqvae/run_behavior_ablation.sh` 的 `t5-only` 分支**，
  该分支在 `run_t5` **之前**即 `exit 0` → A/B/C 段（behavior embedding / fuse / RQ-VAE / codegen）
  **物理上不可达**，而非靠「小心别覆盖」。

**3. 未调整任何超参数** ——
- `model/main.py`、`tools/eval_trained_model.py` 均未修改（`git status` 无改动）。
- 全部 12 次运行只传 `--dataset_path / --code_path / --save_path / --log_path / --seed`，
  其余走默认；评测固定 `--beam_size 30`。
- `topk_list` 沿用 E49 的 `5,10,20,50,100`（已有 Beauty 锚点为 `5,10,20`）——
  已读源码核实该参数只决定 `evaluate()` 打印哪几列，**不影响 NDCG@20 的取值**
  （`tools/eval_trained_model.py` 第 91–101 行）。详见 [beauty_tier2_preflight.md](beauty_tier2_preflight.md) §四-3(d)。

---

## 8. 局限

1. **n = 6**：三个数据集的 ρ 都是 6 个点上的值，方向清晰但**不支持任何显著性/精度断言**。
2. **端点共享**：TA-SID 恒第 1、Shuffled 恒第 6，ρ 被端点主导；真正的自由度只在中间 4 个变体。
3. **AUC 与 NDCG 量纲不同**（见 §3），Pearson 系统性低于 Spearman 属预期，不是数据问题。
4. **Beauty 只有 1 个数据集反例/正例**：Beauty 与 Sport/Toys 的具体名次不同（§4），
   n=3 的数据集数不足以判定「哪种次序才是规律」。
5. 本文件的 NDCG 为 3-seed 均值；`topk_list` 差异（见 §7-3）虽已核实无影响，但仍是
   已有锚点与新增运行之间**唯一**的配置差异，如实记录在此。

## 相关

[beauty_tier1_conclusion.md](beauty_tier1_conclusion.md)（Behavior AUC 来源 / 三级分解）·
[beauty_tier2_preflight.md](beauty_tier2_preflight.md)（§四 四项检查 + 架构指纹）·
[e49_results_corrected.md](e49_results_corrected.md)（Sport/Toys 权威 NDCG）·
[e49grid_diag_summary.md](e49grid_diag_summary.md)（量化端 12 格诊断）·
[beauty_preflight_probe_conclusion.md](beauty_preflight_probe_conclusion.md)（§4.1 Tier-1 裁决闭环）
