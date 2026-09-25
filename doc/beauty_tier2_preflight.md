# Beauty Tier-2 开始前检查报告（§四）

**日期**：2026-09-24　**对应规格**：用户 Tier-2 规格 §四「开始前检查」
**驱动**：`rqvae/run_beauty_tier2_t5.sh`（PID 2911129）· 接力 `rqvae/run_beauty_tier2_chain.sh`（PID 2911653）
**启动时刻**：2026-09-24 08:28:10

---

## 结论：**四项检查全部通过，未发现需要停机的配置不一致。**

---

## §四-1　待补变体的 SID codes 存在性

规格要求确认 `Beauty_{variant}_768d_sid.npy` 存在。**实测 6/6 存在**，且形状/取值完全一致：

| Variant | 文件 | shape | dtype | 取值 | L0distinct | 状态 |
|:--|:--|:--|:--|:--|--:|:--|
| TA-SID（已有） | `Beauty_ta_sid_768d_sid.npy` | (12101, 4) | int64 | [0,255] | 204 | 不重跑 |
| Content-only（已有） | `Beauty_content768d_baseline_sid.npy` | (12101, 4) | int64 | [0,255] | 83 | 不重跑 |
| **Raw-CoOcc（待补）** | `Beauty_rawcooc_768d_sid.npy` | (12101, 4) | int64 | [0,255] | 233 | ✅ |
| **ItemCF（待补）** | `Beauty_itemcf_768d_sid.npy` | (12101, 4) | int64 | [0,255] | 153 | ✅ |
| **Shuffled（待补）** | `Beauty_shuffled_768d_sid.npy` | (12101, 4) | int64 | [0,255] | 214 | ✅ |
| **ZeroPad（待补）** | `Beauty_zeropad_768d_sid.npy` | (12101, 4) | int64 | [0,255] | 156 | ✅ |

- `rows = 12101` 与 `data/Beauty/item_emb_768d.parquet` 的 item 数**一致** ✅
- 取值上限 255 < `vocab_size=1025` **不越界** ✅

## §四-2　T5 训练脚本配置

**唯一权威驱动**：`rqvae/run_behavior_ablation.sh` 的 **`t5-only` 分支**。
本步**逐字复用**该分支（与 Sport/Toys E49 调用的 `run_t5()` 是同一段代码），未复制出新实现。

> 选 `t5-only` 而非全链路模式，是因为该分支在 `run_t5` 之前即 `exit 0`，
> **物理上不可能执行 A/B/C 段（behavior embedding / fuse / RQ-VAE / codegen）** ——
> 这直接坐实了规格 §二-1「不重新训练 RQ-VAE」与 §二-2「不重新生成 SID」。

## §四-3　与 Sport/Toys 的配置一致性

### (a) 架构指纹：**全部 154 份日志只有一个参数量**

```
Total trainable parameters: 4590848     ← 唯一值，出现 154 次（跨全部数据集/变体/seed）
```

### (b) argparse dump：**逐字段完全相同**

| 字段 | Beauty（已有 TA-SID / Content-only） | Sport（E49 rawcooc） | 一致 |
|:--|:--|:--|:--:|
| `num_layers` | 4 | 4 | ✅ |
| `num_decoder_layers` | 4 | 4 | ✅ |
| `d_model` | 128 | 128 | ✅ |
| `d_ff` | 1024 | 1024 | ✅ |
| `num_heads` | 6 | 6 | ✅ |
| `d_kv` | 64 | 64 | ✅ |
| `vocab_size`（tokenizer） | 1025 | 1025 | ✅ |
| `max_len` | 20 | 20 | ✅ |
| `num_epochs` | 200 | 200 | ✅ |
| `lr`（optimizer） | 1e-4 | 1e-4 | ✅ |
| `batch_size` | 256 | 256 | ✅ |
| `infer_size` | 96 | 96 | ✅ |
| `early_stop` | 10 | 10 | ✅ |
| `beam_size` | 30 | 30 | ✅ |

### (c) 评测脚本

`tools/eval_trained_model.py` —— **同一份文件**，未修改（`git status` 无改动），
固定 `--beam_size 30`。

### (d) ⚠️ 一个**不要求检查、但确实存在**的差异（已核实无影响）

| | `topk_list` | eval 输出列 |
|:--|:--|:--|
| 已有 Beauty 锚点（7-8 月） | `5,10,20` | @5 @10 @20 |
| Sport/Toys（E49）与本步新增 | `5,10,20,50,100` | @5 @10 @20 @50 @100 |

**不影响 NDCG@20 的取值**。依据（读 `tools/eval_trained_model.py` 源码）：
`topk_list` 仅被传入 `evaluate(...)` 用于**决定打印哪几列**（第 91–101 行），
top-20 排序与打分本身不依赖它。实测印证：`e49_summary.NDCG_RE`
从两种格式中取到的第 3 列都是 @20（3 列格式 → 第 3 列 = @20；5 列格式 → 第 3 列仍 = @20），
且 `Sport_rawcooc` 的 0.0311 与五列输出 `0.0186 0.0242 0.0311 0.0347 0.0347` 的第 3 位吻合。

⇒ 本步**沿用 E49 的 `5,10,20,50,100`**（与 Sport/Toys 一致，是上游列的超集）。
该差异**不构成规格 §四-3 所列六项（backbone / tokenizer / epoch / optimizer / batch size / eval script）的不一致**，
故未触发「先停止并报告」条件，此处作为**已知且已核实的差异**如实记录。

## §四-4　冲突与覆盖检查

四个新 tag（`rawcooc_768d` / `itemcf_768d` / `shuffled_768d` / `zeropad_768d`）下
`model/ckpt/tiger_*_beauty*.pth` 与 `model/logs/Beauty_*` **均为空** →
既不会误跳过，也不会覆盖任何既有产物。`run_t5` 自带幂等（已存在的 eval 日志跳过），可中断续跑。

## §四-5　预期耗时

实测 `rawcooc_768d` seed2025 前 6 个 epoch ≈ 20.3 s/epoch（14s 训练 + ~15s 验证）。
- 若跑满 200 epoch：≈ 1.1 h/seed
- 参照已有 Beauty 实测（TA-SID seed2026 = 1h13m、seed2027 = 1h01m，早停生效）：**≈ 1 h/seed**
- **合计 4 变体 × 3 seed ≈ 12–13.5 h**

## 相关

用户 Tier-2 规格 §四 · [beauty_tier1_conclusion.md](beauty_tier1_conclusion.md)（Behavior AUC 来源）·
[e49_results_corrected.md](e49_results_corrected.md)（Sport/Toys 权威 NDCG）·
`rqvae/run_behavior_ablation.sh`（E49 权威驱动）
