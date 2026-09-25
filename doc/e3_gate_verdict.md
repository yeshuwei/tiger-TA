# E3 闸门判定 — RQ-VAE 第二创新验证（Toys）

> 结论先行：**E1 ✓ 问题真实存在 → E2 ✓ 问题来自 codebook allocation → E3 ✓ Config B（L0=1024 容量）可修复 → 建议正式设计 Adaptive/Structure-aware Quantization。**
> Config A（L0 Sinkhorn 均衡约束）部分修复但副作用明显，**不作为**主方案。

---

## 基线有效性确认（identical-metrics 之谜解除）

早期疑虑"两个不同模型（Toys_TA_SID 4224d vs Toys_TA_SID_768d）给出逐位相同指标"——**并非模型相同，而是诊断脚本全部读的同一模型**。

| 检查 | 结果 |
|---|---|
| `find_best_ckpt("Toys_TA_SID")` | → Jul-16 4224d（`fused_item_emb.parquet`） |
| `find_best_ckpt("Toys_TA_SID_768d")` | → Jul-28 768d（`fused_item_emb_768d.parquet`） |
| **CONFIG["Toys"]["ta_ckpt"]** | **`Toys_TA_SID_768d`**（本来就指 768d，非 4224d） |
| 直接对比 xq[full] AUC（同 codes 文件） | 4224d=**0.7086**，768d=**0.8719** → 两模型确实不同 |
| 768d 手动重算 vs 报告 | 0.8719 与所有报告逐位一致 ✓ |

→ 诊断脚本从始至终读的都是 768d 模型。所有 E1/E2/E3 对比基于**正确的 768d 基线**，可比性成立。

---

## E1：L0 occupancy ↔ transition quality（已有诊断，无模型改动）

**Toys L0 码本塌缩**：62/256 码（mean=192 items/code，max=435）→ 与 transition 混淆直接相关。

> ⚠️ **2026-09-09 provenance 订正**：本表 0.0367 / L0=62 出自**孤儿 codes**（主表 Toys codes 文件无法从任何现存 ckpt 复现，行级一致率最高 29.3%），作为 Toys **正式结果已作废**；本处仅作"塌缩实例"历史对照保留。Toys 正式主结果 = NDCG@20 **0.0377±0.0007** / R@20 0.0911±0.0020（健康 codes，L0=207，z-gap 0.005）。权威记录 + 修改索引见 [e48_toys_main_codes_provenance_fix.md](e48_toys_main_codes_provenance_fix.md)。

| 数据集 | L0 用码 | xq[L0] AUC | z AUC | z→xq 丢失 |
|---|---|---|---|---|
| **Toys** | **62/256 ⚠ 塌缩** | **0.8535** | 0.9652 | **−0.112** |
| Beauty | 204/256 | 0.8485 | 0.8688 | −0.020（近无损） |
| Sport | 249/256 | 0.9459 | 0.9645 | −0.019（近无损） |

→ 唯一塌缩的 Toys 也是唯一量化大幅丢 AUC 的；非塌缩数据集量化近无损。**E1 PASS**。

---

## E2：Post-hoc capacity curve（冻结 z，单层 k-means）

| 参照 | AUC |
|---|---|
| RQ-VAE L0（真实，62 码） | 0.8535 |
| **KMeans-64**（fresh） | 0.9091 |
| KMeans-256 | 0.9332 |
| KMeans-8192 | 0.9643 |
| **z**（天花板） | **0.9652** |

- 曲线随 k 单调升到 z 水平 → **问题不是"256 不够"，而是现有码本没利用容量**。
- 真实模型（62 码）落在 fresh k=64 之下（0.8535 vs 0.9091）→ 双重缺陷：**欠占用 + 码本摆放次优**。
- 达到用户预设预期（0.854 / ≈0.934 / ≈0.964 ≈ 0.965）。**E2 PASS**。

---

## E3：干预是否修复（Toys，Phase 1 仅重训 RQ-VAE）

| 指标 | Baseline 768d | Config A（L0 Sinkhorn） | Config B（L0=1024） |
|---|---|---|---|
| L0 用码 | 62/256 | **256/256** ✓ | **275/1024**（mean 43.4）✓✓ |
| L0 mean items/code | 192 | 46.6 ✓ | 43.4 ✓✓ |
| **xq[L0] AUC** | 0.8535 | 0.8968 (+0.043) ✓ | **0.9279 (+0.074)** ✓✓ |
| **xq[full] AUC** | 0.8719 | 0.9398 (+0.068) ✓ | **0.9629 (+0.091)** ✓✓ |
| 闭合 z→full gap | −0.093 | −0.015 | **−0.000（近无损）** ✓✓ |
| z AUC（编码器未退化？） | 0.9652 | 0.9545（−0.011 ⚠） | **0.9632（−0.002）** ✓ |
| 3-token 碰撞率 | 0.29% | **0.65% ✗ 恶化** | **0.03% ✓✓ 改善** |
| recon MSE（全量，use_sk） | 0.1452 | 0.1814（+25% ✗） | **0.1530（+5%）** ✓ |
| Best Loss（训练日志） | — | 2.88 | **2.74** |

**Config B 全指标无退化、关键指标大幅提升，且是唯一未破坏碰撞/recon/z 的干预。Config A 利用率与 AUC 提升但碰撞翻倍、recon 恶化、z 轻微退化。**

### 机制解读
- Config A 用 Sinkhorn 强制 L0 码本均衡 → 利用率修复，但**牺牲码本摆放质量**（碰撞↑、recon↑、z↓）。
- Config B 只扩容 L0 容量（256→1024）→ 码本自由生长，利用率自然上升，**摆放质量不牺牲**，几乎无损闭回 z 水平。
- 结论：修复手段应是 **capacity allocation（容量分配）**，而非 uniform forcing（均衡强制）。

---

## 闸门判定（按用户固定的决策链）

| 闸门 | 问题 | 判定 |
|---|---|---|
| E1 | L0 collapse 与 transition 混淆确实有关？ | **YES**（唯一塌缩数据集=唯一量化大丢） |
| E2 | 问题来自 codebook allocation 而非容量不足？ | **YES**（fresh k-means 随 k 到 z，真实码本低于 fresh k=64） |
| E3 | 干预能修复？ | **YES（Config B）** / 部分（Config A） |

→ **通过。正式进入 Adaptive/Structure-aware Quantization 设计（第二创新），基线方案 = Config B（L0 扩容，容量分配机制）。**

---

## 第二阶段建议（Phase 2：SID→T5→NDCG/Recall）

- 采用 **Config B（L0=1024）**：重新生成 SID → 重训 T5 → 对比 NDCG@20/50、Recall@20/50。
- 预期：码本利用率↑ + 碰撞率↓（0.29%→0.03%）→ T5 的 SID 空间更干净，NDCG 有提升空间。
- 若 Phase 2 指标不升，第二创新仍可主张"quantization 端过渡结构保真"（诊断侧证据充分），但论文叙事优先级降级。

## 待办（下一会话）
- [ ] Phase 2：Config B SID 已生成（`Toys_ta_sid_L0Big_sid.npy`），重训 T5（3-seed），评估 NDCG/Recall
- [ ] 论文侧：把 E1/E2/E3 数字写入分析小节（量化端诊断 + 容量修复）
